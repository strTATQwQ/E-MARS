from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient
except Exception:  # FastAPI runtime does not require the optional test client.
    TestClient = None
from slow_planner_frontend.control import (
    MISSION_ROUTE,
    ControlPlaneConfig,
    MissionCommandOutbox,
    MissionControlStore,
)
from slow_planner_frontend.state import FrontendStateStore


ARM_PHRASE = "ARM STRICT REAL GO2"


def mapping(tmp_path: Path, *, enable_dispatch: bool = False) -> dict:
    return {
        "frontend": {
            "control_plane": {
                "enabled": True,
                "navigation_dispatch_enabled": enable_dispatch,
                "human_arm_allowed": enable_dispatch,
                "motion_bridge_enabled": False,
                "command_dir": str(tmp_path / "commands"),
                "state_path": str(tmp_path / "control_state.json"),
                "navigation_state_path": str(tmp_path / "navigation_state.json"),
                "mission_gateway_state_path": str(
                    tmp_path / "mission_gateway_state.json"
                ),
                "audit_path": str(tmp_path / "control_audit.jsonl"),
                "instruction_topic": "/user_instruction",
                "canonical_instruction_topic": "/internvla/mission/canonical",
                "cancel_topic": "/mission/cancel_json",
                "estop_topic": "/operator/estop",
                "mission_deadline_s": 30.0,
                "operator_arm_phrase": ARM_PHRASE,
                "required_components": ["internvla_client_node", "control_mux"],
            }
        }
    }


def client(tmp_path: Path, *, enable_dispatch: bool = False):
    if TestClient is None:
        pytest.skip("FastAPI TestClient dependencies are unavailable")
    from slow_planner_frontend.app import create_app

    config = ControlPlaneConfig.from_mapping(
        mapping(tmp_path, enable_dispatch=enable_dispatch)
    )
    control = MissionControlStore(config)
    return TestClient(
        create_app(store=FrontendStateStore(), mission_control=control)
    )


def healthy_navigation_state(tmp_path: Path) -> None:
    (tmp_path / "navigation_state.json").write_text(
        json.dumps(
            {
                "ready": True,
                "ros_fresh": True,
                "sensors_fresh": True,
                "tf_fresh": True,
                "watchdog_healthy": True,
                "control_mux_healthy": True,
                "estop_latched": False,
                "components": {
                    "internvla_client_node": True,
                    "control_mux": True,
                },
            }
        ),
        encoding="utf-8",
    )


def test_chinese_mission_is_staged_for_step3_without_raw_internvla_route(
    tmp_path: Path,
) -> None:
    api = client(tmp_path)
    response = api.post(
        "/api/v1/missions",
        json={"instruction": "穿过门口，在红色椅子旁边停下", "dispatch": True},
    )
    assert response.status_code == 202
    mission = response.json()["mission"]
    assert mission["route"] == MISSION_ROUTE
    assert mission["internvla_raw_instruction_allowed"] is False
    assert mission["status"] == "STAGED_DISPATCH_BLOCKED"
    assert "raw_instruction" not in mission
    assert not (tmp_path / "commands").exists()
    audit = (tmp_path / "control_audit.jsonl").read_text(encoding="utf-8")
    assert "穿过门口" not in audit


def test_arm_is_fail_closed_while_real_motion_is_disabled(tmp_path: Path) -> None:
    api = client(tmp_path)
    response = api.post("/api/v1/arm", json={"phrase": ARM_PHRASE})
    assert response.status_code == 423
    assert response.json()["detail"]["code"] == "ARM_DISABLED"
    state = api.get("/api/v1/state").json()["control"]
    assert state["armed"] is False
    assert state["estop_latched"] is True


def test_even_armed_dispatch_requires_fresh_navigation_state(tmp_path: Path) -> None:
    api = client(tmp_path, enable_dispatch=True)
    cleared = api.post(
        "/api/v1/estop", json={"action": "clear", "phrase": ARM_PHRASE}
    )
    assert cleared.status_code == 200
    armed = api.post("/api/v1/arm", json={"phrase": ARM_PHRASE})
    assert armed.status_code == 200
    blocked = api.post(
        "/api/v1/missions",
        json={"instruction": "Go to the doorway and stop.", "dispatch": True},
    )
    assert blocked.status_code == 202
    assert "navigation_state_unavailable" in blocked.json()["mission"][
        "dispatch_blockers"
    ]
    assert not (tmp_path / "commands").exists()

    healthy_navigation_state(tmp_path)
    accepted = api.post(
        "/api/v1/missions",
        json={"instruction": "前往门口并停下", "dispatch": True},
    )
    assert accepted.status_code == 202
    mission = accepted.json()["mission"]
    assert mission["status"] == "QUEUED_FOR_STEP3"
    command = json.loads(
        (tmp_path / "commands" / f"mission-{mission['mission_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert command["raw_instruction"] == "前往门口并停下"
    assert command["route"] == MISSION_ROUTE
    assert command["internvla_raw_instruction_allowed"] is False


def test_post_routes_reject_cross_origin_browser_requests(tmp_path: Path) -> None:
    api = client(tmp_path)
    response = api.post(
        "/api/v1/missions",
        json={"instruction": "Stop by the door."},
        headers={"Origin": "https://attacker.invalid", "Host": "panel.local"},
    )
    assert response.status_code == 403


def test_cancel_is_idempotent_and_never_publishes_velocity(tmp_path: Path) -> None:
    api = client(tmp_path)
    mission = api.post(
        "/api/v1/missions", json={"instruction": "Stop by the door."}
    ).json()["mission"]
    first = api.post(f"/api/v1/missions/{mission['mission_id']}/cancel")
    second = api.post(f"/api/v1/missions/{mission['mission_id']}/cancel")
    assert first.status_code == second.status_code == 200
    assert first.json()["mission"]["status"] == "CANCELED"
    source = Path(__file__).resolve().parents[2] / "slow_planner_frontend/control.py"
    body = source.read_text(encoding="utf-8")
    assert "cmd_vel" not in body
    assert "NavigationCommand" not in body


def test_outbox_publishes_step3_mission_exactly_once_across_restart(
    tmp_path: Path,
) -> None:
    config = ControlPlaneConfig.from_mapping(
        mapping(tmp_path, enable_dispatch=True)
    )
    now_ns = 123_000_000_000
    config.command_dir.mkdir(parents=True)
    command = {
        "route": MISSION_ROUTE,
        "mission_id": "m-12345678",
        "raw_instruction": "去门口",
        "internvla_raw_instruction_allowed": False,
        "config_sha256": config.config_sha256,
        "status": "QUEUED_FOR_STEP3",
        "deadline_wall_monotonic_ns": now_ns + 1_000_000,
    }
    (config.command_dir / "mission-m-12345678.json").write_text(
        json.dumps(command, ensure_ascii=False), encoding="utf-8"
    )
    published: list[dict] = []
    first = MissionCommandOutbox(
        config,
        publish_mission=published.append,
        publish_cancel=lambda _value: None,
        publish_estop=lambda _value: None,
        monotonic_ns=lambda: now_ns,
    )
    assert first.drain() == [
        {"filename": "mission-m-12345678.json", "kind": "mission"}
    ]
    second = MissionCommandOutbox(
        config,
        publish_mission=published.append,
        publish_cancel=lambda _value: None,
        publish_estop=lambda _value: None,
        monotonic_ns=lambda: now_ns,
    )
    assert second.drain() == []
    assert published == [command]


def test_outbox_rejects_expired_mission_without_publishing(tmp_path: Path) -> None:
    config = ControlPlaneConfig.from_mapping(
        mapping(tmp_path, enable_dispatch=True)
    )
    config.command_dir.mkdir(parents=True)
    command = {
        "route": MISSION_ROUTE,
        "mission_id": "m-87654321",
        "raw_instruction": "到门口",
        "internvla_raw_instruction_allowed": False,
        "config_sha256": config.config_sha256,
        "status": "QUEUED_FOR_STEP3",
        "deadline_wall_monotonic_ns": 9,
    }
    (config.command_dir / "mission-m-87654321.json").write_text(
        json.dumps(command, ensure_ascii=False), encoding="utf-8"
    )
    published: list[dict] = []
    outbox = MissionCommandOutbox(
        config,
        publish_mission=published.append,
        publish_cancel=lambda _value: None,
        publish_estop=lambda _value: None,
        monotonic_ns=lambda: 10,
    )
    assert outbox.drain() == [
        {
            "filename": "mission-m-87654321.json",
            "kind": "rejected",
            "reason": "ValueError",
        }
    ]
    assert published == []


def test_control_projects_only_step3_canonical_gateway_state(tmp_path: Path) -> None:
    config = ControlPlaneConfig.from_mapping(mapping(tmp_path))
    store = MissionControlStore(config)
    config.mission_gateway_state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "CANONICAL_READY",
                "identity": "real::m-12345678::0::0",
                "raw_instruction": "不得出现在前端",
                "mission": {
                    "mission_id": "m-12345678",
                    "episode_id": "real-m-12345678",
                    "reset_generation": 0,
                    "sequence_id": 0,
                    "source_language": "zh",
                    "canonical_instruction": "Go to the doorway and stop.",
                    "target_description": "the doorway",
                    "constraints": ["Stop at the doorway"],
                    "confidence": 0.91,
                    "config_sha256": config.config_sha256,
                    "internvla_raw_instruction_allowed": False,
                },
                "internvla": {
                    "raw_instruction_allowed": False,
                    "canonical_instruction": "Go to the doorway and stop.",
                },
                "step3": {
                    "route": MISSION_ROUTE,
                    "model_variant": "Step3-VL-10B",
                    "end_to_end_ms": 8100.0,
                    "raw_text": "private generation",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    public = store.state()["gateway"]
    assert public["canonical_ready"] is True
    assert public["mission"]["canonical_instruction"] == (
        "Go to the doorway and stop."
    )
    serialized = json.dumps(public, ensure_ascii=False)
    assert "不得出现在前端" not in serialized
    assert "private generation" not in serialized
