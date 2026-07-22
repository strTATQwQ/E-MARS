from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sensor_runtime.graph_handshake import (
    BRIDGE_GRAPH_REQUIREMENTS,
    INNER_HANDSHAKE_ARTIFACTS,
    ROLE_OWNED_QOS_CONTRACT,
    SIDECAR_GRAPH_REQUIREMENTS,
    ingest_observed_transforms,
    make_exact_ros_time,
    make_steady_control_clock,
    summarize_pending_batches,
    validate_graph_observation,
    validate_inner_handshake_documents,
)


def _complete(requirements: dict[str, dict]) -> dict[str, dict]:
    observed = {}
    for topic, item in requirements.items():
        qos = item["qos"]
        count = item["minimum_subscription_count"]
        observed[topic] = {
            "subscription_count": count,
            "subscriptions": [
                {
                    "reliability": qos["reliability"],
                    "durability": qos["durability"],
                    "history": qos["history"],
                    "depth": qos["minimum_depth"],
                }
                for _ in range(count)
            ],
        }
    return observed


@pytest.mark.parametrize("requirements", [SIDECAR_GRAPH_REQUIREMENTS, BRIDGE_GRAPH_REQUIREMENTS])
def test_graph_handshake_requires_every_topic_count_and_compatible_qos(requirements: dict) -> None:
    observed = _complete(requirements)
    assert validate_graph_observation(requirements, observed)["ready"] is True

    first = next(iter(requirements))
    missing = dict(observed)
    del missing[first]
    assert validate_graph_observation(requirements, missing)["ready"] is False

    zero = _complete(requirements)
    zero[first] = {"subscription_count": 0, "subscriptions": []}
    assert validate_graph_observation(requirements, zero)["ready"] is False

    wrong_qos = _complete(requirements)
    wrong_qos[first]["subscriptions"][0]["reliability"] = "RELIABLE"
    assert validate_graph_observation(requirements, wrong_qos)["ready"] is False


def test_inner_handshake_requires_three_roles_and_both_graph_artifacts() -> None:
    documents = {}
    for name, (status, role) in INNER_HANDSHAKE_ARTIFACTS.items():
        documents[name] = {
            "schema_version": 2 if status == "ROLE_READY" else 1,
            "status": status,
            "role": role,
            "ready": status == "GRAPH_READY",
        }
        if status == "ROLE_READY":
            documents[name]["owned_qos"] = ROLE_OWNED_QOS_CONTRACT[role]
        else:
            documents[name]["owned_qos_proof_required"] = True
            documents[name]["rmw_history_depth_unavailable_count"] = 1
    assert validate_inner_handshake_documents(documents)["ready"] is True
    del documents["bridge_graph_ready.json"]
    assert validate_inner_handshake_documents(documents)["ready"] is False


def test_sidecar_owned_tf_offer_is_reliable_for_nav2_and_audited_separately() -> None:
    tf_qos = ROLE_OWNED_QOS_CONTRACT["sensor_ros_sidecar"]["tf_dynamic"]
    assert tf_qos == {
        "reliability": "RELIABLE",
        "durability": "VOLATILE",
        "history": "KEEP_LAST",
        "depth": 100,
    }
    root = Path(__file__).resolve().parents[1]
    sidecar = (root / "sensor_runtime/ros_sidecar.py").read_text(encoding="utf-8")
    assert 'self.create_publisher(TFMessage, "/tf", tf_qos)' in sidecar


def test_inner_handshake_rejects_missing_or_mutated_owned_qos_proof() -> None:
    documents = {}
    for name, (status, role) in INNER_HANDSHAKE_ARTIFACTS.items():
        documents[name] = {
            "schema_version": 2 if status == "ROLE_READY" else 1,
            "status": status,
            "role": role,
            "ready": status == "GRAPH_READY",
            "owned_qos_proof_required": status == "GRAPH_READY",
            "rmw_history_depth_unavailable_count": 1 if status == "GRAPH_READY" else 0,
        }
        if status == "ROLE_READY":
            documents[name]["owned_qos"] = ROLE_OWNED_QOS_CONTRACT[role]
    documents["sidecar_role_ready.json"]["owned_qos"] = {}
    verdict = validate_inner_handshake_documents(documents)
    assert verdict["ready"] is False
    assert any("owned_qos" in error for error in verdict["errors"])


def test_rmw_unknown_history_depth_requires_exact_compatibility_policies() -> None:
    observed = _complete(BRIDGE_GRAPH_REQUIREMENTS)
    for row in observed.values():
        for endpoint in row["subscriptions"]:
            endpoint["history"] = "UNKNOWN"
            endpoint["depth"] = 0
    verdict = validate_graph_observation(BRIDGE_GRAPH_REQUIREMENTS, observed)
    assert verdict["ready"] is True
    assert verdict["owned_qos_proof_required"] is True
    assert verdict["rmw_history_depth_unavailable_count"] == len(observed)

    first = next(iter(observed.values()))["subscriptions"][0]
    first["reliability"] = "RELIABLE"
    assert validate_graph_observation(BRIDGE_GRAPH_REQUIREMENTS, observed)["ready"] is False
    first["reliability"] = "BEST_EFFORT"
    first["depth"] = 1
    assert validate_graph_observation(BRIDGE_GRAPH_REQUIREMENTS, observed)["ready"] is False


def test_clock_factories_bind_control_to_steady_and_tf_query_to_ros_time() -> None:
    calls: list[tuple[str, object]] = []

    class Types:
        STEADY_TIME = object()
        ROS_TIME = object()

    class FakeClock:
        def __init__(self, *, clock_type: object) -> None:
            calls.append(("clock", clock_type))

    class FakeTime:
        def __init__(self, *, nanoseconds: int, clock_type: object) -> None:
            calls.append((f"time:{nanoseconds}", clock_type))

    make_steady_control_clock(FakeClock, Types)
    make_exact_ros_time(FakeTime, Types, 123)
    assert calls == [("clock", Types.STEADY_TIME), ("time:123", Types.ROS_TIME)]
    with pytest.raises(ValueError):
        make_exact_ros_time(FakeTime, Types, True)


def test_every_ros_control_timer_passes_the_dedicated_control_clock() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "sensor_runtime/ros_sidecar.py",
        root / "sensor_runtime/downstream_recorder.py",
        root / "go2_sensor_bridge/go2_sensor_bridge/bridge_node.py",
    )
    observed = 0
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "create_timer":
                observed += 1
                keyword = next((item for item in node.keywords if item.arg == "clock"), None)
                assert keyword is not None
                assert isinstance(keyword.value, ast.Attribute) and keyword.value.attr == "_control_clock"
    assert observed >= 9


def test_observed_tf_callbacks_feed_the_same_buffer_without_an_implicit_listener() -> None:
    class FakeBuffer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object, str]] = []

        def set_transform(self, transform: object, authority: str) -> None:
            self.calls.append(("dynamic", transform, authority))

        def set_transform_static(self, transform: object, authority: str) -> None:
            self.calls.append(("static", transform, authority))

    buffer = FakeBuffer()
    dynamic = object()
    static = (object(), object())
    authority = "internnav_downstream_recorder_observed_tf"

    assert ingest_observed_transforms(
        buffer, [dynamic], authority=authority, static=False
    ) == 1
    assert ingest_observed_transforms(
        buffer, static, authority=authority, static=True
    ) == 2
    assert buffer.calls == [
        ("dynamic", dynamic, authority),
        ("static", static[0], authority),
        ("static", static[1], authority),
    ]

    with pytest.raises(ValueError, match="observed TF ingestion inputs are invalid"):
        ingest_observed_transforms(buffer, [], authority=authority, static=False)

    root = Path(__file__).resolve().parents[1]
    recorder = (root / "sensor_runtime/downstream_recorder.py").read_text(
        encoding="utf-8"
    )
    assert "TransformListener" not in recorder
    assert "ingest_observed_transforms" in recorder


def test_pending_batch_diagnostic_reports_exact_missing_parts_without_relaxing_gate() -> None:
    rows = summarize_pending_batches(
        {
            100: {
                "created": 4.5,
                "parts": {"clock": True, "identity": True, "tf": True},
                "details": {
                    "identity": {"generation": 0, "sequence": 0, "render_id": 10}
                },
            },
            200: {
                "created": 4.9,
                "parts": {"clock": True, "identity": True},
                "details": {
                    "identity": {"generation": 0, "sequence": 1, "render_id": 20}
                },
            },
        },
        {"clock", "identity", "tf"},
        now_monotonic=5.0,
    )
    assert [row["stamp_ns"] for row in rows] == [100, 200]
    assert rows[0]["age_sec"] == pytest.approx(0.5)
    assert rows[0]["missing_parts"] == ["tf_static"]
    assert rows[1]["missing_parts"] == ["tf"]
    assert rows[0]["unexpected_parts"] == []

    with pytest.raises(ValueError, match="positive integer"):
        summarize_pending_batches(
            {0: {"created": 0.0, "parts": {}, "details": {}}},
            {"clock"},
            now_monotonic=1.0,
        )


def test_ros_startup_avoids_jazzy_parameter_and_node_attribute_collisions() -> None:
    root = Path(__file__).resolve().parents[1]
    sidecar = (root / "sensor_runtime/ros_sidecar.py").read_text(encoding="utf-8")
    recorder = (root / "sensor_runtime/downstream_recorder.py").read_text(
        encoding="utf-8"
    )
    assert 'if not self.has_parameter("use_sim_time"):' in sidecar
    assert 'self.declare_parameter("use_sim_time", False)' in sidecar
    assert 'self.create_subscription(Clock, "/clock", self._on_clock, qos)' in recorder
    assert "def _on_clock(self, message: Clock)" in recorder
    assert "def _clock(self, message: Clock)" not in recorder
