from __future__ import annotations

from copy import deepcopy
from typing import Any


def apply_benchmark_mode_config(base_config: dict[str, Any], payload: dict[str, Any] | None) -> dict[str, Any]:
    cfg = deepcopy(base_config or {})
    payload = payload or {}
    mode_cfg = payload.get("mode_config") if isinstance(payload.get("mode_config"), dict) else payload

    if "use_step" in mode_cfg:
        cfg.setdefault("step", {})["enabled"] = bool(mode_cfg["use_step"])
    if "use_omninav" in mode_cfg:
        cfg.setdefault("omninav", {})["enabled"] = bool(mode_cfg["use_omninav"])
    if "use_internnav" in mode_cfg:
        cfg.setdefault("internnav", {})["enabled"] = bool(mode_cfg["use_internnav"])

    step_policy = str(mode_cfg.get("step_policy", "event_triggered"))
    if step_policy == "every_decision":
        cfg.setdefault("step", {})["min_interval_sec"] = float(mode_cfg.get("step_min_interval_sec", 1.0))
        cfg.setdefault("runtime_mode", {})["step_only_period_sec"] = float(mode_cfg.get("step_only_period_sec", 1.5))
    elif step_policy == "never":
        cfg.setdefault("step", {})["enabled"] = False

    pending_policy = str(mode_cfg.get("pending_policy", "auto"))
    cfg.setdefault("pending_policy", {})["benchmark_policy"] = pending_policy
    if pending_policy == "stop":
        cfg["pending_policy"]["allow_move_while_step"] = False
    elif pending_policy == "move_slow":
        cfg["pending_policy"]["allow_move_while_step"] = True
        cfg["pending_policy"]["stop_if_step_multimodal"] = False
    elif pending_policy == "safe_scan":
        cfg["pending_policy"]["allow_move_while_step"] = False

    if "stale_gate_enabled" in mode_cfg:
        cfg.setdefault("runtime_mode", {})["stale_gate_enabled"] = bool(mode_cfg["stale_gate_enabled"])
    if "step_triggers" in mode_cfg:
        value = mode_cfg.get("step_triggers") or []
        if isinstance(value, str):
            value = [value]
        cfg.setdefault("runtime_mode", {})["step_triggers"] = [str(item) for item in value]
    if "step_roles_only" in mode_cfg:
        cfg.setdefault("step", {})["roles_only"] = bool(mode_cfg["step_roles_only"])
    if "external_step_triggers_only" in mode_cfg:
        cfg.setdefault("runtime_mode", {})["external_step_triggers_only"] = bool(mode_cfg["external_step_triggers_only"])

    mission_timeout = _mission_timeout_from_mode(mode_cfg)
    if mission_timeout is not None:
        mission_cfg = cfg.setdefault("mission", {})
        current_timeout = _positive_float(mission_cfg.get("max_duration_sec"))
        if current_timeout is None or mission_timeout > current_timeout:
            mission_cfg["max_duration_sec"] = mission_timeout
        cfg.setdefault("runtime_mode", {})["mission_max_duration_sec"] = mission_cfg["max_duration_sec"]

    cfg.setdefault("runtime_mode", {})["mode"] = str(payload.get("mode") or mode_cfg.get("mode") or "")
    cfg.setdefault("runtime_mode", {})["mode_config"] = dict(mode_cfg)
    return cfg


def mode_payload_from_json(raw: str) -> dict[str, Any]:
    import json

    try:
        value = json.loads(raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _mission_timeout_from_mode(mode_cfg: dict[str, Any]) -> float | None:
    explicit = _positive_float(mode_cfg.get("mission_max_duration_sec"))
    if explicit is not None:
        return explicit
    task_timeout = _positive_float(mode_cfg.get("task_timeout_sec"))
    if task_timeout is None:
        task_timeout = _positive_float(mode_cfg.get("timeout_sec"))
    if task_timeout is None:
        return None
    margin = _positive_float(mode_cfg.get("mission_timeout_margin_sec"))
    if margin is None:
        margin = 5.0
    return task_timeout + margin
