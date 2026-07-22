"""Isaac-only bounded relay that keeps Collision Monitor observational.

Collision Monitor receives the Nav2 command but publishes to an unconsumed
diagnostic topic.  This relay independently enforces immutable velocity,
command-age, and simulation-estop bounds before publishing ``/cmd_vel_safe``.
It is intentionally unavailable outside the managed completion_sim companion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple


MAX_LINEAR_MPS = 0.25
MAX_ANGULAR_RPS = 1.0
COMMAND_TIMEOUT_SEC = 0.30


@dataclass(frozen=True)
class RelayDecision:
    linear_x: float
    angular_z: float
    stopped: bool
    reason: str


def bounded_command(
    linear_x: float,
    angular_z: float,
    command_age_sec: float,
    estop_seen: bool,
    estop_active: bool,
) -> RelayDecision:
    values = (linear_x, angular_z, command_age_sec)
    if any(not math.isfinite(float(value)) for value in values):
        return RelayDecision(0.0, 0.0, True, "nonfinite_input")
    if not estop_seen:
        return RelayDecision(0.0, 0.0, True, "estop_unobserved")
    if estop_active:
        return RelayDecision(0.0, 0.0, True, "simulation_estop")
    if command_age_sec < 0.0 or command_age_sec > COMMAND_TIMEOUT_SEC:
        return RelayDecision(0.0, 0.0, True, "stale_command")
    return RelayDecision(
        max(-MAX_LINEAR_MPS, min(MAX_LINEAR_MPS, float(linear_x))),
        max(-MAX_ANGULAR_RPS, min(MAX_ANGULAR_RPS, float(angular_z))),
        False,
        "bounded_forward",
    )


def interlock_state(
    stop_seen: bool,
    stop_active: bool,
    motion_seen: bool,
    motion_enabled: bool,
) -> tuple[bool, bool]:
    """Combine model STOP and typed Nav2 motion-enable into one sim interlock."""
    seen = bool(stop_seen or motion_seen)
    active = bool(
        (stop_seen and stop_active) or (motion_seen and not motion_enabled)
    )
    return seen, active


def validate_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("warn relay evidence is missing or unsafe")
    reason_counts: dict[str, int] = {}
    expected_sequence = 1
    nonzero_output_count = 0
    bounded_forward_count = 0
    maximum_abs_linear = 0.0
    maximum_abs_angular = 0.0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if len(raw_line.encode("utf-8")) > 65536:
                raise RuntimeError("warn relay evidence line is too large")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"warn relay evidence line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(record, dict) or record.get("schema_version") != 1:
                raise RuntimeError("warn relay evidence schema changed")
            if int(record.get("sequence", -1)) != expected_sequence:
                raise RuntimeError("warn relay evidence sequence is not contiguous")
            expected_sequence += 1
            reason = str(record.get("reason", ""))
            if reason not in {
                "startup",
                "bounded_forward",
                "stale_command",
                "estop_unobserved",
                "simulation_estop",
                "nonfinite_input",
                "shutdown",
            }:
                raise RuntimeError("warn relay evidence reason is unknown")
            linear = float(record.get("linear_x", math.nan))
            angular = float(record.get("angular_z", math.nan))
            if not math.isfinite(linear) or not math.isfinite(angular):
                raise RuntimeError("warn relay evidence contains nonfinite output")
            if abs(linear) > MAX_LINEAR_MPS + 1e-12:
                raise RuntimeError("warn relay linear output exceeded its bound")
            if abs(angular) > MAX_ANGULAR_RPS + 1e-12:
                raise RuntimeError("warn relay angular output exceeded its bound")
            stopped = record.get("stopped")
            if not isinstance(stopped, bool):
                raise RuntimeError("warn relay stopped flag is not boolean")
            if stopped and (abs(linear) > 1e-12 or abs(angular) > 1e-12):
                raise RuntimeError("warn relay emitted motion while stopped")
            if reason == "bounded_forward" and stopped:
                raise RuntimeError("bounded relay output is incorrectly stopped")
            if reason != "bounded_forward" and not stopped:
                raise RuntimeError("non-forward relay state is not stopped")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            bounded_forward_count += int(reason == "bounded_forward")
            nonzero_output_count += int(abs(linear) > 1e-4 or abs(angular) > 1e-4)
            maximum_abs_linear = max(maximum_abs_linear, abs(linear))
            maximum_abs_angular = max(maximum_abs_angular, abs(angular))
    record_count = expected_sequence - 1
    if record_count < 2 or reason_counts.get("startup", 0) != 1:
        raise RuntimeError("warn relay evidence is incomplete")
    if bounded_forward_count < 1:
        raise RuntimeError("warn relay never observed a bounded command")
    warnings = []
    if nonzero_output_count < 1:
        # A frozen ablation arm may legitimately command only zero motion (for
        # example, when removing history collapses the policy to STOP).  This
        # is functional evidence, not a violation of the immutable velocity
        # bounds or simulation-estop interlock.  Preserve it as an explicit
        # completion_sim warning so the online batch can continue.
        warnings.append("no_nonzero_bounded_command_observed")
    return {
        "schema_version": 1,
        "status": "PASS",
        "record_count": record_count,
        "bounded_forward_count": bounded_forward_count,
        "nonzero_output_count": nonzero_output_count,
        "maximum_abs_linear_mps": maximum_abs_linear,
        "maximum_abs_angular_rps": maximum_abs_angular,
        "reason_counts": dict(sorted(reason_counts.items())),
        "warnings": warnings,
    }


def require_simulation_boundary() -> None:
    if os.name != "posix":
        raise RuntimeError("warn-only relay requires the POSIX simulation host")
    if os.environ.get("INTERNNAV_RUNTIME_POLICY") != "completion_sim":
        raise RuntimeError("warn-only relay requires completion_sim")
    if os.environ.get("INTERNNAV_SIMULATION_TARGET") != "isaac":
        raise RuntimeError("warn-only relay rejects non-Isaac targets")
    if os.environ.get("INTERNNAV_T4_MAP_COMPANION_ACK") != "1":
        raise RuntimeError("warn-only relay requires the managed companion ack")


def run_ros(result_dir: Path, ros_args: List[str]) -> int:
    require_simulation_boundary()
    result_dir.mkdir(parents=True, exist_ok=True)
    evidence = result_dir / "warn_only_relay.jsonl"
    if evidence.exists():
        raise FileExistsError("refusing to append warn-only relay evidence")

    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import Bool

    class WarnOnlyRelay(Node):
        def __init__(self) -> None:
            super().__init__("internnav_completion_warn_only_relay")
            self.raw: Tuple[float, float] = (0.0, 0.0)
            self.raw_monotonic = 0.0
            self.estop_seen = False
            self.estop_active = False
            self.motion_seen = False
            self.motion_enabled = False
            self.monitor: Tuple[float, float] = (0.0, 0.0)
            self.last_reason = "startup"
            self.sequence = 0
            self.stream = evidence.open("x", encoding="utf-8", newline="\n")
            # These Nav2 data-plane names are deliberately relative.  T4's
            # empty namespace still resolves them to the historical root
            # topics, while each T5 lane resolves them below its own namespace.
            self.publisher = self.create_publisher(Twist, "cmd_vel_safe", 10)
            self.create_subscription(Twist, "cmd_vel_nav", self.on_raw, 10)
            self.create_subscription(
                Twist,
                "completion_sim/collision_monitor/warn_only_cmd_vel",
                self.on_monitor,
                10,
            )
            self.create_subscription(Bool, "/internvla/stop", self.on_estop, 10)
            self.create_subscription(
                Bool,
                "/internvla/nav2_motion_enabled",
                self.on_motion_enabled,
                10,
            )
            self.create_timer(0.02, self.tick)
            self.record("startup", RelayDecision(0.0, 0.0, True, "startup"))

        def on_raw(self, message: Twist) -> None:
            self.raw = (float(message.linear.x), float(message.angular.z))
            self.raw_monotonic = time.monotonic()

        def on_monitor(self, message: Twist) -> None:
            self.monitor = (float(message.linear.x), float(message.angular.z))

        def on_estop(self, message: Bool) -> None:
            self.estop_seen = True
            self.estop_active = bool(message.data)

        def on_motion_enabled(self, message: Bool) -> None:
            self.motion_seen = True
            self.motion_enabled = bool(message.data)

        def record(self, event: str, decision: RelayDecision) -> None:
            self.sequence += 1
            payload = {
                "schema_version": 1,
                "event": event,
                "sequence": self.sequence,
                "reason": decision.reason,
                "stopped": decision.stopped,
                "linear_x": decision.linear_x,
                "angular_z": decision.angular_z,
                "monitor_differs_from_raw": self.monitor != self.raw,
                "monotonic_sec": time.monotonic(),
            }
            self.stream.write(json.dumps(payload, sort_keys=True) + "\n")
            self.stream.flush()

        def tick(self) -> None:
            age = (
                time.monotonic() - self.raw_monotonic
                if self.raw_monotonic > 0.0
                else float("inf")
            )
            interlock_seen, interlock_active = interlock_state(
                self.estop_seen,
                self.estop_active,
                self.motion_seen,
                self.motion_enabled,
            )
            decision = bounded_command(
                self.raw[0], self.raw[1], age, interlock_seen, interlock_active
            )
            output = Twist()
            output.linear.x = decision.linear_x
            output.angular.z = decision.angular_z
            self.publisher.publish(output)
            if decision.reason != self.last_reason:
                self.record("state_change", decision)
                self.last_reason = decision.reason

        def close(self) -> None:
            stopped = Twist()
            for _ in range(3):
                self.publisher.publish(stopped)
            self.record("shutdown", RelayDecision(0.0, 0.0, True, "shutdown"))
            self.stream.close()

    rclpy.init(args=ros_args)
    node = WarnOnlyRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--validate-evidence", type=Path)
    parser.add_argument("--summary", type=Path)
    args, ros_args = parser.parse_known_args(argv)
    if args.validate_evidence is not None:
        if args.result_dir is not None or args.summary is None or ros_args:
            parser.error("evidence validation requires only --validate-evidence and --summary")
        try:
            payload = validate_evidence(args.validate_evidence)
        except Exception as exc:
            payload = {
                "schema_version": 1,
                "status": "FAIL",
                "error": str(exc),
            }
            args.summary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(payload, sort_keys=True), file=sys.stderr)
            return 2
        args.summary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.result_dir is None or args.summary is not None:
        parser.error("ROS mode requires --result-dir")
    try:
        return run_ros(args.result_dir, ros_args)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
