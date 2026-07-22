#!/usr/bin/env python3
"""Validate one T4.5 profile and materialize its Nav2 recovery speed overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from t4_completion.recovery.config import load_recovery_config  # noqa: E402


def canonical_sha256(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Nav2 recovery parameter is not unique: {old.strip()}")
    return text.replace(old, new)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--profile", type=Path, required=True)
    value.add_argument("--nav2-input", type=Path, required=True)
    value.add_argument("--nav2-output", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--format", choices=("json", "fields"), default="json")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        profile = load_recovery_config(args.profile)
        if args.nav2_output.exists() or args.manifest.exists():
            raise FileExistsError("recovery runtime outputs must be fresh")
        source = args.nav2_input.read_text(encoding="utf-8")
        output = replace_once(
            source,
            "    max_rotational_vel: 1.0",
            f"    max_rotational_vel: {profile.scan_angular_speed_rps:.8g}",
        )
        output = replace_once(
            output,
            "    min_rotational_vel: 0.4",
            "    min_rotational_vel: 0.15",
        )
        output = replace_once(
            output,
            "    rotational_acc_lim: 3.2",
            "    rotational_acc_lim: 0.7",
        )
        args.nav2_output.parent.mkdir(parents=True, exist_ok=True)
        args.nav2_output.write_text(output, encoding="utf-8", newline="\n")
        profile_sha256 = canonical_sha256(args.profile)
        nav2_sha256 = hashlib.sha256(args.nav2_output.read_bytes()).hexdigest()
        summary = {
            "schema_version": 1,
            "status": "RECOVERY_RUNTIME_READY",
            "runtime_policy": "completion_sim",
            "runtime_target": "isaac_simulation",
            "profile_id": profile.profile_id,
            "profile_sha256": profile_sha256,
            "nav2_source": args.nav2_input.as_posix(),
            "nav2_output": args.nav2_output.as_posix(),
            "nav2_output_sha256": nav2_sha256,
            "scan_angular_speed_rps": profile.scan_angular_speed_rps,
            "scan_yaw_rad": profile.scan_yaw_rad,
            "short_backup_enabled": False,
            "retreat_deviation": (
                "disabled_until_rear_clearance_is_observed"
                if profile.retreat_enabled
                else None
            ),
            "parameters": {
                "progress_horizon_sec": profile.no_progress_window_sec,
                "minimum_progress_m": profile.minimum_displacement_m,
                "oscillation_travel_m": profile.loop_min_travel_m,
                "recovery_cooldown_sec": profile.cooldown_sec,
                "maximum_recoveries_per_episode": (
                    profile.maximum_recoveries_per_episode
                ),
                "maximum_recovery_duration_sec": (
                    profile.maximum_recovery_duration_sec
                ),
                "safety_freshness_timeout_sec": (
                    profile.safety_freshness_timeout_sec
                ),
            },
        }
        args.manifest.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if args.format == "fields":
            fields = (
                profile.profile_id,
                profile_sha256,
                nav2_sha256,
                profile.no_progress_window_sec,
                profile.minimum_displacement_m,
                profile.loop_min_travel_m,
                profile.cooldown_sec,
                profile.maximum_recoveries_per_episode,
                profile.scan_yaw_rad,
                profile.scan_angular_speed_rps,
                profile.safety_freshness_timeout_sec,
                profile.maximum_recovery_duration_sec,
            )
            for field in fields:
                print(field)
        else:
            print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            json.dumps(
                {"status": "ERROR", "error": str(exc), "resource_use": "none"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
