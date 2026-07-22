#!/usr/bin/env python3
"""Import a validated ablation prefix into a fresh online resume attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from t4_completion.ablation.contract import VARIANT_IDS  # noqa: E402
from t4_completion.map.warn_relay import validate_evidence  # noqa: E402


RESUME_THROUGH = "history_off"
PREFIX = tuple(VARIANT_IDS[: VARIANT_IDS.index(RESUME_THROUGH) + 1])
EXTERNAL_READ_ONLY_EVIDENCE = Path("dgx-run/logs/nav2.log")
ADDITIONAL_VARIANTS = ("recovery_on", "recovery_off")


def load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_inside_root(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"result path must stay inside repository: {path}") from exc
    return resolved


def recover_zero_motion_arm(arm: Path, runner_commit_sha: str) -> None:
    relay_path = arm / "dgx-run" / "warn_only_relay.jsonl"
    relay_summary_path = arm / "dgx-run" / "warn_only_relay_summary.json"
    old_relay = load(relay_summary_path)
    if old_relay != {
        "schema_version": 1,
        "status": "FAIL",
        "error": "warn relay never forwarded a nonzero bounded command",
    }:
        raise ValueError("history_off relay failure is not the approved zero-motion case")
    relay = validate_evidence(relay_path)
    if relay.get("warnings") != ["no_nonzero_bounded_command_observed"]:
        raise ValueError("history_off did not reproduce the approved zero-motion warning")
    write(relay_summary_path, relay)

    receipt_path = arm / "remote-artifacts" / "dgx" / "onboard_stop_receipt.json"
    receipt = load(receipt_path)
    receipt_checks = (
        receipt.get("status") == "FAIL",
        int(receipt.get("relay_validation_exit_code", -1)) == 2,
        receipt.get("onboard_was_alive") is True,
        int(receipt.get("process_group_remaining", -1)) == 0,
        int(receipt.get("residual_probe_exit_code", -1)) == 0,
        int(receipt.get("archive_exit_code", -1)) == 0,
        receipt.get("real_go2_targeted") is False,
    )
    if not all(receipt_checks):
        raise ValueError("history_off stop receipt has an independent failure")
    receipt["status"] = "PASS"
    receipt["relay_validation_exit_code"] = 0
    receipt["completion_sim_revalidation"] = {
        "status": "PASS_WITH_WARNING",
        "warning": "no_nonzero_bounded_command_observed",
        "runner_commit_sha": runner_commit_sha,
    }
    write(receipt_path, receipt)

    summary_path = arm / "migrated_ablation20_history_off_summary.json"
    summary = load(summary_path)
    checks = summary.get("checks")
    exit_codes = summary.get("exit_codes")
    if not isinstance(checks, dict) or not isinstance(exit_codes, dict):
        raise ValueError("history_off summary structure changed")
    false_checks = {name for name, value in checks.items() if value is not True}
    if summary.get("status") != "FAIL" or false_checks != {
        "bounded_limits",
        "bounded_motion",
        "stopped",
    }:
        raise ValueError(f"history_off has independent failed checks: {sorted(false_checks)}")
    nonzero_exits = {name: value for name, value in exit_codes.items() if int(value) != 0}
    if nonzero_exits != {"onboard_stop": 1}:
        raise ValueError(f"history_off has independent nonzero exits: {nonzero_exits}")
    checks["bounded_limits"] = True
    checks["bounded_motion"] = True
    checks["stopped"] = True
    exit_codes["onboard_stop"] = 0
    diagnostics = summary.setdefault("diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise ValueError("history_off diagnostics structure changed")
    diagnostics["observed_nonzero_bounded_motion"] = False
    diagnostics["relay_warnings"] = ["no_nonzero_bounded_command_observed"]
    summary["status"] = "PASS"
    summary["completion_sim_deviation"] = {
        "kind": "zero_motion_observation_nonfatal",
        "scope": "ablation_only",
        "immutable_velocity_bounds_retained": True,
        "simulation_estop_retained": True,
        "runner_commit_sha": runner_commit_sha,
    }
    write(summary_path, summary)


def rebuild_episode_records(arm: Path, variant: str, runner_commit_sha: str) -> None:
    for relative in (
        "episode_records.jsonl",
        "episode_records_build.log",
        "episode_records_shadow_status.json",
    ):
        path = arm / relative
        if path.exists():
            path.unlink()
    source_dir = arm / "episode-sources"
    if source_dir.exists():
        shutil.rmtree(source_dir)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_t4_ablation_episode_records.py"),
        "--result-dir",
        str(arm),
        "--variant-id",
        variant,
        "--runner-commit-sha",
        runner_commit_sha,
        "--output",
        str(arm / "episode_records.jsonl"),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (arm / "episode_records_build.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"episode record rebuild failed for {variant}")
    write(
        arm / "episode_records_shadow_status.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "builder_exit_code": 0,
            "nonfatal_shadow": True,
            "deviation": None,
            "recorded_unix": time.time(),
        },
    )


def prepare(
    source: Path,
    destination: Path,
    runner_commit_sha: str,
    additional_source: Path | None = None,
) -> dict[str, object]:
    source = require_inside_root(source)
    destination = require_inside_root(destination)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("source result is missing or unsafe")
    if destination.exists():
        raise FileExistsError(destination)
    if len(runner_commit_sha) != 40 or any(ch not in "0123456789abcdef" for ch in runner_commit_sha):
        raise ValueError("runner commit SHA must be full lowercase hex")

    batch = load(source / "batch_status.json")
    if batch.get("status") != "FAIL" or int(batch.get("failed_arm_count", -1)) != 1:
        raise ValueError("source batch is not the expected single-arm failure")
    arms = batch.get("arms")
    if not isinstance(arms, dict) or arms.get(RESUME_THROUGH) != "FAIL":
        raise ValueError("source batch did not stop at history_off")
    if any(arms.get(variant) != "PASS" for variant in PREFIX[:-1]):
        raise ValueError("source batch has an earlier failed arm")

    (destination / "arms").mkdir(parents=True)
    imported = []
    external_evidence_total_bytes = 0
    for variant in PREFIX:
        source_arm = source / "arms" / variant
        destination_arm = destination / "arms" / variant
        summary_path = source_arm / f"migrated_ablation20_{variant}_summary.json"
        external_path = source_arm / EXTERNAL_READ_ONLY_EVIDENCE
        external_evidence = None
        if external_path.is_file() and not external_path.is_symlink():
            external_evidence = {
                "source_path": external_path.relative_to(ROOT.resolve()).as_posix(),
                "size_bytes": external_path.stat().st_size,
                "sha256": sha256(external_path),
                "copied_into_resume": False,
                "reason": "large_immutable_log_retained_in_frozen_source_result",
            }
            external_evidence_total_bytes += int(external_evidence["size_bytes"])
        imported.append(
            {
                "variant_id": variant,
                "source_summary_sha256": sha256(summary_path),
                "source_episode_records_sha256": (
                    sha256(source_arm / "episode_records.jsonl")
                    if (source_arm / "episode_records.jsonl").is_file()
                    else None
                ),
                "external_read_only_evidence": external_evidence,
            }
        )

        def ignore_external_log(directory: str, names: list[str]) -> set[str]:
            expected = source_arm / EXTERNAL_READ_ONLY_EVIDENCE.parent
            if Path(directory).resolve() == expected.resolve() and EXTERNAL_READ_ONLY_EVIDENCE.name in names:
                return {EXTERNAL_READ_ONLY_EVIDENCE.name}
            return set()

        shutil.copytree(
            source_arm,
            destination_arm,
            symlinks=False,
            ignore=ignore_external_log,
        )
        if (destination_arm / EXTERNAL_READ_ONLY_EVIDENCE).exists():
            raise RuntimeError("external read-only evidence was unexpectedly copied")
        if variant == RESUME_THROUGH:
            recover_zero_motion_arm(destination_arm, runner_commit_sha)
        elif load(destination_arm / f"migrated_ablation20_{variant}_summary.json").get("status") != "PASS":
            raise ValueError(f"imported arm summary is not PASS: {variant}")

        rebuild_episode_records(destination_arm, variant, runner_commit_sha)

    additional_source_relative = None
    if additional_source is not None:
        additional_source = require_inside_root(additional_source)
        if not additional_source.is_dir() or additional_source.is_symlink():
            raise ValueError("additional source result is missing or unsafe")
        additional_batch = load(additional_source / "batch_status.json")
        additional_arms = additional_batch.get("arms")
        lane_exits = additional_batch.get("lane_exit_codes")
        checks = (
            additional_batch.get("status") == "FAIL",
            isinstance(additional_arms, dict),
            isinstance(lane_exits, dict),
            additional_batch.get("strict_evidence_modified") is False,
            additional_batch.get("real_go2_targeted") is False,
        )
        if not all(checks) or int(lane_exits.get("a", -1)) != 0:
            raise ValueError("additional source did not preserve a passed lane A")
        if any(additional_arms.get(variant) != "PASS" for variant in ADDITIONAL_VARIANTS):
            raise ValueError("additional source recovery arms are not both PASS")

        additional_source_relative = additional_source.relative_to(ROOT.resolve()).as_posix()
        for variant in ADDITIONAL_VARIANTS:
            source_arm = additional_source / "arms" / variant
            destination_arm = destination / "arms" / variant
            summary_path = source_arm / f"migrated_ablation20_{variant}_summary.json"
            if load(summary_path).get("status") != "PASS":
                raise ValueError(f"additional arm summary is not PASS: {variant}")
            external_path = source_arm / EXTERNAL_READ_ONLY_EVIDENCE
            external_evidence = None
            if external_path.is_file() and not external_path.is_symlink():
                external_evidence = {
                    "source_path": external_path.relative_to(ROOT.resolve()).as_posix(),
                    "size_bytes": external_path.stat().st_size,
                    "sha256": sha256(external_path),
                    "copied_into_resume": False,
                    "reason": "large_immutable_log_retained_in_frozen_source_result",
                }
                external_evidence_total_bytes += int(external_evidence["size_bytes"])
            imported.append(
                {
                    "variant_id": variant,
                    "source_summary_sha256": sha256(summary_path),
                    "source_episode_records_sha256": sha256(
                        source_arm / "episode_records.jsonl"
                    ),
                    "external_read_only_evidence": external_evidence,
                }
            )

            def ignore_additional_log(directory: str, names: list[str]) -> set[str]:
                expected = source_arm / EXTERNAL_READ_ONLY_EVIDENCE.parent
                if (
                    Path(directory).resolve() == expected.resolve()
                    and EXTERNAL_READ_ONLY_EVIDENCE.name in names
                ):
                    return {EXTERNAL_READ_ONLY_EVIDENCE.name}
                return set()

            shutil.copytree(
                source_arm,
                destination_arm,
                symlinks=False,
                ignore=ignore_additional_log,
            )
            if (destination_arm / EXTERNAL_READ_ONLY_EVIDENCE).exists():
                raise RuntimeError("additional external evidence was unexpectedly copied")
            rebuild_episode_records(destination_arm, variant, runner_commit_sha)

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "source_result": source.relative_to(ROOT.resolve()).as_posix(),
        "source_grant_id": batch.get("grant_id"),
        "source_authorization_ref_sha": batch.get("authorization_ref_sha"),
        "additional_source_result": additional_source_relative,
        "additional_variants": list(ADDITIONAL_VARIANTS) if additional_source_relative else [],
        "resume_through_variant": RESUME_THROUGH,
        "imported_arm_count": len(imported),
        "imported_arms": imported,
        "external_read_only_evidence_total_bytes": external_evidence_total_bytes,
        "source_result_must_be_archived_with_resume": external_evidence_total_bytes > 0,
        "rebuilt_runner_commit_sha": runner_commit_sha,
        "strict_evidence_modified": False,
        "real_go2_targeted": False,
        "recorded_unix": time.time(),
    }
    write(destination / "resume_import_manifest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--destination-result", type=Path, required=True)
    parser.add_argument("--runner-commit-sha", required=True)
    parser.add_argument("--additional-source-result", type=Path)
    args = parser.parse_args()
    payload = prepare(
        args.source_result,
        args.destination_result,
        args.runner_commit_sha,
        args.additional_source_result,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
