#!/usr/bin/env python3
"""Fail-closed validator for InternNav worker handoff.json files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TASK_RE = re.compile(r"^(01R|0[2-5]|[1-5]0)$")
TOP_LEVEL_KEYS = {
    "schema_version",
    "task_id",
    "base_sha",
    "head_sha",
    "changed_paths",
    "offline_tests",
    "resource_needs",
    "online_command",
    "thresholds",
    "artifacts",
    "risks",
    "rollback",
}
NEW_WORKER_IDS = {"10", "20", "30", "40", "50"}
INTERFACE_REQUEST_KEYS = {"path", "change", "reason", "blocking"}


def _is_repo_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        return False
    return all(part not in {"", ".", ".."} for part in value.split("/"))


def _exact_keys(value: Any, expected: set[str], label: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return False
    actual = set(value)
    if actual != expected:
        errors.append(
            f"{label} keys mismatch: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )
        return False
    return True


def _paths(values: Any, label: str, errors: list[str], *, nonempty: bool) -> None:
    if not isinstance(values, list) or (nonempty and not values):
        errors.append(f"{label} must be {'a nonempty' if nonempty else 'an'} array")
        return
    if len(values) != len(set(item for item in values if isinstance(item, str))):
        errors.append(f"{label} must not contain duplicate paths")
    for index, value in enumerate(values):
        if not _is_repo_path(value):
            errors.append(f"{label}[{index}] is not a safe repository-relative path")


def validate(
    payload: Any,
    *,
    expected_base: str | None,
    expected_head: str | None,
) -> list[str]:
    errors: list[str] = []
    task_id_hint = payload.get("task_id") if isinstance(payload, dict) else None
    expected_top_level = set(TOP_LEVEL_KEYS)
    if task_id_hint in NEW_WORKER_IDS:
        expected_top_level.add("interface_requests")
    if not _exact_keys(payload, expected_top_level, "handoff", errors):
        return errors

    if payload["schema_version"] != 1:
        errors.append("schema_version must equal 1")
    task_id = payload["task_id"]
    if not isinstance(task_id, str) or not TASK_RE.fullmatch(task_id):
        errors.append("task_id is invalid")

    base_sha = payload["base_sha"]
    head_sha = payload["head_sha"]
    if not isinstance(base_sha, str) or not SHA_RE.fullmatch(base_sha):
        errors.append("base_sha must be a lowercase 40-character Git SHA")
    if head_sha != "SELF" and (
        not isinstance(head_sha, str) or not SHA_RE.fullmatch(head_sha)
    ):
        errors.append("head_sha must be SELF or a lowercase 40-character Git SHA")
    if expected_base is not None and base_sha != expected_base:
        errors.append(f"base_sha does not match reviewed base {expected_base}")
    if expected_head is not None:
        if not SHA_RE.fullmatch(expected_head):
            errors.append("--expected-head is not a lowercase 40-character Git SHA")
        elif head_sha not in {"SELF", expected_head}:
            errors.append(f"head_sha does not resolve to reviewed HEAD {expected_head}")
    elif head_sha == "SELF":
        errors.append("head_sha SELF requires --expected-head for fail-closed resolution")

    changed = payload["changed_paths"]
    _paths(changed, "changed_paths", errors, nonempty=True)
    if isinstance(task_id, str) and isinstance(changed, list):
        required_paths = {f"handoffs/{task_id}/handoff.json"}
        if task_id not in NEW_WORKER_IDS:
            required_paths.add(f"handoffs/{task_id}.md")
        missing = required_paths - set(item for item in changed if isinstance(item, str))
        if missing:
            errors.append(f"changed_paths misses required handoff paths: {sorted(missing)}")

    tests = payload["offline_tests"]
    test_keys = {"name", "command", "status", "exit_code", "evidence"}
    if not isinstance(tests, list) or not tests:
        errors.append("offline_tests must be a nonempty array")
    else:
        for index, test in enumerate(tests):
            label = f"offline_tests[{index}]"
            if not _exact_keys(test, test_keys, label, errors):
                continue
            if not isinstance(test["name"], str) or not test["name"]:
                errors.append(f"{label}.name must be nonempty")
            if not isinstance(test["command"], str) or not test["command"]:
                errors.append(f"{label}.command must be nonempty")
            status = test["status"]
            exit_code = test["exit_code"]
            if status not in {"PASS", "FAIL", "NOT_RUN"}:
                errors.append(f"{label}.status is invalid")
            elif status == "PASS" and exit_code != 0:
                errors.append(f"{label} PASS requires exit_code 0")
            elif status == "FAIL" and (not isinstance(exit_code, int) or exit_code == 0):
                errors.append(f"{label} FAIL requires a nonzero integer exit_code")
            elif status == "NOT_RUN" and exit_code is not None:
                errors.append(f"{label} NOT_RUN requires null exit_code")
            _paths(test["evidence"], f"{label}.evidence", errors, nonempty=False)

    needs_keys = {"resources", "model_loading", "heavy_download", "reason"}
    needs = payload["resource_needs"]
    if _exact_keys(needs, needs_keys, "resource_needs", errors):
        resources = needs["resources"]
        allowed = {"isaac", "dgx", "real_go2"}
        if not isinstance(resources, list) or any(item not in allowed for item in resources):
            errors.append("resource_needs.resources is invalid")
        elif len(resources) != len(set(resources)):
            errors.append("resource_needs.resources must be unique")
        for key in ("model_loading", "heavy_download"):
            if not isinstance(needs[key], bool):
                errors.append(f"resource_needs.{key} must be boolean")
        if not isinstance(needs["reason"], str) or not needs["reason"]:
            errors.append("resource_needs.reason must be nonempty")
        if (needs["model_loading"] or needs["heavy_download"]) and "dgx" not in resources:
            errors.append("model loading or heavy download requires the DGX resource")

    command_keys = {
        "executed_by_worker",
        "resource",
        "command",
        "result_dir",
        "fresh_result_dir_required",
    }
    online = payload["online_command"]
    if _exact_keys(online, command_keys, "online_command", errors):
        resource = online["resource"]
        command = online["command"]
        if online["executed_by_worker"] is not False:
            errors.append("online_command.executed_by_worker must be false")
        if resource not in {"none", "isaac", "dgx", "both", "real_go2"}:
            errors.append("online_command.resource is invalid")
        if resource in {"isaac", "dgx", "both"}:
            if not isinstance(command, str) or "scripts/with_resource_lease.sh" not in command:
                errors.append("leased online command must use scripts/with_resource_lease.sh")
            if not isinstance(online["result_dir"], str) or not online["result_dir"]:
                errors.append("leased online command requires a result_dir")
            if online["fresh_result_dir_required"] is not True:
                errors.append("leased online command requires a fresh result directory")
        elif resource == "none" and (command is not None or online["result_dir"] is not None):
            errors.append("resource none requires null command and result_dir")
        command_text = command if isinstance(command, str) else ""
        if re.search(r"hf_[A-Za-z0-9]+|(?i:password)\s*=", command_text):
            errors.append("online command contains a credential-like literal")

    thresholds = payload["thresholds"]
    if not isinstance(thresholds, dict) or not thresholds:
        errors.append("thresholds must be a nonempty object")
    elif any(isinstance(value, (dict, list)) for value in thresholds.values()):
        errors.append("threshold values must be scalar")

    artifacts = payload["artifacts"]
    artifact_keys = {"path", "description", "required_for_gate"}
    artifact_sha_keys = artifact_keys | {"sha256"}
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts must be a nonempty array")
    else:
        for index, artifact in enumerate(artifacts):
            label = f"artifacts[{index}]"
            if not isinstance(artifact, dict):
                errors.append(f"{label} keys are invalid")
                continue
            keys = frozenset(artifact)
            if keys not in {frozenset(artifact_keys), frozenset(artifact_sha_keys)}:
                errors.append(f"{label} keys are invalid")
                continue
            if not _is_repo_path(artifact["path"]):
                errors.append(f"{label}.path is invalid")
            if not isinstance(artifact["description"], str) or not artifact["description"]:
                errors.append(f"{label}.description must be nonempty")
            if not isinstance(artifact["required_for_gate"], bool):
                errors.append(f"{label}.required_for_gate must be boolean")
            sha256 = artifact.get("sha256")
            if sha256 is not None and (
                not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256)
            ):
                errors.append(f"{label}.sha256 is invalid")

    if task_id in NEW_WORKER_IDS:
        requests = payload["interface_requests"]
        if not isinstance(requests, list):
            errors.append("interface_requests must be an array")
        else:
            for index, request in enumerate(requests):
                label = f"interface_requests[{index}]"
                if not _exact_keys(
                    request, INTERFACE_REQUEST_KEYS, label, errors
                ):
                    continue
                if not _is_repo_path(request["path"]):
                    errors.append(f"{label}.path is invalid")
                for key in ("change", "reason"):
                    if not isinstance(request[key], str) or not request[key]:
                        errors.append(f"{label}.{key} must be nonempty")
                if not isinstance(request["blocking"], bool):
                    errors.append(f"{label}.blocking must be boolean")

    risks = payload["risks"]
    if not isinstance(risks, list) or not risks or any(
        not isinstance(item, str) or not item for item in risks
    ):
        errors.append("risks must be a nonempty array of nonempty strings")

    rollback = payload["rollback"]
    if _exact_keys(rollback, {"strategy", "commands"}, "rollback", errors):
        if not isinstance(rollback["strategy"], str) or not rollback["strategy"]:
            errors.append("rollback.strategy must be nonempty")
        commands = rollback["commands"]
        if not isinstance(commands, list) or any(
            not isinstance(item, str) or not item for item in commands
        ):
            errors.append("rollback.commands must be an array of nonempty strings")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("handoff", type=Path)
    parser.add_argument("--expected-base")
    parser.add_argument("--expected-head")
    args = parser.parse_args()
    try:
        payload = json.loads(args.handoff.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "errors": [str(exc)]}, indent=2))
        return 2
    errors = validate(
        payload,
        expected_base=args.expected_base,
        expected_head=args.expected_head,
    )
    print(
        json.dumps(
            {
                "status": "PASS" if not errors else "FAIL",
                "resolved_head_sha": args.expected_head,
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
