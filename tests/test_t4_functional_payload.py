from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_TOOL = ROOT / "coordination/t4_functional_payload.py"


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dir() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--absolute-git-dir"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _build(tmp_path: Path) -> tuple[Path, Path, str]:
    archive = tmp_path / "payload.tar"
    manifest = tmp_path / "payload_manifest.json"
    head = _head()
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(PAYLOAD_TOOL),
            "build",
            "--git-dir",
            _git_dir(),
            "--ref-sha",
            head,
            "--archive",
            str(archive),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return archive, manifest, head


def _wsl_path(path: Path) -> str:
    return subprocess.run(
        ["wsl", "wslpath", "-u", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_linux_tree(
    archive: Path, manifest: Path, head: str, mutation: str = "none"
) -> subprocess.CompletedProcess[str]:
    if os.name != "nt":
        raise RuntimeError("the current test fixture expects the Windows/WSL host")
    script = r"""
set -euo pipefail
archive="$1"
manifest="$2"
tool="$3"
expected_ref="$4"
mutation="$5"
root="$(mktemp -d)"
trap 'rm -rf -- "$root"' EXIT
tar -C "$root" -xf "$archive"
case "$mutation" in
  none) ;;
  mutate) printf '\n# mutation\n' >>"$root/scripts/run_t4_model_server.sh" ;;
  extra) printf 'pass\n' >"$root/scripts/untracked.py" ;;
  *) exit 64 ;;
esac
python3 "$tool" verify-tree --root "$root" --manifest "$manifest" \
  --expected-ref "$expected_ref"
"""
    script_path = manifest.parent / f"verify_{mutation}.sh"
    script_path.write_text(script, encoding="utf-8", newline="\n")
    return subprocess.run(
        [
            "wsl",
            "bash",
            _wsl_path(script_path),
            _wsl_path(archive),
            _wsl_path(manifest),
            _wsl_path(PAYLOAD_TOOL),
            head,
            mutation,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_functional_payload_is_an_exact_git_object_view(tmp_path: Path) -> None:
    archive, manifest_path, head = _build(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "FUNCTIONAL_PAYLOAD_READY"
    assert manifest["ref_sha"] == head
    assert manifest["model_host"] == "dgx_spark_only"
    assert manifest["runtime_policy"] == "completion_sim"
    assert manifest["strict_evidence_modified"] is False
    assert manifest["real_go2_targeted"] is False
    assert manifest["path_count"] == len(manifest["files"])
    paths = {record["path"] for record in manifest["files"]}
    for required in (
        "internvla_ros2_msgs/srv/RecoveryControl.srv",
        "internvla_ros2/internvla_ros2/recovery_contract.py",
        "internvla_t4_recovery/internvla_t4_recovery/model_node.py",
        "internvla_t4_sensors/internvla_t4_sensors/client_node.py",
        "configs/completion_sim/map/nav2_static_lidar.yaml",
        "scripts/run_t4_model_server.sh",
    ):
        assert required in paths

    verified = _verify_linux_tree(archive, manifest_path, head)
    assert verified.returncode == 0, verified.stdout + verified.stderr


def test_functional_tree_verifier_rejects_mutation_and_extra_files(
    tmp_path: Path,
) -> None:
    archive, manifest_path, head = _build(tmp_path)
    mutated = _verify_linux_tree(archive, manifest_path, head, "mutate")
    assert mutated.returncode != 0
    extra = _verify_linux_tree(archive, manifest_path, head, "extra")
    assert extra.returncode != 0


def test_remote_stage_is_role_locked_isolated_and_combined_lease_only() -> None:
    stage = (ROOT / "coordination/remote_t4_functional_stage.sh").read_text(
        encoding="utf-8"
    )
    dgx_build = (ROOT / "scripts/build_t4_host_ros.sh").read_text(
        encoding="utf-8"
    )
    assert "expected_user=railgun" in stage
    assert "expected_user=song" in stage
    assert 'expected_ip=" 10.100.100.128/"' in stage
    assert 'expected_ip=" 10.100.120.111/"' in stage
    assert 'deployment_parent="$stable_root/.t4-deployments"' in stage
    assert '"$deployment_parent/"*.partial.*' in stage
    assert "test ! -e \"$deployment_root\"" in stage
    assert 'python3 "$payload_tool" verify-tree' in stage
    assert "export PYTHONDONTWRITEBYTECODE=1" in stage
    assert "INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac" in stage
    assert "strict_evidence_modified\":False" in stage
    assert "real_go2_targeted\":False" in stage
    assert "--packages-up-to" in dgx_build
    assert "internvla_t4_sensors" in dgx_build
    assert "internvla_t4_recovery" in dgx_build
    assert "go2_sensor_bridge" in dgx_build


def test_coordinator_prepare_requires_f1_grant_and_dgx_first_combined_lease() -> None:
    runner = (ROOT / "coordination/run_t4_functional_prepare.sh").read_text(
        encoding="utf-8"
    )
    assert '"worker":"00"' in runner
    assert '"resource":"dgx+isaac"' in runner
    assert '"profile":"functional_prepare"' in runner
    assert "F1_PREREQUISITE_PASS" in runner
    assert 'for role in dgx isaac; do' in runner
    assert 'with_resource_lease.sh" both' in runner
    assert "00-functional-prepare" in runner
    assert "payload.tar payload_manifest.json >SHA256SUMS" in runner
    assert "strict_evidence_modified\":False" in runner
    assert "real_go2_targeted\":False" in runner
    assert "ISAAC_PASSWORD_FILE" in runner
    assert "DGX_PASSWORD_FILE" in runner
    assert "BatchMode=yes" in runner
