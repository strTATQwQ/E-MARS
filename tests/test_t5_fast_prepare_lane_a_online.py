from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "coordination" / "run_t5_fast_prepare_lane_a_online.sh"
PREPARE = ROOT / "coordination" / "run_t5_fast_prepare_online.sh"
WORKERS = ROOT / "scripts" / "prepare_t5_isaac_workers.sh"
FAST_LANE = ROOT / "coordination" / "run_t5_fast_lane_online.sh"


def test_explicit_lane_a_prepare_is_gpu0_only_and_dual_default_remains() -> None:
    entry = ENTRY.read_text(encoding="utf-8")
    prepare = PREPARE.read_text(encoding="utf-8")
    workers = WORKERS.read_text(encoding="utf-8")

    assert "INTERNNAV_T5_FAST_PREPARE_SCOPE=lane-a" in entry
    assert "INTERNNAV_T5_LANE_A_PREPARE_ENTRY=1" in entry
    assert "10.100.120.116" not in entry
    assert "isaac_gpu1" not in entry
    assert "internnav_t5_isaac_b" not in entry

    assert 'prepare_scope="${INTERNNAV_T5_FAST_PREPARE_SCOPE:-dual}"' in prepare
    assert "lane_a_pair:lane-a" in prepare
    assert 'run_task lane_a_pair lane-a' in prepare
    assert 'requested_resources")==["dgx_a","isaac_gpu0"]' in prepare
    assert '"prepared_lanes":["a"]' in prepare
    assert 'quarantine_file=/tmp/internnav_isaac_gpu0.quarantine' in prepare
    assert "quarantine_role=x86_gpu0" in prepare
    assert 't5_quarantine_arm "$target" "$quarantine_file" "$quarantine_role"' in prepare
    assert 't5_quarantine_owned_clear "$target" "$quarantine_file" "$quarantine_role"' in prepare
    assert 'INTERNNAV_T5_RESOURCE_LEASE_ACK="$(test "$prepare_scope" = lane-a && printf lane-a || printf isaac)"' in prepare
    assert "write_lane_a_scoped_compute_receipt" not in prepare
    assert 'run_residual_audit "$target" "$role" "$scope_roots" "$ownership_dir"' in prepare
    assert 'runtime_locks = (("/tmp/internnav_t5_isaac_a_runtime.lock",)' in prepare

    x86_a_case = prepare.split("    x86_a)", 1)[1].split("      ;;", 1)[0]
    assert "dgx_b_target" not in x86_a_case
    assert "x86_b_root" not in x86_a_case
    assert "internnav_t5_isaac_b" not in x86_a_case
    assert "isaac_gpu1" not in x86_a_case

    # The historical entry stays dual unless the explicit wrapper selects A.
    assert 'run_task dgx_b dgx-b' in prepare
    assert 'run_task x86 isaac' in prepare
    assert 'prepare_scope="${INTERNVLA_T5_ISAAC_PREPARE_SCOPE:-dual}"' in workers
    assert 'if test "$prepare_scope" = dual; then' in workers
    assert 'prepare_lane a 0 75 "$lane_a_cpuset"' in workers
    assert 'prepare_lane b 1 76 "$lane_b_cpuset"' in workers


def test_fast_lane_consumes_scoped_prepare_without_lane_b_authority() -> None:
    text = FAST_LANE.read_text(encoding="utf-8")

    assert '"$prep_dir" "$code_sha" "$lane"' in text
    assert 'Path("remote/x86_a/dataset_audit.json")' in text
    assert 'Path("remote/x86/dataset_audit.json")' in text
    assert 'raise SystemExit("Lane-A prepare cannot authorize Lane B")' in text
    assert 'raise SystemExit("prepare scope receipts disagree")' in text
    assert 'not dataset_path.is_file() or dataset_path.is_symlink()' in text
    assert '"prepare_scope_compatible"' in text
