from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "coordination/run_t5_dgx_b_coexistence_online.sh"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_coexistence_online_holds_only_dgx_b_and_uses_exact_lane_b_ref() -> None:
    text = _text()
    assert 'with_resource_lease.sh" dgx-b' in text
    assert "10.100.120.122" in text
    assert "T5_DEPLOYMENT_REF" in text
    assert "-${code_sha:0:12}-lane-b" in text
    assert "10.100.100.128" not in text
    assert "10.100.120.123" not in text
    assert "lane-b --owner" not in text
    assert 'quarantine_marker=/tmp/internnav_dgx.quarantine' in text
    assert 't5_quarantine_arm "$target"' in text
    assert 't5_quarantine_owned_clear "$target"' in text


def test_coexistence_online_starts_only_two_real_model_services() -> None:
    text = _text()
    assert 'run_t4_model_server.sh" --ros-args' in text
    assert 'test -f "$deployment/scripts/run_t4_model_server.sh"' in text
    assert "INTERNVLA_BACKEND=real" in text
    assert "INTERNVLA_PRELOAD_MODEL=1" in text
    assert "INTERNVLA_T4_FUNCTIONAL_MODEL=0" in text
    assert "INTERNNAV_T5_RESOURCE_LEASE_ACK=dgx-b" in text
    assert "use_sim_time:=false" in text
    assert 'run_t5_step3_shadow.sh" shadow' in text
    assert 'raw.get("precision_mode") == "bf16"' in text
    assert '{"torch.bfloat16"}' in text
    for forbidden in (
        "run_t5_dgx_lane.sh",
        "run_t4_dgx_onboard.sh",
        "run_t5_distributed_isaac.sh",
        "nav2_bringup",
        "cmd_vel",
    ):
        assert forbidden not in text


def test_coexistence_online_resolves_real_python_pids_and_samples_for_60s() -> None:
    text = _text()
    assert '"internvla_t4_recovery.model_node"' in text
    assert 'validate(service, "slow_planner.serve")' in text
    assert "os.getpgid(pid) == pid" in text
    assert 'test "$internvla_pid" != "$step3_pid"' in text
    assert 'sample_t5_dgx_coexistence.py"' in text
    assert '--internvla-pid "$internvla_pid" --step3-pid "$step3_pid"' in text
    assert "--duration-sec 60 --interval-sec 1" in text
    assert '"not_applicable_unified"' in text
    assert '"system_oom_kill_did_not_increase"' in text
    assert 'preload_memory_summary.json' in text
    assert '--preload-stop-file "$result/preload_memory.stop"' in text
    assert '"preload_memory_sampling_pass"' in text


def test_coexistence_online_cleans_owned_groups_and_related_ports() -> None:
    text = _text()
    assert "stop_group" in text
    assert 'stop_group "$internvla_runner_pid" "$internvla_runner_identity" INT' in text
    assert 'touch "$result/step3/stop.request"' in text
    assert "residual_process_groups" in text
    assert "related_ports_free" in text
    for port in (8200, 8300, 25138, 25239, 25240, 25241):
        assert str(port) in text
    assert 'cleanup.get("occupied_ports") == []' in text
    assert 'step3_cleanup.get("residual_pids") == []' in text
    assert "capture_identity()" in text
    assert 'scope="${2:-member}"' in text
    assert 'scope == "leader" and (pgid != pid or sid != pid)' in text
    assert 'capture_identity "$preload_sampler_pid" leader' in text
    assert 'capture_identity "$internvla_runner_pid" leader' in text
    assert "identity_matches()" in text
    assert 'identity_matches "$pgid" "$identity" || return 75' in text
    assert "starttime" in text and "pgid" in text and "sid" in text
    assert "root_bound" in text


def test_coexistence_online_preserves_machine_readable_failure_evidence() -> None:
    text = _text()
    assert "remote_rc=$?" in text
    assert "collect_rc=$?" in text
    assert 'test "$quarantine_clear_rc" != 0; then' in text
    assert '"remote_coexistence_runtime" if remote_rc != 0' in text
    assert 'else "evidence_collection" if collect_rc != 0' in text
    assert '"remote_exit_code": remote_rc' in text
    assert '"cleanup_status": cleanup.get("status")' in text
    assert '"memory_status": memory.get("status")' in text
    assert '"quarantine_clear_exit_code": quarantine_clear_rc' in text
    assert "exit 75" in text


def test_coexistence_online_has_no_embedded_credentials() -> None:
    text = _text()
    for forbidden in ("HF_TOKEN", "OPENAI_API_KEY", "sshpass", "password"):
        assert forbidden not in text
