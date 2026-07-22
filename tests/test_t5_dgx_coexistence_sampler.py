from __future__ import annotations

from scripts import sample_t5_dgx_coexistence as coexistence


summarize = coexistence.summarize


def _sample(
    *,
    alive=True,
    process_swap=0,
    system_swap=100,
    oom_kills=3,
    gpu_memory=40_000,
    unified_memory=False,
):
    gpu = {
        "gpu_index": 0,
        "gpu_uuid": "GPU-gb10-test",
        "utilization_percent": 12.0,
    }
    if unified_memory:
        gpu.update(
            {
                "memory_used_mib": None,
                "memory_total_mib": None,
                "memory_accounting_status": "not_applicable_unified",
            }
        )
    else:
        gpu.update(
            {
                "memory_used_mib": gpu_memory,
                "memory_total_mib": 120_000,
                "memory_accounting_status": "observed",
            }
        )
    return {
        "processes": {
            "internvla": {"alive": alive, "rss_kib": 10_000, "swap_kib": process_swap},
            "step3": {"alive": alive, "rss_kib": 20_000, "swap_kib": process_swap},
        },
        "system_memory": {
            "mem_available_kib": 50_000,
            "swap_used_kib": system_swap,
            "oom_kill_count": oom_kills,
        },
        "gpu": gpu,
    }


def _preload_sample(*, available=50_000, system_swap=100, oom_kills=3):
    return {
        "system_memory": {
            "mem_available_kib": available,
            "swap_used_kib": system_swap,
            "oom_kill_count": oom_kills,
        }
    }


def test_coexistence_summary_passes_only_live_zero_swap_samples() -> None:
    value = summarize(
        [_sample(), _sample(system_swap=100, gpu_memory=42_000)],
        internvla_pid=101,
        step3_pid=202,
        gpu_index=0,
        minimum_samples=2,
    )
    assert value["status"] == "PASS"
    assert value["process_rss_peak_kib"] == {
        "internvla": 10_000.0,
        "step3": 20_000.0,
    }
    assert value["gpu_memory_used_peak_mib"] == 42_000.0
    assert value["gpu_memory_accounting_status"] == "observed"
    assert value["motion_authority"] == "none"


def test_coexistence_summary_fails_on_process_loss_or_swap_growth() -> None:
    value = summarize(
        [_sample(), _sample(alive=False, process_swap=4, system_swap=104)],
        internvla_pid=101,
        step3_pid=202,
        gpu_index=0,
        minimum_samples=2,
    )
    assert value["status"] == "FAIL"
    assert value["checks"]["internvla_alive_throughout"] is False
    assert value["checks"]["step3_process_swap_zero"] is False
    assert value["checks"]["system_swap_did_not_increase"] is False


def test_coexistence_summary_fails_if_gpu_memory_curve_is_missing() -> None:
    first = _sample()
    second = _sample()
    second["gpu"] = {"sample_error": "FileNotFoundError"}
    value = summarize(
        [first, second],
        internvla_pid=101,
        step3_pid=202,
        gpu_index=0,
        minimum_samples=2,
    )
    assert value["status"] == "FAIL"
    assert value["checks"]["gpu_memory_accounting_valid"] is False


def test_coexistence_summary_accepts_explicit_gb10_unified_memory() -> None:
    value = summarize(
        [_sample(unified_memory=True), _sample(unified_memory=True)],
        internvla_pid=101,
        step3_pid=202,
        gpu_index=0,
        minimum_samples=2,
    )
    assert value["status"] == "PASS"
    assert value["gpu_memory_accounting_status"] == "not_applicable_unified"
    assert value["gpu_memory_used_peak_mib"] is None
    assert value["gpu_memory_total_mib"] is None
    assert value["gpu_uuid"] == "GPU-gb10-test"
    assert value["checks"]["gpu_utilization_observed"] is True


def test_coexistence_summary_does_not_infer_unified_memory_from_missing_values() -> None:
    first = _sample(unified_memory=True)
    second = _sample(unified_memory=True)
    second["gpu"].pop("memory_accounting_status")
    value = summarize(
        [first, second],
        internvla_pid=101,
        step3_pid=202,
        gpu_index=0,
        minimum_samples=2,
    )
    assert value["status"] == "FAIL"
    assert value["gpu_memory_accounting_status"] == "unavailable"
    assert value["checks"]["gpu_memory_accounting_valid"] is False


def test_coexistence_summary_requires_gpu_identity_utilization_and_no_new_oom() -> None:
    first = _sample()
    second = _sample(oom_kills=4)
    second["gpu"]["gpu_uuid"] = ""
    second["gpu"]["utilization_percent"] = None
    value = summarize(
        [first, second],
        internvla_pid=101,
        step3_pid=202,
        gpu_index=0,
        minimum_samples=2,
    )
    assert value["status"] == "FAIL"
    assert value["checks"]["gpu_uuid_observed_and_stable"] is False
    assert value["checks"]["gpu_utilization_observed"] is False
    assert value["checks"]["system_oom_kill_did_not_increase"] is False


def test_gpu_sample_marks_nvidia_memory_na_as_unified(monkeypatch) -> None:
    responses = iter(
        [
            (0, "0, GPU-gb10-test, 7, [N/A], [N/A]\n"),
            (0, "101, python, [N/A]\n"),
        ]
    )

    class Result:
        def __init__(self, returncode, stdout):
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(*args, **kwargs):
        return Result(*next(responses))

    monkeypatch.setattr(coexistence.subprocess, "run", fake_run)
    value = coexistence._gpu_sample(0)
    assert value["gpu_uuid"] == "GPU-gb10-test"
    assert value["utilization_percent"] == 7.0
    assert value["memory_accounting_status"] == "not_applicable_unified"
    assert value["memory_used_mib"] is None
    assert value["memory_total_mib"] is None


def test_preload_summary_covers_peak_memory_swap_and_oom_before_ready() -> None:
    value = coexistence.summarize_preload(
        [
            _preload_sample(),
            _preload_sample(available=30_000),
            _preload_sample(available=40_000),
        ],
        stop_observed=True,
    )
    assert value["status"] == "PASS"
    assert value["system_mem_available_min_kib"] == 30_000.0
    assert value["checks"]["explicit_stop_observed"] is True


def test_preload_summary_fails_on_swap_oom_or_missing_stop() -> None:
    value = coexistence.summarize_preload(
        [
            _preload_sample(),
            _preload_sample(system_swap=104, oom_kills=4),
        ],
        stop_observed=False,
    )
    assert value["status"] == "FAIL"
    assert value["checks"]["system_swap_did_not_increase"] is False
    assert value["checks"]["system_oom_kill_did_not_increase"] is False
    assert value["checks"]["explicit_stop_observed"] is False
