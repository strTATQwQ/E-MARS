from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_go2_entrypoint_can_pin_vulkan_renderer_and_cuda_physics() -> None:
    text = (ROOT / "scripts/run_internnav_go2_entrypoint.py").read_text(
        encoding="utf-8"
    )
    assert 'os.environ.get("INTERNVLA_ISAAC_RENDER_GPU")' in text
    assert 'os.environ.get("INTERNVLA_ISAAC_PHYSICS_GPU")' in text
    assert 'launcher_config["active_gpu"] = int(render_gpu)' in text
    assert 'launcher_config["physics_gpu"] = int(physics_gpu)' in text
    assert '"multi_gpu": False' in text


def test_dual_lane_runner_exports_physical_renderer_and_logical_physics_gpu() -> None:
    text = (ROOT / "scripts/run_t5_distributed_isaac.sh").read_text(
        encoding="utf-8"
    )
    assert 'export INTERNVLA_ISAAC_RENDER_GPU="$gpu"' in text
    assert "export INTERNVLA_ISAAC_PHYSICS_GPU=0" in text
