from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_strict_bootstrap_is_release45_dgx_a_and_fail_closed() -> None:
    text = (ROOT / "coordination/bootstrap_t5_dgx_a_isaac_ros.sh").read_text(
        encoding="utf-8"
    )
    assert 'with_resource_lease.sh" dgx-a' in text
    assert "10.100.100.128" in text
    assert "10.100.120.116" not in text
    assert "release-4.5 noble-fastos" in text
    assert "isaac-ros-cli" in text
    assert "ros-jazzy-isaac-ros-visual-slam" in text
    assert "ros-jazzy-isaac-ros-nvblox" in text
    assert "docker ps -q" in text
    assert "refusing to restart Docker while a container is running" in text
    assert "--gpus all" in text
    assert "NVIDIA GB10" in text


def test_strict_bootstrap_never_puts_password_in_arguments_or_results() -> None:
    text = (ROOT / "coordination/bootstrap_t5_dgx_a_isaac_ros.sh").read_text(
        encoding="utf-8"
    )
    assert "DGX_A_SUDO_PASSWORD" in text
    assert '"$root/.env.local"' in text
    assert 'values.get("DGX_A_PASSWORD","")' in text
    assert '"sudo -S -p \'\' bash -s -- \'$run_id\' \'$remote_result\'"' in text
    for forbidden in ("sshpass", "DGX_A_PASSWORD=", "spark", "OPENAI_API_KEY", "HF_TOKEN"):
        assert forbidden not in text


def test_docker_pull_proxy_is_temporary_and_not_a_codex_proxy() -> None:
    text = (ROOT / "coordination/bootstrap_t5_dgx_a_isaac_ros.sh").read_text(
        encoding="utf-8"
    )
    assert "98-internnav-bootstrap-proxy.conf" in text
    assert 'rm -f -- "$proxy_dropin"' in text
    assert 'test ! -e "$proxy_dropin"' in text
    assert "temporary_proxy_removed" in text
    assert ".codex" not in text


def test_strict_component_prepare_uses_core_nvblox_without_cuda_downgrade() -> None:
    text = (ROOT / "coordination/prepare_t5_dgx_a_strict_component.sh").read_text(
        encoding="utf-8"
    )
    assert 'with_resource_lease.sh" dgx-a' in text
    assert "ros-jazzy-isaac-ros-visual-slam" in text
    assert "nvblox) package=ros-jazzy-nvblox-ros" in text
    assert "packages+=(ros-jazzy-nvblox-nav2)" in text
    assert "ros-jazzy-isaac-ros-nvblox" not in text
    assert "--allow-downgrades" not in text
    assert "timeout --signal=TERM --kill-after=5s 20s" in text
    assert 'test "$smoke_rc" = 124' in text
