from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "internvla_nav2_adapter"))

from internvla_nav2_adapter.system1_target import FrozenSystem1Target  # noqa: E402


PATH_SHA = "ab" * 32


def _target(**overrides: object) -> FrozenSystem1Target:
    values: dict[str, object] = {
        "episode_id": "a::121",
        "reset_generation": 2,
        "source_sequence_id": 1,
        "source_path_sha256": PATH_SHA,
        "valid_until_ns": 40_000_000_000,
        "frame_id": "map",
        "position_xyz": (4.5, 0.25, 0.0),
        "orientation_xyzw": (0.0, 0.0, 0.25, 0.9682458366),
    }
    values.update(overrides)
    return FrozenSystem1Target(**values)  # type: ignore[arg-type]


def test_frozen_target_hash_is_deterministic_and_binds_path_and_pose() -> None:
    first = _target()
    assert first.target_sha256 == _target().target_sha256
    assert first.target_sha256 != _target(source_path_sha256="cd" * 32).target_sha256
    assert first.target_sha256 != _target(position_xyz=(4.6, 0.25, 0.0)).target_sha256


def test_frozen_target_reports_remaining_absolute_xy_distance() -> None:
    target = _target(position_xyz=(4.5, 0.25, 0.0))
    assert target.remaining_xy_distance(4.5, 0.0) == pytest.approx(0.25)
    assert target.remaining_xy_distance(4.35, 0.05) == pytest.approx(0.25)
    assert target.permits_bounded_reissue(4.5, 0.0, 0.25)
    assert not target.permits_bounded_reissue(4.5, 0.01, 0.25)
    with pytest.raises(ValueError, match="NaN/Inf"):
        target.remaining_xy_distance(float("nan"), 0.0)
    with pytest.raises(ValueError, match="invalid"):
        target.permits_bounded_reissue(4.5, 0.0, 0.0)


def test_queue_binding_requires_same_episode_reset_newer_sequence_and_ttl() -> None:
    target = _target()
    target.require_queue_binding(
        episode_id="a::121",
        reset_generation=2,
        sequence_id=2,
        now_ns=39_000_000_000,
    )
    with pytest.raises(ValueError, match="episode/reset"):
        target.require_queue_binding(
            episode_id="a::other",
            reset_generation=2,
            sequence_id=2,
            now_ns=39_000_000_000,
        )
    with pytest.raises(ValueError, match="sequence"):
        target.require_queue_binding(
            episode_id="a::121",
            reset_generation=2,
            sequence_id=1,
            now_ns=39_000_000_000,
        )
    with pytest.raises(ValueError, match="expired"):
        target.require_queue_binding(
            episode_id="a::121",
            reset_generation=2,
            sequence_id=2,
            now_ns=40_000_000_001,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"frame_id": "base_link"},
        {"source_path_sha256": "not-a-sha"},
        {"position_xyz": (float("nan"), 0.0, 0.0)},
        {"orientation_xyzw": (0.0, 0.0, 0.0, 0.0)},
    ],
)
def test_frozen_target_rejects_relative_or_invalid_state(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _target(**overrides)


def test_active_adapter_reissues_only_absolute_identity_bound_target() -> None:
    source = (
        ROOT
        / "internvla_nav2_adapter"
        / "internvla_nav2_adapter"
        / "active_node.py"
    ).read_text(encoding="utf-8")
    queue_branch = source[
        source.index("elif source == SOURCE_SYSTEM1_QUEUE:") :
        source.index("        if self.active_goal is None:", source.index("elif source == SOURCE_SYSTEM1_QUEUE:"))
    ]
    assert "_require_frozen_system1_target(command)" in queue_branch
    assert "command.local_path" not in queue_branch
    assert "_materialize_frozen_system1_target" in source
    assert "preserve_system1_target=True" in source
    assert "permits_bounded_reissue" in source
    assert '"cleared"' in source
    assert "System 1 queue has no identity-bound absolute target" in source
