from __future__ import annotations

import pytest

from slow_benchmark.video import safe_episode_filename, write_episode_video


def test_safe_episode_filename_is_windows_portable() -> None:
    assert safe_episode_filename("val_unseen:2azQ1b91cZZ:42") == "val_unseen__2azQ1b91cZZ__42"
    assert ":" not in safe_episode_filename("x:y")


def test_video_requires_at_least_one_frame(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least one JPEG"):
        write_episode_video(tmp_path, "episode", [], fps=2.0, codec="libx264", quality=7)
