import json

from isaac_visual_debug_overlay.episode_overlay_logger import EpisodeOverlayLogger
from isaac_visual_debug_overlay.marker_builder import MarkerBuilder


def test_visual_overlay_state_logging(tmp_path):
    logger = EpisodeOverlayLogger(tmp_path)
    state = {
        "t": 0.0,
        "robot_pose": [0.0, 0.0, 0.0],
        "target": "blue_box",
        "current_primitive": "move_forward",
        "target_visible": True,
    }
    logger.log_overlay(state)
    rows = [json.loads(line) for line in (tmp_path / "overlay_state.jsonl").read_text().splitlines()]
    assert rows == [state]


def test_marker_builder_writes_png(tmp_path):
    scene = {
        "target": "blue_box",
        "success_radius_m": 2.0,
        "world": {"objects": [{"name": "blue_box", "x": 2.0, "y": 0.0, "color": "blue"}]},
    }
    builder = MarkerBuilder(width=160, height=120, world_scale=20.0)
    state = builder.build_overlay_state(
        t=0.0,
        robot_pose=[0.0, 0.0, 0.0],
        trajectory=[{"x": 0.0, "y": 0.0}],
        scene=scene,
        mode="omninav_only",
        active_subgoal="Move forward",
        primitive="move_forward",
        step_json=None,
        stale_status="accepted",
        safety_status="clear",
        distance_to_target=2.0,
        target_visible=True,
        visible_to_stop_latency=None,
        entered_correct_branch=None,
        stop_decision=None,
    )
    out = tmp_path / "frame.png"
    builder.render_png(out, state, scene, [{"x": 0.0, "y": 0.0}])
    assert out.read_bytes().startswith(b"\x89PNG")
