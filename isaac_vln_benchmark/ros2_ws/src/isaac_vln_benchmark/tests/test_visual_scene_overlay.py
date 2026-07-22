from isaac_vln_benchmark.go2_benchmark_adapter_node import merge_scene_entities


def test_visual_overlay_preserves_geometry_and_replaces_matching_target():
    existing = [
        {"id": "branch_marker", "pose": [4.0, 0.0, 0.0]},
        {"id": "fire_extinguisher_1", "pose": [4.0, 0.0, 0.0]},
    ]
    overlays = [
        {"id": "fire_extinguisher_1", "pose": [2.0, 0.0, 0.0]},
        {"id": "visual_distractor", "pose": [2.5, 1.0, 0.0]},
    ]

    merged = merge_scene_entities(existing, overlays)

    assert next(row for row in merged if row["id"] == "branch_marker")["pose"] == [4.0, 0.0, 0.0]
    assert next(row for row in merged if row["id"] == "fire_extinguisher_1")["pose"] == [2.0, 0.0, 0.0]
    assert len([row for row in merged if row["id"] == "fire_extinguisher_1"]) == 1
