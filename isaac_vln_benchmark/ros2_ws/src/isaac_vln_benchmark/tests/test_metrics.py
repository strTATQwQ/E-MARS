from isaac_vln_benchmark.metrics import aggregate_metrics, distance_xy, path_length, path_obstacle_status


def test_distance_xy():
    assert distance_xy([0, 0], [3, 4]) == 5.0


def test_path_length():
    rows = [{"pose": [0, 0, 0]}, {"pose": [3, 4, 0]}, {"pose": [6, 8, 0]}]
    assert path_length(rows) == 10.0


def test_aggregate_metrics():
    agg = aggregate_metrics(
        [
            {"success": True, "mission_time_sec": 10, "path_length_m": 2, "num_step_calls": 1, "num_omninav_calls": 10},
            {"success": False, "failure_reason": "timeout", "mission_time_sec": 20, "path_length_m": 3, "num_step_calls": 2, "num_omninav_calls": 20},
        ]
    )
    assert agg["episodes"] == 2
    assert agg["success_rate"] == 0.5
    assert agg["failure_top1"] == "timeout"


def test_path_obstacle_status_ignores_side_clearance():
    status = path_obstacle_status(
        [2.83, -0.14, -0.065],
        [{"id": "cone", "pose": [3.5, -0.55, 0.3], "size": [0.25, 0.25, 0.6]}],
        stop_distance_m=0.8,
        path_half_width_m=0.22,
    )
    assert status["local_costmap_clear"] is True


def test_path_obstacle_status_ignores_drifted_side_cone_by_default():
    status = path_obstacle_status(
        [2.8753483295440674, -0.2023811638355255, -0.08260852098464966],
        [{"id": "cone", "pose": [3.5, -0.55, 0.3], "size": [0.25, 0.25, 0.6]}],
    )
    assert status["local_costmap_clear"] is True


def test_path_obstacle_status_blocks_centerline_obstacle():
    status = path_obstacle_status(
        [2.83, 0.0, 0.0],
        [{"id": "box", "pose": [3.4, 0.0, 0.3], "size": [0.5, 0.5, 0.6]}],
        stop_distance_m=0.8,
        path_half_width_m=0.22,
    )
    assert status["local_costmap_clear"] is False
