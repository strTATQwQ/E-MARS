from isaac_vln_benchmark.metrics import SuccessJudgeCore


def test_success_judge_requires_stop_hold():
    task = {"success": {"distance_to_target_m": 2.0, "target_visible": False, "stop_required": True}}
    scene = {"bounds": [-5, -5, 5, 5], "obstacles": []}
    target = {"id": "target", "pose": [1.0, 0.0, 0.0]}
    judge = SuccessJudgeCore(success_hold_sec=1.0)
    first = judge.evaluate(task, scene, [0.0, 0.0, 0.0], target, now_sec=0.1, cmd_vel={"linear_x": 0.0, "angular_z": 0.0})
    second = judge.evaluate(task, scene, [0.0, 0.0, 0.0], target, now_sec=1.2, cmd_vel={"linear_x": 0.0, "angular_z": 0.0})
    assert not first["done"]
    assert second["success"]


def test_success_judge_collision_fails():
    task = {"success": {"distance_to_target_m": 2.0, "target_visible": False, "stop_required": False}}
    scene = {"bounds": [-5, -5, 5, 5], "obstacles": []}
    target = {"id": "target", "pose": [1.0, 0.0, 0.0]}
    status = SuccessJudgeCore().evaluate(task, scene, [0.0, 0.0, 0.0], target, now_sec=0.0, collision=True)
    assert status["done"]
    assert status["reason"] == "collision"
