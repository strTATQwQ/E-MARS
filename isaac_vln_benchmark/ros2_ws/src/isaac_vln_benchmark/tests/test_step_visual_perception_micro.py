import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "run_step_visual_perception_micro.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_step_visual_perception_micro", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_visual_micro_summary_requires_fresh_images_recognition_and_track():
    module = _module()
    rows = [
        {
            "role": "route_choice",
            "expected": "left",
            "response": {
                "route_choice": "left",
                "multimodal": True,
                "image_snapshot": {"age_sec": 0.1},
                "track": {"confirmed": True},
            },
            "primitives_after_response": [
                {"primitive": "follow_waypoint", "route_choice": "left"}
            ],
            "safe_cmds_after_response": [{"linear_x": 0.1, "angular_z": 0.2}],
        },
        {
            "role": "semantic_stop",
            "expected": True,
            "response": {
                "target_visible": True,
                "multimodal": True,
                "image_snapshot": {"age_sec": 0.2},
                "track": {"confirmed": True},
            },
            "primitives_after_response": [{"primitive": "stop"}],
            "safe_cmds_after_response": [{"linear_x": 0.0, "angular_z": 0.0}],
        },
    ]

    result = module.evaluate_results(rows)

    assert result["pass"] is True
    assert result["fresh_multimodal_images"] == 2
    assert result["step_calls"] == 2
    assert result["route_primitive_correct"] == 1
    assert result["route_safe_cmd_correct"] == 1
    assert result["semantic_chain_correct"] == 1


def test_visual_micro_rejects_false_positive_and_unconfirmed_positive_track():
    module = _module()
    rows = [
        {
            "role": "semantic_stop",
            "expected": True,
            "response": {
                "target_visible": True,
                "multimodal": True,
                "image_snapshot": {"age_sec": 0.1},
                "track": {"confirmed": False},
            },
        },
        {
            "role": "semantic_stop",
            "expected": False,
            "response": {
                "target_visible": True,
                "multimodal": True,
                "image_snapshot": {"age_sec": 0.1},
                "track": {"confirmed": False},
            },
        },
    ]

    result = module.evaluate_results(rows)

    assert result["pass"] is False
    assert "semantic target presence mismatch" in result["failures"]
    assert "positive semantic target was not confirmed across frames" in result["failures"]


def test_visual_micro_route_planning_requires_correct_branch_entry():
    module = _module()
    row = {
        "role": "route_choice",
        "expected": "right",
        "route_planning_required": True,
        "entered_correct_branch": False,
        "response": {
            "route_choice": "right",
            "multimodal": True,
            "image_snapshot": {"age_sec": 0.1},
            "track": {"confirmed": False},
        },
        "primitives_after_response": [
            {"primitive": "follow_waypoint", "route_choice": "right"}
        ],
        "safe_cmds_after_response": [{"linear_x": 0.1, "angular_z": -0.2}],
    }

    result = module.evaluate_results([row])

    assert result["pass"] is False
    assert result["route_planning_total"] == 1
    assert "Step route decision did not enter the correct branch" in result["failures"]


def test_visual_micro_semantic_approach_requires_trigger_and_success_judge():
    module = _module()
    row = {
        "role": "semantic_stop",
        "expected": True,
        "semantic_approach_required": True,
        "approach_trigger_reached": True,
        "semantic_success": False,
        "response": {
            "target_visible": True,
            "multimodal": True,
            "image_snapshot": {"age_sec": 0.1},
            "track": {"confirmed": True},
        },
        "primitives_after_response": [{"primitive": "stop"}],
        "safe_cmds_after_response": [{"linear_x": 0.0, "angular_z": 0.0}],
    }

    result = module.evaluate_results([row])

    assert result["pass"] is False
    assert result["semantic_approach_triggered"] == 1
    assert result["semantic_approach_success"] == 0
    assert "tracked semantic stop did not satisfy the Isaac success judge" in result["failures"]
