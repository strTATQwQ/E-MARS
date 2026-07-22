from isaac_vln_benchmark.public_semantic_heuristic_node import public_instruction_plan


def test_public_instruction_plan_uses_only_language_and_keeps_terminal_target():
    plan = public_instruction_plan(
        "Pass the reception desk, enter the second doorway, then stop at the fire extinguisher."
    )
    assert [row["subgoal_type"] for row in plan] == ["pass", "enter", "approach", "verify"]
    assert plan[-1]["target"] == plan[-2]["target"]
    assert all(row["source"] == "public_instruction_heuristic" for row in plan)
    assert not any(key in row for row in plan for key in ("waypoint", "cmd_vel", "oracle_plan", "judge"))


def test_public_instruction_plan_maps_recovery_words_without_geometry():
    plan = public_instruction_plan(
        "Find the printer beyond the blue sign; if no printer appears, backtrack to the sign."
    )
    assert plan[0]["subgoal_type"] == "find"
    assert plan[0]["recovery"] == "backtrack"
    assert plan[0]["relation"].startswith("beyond")


def test_unparsed_instruction_holds_and_asks():
    plan = public_instruction_plan("Proceed appropriately.")
    assert len(plan) == 1
    assert plan[0]["subgoal_type"] == "ask"
    assert plan[0]["recovery"] == "stop"
