from types import SimpleNamespace

from omninav_step_scheduler.route_choice_verifier_node import RouteChoiceVerifierNode
from omninav_step_scheduler.semantic_stop_verifier_node import SemanticStopVerifierNode


def test_route_choice_verifier_ignores_triggers_when_step_disabled():
    node = object.__new__(RouteChoiceVerifierNode)
    node.active_mode_config = {"use_step": False}

    RouteChoiceVerifierNode.on_trigger(node, SimpleNamespace(data='{"type":"route_choice_upcoming"}'))


def test_semantic_stop_verifier_ignores_triggers_when_step_disabled():
    node = object.__new__(SemanticStopVerifierNode)
    node.active_mode_config = {"use_step": False}

    SemanticStopVerifierNode.on_trigger(node, SimpleNamespace(data='{"type":"candidate_goal_reached"}'))
