from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "internvla_ros2/internvla_ros2/client_node.py"


def test_reset_generation_precedes_every_ready_summary_in_constructor() -> None:
    tree = ast.parse(CLIENT.read_text(encoding="utf-8"))
    client_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InternVLAClientNode"
    )
    constructor = next(
        node
        for node in client_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    reset_assignments = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "reset_generation"
            for target in node.targets
        )
    ]
    ready_calls = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_write_client_summary"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "READY"
    ]
    assert len(reset_assignments) == 1
    assert len(ready_calls) == 1
    assert reset_assignments[0].lineno < ready_calls[0].lineno


def test_ready_summary_records_frozen_lane_qualified_reset_id() -> None:
    source = CLIENT.read_text(encoding="utf-8")
    assert '"reset_id": f"{self.identity_prefix}{self.reset_generation}"' in source
    assert 'self.identity_prefix not in {"", "a::", "b::"}' in source
    assert "getattr(self, \"reset_generation\"" not in source
