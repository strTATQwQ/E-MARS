from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:  # pragma: no cover
    rclpy = None
    Node = object
    String = None


VERB_PATTERNS = (
    ("verify", re.compile(r"\bverify\b", re.IGNORECASE)),
    ("pass", re.compile(r"\b(pass|past|skip)\b", re.IGNORECASE)),
    ("enter", re.compile(r"\b(enter|take|turn into|use)\b", re.IGNORECASE)),
    ("find", re.compile(r"\b(find|locate)\b", re.IGNORECASE)),
    ("approach", re.compile(r"\b(approach|go to|stop at|stop by)\b", re.IGNORECASE)),
)


def public_instruction_plan(instruction: str) -> list[dict[str, Any]]:
    """Build a motion-free baseline plan from public instruction text only."""
    text = " ".join(str(instruction or "").split())
    clauses = [
        clause.strip(" .,;")
        for clause in re.split(r"\bthen\b|[;,]|\band\b", text, flags=re.IGNORECASE)
        if clause.strip(" .,;")
    ]
    plan: list[dict[str, Any]] = []
    for clause in clauses:
        lowered = clause.lower()
        if lowered.startswith(("if ", "otherwise ", "but ")):
            continue
        subgoal_type = next((kind for kind, pattern in VERB_PATTERNS if pattern.search(clause)), "")
        if not subgoal_type:
            continue
        target = _target_from_clause(clause)
        if not target:
            continue
        recovery = _recovery_for_clause(clause, text, subgoal_type)
        plan.append(
            {
                "subgoal_type": subgoal_type,
                "target": target,
                "relation": _relation_from_clause(clause),
                "constraints": _constraints_from_text(text),
                "completion_evidence": _completion_evidence(subgoal_type, target),
                "recovery": recovery,
                "confidence": 0.65,
                "source": "public_instruction_heuristic",
            }
        )
    if not plan:
        plan.append(
            {
                "subgoal_type": "ask",
                "target": "operator clarification",
                "relation": "",
                "constraints": ["hold position"],
                "completion_evidence": "instruction is clarified",
                "recovery": "stop",
                "confidence": 0.65,
                "source": "public_instruction_heuristic",
            }
        )
        return plan
    if plan[-1]["subgoal_type"] != "verify":
        terminal_target = plan[-1]["target"]
        plan.append(
            {
                "subgoal_type": "verify",
                "target": terminal_target,
                "relation": "",
                "constraints": [],
                "completion_evidence": f"{terminal_target} arrival verified"[:120],
                "recovery": "stop",
                "confidence": 0.65,
                "source": "public_instruction_heuristic",
            }
        )
    return plan


def _target_from_clause(clause: str) -> str:
    value = clause
    value = re.sub(r"^\(\d+\)\s*", "", value)
    value = re.sub(r"^(first|finally|next)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"^(verify arrival at|verify|find|locate|approach|go to|stop at|stop by|pass|walk past|go past|enter|take|turn into|use|skip)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.split(r"\bwith recovery\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.split(r"\b(if|but|otherwise)\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    return " ".join(value.strip(" .,;").split())[:120]


def _relation_from_clause(clause: str) -> str:
    match = re.search(
        r"\b(beside|next to|to the left of|to the right of|after|beyond|between|under|above)\b(.+)$",
        clause,
        flags=re.IGNORECASE,
    )
    return " ".join(match.group(0).strip(" .,;").split())[:160] if match else ""


def _constraints_from_text(text: str) -> list[str]:
    match = re.search(r"\b(avoid(?: entering)? [^,;.]+)", text, flags=re.IGNORECASE)
    return [" ".join(match.group(1).split())[:120]] if match else []


def _recovery_for_clause(clause: str, instruction: str, subgoal_type: str) -> str:
    explicit = re.search(r"\bwith recovery\s+(scan|backtrack|ask|stop)\b", clause, flags=re.IGNORECASE)
    if explicit:
        return explicit.group(1).lower()
    if subgoal_type == "verify":
        return "stop"
    context = f"{clause} {instruction}".lower()
    if "ask" in context or "clarification" in context:
        return "ask"
    if "backtrack" in context or "another visible corridor" in context or "choose another corridor" in context:
        return "backtrack"
    if "scan" in context or "disappears" in context or "uncertain" in context:
        return "scan"
    return "backtrack" if subgoal_type == "enter" else "scan"


def _completion_evidence(subgoal_type: str, target: str) -> str:
    suffix = {
        "find": "track confirmed",
        "pass": "landmark passed",
        "enter": "region entered",
        "approach": "target approached",
    }.get(subgoal_type, "stage completed")
    return f"{target} {suffix}"[:120]


class PublicSemanticHeuristicNode(Node):
    def __init__(self) -> None:
        if rclpy is None:
            raise RuntimeError("rclpy is required to run PublicSemanticHeuristicNode")
        super().__init__("public_semantic_heuristic")
        self.enabled = False
        self.episode_id = ""
        self.plan: list[dict[str, Any]] = []
        self.pending_index: int | None = None
        self.published_indices: set[int] = set()
        self.decision_pub = self.create_publisher(String, "/step/semantic_subgoal_json", 10)
        self.metric_pub = self.create_publisher(String, "/metrics/event_jsonl", 100)
        self.create_subscription(String, "/benchmark/mode_json", self.on_mode, 10)
        self.create_subscription(String, "/user_instruction", self.on_instruction, 10)
        self.create_subscription(String, "/scheduler/step_trigger_json", self.on_trigger, 10)

    def on_mode(self, msg: Any) -> None:
        payload = _safe_json(msg.data)
        mode_cfg = payload.get("mode_config") if isinstance(payload.get("mode_config"), dict) else {}
        self.enabled = bool(mode_cfg.get("public_semantic_heuristic", False))
        self.episode_id = str(payload.get("episode_id") or "")
        self.plan = []
        self.pending_index = None
        self.published_indices = set()
        self.publish_metric(
            "public_semantic_heuristic_configured",
            result="ready" if self.enabled else "disabled",
            episode_id=self.episode_id,
            oracle_inputs=False,
        )

    def on_instruction(self, msg: Any) -> None:
        if not self.enabled:
            return
        payload = _safe_json(msg.data)
        instruction = str(payload.get("instruction") or msg.data or "").strip()
        self.plan = public_instruction_plan(instruction)
        self.publish_metric(
            "public_semantic_heuristic_plan",
            result="parsed_public_instruction",
            episode_id=self.episode_id,
            subgoal_count=len(self.plan),
            reads_task_file=False,
            reads_oracle_plan=False,
        )
        if self.pending_index is not None:
            index = self.pending_index
            self.pending_index = None
            self.publish_index(index)

    def on_trigger(self, msg: Any) -> None:
        if not self.enabled:
            return
        event = _safe_json(msg.data)
        if str(event.get("type") or "") not in {"semantic_plan_requested", "semantic_subgoal_completed"}:
            return
        index = int(event.get("next_subgoal_index", 0) or 0)
        if not self.plan:
            self.pending_index = index
            return
        self.publish_index(index)

    def publish_index(self, index: int) -> None:
        if index in self.published_indices or index < 0 or index >= len(self.plan):
            return
        self.published_indices.add(index)
        payload = self._stamp(self.plan[index] | {"subgoal_index": index})
        self.decision_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.publish_metric(
            "public_semantic_heuristic_subgoal",
            result="published",
            episode_id=self.episode_id,
            subgoal_index=index,
            subgoal_type=payload["subgoal_type"],
            requires_stale_gate=True,
            publishes_motion=False,
        )

    def _stamp(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        stamp = float(self.get_clock().now().nanoseconds) * 1e-9
        wall = time.time()
        request_id = f"public_heuristic_{uuid.uuid4().hex[:12]}"
        clock_domain = "ros_sim" if bool(self.get_parameter("use_sim_time").value) else "ros_system"
        timebase = {
            "episode_id": self.episode_id,
            "mission_id": "",
            "request_id": request_id,
            "clock_domain": clock_domain,
            "ros_now_sec": stamp,
            "wall_now_sec": wall,
            "clock_msg_sec": stamp,
            "header_stamp_sec": stamp,
            "source_stamp_sec": stamp,
            "created_ros_time_sec": stamp,
            "created_wall_time_sec": wall,
        }
        result.update(timebase)
        result.update({"request_id": request_id, "timestamp_response": stamp, "timebase": timebase})
        return result

    def publish_metric(self, event_type: str, **kwargs: Any) -> None:
        payload = {"event": event_type, "timestamp": time.time(), **kwargs}
        self.metric_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PublicSemanticHeuristicNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
