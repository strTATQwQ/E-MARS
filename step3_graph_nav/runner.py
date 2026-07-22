from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from slow_benchmark.oracle_graph import MatterportGraph

from .client import Step3GraphNavClient
from .evaluation import CandidateSpec, GraphEpisodeState, candidate_specs
from .io import append_jsonl, read_jsonl, write_json
from .protocol import CandidateView, GraphNavRequest


def render_candidate_views(
    scene: Any,
    graph: MatterportGraph,
    node_id: int,
    yaw_rad: float,
    *,
    width: int,
    height: int,
    save_dir: Path | None = None,
) -> tuple[tuple[CandidateSpec, ...], tuple[CandidateView, ...]]:
    specs = candidate_specs(graph, node_id, yaw_rad)
    position = graph.nodes[node_id].camera_position
    candidates = []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        jpeg, _, _ = scene.render(position, spec.absolute_yaw_rad)
        if save_dir is not None:
            (save_dir / f"candidate_{spec.candidate_id:02d}.jpg").write_bytes(jpeg)
        candidates.append(
            CandidateView(
                candidate_id=spec.candidate_id,
                target_viewpoint_id=spec.target_viewpoint_id,
                relative_heading_deg=spec.relative_heading_deg,
                graph_distance_m=spec.graph_distance_m,
                jpeg=jpeg,
                width=width,
                height=height,
            )
        )
    return specs, tuple(candidates)


def validate_service_health(health: dict[str, Any]) -> None:
    required = {
        "ok": True,
        "ready": True,
        "service": "step3_split_policy",
        "split_protocol_version": 2,
        "model_variant": "step3_vl_10b_bf16",
        "precision_mode": "bf16",
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": 32,
        "reasoning_tokens": 20,
        "batch_size": 1,
        "pacore": False,
        "retries": 0,
        "fallback": "none",
        "reasoning_mode": "bounded_private_then_schema_score",
        "navigation_schema": "move_only_candidate_id",
        "arrival_schema": "arrived_boolean_only",
        "global_stop_logit_bias": False,
        "split_policy": True,
        "stop_logit_bias": False,
    }
    mismatches = {
        key: {"required": value, "actual": health.get(key)}
        for key, value in required.items()
        if health.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Step3 graph-nav health mismatch: {mismatches}")


def run_model_scene(
    *,
    scene: Any,
    graph: MatterportGraph,
    episodes: Iterable[dict[str, Any]],
    client: Step3GraphNavClient,
    config: dict[str, Any],
    output: str | Path,
    run_id: str,
    gate: str,
    max_calls: int,
    same_node_visit_limit: int,
    global_early_stop_after_failures: int | None = None,
) -> dict[str, Any]:
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    episodes_path = output_path / "episodes.jsonl"
    decisions_path = output_path / "decisions.jsonl"
    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])
    gate_stopped = False
    gate_stop_reason = ""

    for episode in episodes:
        state = GraphEpisodeState.from_episode(graph, episode)
        started = time.perf_counter()
        failure_reason = "max_model_calls"
        calls = 0
        history: list[str] = []
        invalid_id = False
        parse_failure = False
        terminal_frame_dir = ""
        episode_dir = output_path / "snapshots" / state.episode_id.replace(":", "__")
        for step_index in range(max_calls):
            snapshot_id = f"{state.episode_id}:{step_index}:{state.current_node}"
            frame_dir = episode_dir / f"step_{step_index:02d}_node_{state.current_node:04d}"
            specs, candidates = render_candidate_views(
                scene,
                graph,
                state.current_node,
                state.yaw_rad,
                width=width,
                height=height,
                save_dir=frame_dir,
            )
            request = GraphNavRequest(
                episode_id=state.episode_id,
                snapshot_id=snapshot_id,
                instruction=state.instruction,
                current_viewpoint_id=state.current_node,
                step_index=step_index,
                candidates=candidates,
                history=tuple(history[-6:]),
            )
            response = client.decide(request)
            calls += 1
            action = response.get("action")
            positive_before = state.goal_positive
            decision_row = {
                "run_id": run_id,
                "gate": gate,
                "episode_id": state.episode_id,
                "scene_id": graph.scene_id,
                "snapshot_id": snapshot_id,
                "step_index": step_index,
                "current_viewpoint_id": state.current_node,
                "yaw_rad": state.yaw_rad,
                "goal_positive": positive_before,
                "goal_ne_m": state.ne_m,
                "candidate_order": [spec.to_mapping() for spec in specs],
                "raw_output": response.get("raw_output", ""),
                "parse_ok": bool(response.get("parse_ok")),
                "candidate_id_valid": bool(response.get("candidate_id_valid")),
                "action": action,
                "parse_error": response.get("parse_error", ""),
                "error_type": response.get("error_type", ""),
                "metrics": response.get("metrics", {}),
                "frame_dir": str(frame_dir.relative_to(output_path)),
                "frame_role": "first_decision" if step_index == 0 else "decision",
            }
            if not response.get("parse_ok"):
                invalid_id = response.get("error_type") == "InvalidCandidateId"
                parse_failure = not invalid_id
                failure_reason = "invalid_candidate_id" if invalid_id else "parse_failure"
                gate_stopped = invalid_id
                gate_stop_reason = failure_reason if invalid_id else ""
                decision_row["frame_role"] = "terminal"
                terminal_frame_dir = str(frame_dir.relative_to(output_path))
                append_jsonl(decisions_path, decision_row)
                break
            if action["action"] == "stop":
                state.stop()
                failure_reason = "" if state.success else "false_stop"
                decision_row["frame_role"] = "terminal"
                terminal_frame_dir = str(frame_dir.relative_to(output_path))
                append_jsonl(decisions_path, decision_row)
                break
            candidate_id = int(action["candidate_id"])
            spec = specs[candidate_id]
            state.move(spec)
            history.append(
                f"moved candidate_id={candidate_id}; relative_heading_deg={spec.relative_heading_deg:.1f}"
            )
            append_jsonl(decisions_path, decision_row)
            if Counter(state.visited_nodes)[state.current_node] >= same_node_visit_limit:
                failure_reason = "loop_same_node_visit_limit"
                break
        if not terminal_frame_dir:
            terminal_dir = episode_dir / f"terminal_node_{state.current_node:04d}"
            terminal_specs, _ = render_candidate_views(
                scene,
                graph,
                state.current_node,
                state.yaw_rad,
                width=width,
                height=height,
                save_dir=terminal_dir,
            )
            terminal_frame_dir = str(terminal_dir.relative_to(output_path))
            append_jsonl(
                output_path / "terminal_frames.jsonl",
                {
                    "run_id": run_id,
                    "gate": gate,
                    "episode_id": state.episode_id,
                    "scene_id": graph.scene_id,
                    "node_id": state.current_node,
                    "frame_dir": terminal_frame_dir,
                    "frame_role": "terminal",
                    "candidate_order": [spec.to_mapping() for spec in terminal_specs],
                    "failure_reason": failure_reason,
                },
            )
        result = state.result(
            calls=calls,
            failure_reason=failure_reason,
            wall_seconds=time.perf_counter() - started,
        )
        result.update(
            {
                "run_id": run_id,
                "gate": gate,
                "source_episode_id": episode.get("episode_id"),
                "invalid_candidate_id": invalid_id,
                "parse_failure": parse_failure,
                "terminal_frame_dir": terminal_frame_dir,
            }
        )
        append_jsonl(episodes_path, result)
        print(json.dumps(result, separators=(",", ":")), flush=True)
        if gate_stopped:
            break
        if global_early_stop_after_failures:
            rows = read_jsonl(episodes_path)
            run_rows = [row for row in rows if row.get("run_id") == run_id]
            if len(run_rows) >= global_early_stop_after_failures and not any(
                row.get("success") for row in run_rows[:global_early_stop_after_failures]
            ):
                gate_stopped = True
                gate_stop_reason = f"first_{global_early_stop_after_failures}_all_failed"
                break

    summary = summarize_run(episodes_path, decisions_path, run_id=run_id)
    summary.update({"gate_stopped": gate_stopped, "gate_stop_reason": gate_stop_reason})
    write_json(output_path / "summary.json", summary)
    return summary


def summarize_run(episodes_path: str | Path, decisions_path: str | Path, *, run_id: str) -> dict[str, Any]:
    episodes = [row for row in read_jsonl(episodes_path) if row.get("run_id") == run_id]
    decisions = [row for row in read_jsonl(decisions_path) if row.get("run_id") == run_id]
    latencies = sorted(float(row["metrics"].get("client_roundtrip_ms") or 0.0) for row in decisions)
    ttft = sorted(float(row["metrics"].get("prefill_ttft_ms") or 0.0) for row in decisions)
    decode_rates = [float(row["metrics"].get("decode_tokens_per_s") or 0.0) for row in decisions]

    def percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        index = (len(values) - 1) * q
        low = int(index)
        high = min(low + 1, len(values) - 1)
        fraction = index - low
        return values[low] * (1.0 - fraction) + values[high] * fraction

    return {
        "schema_version": 1,
        "run_id": run_id,
        "episodes": len(episodes),
        "successes": sum(bool(row.get("success")) for row in episodes),
        "success_rate": sum(bool(row.get("success")) for row in episodes) / max(len(episodes), 1),
        "oracle_successes": sum(bool(row.get("oracle_success")) for row in episodes),
        "oracle_success_rate": sum(bool(row.get("oracle_success")) for row in episodes) / max(len(episodes), 1),
        "mean_spl": sum(float(row.get("spl") or 0.0) for row in episodes) / max(len(episodes), 1),
        "mean_ne_m": sum(float(row.get("ne_m") or 0.0) for row in episodes) / max(len(episodes), 1),
        "stop_fp": sum(int(row.get("stop_fp") or 0) for row in episodes),
        "stop_fn": sum(int(row.get("stop_fn") or 0) for row in episodes),
        "loop_episodes": sum(row.get("failure_reason") == "loop_same_node_visit_limit" for row in episodes),
        "loop_rate": sum(row.get("failure_reason") == "loop_same_node_visit_limit" for row in episodes) / max(len(episodes), 1),
        "parse_failures": sum(not bool(row.get("parse_ok")) for row in decisions),
        "parse_failure_rate": sum(not bool(row.get("parse_ok")) for row in decisions) / max(len(decisions), 1),
        "invalid_candidate_ids": sum(row.get("error_type") == "InvalidCandidateId" for row in decisions),
        "model_calls": len(decisions),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "ttft_ms_p50": percentile(ttft, 0.50),
        "ttft_ms_p95": percentile(ttft, 0.95),
        "decode_tokens_per_s_mean": sum(decode_rates) / max(len(decode_rates), 1),
        "episode_wall_seconds": [float(row.get("wall_seconds") or 0.0) for row in episodes],
    }
