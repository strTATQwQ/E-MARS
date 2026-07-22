#!/usr/bin/env python3
"""Convert an InternNav progress log into a safe per-episode JSON summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FINISH = re.compile(
    r"\[(?P<ordinal>\d+)/(?P<total>\d+)\]"
    r"\[step_index:(?P<step_index>\d+)\] finish: "
    r"\[trajectory_id:(?P<trajectory_id>[^\]]+)\]"
    r"\[duration:(?P<duration>[0-9.]+) s\]"
    r"\[step_count:(?P<step_count>\d+)\]"
    r"\[fps:(?P<fps>[0-9.]+)\]"
    r"\[result:(?P<result>[^\]]+)\]"
)


def _metric_record(value: object, trajectory_id: str) -> dict[str, object] | None:
    if isinstance(value, dict):
        if {"NE", "success", "osr", "spl"}.issubset(value):
            observed = f"{value.get('trajectory_id')}_{value.get('episode_id')}"
            if observed == trajectory_id:
                return value
        for child in value.values():
            matched = _metric_record(child, trajectory_id)
            if matched is not None:
                return matched
    elif isinstance(value, list):
        for child in value:
            matched = _metric_record(child, trajectory_id)
            if matched is not None:
                return matched
    return None


def _metric_key(value: dict[str, object]) -> str | None:
    if not {"NE", "success", "osr", "spl"}.issubset(value):
        return None
    trajectory = value.get("trajectory_id")
    episode = value.get("episode_id")
    if trajectory is None or episode is None:
        return None
    return f"{trajectory}_{episode}"


def _collect_metric_records(
    value: object, output: dict[str, dict[str, object]]
) -> None:
    if isinstance(value, dict):
        key = _metric_key(value)
        if key is not None:
            previous = output.get(key)
            if previous is not None and previous != value:
                raise ValueError(f"conflicting evaluator metrics for {key}")
            output[key] = value
        for child in value.values():
            _collect_metric_records(child, output)
    elif isinstance(value, list):
        for child in value:
            _collect_metric_records(child, output)


def metric_records(path: Path) -> dict[str, dict[str, object]]:
    """Return exact evaluator records even when Kit prefixes/trails log text."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    decoder = json.JSONDecoder()
    output: dict[str, dict[str, object]] = {}
    for start, line in enumerate(lines):
        marker = line.find("{")
        if marker < 0:
            continue
        candidate = "\n".join(
            [line[marker:], *lines[start + 1 : min(len(lines), start + 2000)]]
        )
        try:
            value, _end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        _collect_metric_records(value, output)
    return output


def attach_official_metrics(
    episode: dict[str, object], metric: dict[str, object]
) -> None:
    official = {
        "sr": int(float(metric["success"]) > 0.0),
        "os": int(float(metric["osr"]) > 0.0),
        "spl": float(metric["spl"]),
        "ne_m": float(metric["NE"]),
    }
    previous = episode.get("official_metrics")
    if previous is not None and previous != official:
        raise ValueError(
            f"conflicting attached evaluator metrics for {episode['trajectory_id']}"
        )
    episode["official_metrics"] = official
    for name in ("TL", "ndtw", "shortest_path_length"):
        if name not in metric:
            continue
        value = float(metric[name])
        if name in episode and episode[name] != value:
            raise ValueError(
                f"conflicting attached evaluator {name} for "
                f"{episode['trajectory_id']}"
            )
        episode[name] = value


def _metrics_before_finish(
    lines: list[str], finish_index: int, trajectory_id: str
) -> dict[str, object] | None:
    # InternNav prints the authoritative evaluator JSON immediately before the
    # progress finish line.  Walk backwards over a bounded window and accept
    # only a complete JSON object carrying the same trajectory+episode ID.
    lower = max(0, finish_index - 1000)
    decoder = json.JSONDecoder()
    for start in range(finish_index - 1, lower - 1, -1):
        marker = lines[start].find("{")
        if marker < 0:
            continue
        candidate = "\n".join([lines[start][marker:], *lines[start + 1 : finish_index]])
        try:
            # Isaac/Kit may append timestamped diagnostic lines between the
            # evaluator's authoritative JSON object and the finish record.
            # Decode exactly the first complete JSON value and ignore only the
            # trailing log text; never synthesize metrics from aggregates.
            value, _end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        matched = _metric_record(value, trajectory_id)
        if matched is not None:
            return matched
    return None


def summarize(path: Path) -> dict[str, object]:
    episodes: list[dict[str, object]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line_index, line in enumerate(lines):
        match = FINISH.search(line)
        if match is None:
            continue
        item = match.groupdict()
        result = item["result"]
        episode: dict[str, object] = {
            "ordinal": int(item["ordinal"]),
            "trajectory_id": item["trajectory_id"],
            "step_index": int(item["step_index"]),
            "step_count": int(item["step_count"]),
            "duration_sec": float(item["duration"]),
            "fps": float(item["fps"]),
            "termination_reason": result,
            "success": result == "success",
        }
        metric = _metrics_before_finish(lines, line_index, item["trajectory_id"])
        if metric is not None:
            attach_official_metrics(episode, metric)
        episodes.append(episode)
    totals = {
        int(match.group("total"))
        for line in lines
        if (match := FINISH.search(line)) is not None
    }
    expected = totals.pop() if len(totals) == 1 else 0
    completed = len(episodes)
    success_count = sum(bool(item["success"]) for item in episodes)
    return {
        "schema_version": 1,
        "expected_episode_count": expected,
        "completed_episode_count": completed,
        "success_count": success_count,
        "success_rate": success_count / completed if completed else 0.0,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("progress_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = summarize(args.progress_log.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["completed_episode_count"] != payload["expected_episode_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
