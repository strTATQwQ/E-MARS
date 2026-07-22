#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import statistics
import time
from pathlib import Path
from typing import Any


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]


def summarize(samples: list[dict[str, Any]], *, warmup_sec: float) -> dict[str, Any]:
    usable = [row for row in samples if float(row["elapsed_sec"]) >= warmup_sec]
    speeds = [math.hypot(float(row["vx"]), float(row["vy"])) for row in usable]
    yaw_rates = [abs(float(row["wz"])) for row in usable]
    heights = [float(row["z"]) for row in samples]
    progress = 0.0
    if len(usable) >= 2:
        progress = math.hypot(
            float(usable[-1]["x"]) - float(usable[0]["x"]),
            float(usable[-1]["y"]) - float(usable[0]["y"]),
        )
    return {
        "samples": len(samples),
        "usable_samples": len(usable),
        "progress_m": round(progress, 6),
        "speed_mean_mps": round(statistics.fmean(speeds), 6) if speeds else None,
        "speed_p95_mps": round(float(percentile(speeds, 0.95)), 6) if speeds else None,
        "speed_max_mps": round(max(speeds), 6) if speeds else None,
        "yaw_p95_radps": round(float(percentile(yaw_rates, 0.95)), 6) if yaw_rates else None,
        "yaw_max_radps": round(max(yaw_rates), 6) if yaw_rates else None,
        "min_root_z_m": round(min(heights), 6) if heights else None,
        "fell": bool(heights and min(heights) < 0.18),
    }


def drain(sock: socket.socket, rows: list[dict[str, Any]], started: float, label: str) -> None:
    while True:
        try:
            payload, _addr = sock.recvfrom(65535)
        except BlockingIOError:
            return
        data = json.loads(payload.decode("utf-8"))
        pose = data.get("pose") or []
        linear = data.get("linear_velocity") or []
        angular = data.get("angular_velocity") or []
        if len(pose) < 3 or len(linear) < 2 or len(angular) < 3:
            continue
        rows.append(
            {
                "label": label,
                "elapsed_sec": time.monotonic() - started,
                "seq": data.get("seq"),
                "event": data.get("event"),
                "x": pose[0],
                "y": pose[1],
                "yaw": pose[2],
                "z": data.get("z"),
                "vx": linear[0],
                "vy": linear[1],
                "wz": angular[2],
            }
        )


def send(sock: socket.socket, address: tuple[str, int], payload: dict[str, Any]) -> None:
    sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), address)


def run_stage(
    *,
    tx: socket.socket,
    rx: socket.socket,
    control_address: tuple[str, int],
    command_address: tuple[str, int],
    label: str,
    vx: float,
    wz: float,
    settle_sec: float,
    duration_sec: float,
    hz: float,
) -> list[dict[str, Any]]:
    send(
        tx,
        control_address,
        {"event": "reset", "episode_id": f"policy_probe_{label}", "task_id": label, "scene_id": "flat_ground", "pose": [0.0, 0.0, 0.0]},
    )
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    seq = 0
    period = 1.0 / hz
    while time.monotonic() - started < settle_sec + duration_sec:
        elapsed = time.monotonic() - started
        command_vx = 0.0 if elapsed < settle_sec else vx
        command_wz = 0.0 if elapsed < settle_sec else wz
        send(
            tx,
            command_address,
            {"vx": command_vx, "vy": 0.0, "wz": command_wz, "seq": seq, "timestamp": time.time(), "source": "policy_velocity_probe"},
        )
        seq += 1
        drain(rx, rows, started, label)
        if rows and float(rows[-1]["z"]) < 0.17:
            break
        time.sleep(period)
    drain(rx, rows, started, label)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Characterize the Isaac Go2 locomotion policy response to raw velocity commands.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=15002)
    parser.add_argument("--control-port", type=int, default=15011)
    parser.add_argument("--telemetry-port", type=int, default=15010)
    parser.add_argument("--commands", default="0.20,0.26,0.30,0.35,0.40,0.45")
    parser.add_argument("--settle-sec", type=float, default=5.0)
    parser.add_argument("--duration-sec", type=float, default=12.0)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((args.host, args.telemetry_port))
    rx.setblocking(False)
    all_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    try:
        for command in [float(item) for item in args.commands.split(",") if item.strip()]:
            label = f"vx_{command:.2f}".replace(".", "p")
            rows = run_stage(
                tx=tx,
                rx=rx,
                control_address=(args.host, args.control_port),
                command_address=(args.host, args.command_port),
                label=label,
                vx=command,
                wz=0.0,
                settle_sec=args.settle_sec,
                duration_sec=args.duration_sec,
                hz=args.hz,
            )
            all_rows.extend(rows)
            summary = summarize(rows, warmup_sec=args.settle_sec + args.warmup_sec)
            results.append({"label": label, "command_vx_mps": command, "command_wz_radps": 0.0, **summary})
            print(json.dumps(results[-1], sort_keys=True), flush=True)
    finally:
        tx.close()
        rx.close()

    with (output / "telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]) if all_rows else ["label"])
        writer.writeheader()
        writer.writerows(all_rows)
    report = {"schema_version": 1, "qualification_evidence": False, "diagnostic_only": True, "results": results}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
