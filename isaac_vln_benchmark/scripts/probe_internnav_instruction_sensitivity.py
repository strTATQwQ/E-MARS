#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import pickle
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - allows --help and dry inspection on minimal hosts
    np = None

try:
    from omninav_step_scheduler.internnav_go2_client_node import make_agent_config
except Exception:  # pragma: no cover
    make_agent_config = None


INSTRUCTIONS = [
    ("forward", "Move forward to the blue box."),
    ("left", "Turn left toward the red cone."),
    ("right", "Turn right toward the blue box."),
    ("stop", "Stop near the fire extinguisher."),
    ("look", "Look around and do not move forward."),
    ("stop_ignore", "Ignore the target and stop."),
]

DEFAULT_TOKENS = [101, 102, 103, 104, 105]


def post_json(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {body}") from exc


def try_post_json(url: str, payload: dict[str, Any], timeout_sec: float) -> tuple[bool, dict[str, Any] | str]:
    try:
        return True, post_json(url, payload, timeout_sec)
    except Exception as exc:
        return False, repr(exc)


def serialize_obs(obs: Any) -> str:
    return base64.b64encode(pickle.dumps(obs)).decode("utf-8")


def observation_for_server(obs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for item in obs:
        clean = dict(item)
        clean.pop("instruction_context", None)
        sanitized.append(clean)
    return sanitized


def synthetic_rgb(sample_index: int, height: int, width: int) -> np.ndarray:
    if np is None:
        return [
            [[int((x + sample_index) % 255), int((y + sample_index) % 255), 90] for x in range(width)]
            for y in range(height)
        ]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[..., 0] = np.mod(x + 0.017 * sample_index, 1.0)
    img[..., 1] = np.mod(y + 0.013 * sample_index, 1.0)
    img[..., 2] = 0.35
    img[:, width // 2 - 8 : width // 2 + 8, 1] = 0.95
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def synthetic_depth(sample_index: int, height: int, width: int) -> np.ndarray:
    if np is None:
        return [[[min(1.0, 0.08 + 0.47 * y / max(1, height - 1))] for _x in range(width)] for y in range(height)]
    base = np.linspace(0.08, 0.55, height, dtype=np.float32)[:, None]
    ripple = 0.015 * math.sin(sample_index * 0.5)
    return np.repeat(np.clip(base + ripple, 0.0, 1.0), width, axis=1)[..., None].astype(np.float32)


def make_observation(sample_index: int, instruction: str, height: int, width: int) -> list[dict[str, Any]]:
    rgb = synthetic_rgb(sample_index, height, width)
    depth = synthetic_depth(sample_index, height, width)
    return make_observation_from_arrays(
        instruction,
        rgb,
        depth,
        [float(sample_index) * 0.01, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )


def make_observation_from_arrays(
    instruction: str,
    rgb: Any,
    depth: Any,
    gps: list[float],
    rotation: list[float],
) -> list[dict[str, Any]]:
    return [
        {
            "instruction": instruction,
            "instruction_tokens": list(DEFAULT_TOKENS),
            "instruction_context": {
                "instruction": instruction,
                "subgoal": instruction,
                "success_condition": "probe only",
                "token_source": "fallback_static_cma_tokens",
                "token_count": len(DEFAULT_TOKENS),
            },
            "rgb": array_like_uint8(rgb),
            "depth": array_like_float32(depth),
            "globalgps": gps,
            "globalrotation": rotation,
        }
    ]


def array_like_uint8(value: Any) -> Any:
    if np is None:
        return value
    return value.astype(np.uint8, copy=False)


def array_like_float32(value: Any) -> Any:
    if np is None:
        return value
    return value.astype(np.float32, copy=False)


def extract_action(response: dict[str, Any]) -> str:
    value: Any = response.get("action")
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        value = value.get("action")
    if isinstance(value, list) and value:
        value = value[0]
    try:
        code = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return {-1: "stop", 0: "unknown", 1: "forward", 2: "left", 3: "right"}.get(code, "unknown")


def capture_ros_observations(
    *,
    count: int,
    height: int,
    width: int,
    timeout_sec: float,
    image_topic: str,
    depth_topic: str,
    odom_topic: str,
) -> list[dict[str, Any]]:
    if np is None:
        raise RuntimeError("ROS snapshot mode requires numpy")
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    class SnapshotNode(Node):
        def __init__(self) -> None:
            super().__init__("internnav_instruction_sensitivity_snapshot")
            self.rgb_msg = None
            self.depth_msg = None
            self.odom_msg = None
            self.create_subscription(Image, image_topic, self.on_rgb, 2)
            self.create_subscription(Image, depth_topic, self.on_depth, 2)
            self.create_subscription(Odometry, odom_topic, self.on_odom, 10)

        def on_rgb(self, msg):
            self.rgb_msg = msg

        def on_depth(self, msg):
            self.depth_msg = msg

        def on_odom(self, msg):
            self.odom_msg = msg

    if not rclpy.ok():
        rclpy.init(args=None)
        owns_rclpy = True
    else:
        owns_rclpy = False
    node = SnapshotNode()
    snapshots: list[dict[str, Any]] = []
    try:
        deadline = time.monotonic() + timeout_sec
        while len(snapshots) < count and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.rgb_msg is None or node.depth_msg is None or node.odom_msg is None:
                continue
            rgb = resize_nearest(ros_image_to_rgb(node.rgb_msg), height, width)
            depth = resize_nearest(ros_image_to_depth(node.depth_msg), height, width)
            pose = node.odom_msg.pose.pose
            snapshots.append(
                {
                    "rgb": rgb,
                    "depth": depth,
                    "gps": [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
                    "rotation": [
                        float(pose.orientation.x),
                        float(pose.orientation.y),
                        float(pose.orientation.z),
                        float(pose.orientation.w),
                    ],
                    "source": "ros_snapshot",
                }
            )
            node.rgb_msg = None
            node.depth_msg = None
            node.odom_msg = None
        if len(snapshots) < count:
            raise TimeoutError(f"only captured {len(snapshots)}/{count} ROS snapshots in {timeout_sec:.1f}s")
        return snapshots
    finally:
        node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def ros_image_to_rgb(msg: Any) -> Any:
    encoding = str(getattr(msg, "encoding", "rgb8")).lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if encoding in {"rgb8", "bgr8"}:
        channels = 3
    elif encoding in {"rgba8", "bgra8"}:
        channels = 4
    elif encoding in {"mono8", "8uc1"}:
        channels = 1
    else:
        raise ValueError(f"unsupported rgb encoding: {encoding!r}")
    arr = data.reshape((height, step))[:, : width * channels].reshape((height, width, channels))
    if channels == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif encoding in {"bgr8", "bgra8"}:
        arr = arr[..., :3][..., ::-1]
    else:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=True)


def ros_image_to_depth(msg: Any) -> Any:
    encoding = str(getattr(msg, "encoding", "32FC1")).lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    if encoding in {"32fc1", "32fc"}:
        raw = np.frombuffer(msg.data, dtype=np.float32)
        arr = raw.reshape((height, step // 4))[:, :width]
    elif encoding in {"16uc1", "mono16"}:
        raw = np.frombuffer(msg.data, dtype=np.uint16)
        arr = raw.reshape((height, step // 2))[:, :width].astype(np.float32) / 1000.0
    elif encoding in {"mono8", "8uc1"}:
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        arr = raw.reshape((height, step))[:, :width].astype(np.float32) / 255.0 * 5.0
    else:
        raise ValueError(f"unsupported depth encoding: {encoding!r}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=0.0)
    return np.clip(arr / 10.0, 0.0, 1.0).astype(np.float32)[..., None]


def resize_nearest(arr: Any, height: int, width: int) -> Any:
    if arr.shape[0] == height and arr.shape[1] == width:
        return arr.copy()
    ys = np.linspace(0, arr.shape[0] - 1, height).astype(np.int64)
    xs = np.linspace(0, arr.shape[1] - 1, width).astype(np.int64)
    return arr[ys][:, xs].copy()


def mutual_information(records: list[dict[str, Any]]) -> float:
    total = len(records)
    if not total:
        return 0.0
    joint = Counter((row["instruction_label"], row["model_action"]) for row in records)
    labels = Counter(row["instruction_label"] for row in records)
    actions = Counter(row["model_action"] for row in records)
    mi = 0.0
    for (label, action), count in joint.items():
        p_xy = count / total
        p_x = labels[label] / total
        p_y = actions[action] / total
        if p_xy > 0.0 and p_x > 0.0 and p_y > 0.0:
            mi += p_xy * math.log(p_xy / (p_x * p_y), 2)
    return mi


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_sample[row["sample_id"]].append(row)

    comparisons = 0
    changes = 0
    same = 0
    for rows in by_sample.values():
        forward_actions = [row["model_action"] for row in rows if row["instruction_label"] == "forward"]
        if not forward_actions:
            continue
        baseline = forward_actions[0]
        for row in rows:
            if row["instruction_label"] == "forward":
                continue
            comparisons += 1
            if row["model_action"] == baseline:
                same += 1
            else:
                changes += 1

    def hit_rate(label: str, action: str) -> float:
        rows = [row for row in records if row["instruction_label"] == label]
        return sum(1 for row in rows if row["model_action"] == action) / max(len(rows), 1)

    same_action_rate = same / max(comparisons, 1)
    summary = {
        "records": len(records),
        "sample_count": len(by_sample),
        "action_change_rate": changes / max(comparisons, 1),
        "same_action_rate": same_action_rate,
        "instruction_action_mutual_info": mutual_information(records),
        "left_instruction_left_rate": hit_rate("left", "left"),
        "right_instruction_right_rate": hit_rate("right", "right"),
        "stop_instruction_stop_rate": hit_rate("stop", "stop"),
        "forward_instruction_forward_rate": hit_rate("forward", "forward"),
        "action_counts": dict(Counter(row["model_action"] for row in records)),
    }
    if same_action_rate > 0.8:
        conclusion = "InternNav current deployment appears largely insensitive to language instruction."
    elif summary["action_change_rate"] > 0.4 and summary["instruction_action_mutual_info"] > 0.1:
        conclusion = "InternNav behaves like a weak language-conditioned executor on this probe."
    else:
        conclusion = "InternNav appears partially language-conditioned or weakly sensitive on this probe."
    summary["conclusion"] = conclusion
    return summary


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# InternNav Instruction Sensitivity Summary",
        "",
        f"- records: {summary['records']}",
        f"- sample_count: {summary['sample_count']}",
        f"- action_change_rate: {summary['action_change_rate']:.3f}",
        f"- same_action_rate: {summary['same_action_rate']:.3f}",
        f"- instruction_action_mutual_info: {summary['instruction_action_mutual_info']:.4f}",
        f"- left_instruction_left_rate: {summary['left_instruction_left_rate']:.3f}",
        f"- right_instruction_right_rate: {summary['right_instruction_right_rate']:.3f}",
        f"- stop_instruction_stop_rate: {summary['stop_instruction_stop_rate']:.3f}",
        f"- forward_instruction_forward_rate: {summary['forward_instruction_forward_rate']:.3f}",
        f"- action_counts: `{json.dumps(summary['action_counts'], sort_keys=True)}`",
        "",
        f"Conclusion: {summary['conclusion']}",
        "",
        "Caveat: this probes the deployed CmaAgent/system1 path with fallback_static_cma_tokens unless the audit file proves otherwise.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_agent_config(args: argparse.Namespace, server_host: str, server_port: int) -> dict[str, Any]:
    if make_agent_config is not None:
        return make_agent_config(
            server_host=server_host,
            server_port=server_port,
            model_name=args.agent_name,
            ckpt_path=args.ckpt_path,
        )
    return {
        "server_host": server_host,
        "server_port": server_port,
        "model_name": args.agent_name,
        "ckpt_path": args.ckpt_path,
        "model_settings": {"env_num": 1, "proc_num": 1},
    }


def ensure_agent(base_url: str, args: argparse.Namespace, server_host: str, server_port: int) -> str:
    agent_name = str(args.agent_name)
    if not args.force_init:
        ok, response = try_post_json(
            f"{base_url}/agent/{agent_name}/reset",
            {"reset_index": None},
            args.timeout_sec,
        )
        if ok:
            return agent_name
        print(f"Existing agent reset did not succeed; initializing {agent_name}: {response}")

    init_payload = {"agent_config": build_agent_config(args, server_host, server_port)}
    init_response = post_json(f"{base_url}/agent/init", init_payload, args.timeout_sec)
    return str(init_response.get("agent_name") or agent_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internnav-server", required=True)
    parser.add_argument("--isaac-host", default="")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--ros-snapshot", action="store_true")
    parser.add_argument("--ros-snapshot-timeout-sec", type=float, default=60.0)
    parser.add_argument("--image-topic", default="/camera/front/image")
    parser.add_argument("--depth-topic", default="/camera/front/depth")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--server-model-class", default="CmaAgent")
    parser.add_argument("--server-mode", default="system1")
    parser.add_argument("--agent-name", default="cma")
    parser.add_argument("--ckpt-path", default="checkpoints/r2r/fine_tuned/cma_plus")
    parser.add_argument("--force-init", action="store_true")
    parser.add_argument("--reset-between-actions", action="store_true", default=True)
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base_url = args.internnav_server.rstrip("/")

    server_host, server_port = parse_server_host_port(base_url)
    agent_name = ensure_agent(base_url, args, server_host, server_port)

    if args.ros_snapshot:
        snapshots = capture_ros_observations(
            count=args.samples,
            height=args.height,
            width=args.width,
            timeout_sec=args.ros_snapshot_timeout_sec,
            image_topic=args.image_topic,
            depth_topic=args.depth_topic,
            odom_topic=args.odom_topic,
        )
    else:
        snapshots = [
            {
                "rgb": synthetic_rgb(sample_index, args.height, args.width),
                "depth": synthetic_depth(sample_index, args.height, args.width),
                "gps": [float(sample_index) * 0.01, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "source": "synthetic_probe",
            }
            for sample_index in range(args.samples)
        ]

    records: list[dict[str, Any]] = []
    forward_by_sample: dict[tuple[str, int], str] = {}
    jsonl_path = output / "internnav_instruction_sensitivity.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample_index in range(args.samples):
            sample_id = f"obs_{sample_index:04d}"
            snapshot = snapshots[sample_index]
            for repeat_index in range(args.repeats):
                for label, instruction in INSTRUCTIONS:
                    if args.reset_between_actions:
                        post_json(f"{base_url}/agent/{agent_name}/reset", {"reset_index": None}, args.timeout_sec)
                    obs = make_observation_from_arrays(
                        instruction,
                        snapshot["rgb"],
                        snapshot["depth"],
                        snapshot["gps"],
                        snapshot["rotation"],
                    )
                    t0 = time.perf_counter()
                    response = post_json(
                        f"{base_url}/agent/{agent_name}/step",
                        {"observation": serialize_obs(observation_for_server(obs))},
                        args.timeout_sec,
                    )
                    latency = time.perf_counter() - t0
                    action = extract_action(response)
                    key = (sample_id, repeat_index)
                    if label == "forward":
                        forward_by_sample[key] = action
                    record = {
                        "sample_id": sample_id,
                        "repeat_index": repeat_index,
                        "instruction_label": label,
                        "instruction": instruction,
                        "model_action": action,
                        "applied_action": action,
                        "latency_sec": round(latency, 6),
                        "token_source": "fallback_static_cma_tokens",
                        "same_as_forward_instruction": None,
                        "server_model_class": args.server_model_class,
                        "server_mode": args.server_mode,
                        "observation_source": snapshot.get("source", "unknown"),
                        "raw_action": response.get("action"),
                    }
                    baseline = forward_by_sample.get(key)
                    if baseline is not None and label != "forward":
                        record["same_as_forward_instruction"] = action == baseline
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = summarize(records)
    (output / "internnav_instruction_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(output / "internnav_instruction_sensitivity_summary.md", summary)
    print(json.dumps({"output": str(output), **summary}, indent=2, sort_keys=True))
    return 0


def parse_server_host_port(base_url: str) -> tuple[str, int]:
    hostport = base_url.replace("http://", "").replace("https://", "").split("/", 1)[0]
    if ":" not in hostport:
        return hostport, 8087
    host, port = hostport.rsplit(":", 1)
    try:
        return host, int(port)
    except ValueError:
        return host, 8087


if __name__ == "__main__":
    raise SystemExit(main())
