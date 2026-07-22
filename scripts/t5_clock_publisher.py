#!/usr/bin/env python3
"""Publish the sole T5 /clock from simulator-step datagrams on ISAAC_X86."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class T5ClockPublisher(Node):
    def __init__(self, port: int, result: Path, state: Path):
        super().__init__("internnav_t5_isaac_clock")
        self.publisher = self.create_publisher(Clock, "/clock", 10)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.result = result
        self.state = state
        self.started_unix = time.time()
        self.last_clock_ns = time.time_ns()
        self.received_any = False
        self.received_count = 0
        self.publish_count = 0
        self.regression_count = 0
        self.invalid_count = 0
        self.last_state_write_monotonic = 0.0
        self.timer = self.create_timer(0.02, self._tick)

    def _write_state(self, status: str) -> None:
        _atomic_json(
            self.state,
            {
                "schema_version": 1,
                "status": status,
                "clock_source_host_role": "isaac_x86",
                "started_unix": self.started_unix,
                "updated_unix": time.time(),
                "last_clock_ns": self.last_clock_ns,
                "received_step_count": self.received_count,
                "publish_count": self.publish_count,
                "regression_count": self.regression_count,
                "invalid_count": self.invalid_count,
            },
        )

    def _tick(self) -> None:
        while True:
            try:
                payload, address = self.sock.recvfrom(128)
            except BlockingIOError:
                break
            if address[0] != "127.0.0.1":
                self.invalid_count += 1
                continue
            try:
                value = int(payload.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                self.invalid_count += 1
                continue
            if value <= 0:
                self.invalid_count += 1
                continue
            # Bootstrap publishes wall-derived ticks so ROS lifecycle timers
            # can start before Isaac emits physics steps.  The first real
            # simulator step establishes the authoritative clock baseline;
            # only subsequent simulator steps are subject to monotonicity.
            if self.received_any and value <= self.last_clock_ns:
                self.regression_count += 1
                continue
            self.last_clock_ns = value
            self.received_count += 1
            self.received_any = True
        if not self.received_any:
            # Allow ROS lifecycle timers to progress while Isaac is starting.
            # Once the first physics step arrives, only simulator steps advance
            # the clock; model pauses therefore remain simulation-time pauses.
            self.last_clock_ns = max(self.last_clock_ns + 20_000_000, time.time_ns())
        message = Clock()
        message.clock.sec = self.last_clock_ns // 1_000_000_000
        message.clock.nanosec = self.last_clock_ns % 1_000_000_000
        self.publisher.publish(message)
        self.publish_count += 1
        now = time.monotonic()
        if now - self.last_state_write_monotonic >= 0.5:
            self._write_state("RUNNING")
            self.last_state_write_monotonic = now

    def finish(self) -> None:
        self.sock.close()
        self._write_state("STOPPED")
        _atomic_json(
            self.result,
            {
                "schema_version": 1,
                "status": "PASS"
                if self.publish_count > 0
                and self.received_count > 0
                and self.regression_count == 0
                and self.invalid_count == 0
                else "FAIL",
                "clock_source_host_role": "isaac_x86",
                "bind": "127.0.0.1",
                "started_unix": self.started_unix,
                "finished_unix": time.time(),
                "last_clock_ns": self.last_clock_ns,
                "received_step_count": self.received_count,
                "publish_count": self.publish_count,
                "regression_count": self.regression_count,
                "invalid_count": self.invalid_count,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=25141)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    arguments, ros_args = parser.parse_known_args()
    if not 1024 <= arguments.port <= 65535:
        raise SystemExit("invalid clock UDP port")
    arguments.result.parent.mkdir(parents=True, exist_ok=True)
    arguments.state.parent.mkdir(parents=True, exist_ok=True)
    if arguments.result.exists():
        raise SystemExit("refusing to overwrite clock result")
    if arguments.state.exists():
        raise SystemExit("refusing to overwrite clock live state")
    rclpy.init(args=ros_args)
    node = T5ClockPublisher(
        arguments.port, arguments.result.resolve(), arguments.state.resolve()
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
