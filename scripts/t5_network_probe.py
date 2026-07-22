#!/usr/bin/env python3
"""Measure the frozen T5 three-host network contract under a resource lease."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY_PATH = ROOT / "configs/internnav_t5/topology.json"
SSH_OPTIONS = (
    "-T",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "ServerAliveInterval=5",
    "-o",
    "ServerAliveCountMax=2",
    "-o",
    "StrictHostKeyChecking=accept-new",
)
PING_RE = re.compile(
    r"(?P<tx>\d+) packets transmitted, (?P<rx>\d+) received, "
    r"(?P<loss>[0-9.]+)% packet loss"
)
RTT_RE = re.compile(
    r"(?:rtt|round-trip) min/avg/max/(?:mdev|stddev) = "
    r"(?P<minimum>[0-9.]+)/(?P<average>[0-9.]+)/"
    r"(?P<maximum>[0-9.]+)/(?P<jitter>[0-9.]+) ms"
)


@dataclass(frozen=True)
class Host:
    role: str
    user: str
    ip: str

    @property
    def target(self) -> str:
        return f"{self.user}@{self.ip}"


def _run(
    argv: list[str] | tuple[str, ...],
    *,
    timeout: float = 30.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=check,
    )


def _ssh(host: Host, command: str, *, timeout: float = 30.0) -> str:
    completed = _run(
        ["ssh", *SSH_OPTIONS, host.target, command], timeout=timeout, check=True
    )
    return completed.stdout


def _ssh_popen(host: Host, command: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        ["ssh", *SSH_OPTIONS, host.target, command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _remote_python(host: Host, source: str, *, timeout: float = 30.0) -> str:
    return _ssh(host, f"python3 -c {shlex.quote(source)}", timeout=timeout)


def parse_ping(output: str) -> dict[str, float | int]:
    packets = PING_RE.search(output)
    rtt = RTT_RE.search(output)
    if packets is None or rtt is None:
        raise ValueError(f"unrecognized ping output: {output[-500:]}")
    return {
        "transmitted": int(packets.group("tx")),
        "received": int(packets.group("rx")),
        "packet_loss_percent": float(packets.group("loss")),
        "rtt_min_ms": float(rtt.group("minimum")),
        "rtt_avg_ms": float(rtt.group("average")),
        "rtt_max_ms": float(rtt.group("maximum")),
        "jitter_ms": float(rtt.group("jitter")),
    }


def _inventory(host: Host) -> dict[str, Any]:
    source = r'''
import json, os, pathlib, shutil, socket, subprocess

def command(argv):
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

addresses = json.loads(command(["ip", "-j", "address"]) or "[]")
routes = json.loads(command(["ip", "-j", "route"]) or "[]")
expected = os.environ.get("T5_EXPECTED_IP", "")
selected = None
for item in addresses:
    for info in item.get("addr_info", []):
        if info.get("local") == expected:
            selected = {
                "ifname": item.get("ifname"),
                "mtu": item.get("mtu"),
                "operstate": item.get("operstate"),
                "address": expected,
            }
process_counts = {"model": 0, "nav2": 0, "isaac": 0}
patterns = {
    "model": ("internvla_ros2.model_node", "internvla_t4_recovery.model_node"),
    "nav2": ("nav2_", "component_container"),
    "isaac": ("isaac-sim", "isaacsim", "omni.isaac"),
}
skip_pids = set()
pid = os.getpid()
while pid > 1 and pid not in skip_pids:
    skip_pids.add(pid)
    try:
        status = (pathlib.Path("/proc") / str(pid) / "status").read_text()
        pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
    except Exception:
        break
for proc in pathlib.Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    if int(proc.name) in skip_pids:
        continue
    try:
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
    except Exception:
        continue
    for name, values in patterns.items():
        if any(value in cmdline for value in values):
            process_counts[name] += 1
host_ros = pathlib.Path("/opt/ros/jazzy/setup.bash").is_file()
container = "internnav_t4_isaac_ros"
container_running = command(["docker", "inspect", "-f", "{{.State.Running}}", container]) == "true"
container_ros = False
if container_running:
    container_ros = command(["docker", "exec", container, "test", "-f", "/opt/ros/jazzy/setup.bash"]) == ""
payload = {
    "user": command(["id", "-un"]),
    "hostname": socket.gethostname(),
    "kernel": command(["uname", "-r"]),
    "selected_interface": selected,
    "default_routes": [item for item in routes if item.get("dst") == "default"],
    "gpu_csv": command(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"]),
    "memory_available_kib": next((int(line.split()[1]) for line in pathlib.Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")), None),
    "ntp_synchronized": command(["timedatectl", "show", "-p", "NTPSynchronized", "--value"]),
    "ros_jazzy_present": host_ros or container_ros,
    "ros_runtime": "host" if host_ros else ("docker_exec" if container_ros else "missing"),
    "ros_container": container if container_ros else None,
    "ros_container_running": container_running,
    "python3": shutil.which("python3"),
    "iperf3": shutil.which("iperf3"),
    "process_counts": process_counts,
}
print(json.dumps(payload, sort_keys=True))
'''
    command = f"T5_EXPECTED_IP={shlex.quote(host.ip)} python3 -c {shlex.quote(source)}"
    return json.loads(_ssh(host, command, timeout=30.0))


def _ping(source: Host, destination: Host) -> dict[str, float | int | str]:
    output = _ssh(
        source,
        f"ping -n -q -c 20 -i 0.05 -W 1 {shlex.quote(destination.ip)}",
        timeout=15.0,
    )
    return {
        "source": source.role,
        "destination": destination.role,
        **parse_ping(output),
    }


def _clock_pair(
    source: Host, reference: Host, port: int, count: int = 20
) -> dict[str, Any]:
    server_source = f'''
import socket, time
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind(("0.0.0.0",{port})); s.listen(4); s.settimeout(15)
for _ in range({count}):
    c,_=s.accept(); c.settimeout(5); c.recv(1); c.sendall((str(time.time_ns())+"\\n").encode()); c.close()
s.close()
'''
    client_source = f'''
import json, socket, time
samples=[]
for _ in range({count}):
    started=time.time_ns(); s=socket.create_connection(({reference.ip!r},{port}),timeout=5)
    s.sendall(b"x"); data=b""
    while not data.endswith(b"\\n"): data += s.recv(64)
    ended=time.time_ns(); s.close(); remote=int(data.strip()); midpoint=(started+ended)//2
    samples.append({{"rtt_ns":ended-started,"reference_minus_source_midpoint_ns":remote-midpoint}})
print(json.dumps(samples))
'''
    server = _ssh_popen(reference, f"timeout 20 python3 -c {shlex.quote(server_source)}")
    try:
        time.sleep(0.5)
        samples = json.loads(_remote_python(source, client_source, timeout=20.0))
        _, server_stderr = server.communicate(timeout=20.0)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                server.kill()
    if server.returncode != 0:
        raise RuntimeError(f"clock server failed {reference.role}: {server_stderr[-500:]}")
    best = min(samples, key=lambda item: item["rtt_ns"])
    offsets = sorted(item["reference_minus_source_midpoint_ns"] for item in samples)
    return {
        "source": source.role,
        "reference": reference.role,
        "sample_count": len(samples),
        "best_rtt_ns": best["rtt_ns"],
        "best_offset_ns": best["reference_minus_source_midpoint_ns"],
        "median_offset_ns": offsets[len(offsets) // 2],
        "samples": samples,
    }


def _bandwidth(source: Host, destination: Host, port: int) -> dict[str, Any]:
    byte_count = 32 * 1024 * 1024
    server_source = f'''
import json, socket, time
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind(("0.0.0.0",{port})); s.listen(1); s.settimeout(15)
c,_=s.accept(); c.settimeout(15); total=0; started=time.monotonic_ns()
while total < {byte_count}:
    data=c.recv(min(1048576,{byte_count}-total))
    if not data: break
    total += len(data)
ended=time.monotonic_ns(); c.close(); s.close()
print(json.dumps({{"bytes":total,"elapsed_ns":ended-started,"bits_per_second":total*8e9/max(1,ended-started)}}))
'''
    client_source = f'''
import json, socket, time
s=socket.create_connection(({destination.ip!r},{port}),timeout=10); s.settimeout(15)
chunk=b"\\0"*1048576; total=0; started=time.monotonic_ns()
while total < {byte_count}:
    count=min(len(chunk),{byte_count}-total); s.sendall(chunk[:count]); total += count
ended=time.monotonic_ns(); s.shutdown(socket.SHUT_WR); s.close()
print(json.dumps({{"bytes":total,"elapsed_ns":ended-started,"bits_per_second":total*8e9/max(1,ended-started)}}))
'''
    server = _ssh_popen(
        destination,
        f"timeout 20 python3 -c {shlex.quote(server_source)}",
    )
    try:
        time.sleep(0.5)
        client = json.loads(_remote_python(source, client_source, timeout=20.0))
        server_stdout, server_stderr = server.communicate(timeout=20.0)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                server.kill()
    if server.returncode != 0:
        raise RuntimeError(
            f"bandwidth server failed {destination.role}: {server_stderr[-500:]}"
        )
    received = json.loads(server_stdout)
    if client["bytes"] != byte_count or received["bytes"] != byte_count:
        raise RuntimeError("bandwidth byte count mismatch")
    return {
        "source": source.role,
        "destination": destination.role,
        "port": port,
        "bytes": byte_count,
        "client_bits_per_second": client["bits_per_second"],
        "server_bits_per_second": received["bits_per_second"],
    }


def _ros_command(host: Host, domain: int, command: str) -> str:
    body = (
        "set -eo pipefail; set +u; source /opt/ros/jazzy/setup.bash; set -u; "
        f"export ROS_DOMAIN_ID={domain} ROS_LOCALHOST_ONLY=0; {command}"
    )
    if host.role == "isaac_x86":
        return (
            "docker exec --user admin -e ROS_DOMAIN_ID="
            f"{domain} -e ROS_LOCALHOST_ONLY=0 internnav_t4_isaac_ros "
            f"bash -lc {shlex.quote(body)}"
        )
    return f"bash -lc {shlex.quote(body)}"


def _dds_topic_probe(
    publisher: Host,
    subscribers: list[Host],
    *,
    domain: int,
    topic: str,
    message_type: str,
    value: str,
) -> dict[str, Any]:
    publish = _ssh_popen(
        publisher,
        _ros_command(
            publisher,
            domain,
            f"timeout 12 ros2 topic pub -r 5 {shlex.quote(topic)} "
            f"{shlex.quote(message_type)} {shlex.quote(value)}",
        ),
    )
    observed: dict[str, bool] = {}
    errors: dict[str, str] = {}
    try:
        time.sleep(2.0)
        for host in subscribers:
            try:
                output = _ssh(
                    host,
                    _ros_command(
                        host,
                        domain,
                        f"timeout 8 ros2 topic echo --once {shlex.quote(topic)} "
                        f"{shlex.quote(message_type)}",
                    ),
                    timeout=12.0,
                )
                observed[host.role] = bool(output.strip())
            except subprocess.CalledProcessError as exc:
                observed[host.role] = False
                errors[host.role] = {
                    "type": type(exc).__name__,
                    "returncode": exc.returncode,
                    "stdout_tail": (exc.stdout or "")[-500:],
                    "stderr_tail": (exc.stderr or "")[-500:],
                }
            except Exception as exc:
                observed[host.role] = False
                errors[host.role] = {"type": type(exc).__name__}
    finally:
        try:
            publish.communicate(timeout=15.0)
        except subprocess.TimeoutExpired:
            publish.terminate()
            try:
                publish.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                publish.kill()
        if publish.stdout is not None:
            publish.stdout.close()
        if publish.stderr is not None:
            publish.stderr.close()
    return {
        "publisher": publisher.role,
        "topic": topic,
        "domain": domain,
        "observed": observed,
        "errors": errors,
    }


def evaluate(measurements: dict[str, Any]) -> dict[str, Any]:
    inventories = measurements["inventory"]
    ping = measurements["ping"]
    bandwidth = measurements["bandwidth"]
    clocks = measurements["clock"]
    dds = measurements["dds"]
    role_forbidden = {
        "dgx_model": ("nav2", "isaac"),
        "dgx_edge": ("model", "isaac"),
        "isaac_x86": ("model", "nav2"),
    }
    checks = {
        "host_identity": all(
            inventories[role]["user"] == expected
            for role, expected in {
                "dgx_model": "railgun",
                "dgx_edge": "rail",
                "isaac_x86": "song",
            }.items()
        ),
        "expected_interface": all(
            inventories[role]["selected_interface"] is not None
            for role in role_forbidden
        ),
        "ros_jazzy": all(inventories[role]["ros_jazzy_present"] for role in role_forbidden),
        "role_process_isolation": all(
            int(inventories[role]["process_counts"][name]) == 0
            for role, names in role_forbidden.items()
            for name in names
        ),
        "pairwise_packet_loss_zero": all(
            item["packet_loss_percent"] == 0.0 for item in ping
        ),
        "pairwise_rtt_max_under_20_ms": all(item["rtt_max_ms"] < 20.0 for item in ping),
        "pairwise_jitter_under_5_ms": all(item["jitter_ms"] < 5.0 for item in ping),
        "bandwidth_at_least_100_mbps": all(
            min(item["client_bits_per_second"], item["server_bits_per_second"])
            >= 100_000_000
            for item in bandwidth
        ),
        "wall_clock_offsets_under_100_ms": all(
            abs(item["best_offset_ns"]) <= 100_000_000 for item in clocks
        ),
        "dds_discovery": all(
            all(item["observed"].values()) for item in dds
        ),
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "minimum_bandwidth_mbps": min(
            min(item["client_bits_per_second"], item["server_bits_per_second"])
            for item in bandwidth
        )
        / 1_000_000,
        "maximum_rtt_ms": max(item["rtt_max_ms"] for item in ping),
        "maximum_jitter_ms": max(item["jitter_ms"] for item in ping),
        "maximum_absolute_clock_offset_ms": max(
            abs(item["best_offset_ns"]) for item in clocks
        )
        / 1_000_000,
    }


def run_probe(result_dir: Path, topology_path: Path = TOPOLOGY_PATH) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    roles = topology["roles"]
    hosts = {
        "dgx_model": Host("dgx_model", roles["dgx_model"]["user"], roles["dgx_model"]["host"]),
        "dgx_edge": Host("dgx_edge", roles["dgx_edge"]["user"], roles["dgx_edge"]["host"]),
        "isaac_x86": Host("isaac_x86", roles["isaac_x86"]["user"], roles["isaac_x86"]["host"]),
    }
    ordered = [hosts["dgx_model"], hosts["dgx_edge"], hosts["isaac_x86"]]
    measurements: dict[str, Any] = {
        "schema_version": 1,
        "topology": topology,
        "inventory": {host.role: _inventory(host) for host in ordered},
        "ping": [
            _ping(source, destination)
            for source in ordered
            for destination in ordered
            if source != destination
        ],
        "clock": [
            _clock_pair(hosts["dgx_model"], hosts["isaac_x86"], 25210),
            _clock_pair(hosts["dgx_edge"], hosts["isaac_x86"], 25211),
        ],
        "bandwidth": [
            _bandwidth(hosts["isaac_x86"], hosts["dgx_edge"], 25201),
            _bandwidth(hosts["dgx_edge"], hosts["dgx_model"], 25202),
            _bandwidth(hosts["dgx_model"], hosts["dgx_edge"], 25203),
        ],
        "dds": [
            _dds_topic_probe(
                hosts["isaac_x86"],
                [hosts["dgx_edge"], hosts["dgx_model"]],
                domain=topology["lanes"]["primary"]["ros_domain_id"],
                topic="/t5/network_probe",
                message_type="std_msgs/msg/String",
                value="{data: t5-network-probe}",
            ),
            _dds_topic_probe(
                hosts["isaac_x86"],
                [hosts["dgx_edge"], hosts["dgx_model"]],
                domain=topology["lanes"]["primary"]["ros_domain_id"],
                topic="/clock",
                message_type="rosgraph_msgs/msg/Clock",
                value="{clock: {sec: 1, nanosec: 0}}",
            ),
        ],
    }
    summary = evaluate(measurements)
    (result_dir / "network_measurements.json").write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (result_dir / "network_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--topology", type=Path, default=TOPOLOGY_PATH)
    args = parser.parse_args()
    summary = run_probe(args.result_dir, args.topology)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
