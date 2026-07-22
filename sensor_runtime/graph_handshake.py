"""ROS-graph readiness contract with a ROS-less validation surface."""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping


BEST_EFFORT_QOS = {
    "reliability": "BEST_EFFORT",
    "durability": "VOLATILE",
    "history": "KEEP_LAST",
    "minimum_depth": 4,
}
STATIC_QOS = {
    "reliability": "RELIABLE",
    "durability": "TRANSIENT_LOCAL",
    "history": "KEEP_LAST",
    "minimum_depth": 1,
}
ROLE_OWNED_QOS_CONTRACT = {
    "sensor_ros_sidecar": {
        "dynamic": {
            "reliability": "BEST_EFFORT",
            "durability": "VOLATILE",
            "history": "KEEP_LAST",
            "depth": 4,
        },
        "tf_dynamic": {
            "reliability": "RELIABLE",
            "durability": "VOLATILE",
            "history": "KEEP_LAST",
            "depth": 100,
        },
        "static": {
            "reliability": "RELIABLE",
            "durability": "TRANSIENT_LOCAL",
            "history": "KEEP_LAST",
            "depth": 1,
        },
    },
    "go2_sensor_bridge": {
        "dynamic": {
            "reliability": "BEST_EFFORT",
            "durability": "VOLATILE",
            "history": "KEEP_LAST",
            "depth": 4,
        },
    },
    "downstream_recorder": {
        "dynamic": {
            "reliability": "BEST_EFFORT",
            "durability": "VOLATILE",
            "history": "KEEP_LAST",
            "depth": 8,
        },
        "static": {
            "reliability": "RELIABLE",
            "durability": "TRANSIENT_LOCAL",
            "history": "KEEP_LAST",
            "depth": 1,
        },
    },
}


def _require(topic: str, minimum: int, qos: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    return topic, {"minimum_subscription_count": minimum, "qos": dict(qos)}


SIDECAR_GRAPH_REQUIREMENTS = dict(
    (
        _require("/clock", 2, BEST_EFFORT_QOS),
        _require("/internnav/sensor_generation", 1, BEST_EFFORT_QOS),
        _require("/internnav/sensor_frame_identity", 2, BEST_EFFORT_QOS),
        _require("/tf", 2, BEST_EFFORT_QOS),
        _require("/tf_static", 1, STATIC_QOS),
        _require("/go2/d435i/color/image_raw", 1, BEST_EFFORT_QOS),
        _require("/go2/d435i/color/camera_info", 1, BEST_EFFORT_QOS),
        _require("/go2/d435i/depth/image_rect", 1, BEST_EFFORT_QOS),
        _require("/go2/d435i/depth/camera_info", 1, BEST_EFFORT_QOS),
        _require("/go2/depth/points", 2, BEST_EFFORT_QOS),
        _require("/internvla_t4/go2/lidar_raw", 1, BEST_EFFORT_QOS),
        _require("/internvla_t4/go2/front_rgb_raw", 1, BEST_EFFORT_QOS),
        _require("/internvla_t4/go2/front_rgb_camera_info_raw", 1, BEST_EFFORT_QOS),
        _require("/go2/imu/data", 1, BEST_EFFORT_QOS),
        _require("/internvla_t4/go2/self_filter_link_centers", 1, BEST_EFFORT_QOS),
        _require("/odom", 1, BEST_EFFORT_QOS),
    )
)

BRIDGE_GRAPH_REQUIREMENTS = dict(
    (
        _require("/go2/lidar/points", 1, BEST_EFFORT_QOS),
        _require("/go2/lidar/points_base", 1, BEST_EFFORT_QOS),
        _require("/go2/safety/points", 1, BEST_EFFORT_QOS),
        _require("/go2/front_rgb/image_raw", 1, BEST_EFFORT_QOS),
        _require("/go2/front_rgb/camera_info", 1, BEST_EFFORT_QOS),
    )
)


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    text = str(value).upper()
    return text.rsplit(".", 1)[-1]


def qos_snapshot(profile: Any) -> dict[str, Any]:
    return {
        "reliability": _enum_name(profile.reliability),
        "durability": _enum_name(profile.durability),
        "history": _enum_name(profile.history),
        "depth": int(profile.depth),
    }


def observe_publishers(
    node: Any,
    publishers: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Capture actual matched subscription counts and endpoint QoS."""

    observed: dict[str, dict[str, Any]] = {}
    for topic, publisher in publishers.items():
        endpoints = node.get_subscriptions_info_by_topic(topic)
        observed[topic] = {
            "subscription_count": int(publisher.get_subscription_count()),
            "subscriptions": [qos_snapshot(item.qos_profile) for item in endpoints],
        }
    return observed


def validate_graph_observation(
    requirements: Mapping[str, Mapping[str, Any]],
    observed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a machine-readable verdict; malformed observations fail closed."""

    topics: dict[str, Any] = {}
    ready = True
    unavailable_total = 0
    for topic, required in requirements.items():
        row = observed.get(topic)
        minimum = int(required["minimum_subscription_count"])
        expected_qos = required["qos"]
        errors: list[str] = []
        compatible = 0
        history_depth_unavailable = 0
        count = 0
        subscriptions: list[Any] = []
        if not isinstance(row, Mapping):
            errors.append("missing_topic_observation")
        else:
            raw_count = row.get("subscription_count")
            if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
                errors.append("invalid_subscription_count")
            else:
                count = raw_count
                if count < minimum:
                    errors.append("subscription_count_below_minimum")
            raw_subscriptions = row.get("subscriptions")
            if not isinstance(raw_subscriptions, list):
                errors.append("invalid_endpoint_qos_list")
            else:
                subscriptions = raw_subscriptions
                for endpoint in subscriptions:
                    if not isinstance(endpoint, Mapping):
                        continue
                    reliability_and_durability_match = (
                        endpoint.get("reliability") == expected_qos["reliability"]
                        and endpoint.get("durability") == expected_qos["durability"]
                    )
                    depth = endpoint.get("depth")
                    depth_is_integer = isinstance(depth, int) and not isinstance(
                        depth, bool
                    )
                    full_history_depth_match = (
                        endpoint.get("history") == expected_qos["history"]
                        and depth_is_integer
                        and int(depth) >= int(expected_qos["minimum_depth"])
                    )
                    rmw_history_depth_unavailable = (
                        endpoint.get("history") == "UNKNOWN"
                        and depth_is_integer
                        and int(depth) == 0
                    )
                    if reliability_and_durability_match and (
                        full_history_depth_match or rmw_history_depth_unavailable
                    ):
                        compatible += 1
                        if rmw_history_depth_unavailable:
                            history_depth_unavailable += 1
                if compatible < minimum:
                    errors.append("compatible_endpoint_qos_below_minimum")
        if errors:
            ready = False
        topics[topic] = {
            "minimum_subscription_count": minimum,
            "subscription_count": count,
            "compatible_qos_count": compatible,
            "history_depth_unavailable_count": history_depth_unavailable,
            "required_qos": dict(expected_qos),
            "subscriptions": subscriptions,
            "errors": errors,
        }
        unavailable_total += history_depth_unavailable
    return {
        "schema_version": 1,
        "status": "GRAPH_READY" if ready else "WAITING",
        "ready": ready,
        "owned_qos_proof_required": unavailable_total > 0,
        "rmw_history_depth_unavailable_count": unavailable_total,
        "observed_wall_monotonic_ns": time.monotonic_ns(),
        "topics": topics,
    }


def make_steady_control_clock(clock_class: Any, clock_type: Any) -> Any:
    """Construct the only clock allowed for lifecycle/control-plane timers."""

    return clock_class(clock_type=clock_type.STEADY_TIME)


def make_exact_ros_time(time_class: Any, clock_type: Any, stamp_ns: int) -> Any:
    """Construct an exact simulated timestamp for tf2 lookup."""

    if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns <= 0:
        raise ValueError("TF query stamp must be a positive integer nanosecond value")
    return time_class(nanoseconds=stamp_ns, clock_type=clock_type.ROS_TIME)


def ingest_observed_transforms(
    buffer: Any,
    transforms: Iterable[Any],
    *,
    authority: str,
    static: bool,
) -> int:
    """Insert the transforms from one already-observed subscription into tf2."""

    observed = tuple(transforms)
    if (
        not observed
        or not isinstance(authority, str)
        or not authority
        or type(static) is not bool
    ):
        raise ValueError("observed TF ingestion inputs are invalid")
    method_name = "set_transform_static" if static else "set_transform"
    setter = getattr(buffer, method_name, None)
    if not callable(setter):
        raise TypeError(f"TF buffer lacks callable {method_name}")
    for transform in observed:
        setter(transform, authority)
    return len(observed)


def summarize_pending_batches(
    pending: Mapping[int, Mapping[str, Any]],
    required_parts: Iterable[str],
    *,
    now_monotonic: float,
) -> list[dict[str, Any]]:
    """Describe the exact incomplete batch state without changing its gate."""

    if isinstance(now_monotonic, bool) or not isinstance(now_monotonic, (int, float)):
        raise TypeError("pending batch diagnostic time must be numeric")
    required_base = set(required_parts)
    if not required_base or not all(isinstance(item, str) and item for item in required_base):
        raise ValueError("pending batch diagnostic requires named parts")
    rows: list[dict[str, Any]] = []
    for stamp in sorted(pending):
        if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= 0:
            raise ValueError("pending batch diagnostic stamp must be a positive integer")
        entry = pending[stamp]
        created = entry.get("created")
        parts = entry.get("parts")
        details = entry.get("details")
        if (
            isinstance(created, bool)
            or not isinstance(created, (int, float))
            or not isinstance(parts, Mapping)
            or not isinstance(details, Mapping)
        ):
            raise ValueError("pending batch diagnostic entry is malformed")
        identity = details.get("identity")
        required = set(required_base)
        if isinstance(identity, Mapping) and identity.get("sequence") == 0:
            required.add("tf_static")
        observed = set(parts)
        rows.append(
            {
                "stamp_ns": stamp,
                "age_sec": max(0.0, float(now_monotonic) - float(created)),
                "identity": dict(identity) if isinstance(identity, Mapping) else None,
                "observed_parts": sorted(observed),
                "missing_parts": sorted(required - observed),
                "unexpected_parts": sorted(observed - required),
            }
        )
    return rows


INNER_HANDSHAKE_ARTIFACTS = {
    "sidecar_role_ready.json": ("ROLE_READY", "sensor_ros_sidecar"),
    "bridge_role_ready.json": ("ROLE_READY", "go2_sensor_bridge"),
    "downstream_role_ready.json": ("ROLE_READY", "downstream_recorder"),
    "sidecar_graph_ready.json": ("GRAPH_READY", "sensor_ros_sidecar"),
    "bridge_graph_ready.json": ("GRAPH_READY", "go2_sensor_bridge"),
}


def validate_inner_handshake_documents(
    documents: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all role/graph readiness documents without importing ROS."""

    errors: list[str] = []
    for filename, (status, role) in INNER_HANDSHAKE_ARTIFACTS.items():
        document = documents.get(filename)
        if not isinstance(document, Mapping):
            errors.append(f"{filename}:missing_or_malformed")
            continue
        if document.get("status") != status:
            errors.append(f"{filename}:status")
        if document.get("role") != role:
            errors.append(f"{filename}:role")
        if status == "ROLE_READY":
            if document.get("schema_version") != 2:
                errors.append(f"{filename}:schema_version")
            if document.get("owned_qos") != ROLE_OWNED_QOS_CONTRACT[role]:
                errors.append(f"{filename}:owned_qos")
        else:
            if document.get("schema_version") != 1:
                errors.append(f"{filename}:schema_version")
            if document.get("ready") is not True:
                errors.append(f"{filename}:ready")
            if not isinstance(document.get("owned_qos_proof_required"), bool):
                errors.append(f"{filename}:owned_qos_proof_required")
            unavailable = document.get("rmw_history_depth_unavailable_count")
            if (
                isinstance(unavailable, bool)
                or not isinstance(unavailable, int)
                or unavailable < 0
                or document.get("owned_qos_proof_required") != (unavailable > 0)
            ):
                errors.append(f"{filename}:rmw_history_depth_unavailable_count")
    return {
        "status": "READY" if not errors else "WAITING",
        "ready": not errors,
        "errors": errors,
        "artifacts": sorted(INNER_HANDSHAKE_ARTIFACTS),
    }
