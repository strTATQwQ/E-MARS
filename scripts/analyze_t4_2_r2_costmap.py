#!/usr/bin/env python3
"""Summarize T4.2-R2 three-stage costmap evidence and classify Gate 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def range_or_none(values: list[float | int]) -> dict[str, float | int | None]:
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize_stream(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"record_count": len(records)}
    for stage_key in ("stage_b_pre_inflation", "stage_c_final_after_inflation"):
        stage_records = [record[stage_key] for record in records]
        result[stage_key] = {
            "free_count": range_or_none(
                [int(record["counts"]["free_count"]) for record in stage_records]
            ),
            "occupied_count": range_or_none(
                [int(record["counts"]["occupied_count"]) for record in stage_records]
            ),
            "unknown_count": range_or_none(
                [int(record["counts"]["unknown_count"]) for record in stage_records]
            ),
            "robot_cell_classes": sorted(
                {
                    str(record["spatial"]["robot_cell_class"])
                    for record in stage_records
                }
            ),
            "footprint_all_traversable_count": sum(
                bool(record["spatial"]["footprint"]["all_traversable"])
                for record in stage_records
            ),
            "connected_free_cell_count": range_or_none(
                [
                    int(record["spatial"]["connected_free_cell_count"])
                    for record in stage_records
                ]
            ),
            "forward_connected_count": {
                distance: sum(
                    bool(
                        record["spatial"]["forward"][distance][
                            "connected_free_from_robot"
                        ]
                    )
                    for record in stage_records
                )
                for distance in ("0.5_m", "1.0_m", "1.5_m")
            },
        }
    result["maximum_consecutive_final_connected_free"] = max(
        (
            int(
                record["stage_c_final_after_inflation"][
                    "consecutive_updates_with_connected_free"
                ]
            )
            for record in records
        ),
        default=0,
    )
    result["frames"] = sorted(
        {str(record["metadata"]["frame"]) for record in records}
    )
    result["resolutions_m"] = sorted(
        {float(record["metadata"]["resolution_m"]) for record in records}
    )
    result["origins_observed"] = len(
        {
            tuple(float(value) for value in record["metadata"]["origin_xyz_m"])
            for record in records
        }
    )
    result["backing_distance_m"] = {
        "minimum_observed": min(
            (
                float(record["backing_slice_distance_m"]["min"])
                for record in records
                if record["backing_slice_distance_m"]["min"] is not None
            ),
            default=None,
        ),
        "maximum_observed": max(
            (
                float(record["backing_slice_distance_m"]["max"])
                for record in records
                if record["backing_slice_distance_m"]["max"] is not None
            ),
            default=None,
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw_records = load_jsonl(args.result_dir / "nvblox_distance_slice_stage_a.jsonl")
    topic_stage_path = args.result_dir / "costmap_stage_records.jsonl"
    all_stage_records = load_jsonl(topic_stage_path) if topic_stage_path.is_file() else []
    trace_errors = [
        record for record in all_stage_records if record.get("event") == "trace_error"
    ]
    pairs = [
        record
        for record in all_stage_records
        if record.get("event") == "costmap_stage_pair"
    ]
    service_path = args.result_dir / "costmap_service_stage_records.jsonl"
    service_records = load_jsonl(service_path) if service_path.is_file() else []
    for record in service_records:
        if record.get("event") != "actual_costmap_service_stage_pair":
            continue
        pairs.append(
            {
                **record,
                "event": "costmap_stage_pair",
                "metadata": record["stage_c_metadata"],
                "stage_b_pre_inflation": record["stage_b_actual_pre_inflation"],
                "stage_c_final_after_inflation": record["stage_c_actual_final"],
            }
        )
    streams = {
        name: [record for record in pairs if record.get("stream") == name]
        for name in (
            "local",
            "global",
            "local_grid",
            "global_grid",
            "service_local",
            "service_global",
        )
    }
    summaries = {name: summarize_stream(records) for name, records in streams.items()}

    def mature(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            record
            for record in records
            if int(record["stage_b_pre_inflation"]["counts"]["free_count"])
            + int(record["stage_b_pre_inflation"]["counts"]["occupied_count"])
            > 0
        ]

    selected_streams: dict[str, str] = {}
    stable_records: dict[str, list[dict[str, Any]]] = {}
    for logical in ("local", "global"):
        service_name = f"service_{logical}"
        grid_name = f"{logical}_grid"
        if len(mature(streams[service_name])) >= 5:
            selected = service_name
        elif len(mature(streams[grid_name])) >= 5:
            selected = grid_name
        else:
            selected = logical
        selected_streams[logical] = selected
        stable_records[logical] = mature(streams[selected])[-20:]

    stable_pairs = stable_records["local"] + stable_records["global"]
    stage_b_free = [
        int(record["stage_b_pre_inflation"]["counts"]["free_count"])
        for record in stable_pairs
    ]
    stage_c_free = [
        int(record["stage_c_final_after_inflation"]["counts"]["free_count"])
        for record in stable_pairs
    ]
    stage_c_connected = [
        int(
            record["stage_c_final_after_inflation"]["spatial"][
                "connected_free_cell_count"
            ]
        )
        for record in stable_pairs
    ]
    stage_b_footprint_traversable = [
        bool(
            record["stage_b_pre_inflation"]["spatial"]["footprint"][
                "all_traversable"
            ]
        )
        for record in stable_pairs
    ]
    stable_streams_complete = all(
        len(records) >= 5 for records in stable_records.values()
    )
    if not stable_streams_complete:
        classification = "INSUFFICIENT_MATURE_LOCAL_OR_GLOBAL_EVIDENCE"
        permitted_next_action = "capture_periodic_local_and_global_final_costmaps"
    elif stable_pairs and all(value == 0 for value in stage_b_free):
        classification = "PRE_INFLATION_HAS_NO_FREE"
        permitted_next_action = "inspect_binary_slice_threshold_frame_origin_or_unknown"
    elif (
        pairs
        and all(value > 0 for value in stage_b_free)
        and all(value == 0 for value in stage_c_connected)
        and max(stage_c_free, default=0) <= 1
    ):
        if not any(stage_b_footprint_traversable):
            classification = (
                "INFLATION_PRIMARY_FOR_FREE_CELL_DISAPPEARANCE_WITH_"
                "PREINFLATION_UNKNOWN_FOOTPRINT_BLOCKER"
            )
            permitted_next_action = (
                "scan_inflation_radius_0.35_0.32_0.30_then_require_"
                "preinflation_footprint_connectivity"
            )
        else:
            classification = "INFLATION_PRIMARY_FOR_FREE_CELL_DISAPPEARANCE"
            permitted_next_action = "scan_inflation_radius_0.35_0.32_0.30"
    else:
        classification = "MIXED_OR_AMBIGUOUS"
        permitted_next_action = "do_not_tune_until_pairwise_variation_is_explained"

    required_fields_ok = all(
        record.get("metadata", {}).get("frame")
        and record.get("metadata", {}).get("resolution_m")
        and (
            record.get("stage_b_pre_inflation", {}).get("provenance")
            == "official_plugin_exact_reconstruction"
            or str(record.get("stage_b_pre_inflation", {}).get("provenance", ""))
            .endswith("/get_nvblox_layer")
        )
        and (
            record.get("stage_c_final_after_inflation", {}).get("provenance")
            == "received_nav2_costmap_raw"
            or str(
                record.get("stage_c_final_after_inflation", {}).get(
                    "provenance", ""
                )
            ).endswith("/get_costmap")
        )
        for record in pairs
    )
    gate1_pass = (
        len(raw_records) >= 5
        and stable_streams_complete
        and not trace_errors
        and required_fields_ok
        and classification != "MIXED_OR_AMBIGUOUS"
        and classification != "INSUFFICIENT_MATURE_LOCAL_OR_GLOBAL_EVIDENCE"
    )
    known_distance_maxima = [
        float(record["known_distance_m"]["max"])
        for record in raw_records
        if record.get("known_distance_m", {}).get("max") is not None
    ]
    output = {
        "schema_version": 2,
        "gate": "T4.2-R2 Gate 1",
        "status": "PASS" if gate1_pass else "FAIL",
        "classification": classification,
        "permitted_next_action": permitted_next_action,
        "raw_slice": {
            "record_count": len(raw_records),
            "maximum_known_distance_m": max(known_distance_maxima, default=None),
            "free_positive_count": range_or_none(
                [int(record["free_positive_count"]) for record in raw_records]
            ),
            "occupied_nonpositive_count": range_or_none(
                [int(record["occupied_nonpositive_count"]) for record in raw_records]
            ),
            "unknown_count": range_or_none(
                [int(record["unknown_count"]) for record in raw_records]
            ),
        },
        "streams": summaries,
        "selected_final_streams": selected_streams,
        "stable_mature_record_count": {
            name: len(records) for name, records in stable_records.items()
        },
        "stable_pair_evidence": {
            "stage_b_free_count": range_or_none(stage_b_free),
            "stage_c_free_count": range_or_none(stage_c_free),
            "stage_c_connected_free_cell_count": range_or_none(
                stage_c_connected
            ),
            "stage_b_footprint_all_traversable_count": sum(
                stage_b_footprint_traversable
            ),
            "preinflation_unknown_footprint_blocker": bool(stable_pairs)
            and not any(stage_b_footprint_traversable),
        },
        "trace_error_count": len(trace_errors),
        "trace_errors": trace_errors[:20],
        "required_fields_ok": required_fields_ok,
    }
    output_path = args.output or args.result_dir / "gate1_costmap_analysis.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if gate1_pass else 1)


if __name__ == "__main__":
    main()
