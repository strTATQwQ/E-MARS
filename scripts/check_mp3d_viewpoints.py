#!/usr/bin/env python3
"""Gate 0: validate three official viewpoints in one real Isaac MP3D scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True)
parser.add_argument("--scene-usd", required=True)
parser.add_argument("--connectivity", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--node-ids", type=int, nargs="+", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import hashlib
import json
import math
import numpy as np

from slow_benchmark.oracle_graph import MatterportGraph
from step3_graph_nav.evaluation import candidate_specs
from step3_graph_nav.io import load_yaml, sha256_file, write_json
from step3_graph_nav.isaac_scene import IsaacCandidateScene


def main() -> int:
    if len(args_cli.node_ids) != 3 or len(set(args_cli.node_ids)) != 3:
        raise ValueError("Gate 0 requires exactly three distinct viewpoint node IDs per scene")
    config = load_yaml(args_cli.config)
    gate_cfg = config["gate0"]
    graph = MatterportGraph.load(args_cli.connectivity)
    missing = sorted(set(args_cli.node_ids) - graph.nodes.keys())
    if missing:
        raise ValueError(f"viewpoints are absent from connectivity graph: {missing}")
    output = Path(args_cli.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = IsaacCandidateScene(args_cli.scene_usd, config, str(args_cli.device))
    stage = scene.stage_stats()
    checks = {
        "up_axis_z": str(stage["up_axis"]).upper() == str(gate_cfg["require_up_axis"]).upper(),
        "meters_per_unit": math.isclose(
            float(stage["meters_per_unit"]),
            float(gate_cfg["require_meters_per_unit"]),
            abs_tol=1.0e-9,
        ),
        "meshes_present": int(stage["mesh_count"]) > 0,
        "collision_meshes_present": int(stage["collision_mesh_count"]) > 0,
        "materials_present": int(stage["material_count"]) > 0,
        "material_bindings_present": int(stage["bound_material_mesh_count"]) > 0,
    }
    viewpoint_rows = []
    for node_id in args_cli.node_ids:
        node = graph.nodes[node_id]
        specs = candidate_specs(graph, node_id, 0.0)
        if tuple(spec.target_viewpoint_id for spec in specs) != tuple(sorted(node.neighbors)):
            raise RuntimeError("candidate ordering differs from sorted connectivity neighbors")
        render_rows = []
        for spec in specs:
            jpeg, rgb, depth = scene.render(node.camera_position, spec.absolute_yaw_rad)
            image_name = f"node_{node_id:04d}_candidate_{spec.candidate_id:02d}_to_{spec.target_viewpoint_id:04d}.jpg"
            image_path = output / image_name
            image_path.write_bytes(jpeg)
            h, w = depth.shape
            center = depth[int(h * 0.35) : int(h * 0.65), int(w * 0.35) : int(w * 0.65)]
            finite = center[np.isfinite(center)]
            center_depth_p05 = float(np.percentile(finite, 5.0)) if finite.size else None
            rgb_std = float(np.std(rgb))
            render_pass = bool(
                finite.size
                and center_depth_p05 is not None
                and center_depth_p05 >= float(gate_cfg["min_center_depth_m"])
                and rgb_std >= float(gate_cfg["min_rgb_std"])
            )
            render_rows.append(
                {
                    **spec.to_mapping(),
                    "rgb_file": image_name,
                    "rgb_sha256": hashlib.sha256(jpeg).hexdigest(),
                    "rgb_mean": float(np.mean(rgb)),
                    "rgb_std": rgb_std,
                    "depth_finite_fraction": float(np.isfinite(depth).mean()),
                    "center_depth_p05_m": center_depth_p05,
                    "camera_not_in_wall": bool(
                        center_depth_p05 is not None
                        and center_depth_p05 >= float(gate_cfg["min_center_depth_m"])
                    ),
                    "render_pass": render_pass,
                }
            )
        viewpoint_rows.append(
            {
                "node_id": node_id,
                "image_id": node.image_id,
                "camera_position": list(node.camera_position),
                "raw_connectivity_neighbors": list(node.neighbors),
                "fixed_candidate_targets": [spec.target_viewpoint_id for spec in specs],
                "neighbor_set_correct": set(node.neighbors)
                == {spec.target_viewpoint_id for spec in specs},
                "heading_pose_convention": "camera_at_connectivity_pose_looking_current_to_neighbor",
                "renders": render_rows,
            }
        )
    checks["viewpoint_count"] = len(viewpoint_rows) == 3
    checks["neighbor_sets"] = all(row["neighbor_set_correct"] for row in viewpoint_rows)
    checks["camera_and_rgb"] = all(
        render["render_pass"] for row in viewpoint_rows for render in row["renders"]
    )
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "gate": "gate0",
        "status": "PASS" if passed else "BLOCKED",
        "scene_id": graph.scene_id,
        "scene_usd": str(Path(args_cli.scene_usd).resolve()),
        "scene_usd_sha256": sha256_file(args_cli.scene_usd),
        "connectivity": str(Path(args_cli.connectivity).resolve()),
        "stage": stage,
        "checks": checks,
        "viewpoints": viewpoint_rows,
        "rgb_source": "isaac_render_product",
    }
    write_json(output / "gate0_report.json", report)
    print(json.dumps(report, separators=(",", ":")), flush=True)
    return 0 if passed else 2


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
