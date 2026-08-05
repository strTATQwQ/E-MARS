#!/usr/bin/env bash
set -euo pipefail

die() {
  printf '01R Isaac startup diagnostic: %s\n' "$*" >&2
  exit 64
}

sshpass_prefix() {
  SSH_AUTH=()
  if [[ -n "${ISAAC_PASSWORD_FILE:-}" ]]; then
    [[ -r "$ISAAC_PASSWORD_FILE" ]] || die "ISAAC_PASSWORD_FILE is unreadable"
    SSH_AUTH=(sshpass -f "$ISAAC_PASSWORD_FILE")
  elif [[ -n "${ISAAC_PASSWORD:-}" ]]; then
    export SSHPASS="$ISAAC_PASSWORD"
    SSH_AUTH=(sshpass -e)
  fi
}

run_under_lease() {
  local mode="$1"
  local host="${ISAAC_HOST:-10.100.120.111}"
  local user="${ISAAC_USER:-song}"
  local port="${ISAAC_PORT:-22}"
  local target="${user}@${host}"
  local python_b64 remote_b64
  local -a ssh_options

  sshpass_prefix
  ssh_options=(-T -p "$port" -o ConnectTimeout=8 -o ServerAliveInterval=5
    -o ServerAliveCountMax=2 -o StrictHostKeyChecking=accept-new)

  if [[ "$mode" == "inventory" ]]; then
    read -r -d '' python_program <<'PY' || true
from __future__ import annotations

import ast
import json
from pathlib import Path
import tomllib

package = Path("/home/song/env_isaacsim/lib/python3.12/site-packages/isaacsim")
apps = package / "apps"
simulation_app = (
    package
    / "exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py"
)
source = simulation_app.read_text(encoding="utf-8")
source_lines = source.splitlines()
interesting = [
    line.strip()
    for line in source.splitlines()
    if "experience" in line or "hide_ui" in line or "headless" in line
]
default_block: list[str] = []
for index, line in enumerate(source_lines):
    if 'if experience == "":' in line:
        default_block = [
            f"{line_number + 1}: {source_lines[line_number]}"
            for line_number in range(index, min(index + 35, len(source_lines)))
        ]
        break
experience_summaries = {}
for name in (
    "isaacsim.exp.base.kit",
    "isaacsim.exp.base.python.kit",
    "isaacsim.exp.base.zero_delay.kit",
):
    path = apps / name
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    experience_summaries[name] = {
        "dependencies": sorted(data.get("dependencies", {})),
        "package": data.get("package", {}),
        "settings": data.get("settings", {}),
    }
all_kits = sorted(package.parent.rglob("*.kit"))
candidate_kits = [
    str(path)
    for path in all_kits
    if any(
        token in path.name.lower()
        for token in ("minimal", "headless", "python", "test", "base")
    )
]
extension_summaries = {}
for extension_name in (
    "isaacsim.core.api",
    "isaacsim.core.prims",
    "isaacsim.core.utils",
    "isaacsim.sensors.camera",
    "isaacsim.sensors.experimental.physics",
    "isaacsim.simulation_app",
    "omni.kit.loop-isaac",
):
    matches = [
        path
        for path in package.rglob("extension.toml")
        if path.parent.parent.name == extension_name
    ]
    if len(matches) != 1:
        extension_summaries[extension_name] = {
            "error": f"expected one config, found {len(matches)}"
        }
        continue
    data = tomllib.loads(matches[0].read_text(encoding="utf-8"))
    extension_summaries[extension_name] = {
        "config": str(matches[0]),
        "dependencies": sorted(data.get("dependencies", {})),
    }
camera_method_sources = {}
for camera_source in package.rglob("camera.py"):
    if camera_source.as_posix().endswith(
        "/isaacsim.sensors.camera/isaacsim/sensors/camera/camera.py"
    ):
        source_text = camera_source.read_text(encoding="utf-8")
        source_lines = source_text.splitlines()
        tree = ast.parse(source_text)
        selected = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "Camera":
                selected["class_header"] = source_lines[node.lineno - 1]
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in {
                        "initialize",
                        "get_current_frame",
                        "_data_acquisition_callback",
                        "attach_annotator",
                        "resume",
                        "pause",
                        "get_rgba",
                        "get_rgb",
                        "add_distance_to_image_plane_to_frame",
                    }:
                        selected[child.name] = "\n".join(
                            source_lines[child.lineno - 1 : child.end_lineno]
                        )
        camera_method_sources[str(camera_source)] = selected
runtime_method_sources = {}
runtime_source_specs = {
    "simulation_context.py": {
        "class": "SimulationContext",
        "methods": {"step", "render", "play", "pause", "stop"},
    },
    "world.py": {
        "class": "World",
        "methods": {"step", "render", "play", "pause", "stop"},
    },
    "simulation_manager.py": {
        "class": "SimulationManager",
        "methods": None,
    },
}
for source_name, spec in runtime_source_specs.items():
    for runtime_source in package.rglob(source_name):
        source_text = runtime_source.read_text(encoding="utf-8")
        if spec["class"] not in source_text:
            continue
        source_lines = source_text.splitlines()
        tree = ast.parse(source_text)
        selected = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == spec["class"]:
                selected["class_header"] = source_lines[node.lineno - 1]
                for child in node.body:
                    if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    include = (
                        child.name in spec["methods"]
                        if spec["methods"] is not None
                        else any(
                            token in child.name.lower()
                            for token in ("physics", "simulation", "fabric", "timeline")
                        )
                    )
                    if include:
                        selected[child.name] = "\n".join(
                            source_lines[child.lineno - 1 : child.end_lineno]
                        )
        if len(selected) > 1:
            runtime_method_sources[str(runtime_source)] = selected
articulation_sources = {}
for filename in ("single_articulation.py", "articulation_controller.py"):
    matches = sorted(package.rglob(filename))
    articulation_sources[filename] = [
        {
            "path": str(path),
            "relevant_lines": [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.lstrip().startswith("def ")
                and any(token in line for token in ("joint", "action", "controller"))
            ],
        }
        for path in matches
    ]
print(
    "INTERNNAV_ISAAC_STARTUP_DIAGNOSTIC_PASS="
    + json.dumps(
        {
            "schema_version": 1,
            "status": "PASS",
            "apps": sorted(path.name for path in apps.glob("*.kit")),
            "articulation_sources": articulation_sources,
            "all_kit_count": len(all_kits),
            "candidate_kit_paths": candidate_kits,
            "default_experience_block": default_block,
            "experience_summaries": experience_summaries,
            "extension_summaries": extension_summaries,
            "camera_method_sources": camera_method_sources,
            "runtime_method_sources": runtime_method_sources,
            "simulation_app_source": str(simulation_app),
            "relevant_source_lines": interesting,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
  else
    read -r -d '' python_program <<'PY' || true
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time
import traceback

mode = os.environ["STARTUP_MODE"]
started = time.monotonic()
app = None
backend = None
temporary = None
capture_not_ready_reasons = {}
rgb_content_samples = []
try:
    import isaacsim
    from isaacsim import SimulationApp

    imported = time.monotonic()
    experience = ""
    if mode in (
        "minimal",
        "minimal-backend",
        "minimal-backend-fixed-api",
        "minimal-backend-fixed-api-prewarm",
    ):
        temporary = tempfile.TemporaryDirectory(prefix="internnav-01r-kit-")
        experience_path = Path(temporary.name) / "internnav.model_free_sensor.kit"
        experience_path.write_text(
            '''[package]
title = "InternNav Model-Free Sensor"
description = "Minimal headless physics and camera experience for 01R"
version = "1.0.0"

[dependencies]
"isaacsim.core.api" = {}
"isaacsim.sensors.camera" = {}
"isaacsim.sensors.experimental.physics" = {}
"isaacsim.simulation_app" = {}
"omni.kit.loop-isaac" = {}

[settings]
app.name = "InternNav Model-Free Sensor"
app.version = "1.0.0"
app.vulkan = true
app.fastShutdown = true
app.enableDeveloperWarnings = false
app.content.emptyStageOnStart = true
app.file.ignoreUnsavedStage = true
app.gatherRenderResults = true
app.asyncRendering = false
app.useFabricSceneDelegate = true
app.hydraEngine.waitIdle = true
app.updateOrder.checkForHydraRenderComplete = 1000
app.settings.persistent = false
app.settings.fabricDefaultStageFrameHistoryCount = 3
app.runLoops.main.manualModeEnabled = true
app.runLoops.main.rateLimitEnabled = false
app.renderer.resolution.width = 640
app.renderer.resolution.height = 480
app.renderer.skipWhileMinimized = false
app.renderer.sleepMsOnFocus = 0
app.renderer.sleepMsOutOfFocus = 0
renderer.asyncInit = true
renderer.gpuEnumeration.glInterop.enabled = false
omni.replicator.asyncRendering = false
persistent.omni.replicator.captureOnPlay = true
rtx.hydra.supportMultiTickRate = true
rtx.rendering.perSensorTickTlas = true
''',
            encoding="utf-8",
        )
        experience = str(experience_path)
    launch_config = {
        "headless": True,
        "anti_aliasing": 0,
        "hide_ui": True,
        "multi_gpu": False,
    }
    if mode in (
        "minimal",
        "minimal-backend",
        "minimal-backend-fixed-api",
        "minimal-backend-fixed-api-prewarm",
    ):
        package_root = Path(isaacsim.__file__).resolve().parent
        launch_config["extra_args"] = [
            "--ext-folder",
            str(package_root / "extsDeprecated"),
            "--ext-folder",
            str(package_root / "extscache"),
        ]
    app = SimulationApp(
        launch_config,
        experience=experience,
    )
    app_ready = time.monotonic()
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.sensors.camera")
    enable_extension("isaacsim.sensors.experimental.physics")
    app.update()
    extensions_ready = time.monotonic()
    backend_evidence = None
    if mode in (
        "minimal-backend",
        "minimal-backend-fixed-api",
        "minimal-backend-fixed-api-prewarm",
    ):
        from sensor_runtime.isaac_backend import IsaacModelFreeBackend
        from sensor_runtime.workload import CaptureNotReady

        import sensor_runtime.isaac_backend as isaac_backend_module

        original_rgb_content_evidence = isaac_backend_module.rgb_content_evidence

        def observed_rgb_content_evidence(value):
            evidence = original_rgb_content_evidence(value)
            rgb_content_samples.append(
                {"call_index": len(rgb_content_samples), **evidence}
            )
            return evidence

        isaac_backend_module.rgb_content_evidence = observed_rgb_content_evidence

        if mode in (
            "minimal-backend-fixed-api",
            "minimal-backend-fixed-api-prewarm",
        ):
            from isaacsim.core.utils.types import ArticulationAction

            def apply_safe_stop(self, *, count_physics_step: bool) -> None:
                self.articulation.apply_action(
                    ArticulationAction(
                        joint_positions=self._stand,
                        joint_velocities=self._zeros,
                        joint_indices=self._dof_indices,
                    )
                )
                if count_physics_step:
                    self._global_safe_steps += 1

            IsaacModelFreeBackend._apply_safe_stop = apply_safe_stop

        if mode == "minimal-backend-fixed-api-prewarm":
            def render_metadata_v6(frame, camera_name):
                if "rendering_frame" not in frame or "rendering_time" not in frame:
                    raise RuntimeError(f"{camera_name} frame lacks real render metadata")
                raw_identity = frame["rendering_frame"]
                if isinstance(raw_identity, dict):
                    if set(raw_identity) != {
                        "referenceTimeNumerator",
                        "referenceTimeDenominator",
                    }:
                        raise RuntimeError(
                            f"{camera_name} ReferenceTime schema is invalid: {raw_identity!r}"
                        )
                    render_id = int(raw_identity["referenceTimeNumerator"])
                    denominator = int(raw_identity["referenceTimeDenominator"])
                    if denominator <= 0:
                        raise RuntimeError(
                            f"{camera_name} ReferenceTime denominator is invalid"
                        )
                else:
                    render_id = int(raw_identity)
                render_time = float(frame["rendering_time"])
                if render_id <= 0 or not math.isfinite(render_time) or render_time < 0.0:
                    raise RuntimeError(f"{camera_name} frame has invalid render metadata")
                return render_id, render_time

            IsaacModelFreeBackend._render_metadata = staticmethod(render_metadata_v6)
            original_step_safe_stop = IsaacModelFreeBackend.step_safe_stop

            def step_safe_stop_with_sync_evidence(self):
                try:
                    return original_step_safe_stop(self)
                except RuntimeError as exc:
                    if str(exc) != "camera render time does not match the current world step":
                        raise
                    frames = {
                        "color": self.color_camera.get_current_frame(),
                        "depth": self.depth_camera.get_current_frame(),
                        "front": self.front_camera.get_current_frame(),
                    }
                    observed = {
                        name: {
                            "rendering_frame": frame.get("rendering_frame"),
                            "rendering_time": frame.get("rendering_time"),
                        }
                        for name, frame in frames.items()
                    }
                    raise RuntimeError(
                        "camera render time does not match the current world step: "
                        + json.dumps(
                            {
                                "world_current_time": float(self.world.current_time),
                                "observed": observed,
                            },
                            sort_keys=True,
                        )
                    ) from exc

            IsaacModelFreeBackend.step_safe_stop = step_safe_stop_with_sync_evidence
            original_init = IsaacModelFreeBackend.__init__

            def init_with_safe_render_prewarm(self, *args, **kwargs) -> None:
                original_init(self, *args, **kwargs)
                from isaacsim.core.utils.stage import get_current_stage
                from pxr import Gf, UsdGeom

                front_camera_prim = get_current_stage().GetPrimAtPath(
                    "/World/Go2/base/go2_front_rgb"
                )
                front_camera_schema = UsdGeom.Camera(front_camera_prim)
                clipping_before = front_camera_schema.GetClippingRangeAttr().Get()
                front_camera_schema.GetClippingRangeAttr().Set(
                    Gf.Vec2f(0.20, 1_000_000.0)
                )
                self._diagnostic_front_clipping = {
                    "before": [float(clipping_before[0]), float(clipping_before[1])],
                    "after": [0.20, 1_000_000.0],
                }
                for camera in (
                    self.color_camera,
                    self.depth_camera,
                    self.front_camera,
                ):
                    camera._frequency = -1
                    original_get_current_frame = camera.get_current_frame

                    def get_current_frame_with_v6_rgb_alias(
                        clone=False,
                        *,
                        original=original_get_current_frame,
                    ):
                        frame = original(clone=clone)
                        if "rgb" in frame and "rgba" not in frame:
                            return {**frame, "rgba": frame["rgb"]}
                        return frame

                    camera.get_current_frame = get_current_frame_with_v6_rgb_alias
                original_world_step = self.world.step
                original_world_render = self.world.render
                sync_records = []
                from isaacsim.core.simulation_manager import SimulationManager

                def refresh_camera_from_completed_render(camera):
                    camera._og_controller.evaluate_sync(
                        graph_id=camera._sdg_graph_pipeline
                    )
                    frame_number = camera._fabric_time_annotator.get_data()
                    current_time = (
                        SimulationManager._simulation_manager_interface
                        .get_simulation_time_at_time(
                            (
                                frame_number["referenceTimeNumerator"],
                                frame_number["referenceTimeDenominator"],
                            )
                        )
                    )
                    camera._current_frame["rendering_frame"] = frame_number
                    camera._current_frame["rendering_time"] = current_time
                    for key in tuple(camera._current_frame):
                        if key not in {"rendering_time", "rendering_frame"}:
                            camera._current_frame[key] = (
                                camera._custom_annotators[key].get_data()
                            )
                    camera._previous_time = current_time
                    camera._elapsed_time = 0

                def synchronized_world_step(*step_args, **step_kwargs):
                    render_requested = (
                        bool(step_kwargs["render"])
                        if "render" in step_kwargs
                        else (bool(step_args[0]) if step_args else True)
                    )
                    physics_before = float(self.world.current_time)
                    physics_attempts = []
                    for latch_count in range(4):
                        original_world_step(render=False)
                        physics_after = float(self.world.current_time)
                        delta = physics_after - physics_before
                        physics_attempts.append(physics_after)
                        if abs(delta - 1.0 / 200.0) <= 1e-6:
                            break
                        if abs(delta) > 1e-9:
                            raise RuntimeError(
                                "one safe-stop step advanced an invalid amount of physics time: "
                                + json.dumps(
                                    {
                                        "before": physics_before,
                                        "after": physics_after,
                                        "attempts": physics_attempts,
                                    },
                                    sort_keys=True,
                                )
                            )
                    else:
                        raise RuntimeError(
                            "timeline resume did not produce one physics step: "
                            + json.dumps(
                                {
                                    "before": physics_before,
                                    "attempts": physics_attempts,
                                },
                                sort_keys=True,
                            )
                        )
                    if not render_requested:
                        return
                    target_time = float(self.world.current_time)
                    attempts = []
                    for drain_count in range(65):
                        frames = {
                            "color": self.color_camera.get_current_frame(),
                            "depth": self.depth_camera.get_current_frame(),
                            "front": self.front_camera.get_current_frame(),
                        }
                        observed_times = {}
                        for name, frame in frames.items():
                            try:
                                observed_times[name] = float(
                                    frame.get("rendering_time", -1.0)
                                )
                            except (TypeError, ValueError):
                                observed_times[name] = -1.0
                        attempts.append(observed_times)
                        if (
                            len(set(observed_times.values())) == 1
                            and all(math.isfinite(value) for value in observed_times.values())
                            and abs(next(iter(observed_times.values())) - target_time)
                            <= 1.0 / 200.0 + 1e-6
                        ):
                            sync_records.append(
                                {
                                    "target_time": target_time,
                                    "drain_count": drain_count,
                                    "final_time": next(iter(observed_times.values())),
                                    "attempts": attempts,
                                    "physics_latch_count": latch_count,
                                    "physics_attempts": physics_attempts,
                                }
                            )
                            return
                        if drain_count < 64:
                            before_render_only = float(self.world.current_time)
                            original_world_render()
                            for camera in (
                                self.color_camera,
                                self.depth_camera,
                                self.front_camera,
                            ):
                                refresh_camera_from_completed_render(camera)
                            after_render_only = float(self.world.current_time)
                            if abs(after_render_only - before_render_only) > 1e-9:
                                raise RuntimeError(
                                    "render-only synchronization advanced physics time: "
                                    + json.dumps(
                                        {
                                            "before": before_render_only,
                                            "after": after_render_only,
                                        },
                                        sort_keys=True,
                                    )
                                )
                    raise RuntimeError(
                        "render-only drain did not synchronize camera time: "
                        + json.dumps(
                            {"target_time": target_time, "attempts": attempts},
                            sort_keys=True,
                        )
                    )

                self.world.step = synchronized_world_step
                self._diagnostic_render_sync_records = sync_records
                prewarm_started = time.monotonic()
                samples = []
                for attempt in range(1, 21):
                    self._apply_safe_stop(count_physics_step=False)
                    self.world.step(render=True)
                    frames = {
                        "color": self.color_camera.get_current_frame(),
                        "depth": self.depth_camera.get_current_frame(),
                        "front": self.front_camera.get_current_frame(),
                    }
                    observed = {}
                    for name, frame in frames.items():
                        try:
                            raw_identity = frame.get("rendering_frame", -1)
                            if isinstance(raw_identity, dict):
                                render_id = int(raw_identity["referenceTimeNumerator"])
                                denominator = int(raw_identity["referenceTimeDenominator"])
                            else:
                                render_id = int(raw_identity)
                                denominator = 1
                            render_time = float(frame.get("rendering_time", -1.0))
                        except (KeyError, TypeError, ValueError):
                            render_id, denominator, render_time = -1, -1, -1.0
                        observed[name] = [render_id, denominator, render_time]
                    samples.append(observed)
                    identities = {tuple(value) for value in observed.values()}
                    if (
                        len(identities) == 1
                        and next(iter(identities))[0] > 0
                        and next(iter(identities))[1] > 0
                        and math.isfinite(next(iter(identities))[2])
                        and next(iter(identities))[2] >= 0.0
                    ):
                        self._diagnostic_prewarm_evidence = {
                            "render_steps": attempt,
                            "elapsed_sec": time.monotonic() - prewarm_started,
                            "final_identity": next(iter(identities)),
                            "samples": samples,
                        }
                        break
                else:
                    raise RuntimeError(
                        "safe render prewarm did not reach one positive shared camera identity: "
                        + json.dumps(samples, sort_keys=True)
                    )

            IsaacModelFreeBackend.__init__ = init_with_safe_render_prewarm

            original_rgb8 = IsaacModelFreeBackend._rgb8

            def rgb8_with_startup_not_ready(value, expected):
                try:
                    return original_rgb8(value, expected)
                except ValueError as exc:
                    if str(exc).startswith("unexpected rendered RGB shape:"):
                        raise CaptureNotReady(str(exc)) from exc
                    raise

            IsaacModelFreeBackend._rgb8 = staticmethod(rgb8_with_startup_not_ready)

        backend_started = time.monotonic()
        backend = IsaacModelFreeBackend(Path(os.environ["WRAPPER_USD"]).resolve())
        backend_constructed = time.monotonic()
        backend.reset(0)
        first_capture = None
        capture_not_ready_count = 0
        for _ in range(200):
            safe_step = backend.step_safe_stop()
            if not safe_step.rendered:
                continue
            try:
                first_capture = dict(backend.capture(safe_step))
            except CaptureNotReady as exc:
                capture_not_ready_count += 1
                reason = str(exc)
                capture_not_ready_reasons[reason] = (
                    capture_not_ready_reasons.get(reason, 0) + 1
                )
                continue
            break
        if first_capture is None:
            raise RuntimeError("minimal backend did not produce a valid capture in 200 steps")
        captured = time.monotonic()
        backend_evidence = {
            "backend_constructed_elapsed_sec": backend_constructed - started,
            "first_capture_elapsed_sec": captured - started,
            "backend_construct_duration_sec": backend_constructed - backend_started,
            "capture_not_ready_count": capture_not_ready_count,
            "stream_names": sorted(first_capture),
            "prewarm": getattr(backend, "_diagnostic_prewarm_evidence", None),
            "render_sync_records": getattr(
                backend, "_diagnostic_render_sync_records", None
            ),
            "front_clipping": getattr(
                backend, "_diagnostic_front_clipping", None
            ),
        }
    print(
        "INTERNNAV_ISAAC_STARTUP_DIAGNOSTIC_PASS="
        + json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "mode": mode,
                "config": {
                    "headless": True,
                    "anti_aliasing": 0,
                    "hide_ui": True,
                    "multi_gpu": False,
                },
                "import_elapsed_sec": imported - started,
                "simulation_app_elapsed_sec": app_ready - started,
                "extensions_update_elapsed_sec": extensions_ready - started,
                "backend_evidence": backend_evidence,
            },
            sort_keys=True,
        ),
        flush=True,
    )
except BaseException as exc:
    print(
        "INTERNNAV_ISAAC_STARTUP_DIAGNOSTIC_FAIL="
        + json.dumps(
            {
                "schema_version": 1,
                "status": "FAIL",
                "mode": mode,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "elapsed_sec": time.monotonic() - started,
                "prewarm": (
                    getattr(backend, "_diagnostic_prewarm_evidence", None)
                    if backend is not None
                    else None
                ),
                "render_sync_records": (
                    getattr(backend, "_diagnostic_render_sync_records", None)
                    if backend is not None
                    else None
                ),
                "capture_not_ready_reasons": capture_not_ready_reasons,
                "rgb_content_samples": rgb_content_samples[-8:],
                "front_clipping": (
                    getattr(backend, "_diagnostic_front_clipping", None)
                    if backend is not None
                    else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    traceback.print_exc()
    raise
finally:
    if backend is not None:
        backend.close()
    if app is not None and app.is_running():
        app.close()
    if temporary is not None:
        temporary.cleanup()
PY
  fi
  python_b64="$(printf '%s' "$python_program" | base64 | tr -d '\r\n')"

  read -r -d '' remote_program <<'REMOTE' || true
set -euo pipefail
python_b64="$1"
mode="$2"
export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export STARTUP_MODE="$mode"
export CONTROL_ROOT=/home/song/internnav-t1-t2
export WRAPPER_USD="$CONTROL_ROOT/results/parallel/sensor_producer/online-bootstrap-01r-g20260717t011738z-cf8e1164/runtime/go2_model_free.usda"
export PYTHONPATH="$CONTROL_ROOT"
case "$mode" in minimal-backend|minimal-backend-fixed-api|minimal-backend-fixed-api-prewarm) test -f "$WRAPPER_USD" ;; esac
marker="$(mktemp /tmp/internnav-isaac-startup-marker.XXXXXX)"
trap 'rm -f -- "$marker"' EXIT
set +e
printf '%s' "$python_b64" | base64 -d | /home/song/env_isaacsim/bin/python - | tee "$marker"
python_rc="${PIPESTATUS[2]}"
set -e
(( python_rc == 0 )) || exit "$python_rc"
grep -q '^INTERNNAV_ISAAC_STARTUP_DIAGNOSTIC_PASS=' "$marker"
REMOTE
  remote_b64="$(printf '%s' "$remote_program" | base64 | tr -d '\r\n')"

  "${SSH_AUTH[@]}" ssh "${ssh_options[@]}" "$target" \
    "bash -c \"\$(printf '%s' '$remote_b64' | base64 -d)\" diag '$python_b64' '$mode'"
}

if [[ "${1:-}" == "_under_lease" ]]; then
  [[ $# -eq 2 ]] || die "invalid internal invocation"
  run_under_lease "$2"
  exit $?
fi

[[ $# -ge 1 && $# -le 2 ]] || \
  die "usage: diagnose_01r_isaac_startup.sh DIAGNOSTIC_ID [hide-ui|inventory|minimal|minimal-backend|minimal-backend-fixed-api|minimal-backend-fixed-api-prewarm]"
diag_id="$1"
mode="${2:-hide-ui}"
[[ "$diag_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || die "unsafe diagnostic id"
case "$mode" in hide-ui|inventory|minimal|minimal-backend|minimal-backend-fixed-api|minimal-backend-fixed-api-prewarm) ;; *) die "unsupported diagnostic mode" ;; esac
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(readlink -f "$(dirname "$script_path")/..")"
log_dir="$root/results/parallel/sensor_producer/lease-isaac-startup-${diag_id}"
[[ ! -e "$log_dir" ]] || die "diagnostic result path already exists"

bash "$root/scripts/with_resource_lease.sh" isaac \
  --owner codex-00 \
  --task "01r-isaac-startup-${diag_id}" \
  --log-dir "$log_dir" \
  -- bash "$script_path" _under_lease "$mode"
