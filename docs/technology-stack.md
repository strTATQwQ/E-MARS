# E-MARS Technology Stack

This document lists the public software and model stack used by the E-MARS
simulation and edge-agent architecture. Versioned configuration files and
`dependencies.lock.yaml` remain authoritative for machine execution. The
versions below describe the integration baseline; component-specific isolated
environments may keep narrower checked-in pins.

## Platforms, SDKs, models, and middleware

| Category | Technology / model | Role in E-MARS |
| --- | --- | --- |
| NVIDIA SDK | Isaac Sim 6.0.0.1 | Go2, USD scenes, physics, cameras, and sensor simulation. |
| NVIDIA SDK | Isaac Lab 6.1.14 | Go2 environment, sensors, and simulation-task packaging. |
| NVIDIA SDK | Isaac ROS release 4.5 | NVIDIA-accelerated ROS 2 perception and integration components. |
| NVIDIA SDK | Isaac ROS Nvblox | RGB-D/LiDAR spatial fusion and Nav2 costmap input. |
| NVIDIA SDK | Isaac ROS Visual SLAM / cuVSLAM | Visual localization, odometry, and a selectable pose source. |
| NVIDIA SDK | Omniverse Kit / USD / PhysX | Scene management, rendering, USD assets, collision, and physics queries. |
| NVIDIA platform | DGX Spark / NVIDIA GB10 | Local multimodal-model inference, Agent services, and ROS 2 navigation compute. |
| NVIDIA platform | CUDA 13 | GPU inference and accelerated perception. |
| NVIDIA model | Cosmos Reason2 32B BF16 | Multi-view slow-planner research route and semantic candidate comparison. |
| StepFun model | Step3-VL-10B BF16 | Multi-view scene understanding, candidate viewpoint selection, and high-level navigation decisions. |
| StepFun model | Step 3.7 Flash | Dual-node high-level Agent route for environment exploration, task decomposition, and summaries. |
| Navigation model | InternVLA / InternNav | High-frequency visual-language navigation and local trajectory/action generation. |
| Robot middleware | ROS 2 Jazzy | Component communication, messages, services, actions, lifecycle, and identity propagation. |
| Navigation framework | Nav2 | Maps, planning, control, costmaps, recovery, and bounded goal execution. |
| Inference framework | PyTorch 2.12.1+cu130 + Transformers 4.57.6 | Local multimodal checkpoint loading and generation. |
| Communication | ROS 2 DDS, TCP/ZeroMQ, UDP | Model, sensor, clock, telemetry, and bounded control data exchange. |
| Robot | Unitree Go2 | Simulation navigation target and separately gated physical robot platform. |

## Fixed model and upstream revisions

| Dependency | Fixed revision | Repository/config evidence |
| --- | --- | --- |
| Step3-VL-10B | `5026053b0c2f5dfaa08fc2d149384162c3c8bca1` | `configs/slow_models/step3_vl_10b_bf16.yaml` |
| Cosmos Reason2 32B | `4ed9828334c4397ace8b0c62134961adbe5aed0e` | `configs/slow_models/cosmos_reason2_32b_bf16.yaml` |
| InternNav upstream | `7a5c62400ac45b313d9b709c740b64191556a242` | `dependencies.lock.yaml` |

Fixed revisions make the model identity auditable across local services,
offline replay, simulation, and navigation. They do not imply that model
weights are redistributed by this repository.

## Step3-VL in the control chain

Step3-VL is the multi-view semantic slow-planning layer. It combines a natural
language goal, scene images, navigation history, and reachable candidates to
compare frontier, viewpoint, or motion-primitive choices. Its typed semantic
decision converges with the InternVLA fast path in the command resolver. Nav2,
recovery, watchdog, and bounded command components retain motion authority.
