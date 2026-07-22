# E-MARS

**E-MARS** — **Edge-deployed Multimodal Agent for Robotic Search-and-rescue**

E-MARS is a ROS 2 navigation stack that combines multimodal language-guided
planning, InternVLA, one selected slow advisor (Step3-VL-10B or
Step-3.7-Flash), Nav2, recovery behaviors, watchdogs, and bounded robot
control. The project targets edge deployment on NVIDIA DGX-class hardware
with Isaac Sim providing RGB-D, LiDAR, IMU, and Go2 simulation during
development.

## Branches

- `sim`: Isaac Sim development and evaluation stack. This is the initial
  public branch.
- `real-go2`: strict physical Go2 integration. It is kept separate from
  simulation safety relaxations and will be published after stationary
  hardware integration is ready.

Simulation-only settings must never be copied into the physical-robot branch.
In particular, simulated pose sources, simplified motion, and relaxed
freshness or collision policies are not valid real-robot defaults.

## Architecture

```text
Natural-language task
  -> InternVLA policy
  -> one bounded slow advisor: Step3-VL-10B or Step-3.7-Flash
  -> ROS 2 / Nav2 / recovery / watchdog
  -> bounded velocity control
  -> Isaac Go2 simulation

Isaac RGB-D + LiDAR + IMU
  -> ROS 2 sensor bridge
  -> localization / mapping / local costmap
  -> InternVLA and Nav2
```

The major source packages are:

- `internvla_ros2` and `internvla_ros2_msgs`: typed model and client protocol;
- `internvla_nav2_adapter`: Nav2 command resolution;
- `internvla_t4_recovery`: bounded recovery behavior;
- `internvla_t4_sensors`: sensor and odometry integration;
- `internvla_go2_controller`: bounded simulated Go2 control;
- `slow_planner` and `step3_graph_nav`: slow-planner components for the
  selected advisor;
- `isaac_vln_benchmark`: Isaac/ROS 2 simulation runtime;
- `slow_planner_frontend`: operator panel;
- `configs` and `scripts`: launch configuration and orchestration.

### Slow-advisor selection

The slow-advisor stage is required. Select exactly one model for a deployment:
either **Step3-VL-10B** or **Step-3.7-Flash**. The two choices are mutually
exclusive and must not be enabled together.

## Hardware

The physical Go2 hardware design, including power, cameras, installation
photos, and CAD models, is maintained in
[`railgunqaq/unitree-go2-edge-ai-hardware`](https://github.com/railgunqaq/unitree-go2-edge-ai-hardware).
The relative [`hardware`](hardware) symbolic link points to a sibling checkout
of that repository, so the hardware content can be updated independently.

Clone the software and hardware repositories side by side:

```bash
git clone https://github.com/strTATQwQ/E-MARS.git
git clone https://github.com/railgunqaq/unitree-go2-edge-ai-hardware.git
```

## Repository policy

Model weights, datasets, scene assets, credentials, machine-local settings,
recorded sensor streams, recorded test data, test results, run logs, reports,
videos, and evaluation artifacts are intentionally excluded from this public
source repository.

Copy `.env.example` to `.env.local` and provide local host names, usernames,
paths, and tokens. Never commit `.env.local`.

## Development

The repository contains Python packages, ROS 2 packages, shell launchers, and
offline unit tests. Exact runtime dependencies are recorded in
`dependencies.lock.yaml`. Hardware-specific assets and model checkpoints must
be provisioned separately.

```bash
python -m pytest -q \
  tests/slow_planner_frontend \
  tests/test_t5_motion_observation_gate.py \
  tests/test_t5_observation_identity_guard.py \
  tests/test_t5_nav2_namespace.py \
  tests/test_t5_trajectory_rerank.py \
  tests/test_t5_step3_deadline_contract.py \
  tests/test_t5_step3_direct.py \
  tests/test_t5_revc_x86_contract.py \
  tests/test_t5_planar_motion.py
```

Real Go2 motion is outside the scope of the `sim` branch.

## License

E-MARS is released under the MIT License.
