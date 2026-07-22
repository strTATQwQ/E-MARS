# E-MARS

**E-MARS** — **Edge-deployed Multimodal Agent for Robotic Search-and-rescue**

E-MARS is a ROS 2 navigation stack that combines multimodal language-guided
planning, InternVLA, optional Step3-VL advice, Nav2, recovery behaviors,
watchdogs, and bounded robot control. The project targets edge deployment on
NVIDIA DGX-class hardware with Isaac Sim providing RGB-D, LiDAR, IMU, and Go2
simulation during development.

## Branches

- `sim`: Isaac Sim development and evaluation stack. This is the initial
  public branch.
- `real-go2`: strict physical Go2 integration. It is kept separate from
  simulation safety relaxations and remains fail-closed until stationary
  hardware, TF, localization, watchdog, and control-mux prerequisites pass.

Simulation-only settings must never be copied into the physical-robot branch.
In particular, simulated pose sources, simplified motion, and relaxed
freshness or collision policies are not valid real-robot defaults.

## Architecture

```text
Multilingual natural-language task
  -> mandatory Step3-VL instruction normalization
  -> bounded English canonical mission
  -> InternVLA policy
  -> optional timeout-only Step3-VL navigation advisor
  -> ROS 2 / Nav2 / recovery / watchdog
  -> bounded velocity control
  -> real Go2 control bridge (disabled until explicitly armed)

Real Go2 + D435 + semantic-camera RGB-D/LiDAR/IMU
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
- `slow_planner` and `step3_graph_nav`: optional slow-planner components;
- `isaac_vln_benchmark`: Isaac/ROS 2 simulation runtime;
- `frontend`: pinned `vla-nav-panel` submodule and canonical operator panel;
- `slow_planner_frontend`: synchronized compatibility copy for existing launchers;
- `hardware`: pinned Unitree Go2 edge-AI hardware-description submodule;
- `configs` and `scripts`: launch configuration and orchestration.

Clone with both pinned dependencies:

```bash
git clone --branch real-go2 --recurse-submodules \
  https://github.com/strTATQwQ/E-MARS.git
cd E-MARS
git submodule update --init --recursive
```

The strict real-Go2 language path never sends raw Chinese or other operator
text directly to InternVLA. Step3 must first return a schema-validated,
identity-bound canonical English mission. Timeout, abstention, stale identity,
or invalid output leaves the robot in safe hold.

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
The `real-go2` branch also starts disarmed with E-stop latched; source presence
alone is not authorization to move physical hardware.

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md). Hardware work is maintained in the
`railgunqaq/unitree-go2-edge-ai-hardware` submodule and remains attributed to
its original authors.

## License

E-MARS is released under the MIT License.
