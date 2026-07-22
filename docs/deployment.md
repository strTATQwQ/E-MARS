# E-MARS Simulation Deployment Guide

## English

### 1. Deployment target

This guide deploys the `sim` branch as a distributed navigation stack. The
recommended topology is:

| Role | Runs | Must not run |
| --- | --- | --- |
| DGX Lane | InternVLA, ROS 2 client/model nodes, localization/mapping, Nav2, recovery, watchdog, optional Step service | Isaac Sim |
| Isaac x86 | Isaac Sim/Isaac Lab, Go2 simulation, RGB-D/LiDAR/IMU, `/clock`, episode/reset/evaluator | InternVLA or Nav2 |
| Operator workstation | Source control, orchestration, log collection, optional web browser | Motion bridge without a lease |

For dual-Lane development, duplicate the complete DGX stack and give each Lane
an isolated Isaac GPU, CPU set, ROS domain, namespace, ports, cache, run root,
and locks. Do not split one final Lane into a permanent “model server” and
“edge server”; the deployment target is one complete edge stack per DGX.

### 2. Prerequisites

Install and verify, using versions compatible with your host image:

- NVIDIA driver, CUDA, and one supported GPU per active Lane;
- Isaac Sim/Isaac Lab on the x86 simulator host;
- Ubuntu/Linux on DGX, ROS 2 Jazzy, Nav2, and the required message packages;
- Python environments for InternVLA and the selected slow model;
- Docker only where an existing launcher explicitly requires it;
- Git, Git LFS where applicable, `flock`, `rsync`, `ssh`, and `jq`;
- routed network connectivity and working ROS 2 DDS discovery between hosts.

Read `dependencies.lock.yaml` before installation. It records upstream commits
and runtime facts; it does not redistribute external repositories or weights.

### 3. Clone and local configuration

```bash
git clone --branch sim --recurse-submodules https://github.com/strTATQwQ/E-MARS.git
cd E-MARS
git submodule update --init --recursive
cp .env.example .env.local
chmod 600 .env.local
```

The canonical operator-panel source is the pinned `frontend` submodule. Install
that checkout rather than copying `slow_planner_frontend` into E-MARS:

```bash
python3 -m venv .venv-panel
source .venv-panel/bin/activate
python -m pip install -e ./frontend
```

Keep host names, usernames, model paths, tokens, ROS domain IDs, and result
roots in `.env.local` or a host-local systemd environment file. Never commit
them. Large Hugging Face downloads may use `HF_ENDPOINT=https://hf-mirror.com`.

Typical machine-local variables include:

```bash
DGX_HOST="<DGX_HOST>"
ISAAC_HOST="<ISAAC_HOST>"
ROS_DOMAIN_ID="<DOMAIN_ID>"
ROS_NAMESPACE="/<LANE_NAMESPACE>"
CUDA_VISIBLE_DEVICES="<GPU_ID>"
INTERNVLA_MODEL_PATH="<MODEL_ROOT>/InternVLA"
STEP3_VL_10B_MODEL_PATH="<MODEL_ROOT>/Step3-VL-10B"
SLOW_BENCHMARK_RESULTS="<NON_GIT_RUN_ROOT>"
```

Do not place generated output under a Git-tracked results/report directory in
the public checkout. Use a machine-local run root outside the repository.

### 4. Python and ROS environments

Use separate environments for large models, ROS 2, and Isaac. A generic source
workspace build is:

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p "<ROS_WS>/src"
# Link or copy only the required ROS packages into <ROS_WS>/src.
cd "<ROS_WS>"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Do not upgrade Transformers, PyTorch, CUDA, or ROS packages merely to match a
global environment. Step3-VL clean loading is pinned to its checked-in runtime
contract; model-specific environments may intentionally differ.

### 5. Model provisioning

Model weights are not in this repository. Provision them to a host-local model
root, record the resolved revision and file hashes, then point `.env.local` to
that root. For Step3-VL-10B, the checked-in configuration requires a clean BF16
load and the pinned Transformers/checkpoint mapping contract.

Before an online run, verify:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
test -d "$INTERNVLA_MODEL_PATH"
test -d "$STEP3_VL_10B_MODEL_PATH"   # only when selected
```

### 6. Optional mission normalization

The frozen simulation baseline leaves mission normalization disabled:

```text
configs/completion_sim/mission_normalization_disabled.yaml
```

To normalize multilingual input, select exactly one provider.

**Local Step3-VL-10B** — reuse the resident Step3 service on TCP 8200:

```bash
python -m slow_planner.serve \
  --config configs/slow_models/step3_vl_10b_bf16.yaml
python scripts/normalize_sim_instruction.py \
  --config configs/completion_sim/mission_normalization_step3_vl.yaml \
  --instruction "<TASK>" --mission-id "<MISSION_ID>" \
  --episode-id "<EPISODE_ID>"
```

**Hosted Step-3.7-Flash** — start the text-only API adapter on TCP 8210:

```bash
export STEPFUN_API_KEY="<LOCAL_SECRET>"
python -m slow_planner.serve \
  --config configs/slow_models/step_3_7_flash_normalizer.yaml
python scripts/normalize_sim_instruction.py \
  --config configs/completion_sim/mission_normalization_step37_flash.yaml \
  --instruction "<TASK>" --mission-id "<MISSION_ID>" \
  --episode-id "<EPISODE_ID>"
```

Call normalization once at mission ingress. Bind the canonical output to the
episode/reset/sequence identity and regenerate matching InternVLA tokens.
Never reuse token IDs from the source instruction. `passthrough` is a
simulation-only failure policy; use `reject` when canonical English is required.

### 7. Bring-up sequence

Acquire the Lane's DGX and Isaac leases before starting heavy services. Bring
up one Lane in this order:

1. Source ROS 2 and the built workspace; set domain and namespace.
2. Start the selected model service and verify its health/config identity.
3. Start InternVLA model and ROS client nodes.
4. Start localization, map/costmaps, Nav2, recovery, and watchdog.
5. Start the bounded command adapter in safe-stop state.
6. Start the Isaac worker with the Lane-specific GPU, CPU set, Kit profile,
   cache, temporary directory, namespace, and ROS domain.
7. Verify `/clock` is published only by Isaac and `use_sim_time=true` on DGX.
8. Verify sensor, TF, episode/reset, model, Nav2, and command topics are fresh.
9. Run a short canary before an episode batch or soak.

Repository launchers include:

- `scripts/run_t5_dgx_lane.sh`
- `scripts/run_t5_distributed_isaac.sh`
- `scripts/run_t5_step3_live_services.sh`
- coordination scripts under `coordination/`

These launchers enforce more parameters than the abbreviated commands in this
guide. Treat their current help text and checked-in configuration as the
machine-executable contract.

### 8. Required health checks

Before READY, confirm:

- one `/clock` authority and advancing simulation time;
- episode/reset/sequence IDs agree across bridge, model, and evaluator;
- RGB-D, LiDAR, IMU, odometry, CameraInfo, and required TF are non-empty;
- command age is bounded and stale-command safe-stop is active;
- Nav2 planner/controller/action endpoints are healthy;
- velocity and acceleration limits match the selected profile;
- no cross-Lane topics, ports, reset events, caches, or process groups exist;
- run root, config SHA, code SHA, model identity, and deviations are recorded
  outside the public repository.

For performance diagnosis, record wall duration, simulation duration, RTF,
commanded/measured yaw rate, command age, GPU utilization/VRAM, CPU utilization,
RAM, and swap. Low RTF indicates simulator/host throughput before it indicates
a navigation-policy failure.

### 9. Optional simulation progress panel

The navigation stack does not require the web panel, but an operator may start
the pinned submodule to monitor a running simulation. Use a Lane-specific ROS
domain and an external run root; the launcher starts only the panel and its ROS
telemetry adapter, not Isaac, InternVLA, Nav2, or a motion publisher.

```bash
export ROS_DOMAIN_ID="<DOMAIN_ID>"
export E_MARS_FRONTEND_PYTHON="$PWD/.venv-panel/bin/python"
export E_MARS_ROS_PYTHON="<ROS_PYTHON>"
export E_MARS_ROS_SETUP="<ROS_WORKSPACE>/install/setup.bash"
bash scripts/run_sim_frontend.sh \
  configs/internnav_t5/lane_b_step3.yaml \
  /var/tmp/e-mars/sim-panel-"$ROS_DOMAIN_ID"
```

Open `http://<DGX_HOST>:8300/` from the trusted LAN. The panel shows
rate-limited camera previews, episode/reset/sequence identity, structured model
decisions, ROS/Nav2/recovery/watchdog state, command feedback, latency, and
resource telemetry. Original RGB-D/LiDAR/IMU streams remain on ROS and are not
routed back into the model through HTTP/JPEG. Stop the foreground launcher with
TERM or Ctrl-C; it performs a bounded cleanup of the two process groups it owns.

For two simultaneous Lanes, give each panel a distinct port in its config,
`ROS_DOMAIN_ID`, run root, and `E_MARS_SIM_FRONTEND_LOCK`. A panel failure is a
monitoring failure and must not terminate the simulator or evaluator.

### 10. Dual-Lane operation

Each Lane must have unique values for:

- `ROS_DOMAIN_ID`, namespace, model/health ports, and request prefix;
- `CUDA_VISIBLE_DEVICES`, CPU affinity, Isaac profile, shader/cache, TMP;
- run root, PID ledger, socket, and lock paths.

Serialize shared asset conversion, shader warm-up, model copying, video
encoding, and large archives. If shared x86 CPU/RAM/SSD/network contention
reduces either Lane by more than the accepted budget, interleave Isaac workers
while keeping both DGX development streams active.

### 11. Stop and cleanup

Stop through the owning coordinator or lease wrapper. Use TERM, a bounded wait,
then precise KILL only for the known run root if required. Verify that the
Lane's PID/PGID, sockets, ports, locks, and simulator processes are zero before
releasing resources. SIGTERM exit 143 is acceptable only when cleanup proves
zero residual state.

### 12. Troubleshooting

| Symptom | Check first |
| --- | --- |
| DDS topics missing | Domain ID, namespace, peer configuration, firewall, NIC selection |
| Sensors visible but frozen | `/clock`, reset generation, source timestamp, QoS |
| Model health fails | checkpoint revision, environment, GPU visibility, port owner |
| Normalizer falls back | provider health, identity, API credential, timeout, JSON schema |
| Nav2 never READY | TF chain, odometry age, map/costmap source, lifecycle state |
| Motion slower than expected | RTF, CPU single-core saturation, GPU/VRAM, command age |
| Residual process blocks next run | owning run root, process group, socket/port, lease ledger |

---

## 中文

### 1. 部署目标

本说明将 `sim` 分支部署为分布式导航栈。推荐拓扑如下：

| 角色 | 运行内容 | 禁止内容 |
| --- | --- | --- |
| DGX Lane | InternVLA、ROS 2 client/model、定位/地图、Nav2、recovery、watchdog、可选 Step 服务 | Isaac Sim |
| Isaac x86 | Isaac Sim/Isaac Lab、Go2 仿真、RGB-D/LiDAR/IMU、`/clock`、episode/reset/evaluator | InternVLA 或 Nav2 |
| 操作员工作站 | 源码管理、编排、日志收集、可选浏览器 | 未持有租约的运动 bridge |

双 Lane 研发时，每个 Lane 都应拥有完整 DGX 栈以及独立 Isaac GPU、CPU 集、
ROS domain、namespace、端口、cache、run root 和锁。最终目标不是长期把一台
DGX 固定为模型服务器、另一台固定为边缘节点，而是每台 DGX 都能独立运行完整
机载栈。

### 2. 前置条件

按照主机镜像兼容性安装并验证：

- NVIDIA driver、CUDA，以及每个活动 Lane 对应的一张 GPU；
- x86 仿真主机上的 Isaac Sim/Isaac Lab；
- DGX Linux、ROS 2 Jazzy、Nav2 和所需消息包；
- InternVLA 与所选慢模型的隔离 Python 环境；
- 仅在既有启动器明确要求时使用 Docker；
- Git、必要时的 Git LFS、`flock`、`rsync`、`ssh`、`jq`；
- 三机路由网络和正常的 ROS 2 DDS discovery。

安装前先读 `dependencies.lock.yaml`。它记录上游 commit 和运行事实，但不会替你
下载外部仓库、模型或资产。

### 3. 克隆与本地配置

```bash
git clone --branch sim --recurse-submodules https://github.com/strTATQwQ/E-MARS.git
cd E-MARS
git submodule update --init --recursive
cp .env.example .env.local
chmod 600 .env.local
```

正式操作员前端来自固定版本的 `frontend` 子模块，不要再把
`slow_planner_frontend` 复制进 E-MARS。安装方式如下：

```bash
python3 -m venv .venv-panel
source .venv-panel/bin/activate
python -m pip install -e ./frontend
```

主机名、用户名、模型路径、token、ROS domain ID 和结果根目录只能放在
`.env.local` 或主机本地 systemd environment file 中，绝不能提交。Hugging Face
大文件下载可以设置 `HF_ENDPOINT=https://hf-mirror.com`。

典型本地变量：

```bash
DGX_HOST="<DGX_HOST>"
ISAAC_HOST="<ISAAC_HOST>"
ROS_DOMAIN_ID="<DOMAIN_ID>"
ROS_NAMESPACE="/<LANE_NAMESPACE>"
CUDA_VISIBLE_DEVICES="<GPU_ID>"
INTERNVLA_MODEL_PATH="<MODEL_ROOT>/InternVLA"
STEP3_VL_10B_MODEL_PATH="<MODEL_ROOT>/Step3-VL-10B"
SLOW_BENCHMARK_RESULTS="<NON_GIT_RUN_ROOT>"
```

生成数据不得写入公共 checkout 内被 Git 跟踪的 results/reports 目录，应使用仓库
外的机器本地 run root。

### 4. Python 与 ROS 环境

大模型、ROS 2 和 Isaac 应使用隔离环境。通用 ROS workspace 构建方式：

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p "<ROS_WS>/src"
# 只将所需 ROS package 链接或复制到 <ROS_WS>/src。
cd "<ROS_WS>"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

不要为了匹配全局环境而随意升级 Transformers、PyTorch、CUDA 或 ROS 包。
Step3-VL clean-load 受仓库内合同约束，不同模型环境有意保持不同版本是正常的。

### 5. 模型准备

仓库不包含模型权重。请把权重部署到主机本地模型目录，记录 revision 与文件 hash，
再由 `.env.local` 指向该目录。Step3-VL-10B 配置要求 clean BF16 load，以及固定
Transformers 和 checkpoint mapping 合同。

在线运行前至少检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
test -d "$INTERNVLA_MODEL_PATH"
test -d "$STEP3_VL_10B_MODEL_PATH"   # 仅在选择该模型时
```

### 6. 可选任务规范化

冻结仿真基线默认关闭规范化：

```text
configs/completion_sim/mission_normalization_disabled.yaml
```

需要处理中文等多语言输入时，只能选择一个 provider。

**本地 Step3-VL-10B**：复用 TCP 8200 常驻服务。

```bash
python -m slow_planner.serve \
  --config configs/slow_models/step3_vl_10b_bf16.yaml
python scripts/normalize_sim_instruction.py \
  --config configs/completion_sim/mission_normalization_step3_vl.yaml \
  --instruction "<TASK>" --mission-id "<MISSION_ID>" \
  --episode-id "<EPISODE_ID>"
```

**托管 Step-3.7-Flash**：启动 TCP 8210 文本专用 API adapter。

```bash
export STEPFUN_API_KEY="<LOCAL_SECRET>"
python -m slow_planner.serve \
  --config configs/slow_models/step_3_7_flash_normalizer.yaml
python scripts/normalize_sim_instruction.py \
  --config configs/completion_sim/mission_normalization_step37_flash.yaml \
  --instruction "<TASK>" --mission-id "<MISSION_ID>" \
  --episode-id "<EPISODE_ID>"
```

规范化只在任务入口调用一次。canonical 输出必须绑定 episode/reset/sequence，
并为 InternVLA 重新生成匹配 token；禁止复用源文本 token ID。`passthrough` 只是
仿真失败策略；必须得到英文 canonical mission 时请使用 `reject`。

### 7. 启动顺序

重任务启动前必须持有 Lane 的 DGX 与 Isaac 租约。单 Lane 按以下顺序启动：

1. source ROS 2 与构建 workspace，设置 domain 和 namespace；
2. 启动所选模型服务，核对 health/config identity；
3. 启动 InternVLA model 与 ROS client；
4. 启动定位、地图/costmap、Nav2、recovery 和 watchdog；
5. 以 safe-stop 状态启动有界命令适配器；
6. 使用 Lane 专用 GPU、CPU、Kit profile、cache、TMP、namespace 和 domain 启动
   Isaac worker；
7. 确认只有 Isaac 发布 `/clock`，DGX 所有仿真节点 `use_sim_time=true`；
8. 确认传感器、TF、episode/reset、模型、Nav2 和命令话题新鲜；
9. 先跑短 canary，再开始 episode batch 或 soak。

仓库入口包括：

- `scripts/run_t5_dgx_lane.sh`
- `scripts/run_t5_distributed_isaac.sh`
- `scripts/run_t5_step3_live_services.sh`
- `coordination/` 下的协调脚本

这些启动器比本文缩略命令检查更多参数。真正执行时，以启动器当前 `--help` 和
仓库内配置为机器可执行合同。

### 8. READY 前检查

- `/clock` 唯一且 sim time 持续前进；
- bridge、模型、evaluator 的 episode/reset/sequence 一致；
- RGB-D、LiDAR、IMU、odometry、CameraInfo 与所需 TF 非空；
- command age 有界，stale-command safe-stop 已启用；
- Nav2 planner/controller/action endpoint 健康；
- 速度和加速度限制与配置一致；
- 无跨 Lane topic、端口、reset、cache 或 process group；
- run root、config SHA、code SHA、model identity 与 deviation 记录在公共仓库外。

性能诊断应同时记录 wall duration、sim duration、RTF、commanded/measured yaw、
command age、GPU/显存、CPU、RAM 和 swap。RTF 低时应先排查仿真主机吞吐，而不是
直接归因导航策略。

### 9. 可选仿真进度前端

导航栈不依赖 Web 前端，但操作员可以启动固定版本的子模块监测正在运行的仿真。
必须使用 Lane 专属 ROS domain 和仓库外 run root。以下启动器只启动前端及其 ROS
遥测 adapter，不会启动 Isaac、InternVLA、Nav2 或运动 publisher：

```bash
export ROS_DOMAIN_ID="<DOMAIN_ID>"
export E_MARS_FRONTEND_PYTHON="$PWD/.venv-panel/bin/python"
export E_MARS_ROS_PYTHON="<ROS_PYTHON>"
export E_MARS_ROS_SETUP="<ROS_WORKSPACE>/install/setup.bash"
bash scripts/run_sim_frontend.sh \
  configs/internnav_t5/lane_b_step3.yaml \
  /var/tmp/e-mars/sim-panel-"$ROS_DOMAIN_ID"
```

在可信局域网中打开 `http://<DGX_HOST>:8300/`。前端显示限频相机预览、
episode/reset/sequence 身份、结构化模型决策、ROS/Nav2/recovery/watchdog 状态、
命令反馈、延迟和资源遥测。原始 RGB-D/LiDAR/IMU 仍通过 ROS 直接进入导航消费者，
不能通过 HTTP/JPEG 回灌模型。使用 TERM 或 Ctrl-C 停止前台 launcher；它只清理自己
拥有的两个进程组，并进行有限等待。

双 Lane 同时使用前端时，必须在配置中设置不同端口，并分别设置 `ROS_DOMAIN_ID`、
run root 和 `E_MARS_SIM_FRONTEND_LOCK`。前端故障只能记为监测故障，不得终止
simulator 或 evaluator。

### 10. 双 Lane 运行

每个 Lane 必须独立设置：

- `ROS_DOMAIN_ID`、namespace、模型/health 端口、request prefix；
- `CUDA_VISIBLE_DEVICES`、CPU affinity、Isaac profile、shader/cache、TMP；
- run root、PID ledger、socket 与 lock。

共享资产转换、shader 预热、模型复制、视频编码和大型归档必须串行。如果 x86
共享 CPU/RAM/SSD/网络让任一 Lane 超过允许退化，则交错运行 Isaac，但两台 DGX
的开发仍可并行。

### 11. 停止与清理

必须通过拥有资源的 coordinator/lease wrapper 停止。先 TERM，有限等待，必要时
只对已知 run root 精确 KILL。释放资源前证明 PID/PGID、socket、port、lock 和
simulator process 为零。SIGTERM 143 只有在零残留成立时才算正常退出。

### 12. 常见故障

| 现象 | 优先检查 |
| --- | --- |
| DDS 看不到话题 | Domain ID、namespace、peer、firewall、NIC |
| 传感器可见但不更新 | `/clock`、reset generation、source stamp、QoS |
| 模型 health 失败 | checkpoint revision、环境、GPU、端口 owner |
| 规范化 fallback | provider health、identity、API key、timeout、JSON schema |
| Nav2 不 READY | TF、odometry age、map/costmap source、lifecycle |
| 运动明显变慢 | RTF、单核 CPU、GPU/显存、command age |
| 下轮被残留阻塞 | run root、process group、socket/port、lease ledger |
