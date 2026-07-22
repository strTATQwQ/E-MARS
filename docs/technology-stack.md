# E-MARS Real-Go2 Technology Stack

## English

### 1. Stack overview

The strict branch combines edge AI, physical sensor drivers, ROS 2 navigation,
typed mission/model protocols, and a fail-closed control boundary. Hardware,
frames, topics, drivers, and calibration identities are explicit parts of the
deployment contract.

| Layer | Technology | Real-Go2 role |
| --- | --- | --- |
| Edge compute | NVIDIA DGX-class computer, CUDA | Co-located InternVLA, Step3, ROS 2, localization/map, Nav2, UI, watchdog/control |
| Robot | Unitree Go2, Unitree/Go2 ROS transport | State, LiDAR/IMU/front camera, bounded physical actuation |
| RGB-D | Intel RealSense D435, librealsense/ROS driver | Color, depth, CameraInfo; hardware capability determines IMU availability |
| Semantic vision | Four V4L2/MJPEG USB cameras | Front-left, front, front-right, rear context |
| Fast navigation | InternVLA / InternNav | Canonical-language-conditioned local navigation policy |
| Language/semantics | Step3-VL-10B BF16 | Mandatory multilingual normalization and optional timeout-only bounded advice |
| Middleware | ROS 2 Jazzy, DDS, TF2, Nav2 | Sensor/state transport, frames, lifecycle, planning, control, recovery |
| Mapping/localization | LiDAR/IMU odometry, cuVSLAM, static map, LiDAR costmap, optional Nvblox | Measured pose and obstacle/map state; each source qualified independently |
| Control | Typed resolver, recovery, watchdog, bounded control mux | Single fail-closed path to the Go2 motion bridge |
| Operator UI | FastAPI/Uvicorn, WebSocket, HTML/CSS/JS | Seven visual views, mission/decision, ROS/Go2/model/Nav2 health |
| Process management | systemd, shell launchers, `flock` | Service ownership, host-local environment, exclusive resources, cleanup |

### 2. Co-located edge architecture

The final deployment places all decision and navigation components on one DGX:

```text
Step3-VL + InternVLA
+ ROS 2 sensor/state bridge
+ localization + map/Nvblox + costmaps
+ Nav2 + recovery + watchdog
+ bounded control mux + Go2 bridge
+ operator backend
```

This reduces raw-sensor network forwarding and keeps command-age enforcement
close to the robot. A second DGX may be used as an independent development Lane,
but a production Lane must not depend on a permanent remote model server.

### 3. Model and dependency identities

| Model/dependency | Identity | Strict use |
| --- | --- | --- |
| Step3-VL-10B | `5026053b0c2f5dfaa08fc2d149384162c3c8bca1` | Mandatory instruction normalizer; clean BF16 load and bounded JSON output |
| InternNav upstream | `7a5c62400ac45b313d9b709c740b64191556a242` | Fast policy dependency recorded in `dependencies.lock.yaml` |
| LongCLIP | `3966af9ae9331666309a22128468b734db4672a7` | Recorded visual-language subdependency |
| go2_ros2_sdk | `4e186b5f89bfec1f32c85676cbe22d4958e4f0fa` | External Go2 ROS integration reference |
| unitree_ros2 | `668d1ec5a05d1c38d3306bdca7d59f2ba3581a88` | External Unitree ROS 2 reference |
| Operator panel | Pinned `frontend` gitlink | Canonical VLA navigation UI |
| Hardware design | Pinned `hardware` gitlink | CAD, mounting, power, camera-placement source |

Local patches and external repositories recorded by `dependencies.lock.yaml`
must be audited before deployment.

### 4. Mandatory language normalization

The raw-language ingress uses `slow_planner_v1` over a bounded local endpoint.
Step3-VL receives raw operator text plus mission identity, but no velocity
authority. It returns exactly the canonical mission schema:

- source language;
- English canonical instruction;
- target description;
- bounded constraints;
- confidence; and
- abstention flag.

The gateway validates duplicate keys, prose/suffixes, ASCII bounds, confidence,
instruction digest, config SHA, mission ID, episode/reset/sequence, age, and
deadline. Only validated canonical text reaches InternVLA. Hidden reasoning and
raw model text are not exposed to the UI.

### 5. Physical sensor stack

**Go2 sources** provide low state, sport-mode state, LiDAR state/cloud/IMU,
odometry, and a front camera stream. Driver topic names are not sufficient; the
publisher, hardware interface, type, QoS, frame, timestamps, and rate must all
be verified.

**D435** provides RGB, rectified depth, and matching CameraInfo. Some D435
variants do not include an IMU; the system must reflect the actual device rather
than publishing synthetic state.

**Four semantic cameras** are bound by stable V4L2 paths and mapped to
front-left/front/front-right/rear only after a physical view check. Their
checked-in horizontal correction is an image-orientation transform, not an
extrinsic calibration.

The seven UI views are Go2 front, D435 RGB, D435 depth, and the four semantic
cameras. Navigation consumes the original ROS streams, never the panel JPEGs.

### 6. ROS 2, DDS, and TF

ROS 2 Jazzy supplies typed messages, services, actions, lifecycle, DDS
discovery, and TF2. The real branch uses wall/monotonic time (`use_sim_time=false`)
and must not accept a simulator `/clock`.

The TF tree must be derived from measured hardware configuration. Required
relationships typically include map/odom/base, LiDAR, D435 color/depth optical
frames, Go2 camera, and semantic camera frames. Nearest/latest TF tolerance
cannot replace a measured extrinsic.

QoS is source-specific. Sensor-data QoS may differ from reliable state/control
QoS, but the chosen profile must be confirmed at both publisher and subscriber.

### 7. Localization and mapping

Physical localization candidates include LiDAR/IMU odometry and cuVSLAM. A
candidate requires measured ATE/RPE or an accepted physical reference, yaw
drift, TF age, tracking loss, recovery behavior, and restart continuity.

Mapping components include:

- static/surveyed global map;
- LiDAR obstacle layer/local costmap;
- Nvblox shadow for observation without authority; and
- Nvblox active-local only after input, TF, reset, and stale behavior pass.

GT pose and simulation-derived transforms are forbidden. A mapping layer cannot
substitute for localization, and stale map data must not preserve old motion.

### 8. Nav2 and recovery

Nav2 provides planner, controller, behavior servers, lifecycle management, and
NavigateToPose/action execution. The typed resolver is the only model-to-Nav2
entry. It accepts current bounded candidates and rejects stale, malformed, or
cross-reset commands.

Recovery monitors no progress and short loops. It can cancel the active goal,
clear trajectory state, scan, and request replanning. Recovery is bounded in
attempt count/state and cannot repeatedly execute an old trajectory.

### 9. Control mux and watchdog

The checked-in strict policy currently bounds:

| Parameter | Public default |
| --- | --- |
| Command TTL | 0.30 s |
| Maximum linear x | 0.20 m/s |
| Maximum angular z | 0.30 rad/s |
| Maximum linear acceleration | 0.20 m/s² |
| Maximum angular acceleration | 0.40 rad/s² |

These limits are reviewed as part of commissioning. The mux also checks human arm, E-stop, required component health,
sensor/TF/localization freshness, command age, mission identity, and feedback.
Any failed prerequisite produces zero command/safe hold.

### 10. Frontend and API

The panel stack uses FastAPI/Uvicorn, WebSocket state streaming, rate-limited
JPEG previews, and structured JSON state. It shows mission/reset/sequence,
canonical instruction, decision/confidence/fallback, camera age, ROS/Go2 health,
Nav2/recovery/watchdog, command feedback, GPU memory, and latency.

The API has read endpoints plus high-level mission, cancel, arm, and E-stop
requests. It has no raw velocity endpoint. Backend policy can keep dispatch and
motion disabled regardless of UI state.

### 11. System services and configuration

Systemd units provide host-local restart and environment integration for the
panel and mission gateway. The checked-in service files must be installed with
reviewed paths/users/environment; do not assume repository placeholders match a
target host.

Primary configuration:

- `configs/strict_real_go2.yaml`
- `configs/internnav_t5/strict_real_go2_mission_ingress.json`
- `configs/slow_models/step3_vl_10b_bf16.yaml`
- `deploy/systemd/vla-nav-panel-strict-real.service`
- `deploy/systemd/internvla-real-mission-gateway.service`

### 12. Security and data policy

- Secrets live only in `.env.local` or protected host service credentials.
- The panel should be exposed only on a trusted LAN or behind authenticated TLS.
- No raw chain-of-thought is stored or displayed.
- Camera previews are rate-limited and should not be archived by default.
- Logs, sensor records, calibration data, test results, and reports stay outside
  the public Git repository.
- Resource leases and exact process ownership prevent one task from killing or
  controlling another task's services.

### 13. Upgrade policy

- Change one driver/model/control layer at a time.
- Pin and record model/config/source identities.
- Re-run stationary validation after any ROS topic, QoS, TF, driver, model, or
  frontend adapter change.
- Treat velocity/acceleration/TTL, arm, E-stop, and motion-bridge changes as
  safety-sensitive policy changes.
- Do not inherit a passing simulation setting into real hardware without a
  dedicated strict review.

---

## 中文

### 1. 技术栈总览

严格分支组合端侧 AI、物理传感器驱动、ROS 2 导航、类型化任务/模型协议和
fail-closed 控制边界。硬件、frame、topic、driver 与 calibration identity 都是
部署合同的显式组成部分。

| 层级 | 技术 | Real-Go2 作用 |
| --- | --- | --- |
| 边缘计算 | NVIDIA DGX 级计算机、CUDA | 共置 InternVLA、Step3、ROS 2、定位/地图、Nav2、UI、watchdog/control |
| 机器人 | Unitree Go2、Unitree/Go2 ROS transport | 状态、LiDAR/IMU/front camera 与有界物理执行 |
| RGB-D | Intel RealSense D435、librealsense/ROS driver | Color、depth、CameraInfo；是否有 IMU 由真实硬件决定 |
| 语义视觉 | 四路 V4L2/MJPEG USB 相机 | 左前、正前、右前、正后上下文 |
| 快速导航 | InternVLA / InternNav | 以 canonical 英文任务为条件的局部导航策略 |
| 语言/语义 | Step3-VL-10B BF16 | 强制多语言规范化与可选 timeout-only 受限辅助 |
| 中间件 | ROS 2 Jazzy、DDS、TF2、Nav2 | 传感器/状态、frame、lifecycle、规划、控制、恢复 |
| 地图/定位 | LiDAR/IMU odometry、cuVSLAM、static map、LiDAR costmap、可选 Nvblox | 实测位姿和障碍/地图状态；每个 source 独立验收 |
| 控制 | Typed resolver、recovery、watchdog、有界 control mux | 唯一 fail-closed Go2 motion bridge 路径 |
| 操作员 UI | FastAPI/Uvicorn、WebSocket、HTML/CSS/JS | 七路视觉、任务/决策、ROS/Go2/model/Nav2 health |
| 进程管理 | systemd、shell launcher、`flock` | 服务所有权、本地环境、独占资源与清理 |

### 2. 共置机载架构

最终一台 DGX 共置：

```text
Step3-VL + InternVLA
+ ROS 2 sensor/state bridge
+ localization + map/Nvblox + costmaps
+ Nav2 + recovery + watchdog
+ bounded control mux + Go2 bridge
+ operator backend
```

这样减少原始传感器跨网转发，并让 command-age enforcement 靠近机器人。第二台
DGX 可以作为独立研发 Lane，但生产 Lane 不能长期依赖远程模型服务器。

### 3. 模型与依赖身份

| 模型/依赖 | 身份 | 严格用途 |
| --- | --- | --- |
| Step3-VL-10B | `5026053b0c2f5dfaa08fc2d149384162c3c8bca1` | 强制 instruction normalizer；clean BF16 load 与有界 JSON |
| InternNav upstream | `7a5c62400ac45b313d9b709c740b64191556a242` | `dependencies.lock.yaml` 中的快速策略依赖 |
| LongCLIP | `3966af9ae9331666309a22128468b734db4672a7` | 视觉语言子依赖 |
| go2_ros2_sdk | `4e186b5f89bfec1f32c85676cbe22d4958e4f0fa` | 外部 Go2 ROS 集成参考 |
| unitree_ros2 | `668d1ec5a05d1c38d3306bdca7d59f2ba3581a88` | 外部 Unitree ROS 2 参考 |
| 操作员前端 | 固定 `frontend` gitlink | 正式 VLA navigation UI |
| 硬件描述 | 固定 `hardware` gitlink | CAD、安装、电源、相机位置来源 |

部署前必须审计 `dependencies.lock.yaml` 记录的本地 patch 与外部仓库。

### 4. 强制语言规范化

原始语言入口通过本地受限 endpoint 使用 `slow_planner_v1`。Step3-VL 接收操作员
原文与 mission identity，但没有速度权限；它返回固定 canonical schema：源语言、
英文指令、目标描述、受限约束、confidence 和 abstain。

Gateway 校验重复 key、prose/suffix、ASCII 边界、confidence、instruction digest、
config SHA、mission ID、episode/reset/sequence、age 与 deadline。只有通过校验的
canonical 文本进入 InternVLA。隐藏推理和模型原文不显示在 UI。

### 5. 物理传感器栈

**Go2** 提供 low state、sport state、LiDAR state/cloud/IMU、odometry 与前视相机。
话题名称不够；必须核对 publisher、硬件接口、type、QoS、frame、stamp 和 rate。

**D435** 提供 RGB、rectified depth 和匹配 CameraInfo。部分 D435 没有 IMU，系统
必须反映真实设备，不能发布合成状态。

**四路语义相机**通过稳定 V4L2 path 绑定，物理检查后映射为左前/正前/右前/正后。
配置中的水平修正只是图像方向变换，不是外参标定。

前端七路视觉为 Go2 front、D435 RGB、D435 depth 与四路语义相机。导航读取原始
ROS stream，绝不能把前端 JPEG 作为模型输入。

### 6. ROS 2、DDS 与 TF

ROS 2 Jazzy 提供 typed message/service/action、lifecycle、DDS discovery 与 TF2。
真实分支使用 wall/monotonic time（`use_sim_time=false`），不能接受 simulator
`/clock`。

TF 必须来自实测硬件配置，通常包括 map/odom/base、LiDAR、D435 color/depth optical、
Go2 camera 和语义相机 frame。nearest/latest TF tolerance 不能代替实测外参。

QoS 因 source 而异。sensor-data QoS 可以不同于可靠 state/control QoS，但必须同时
核对 publisher 和 subscriber。

### 7. 定位与地图

物理定位候选包括 LiDAR/IMU odometry 与 cuVSLAM，需要实测 ATE/RPE 或认可的物理
reference、yaw drift、TF age、tracking loss、recovery 和 restart continuity。

地图组件包括 static/surveyed global map、LiDAR obstacle/local costmap、无权限的
Nvblox shadow，以及在 input/TF/reset/stale 全部通过后的 Nvblox active-local。
禁止 GT pose 与仿真 TF。地图层不能代替定位，stale map 不能维持旧运动。

### 8. Nav2 与 Recovery

Nav2 提供 planner、controller、behavior、lifecycle 和 NavigateToPose/action。
Typed resolver 是唯一模型到 Nav2 入口，只接受当前有界候选，拒绝 stale、malformed
或 cross-reset 命令。

Recovery 监控无进展与短回环，可取消 active goal、清除轨迹状态、扫描并请求新规划。
它必须限制尝试次数/状态，不能重复执行旧轨迹。

### 9. Control mux 与 Watchdog

严格配置数值：

| 参数 | 默认值 |
| --- | --- |
| Command TTL | 0.30 s |
| 最大线速度 x | 0.20 m/s |
| 最大角速度 z | 0.30 rad/s |
| 最大线加速度 | 0.20 m/s² |
| 最大角加速度 | 0.40 rad/s² |

这些限制在 commissioning 中整体审查。Mux 还检查 human arm、E-stop、required
component health、sensor/TF/localization freshness、command age、mission identity
和 feedback。任何前置条件失败都输出零命令/safe hold。

### 10. 前端与 API

前端使用 FastAPI/Uvicorn、WebSocket、限频 JPEG 与结构化 JSON，显示
mission/reset/sequence、canonical instruction、decision/confidence/fallback、
camera age、ROS/Go2 health、Nav2/recovery/watchdog、command feedback、GPU memory
与 latency。

API 包含只读 endpoint，以及高层 mission、cancel、arm、E-stop 请求；没有 raw
velocity endpoint。无论 UI 状态如何，后端策略都可以保持 dispatch/motion 关闭。

### 11. Systemd 与配置

Systemd unit 负责前端和 mission gateway 的主机本地环境与重启。安装时必须审核
path/user/environment，不能假设仓库占位符匹配目标主机。

主要配置：

- `configs/strict_real_go2.yaml`
- `configs/internnav_t5/strict_real_go2_mission_ingress.json`
- `configs/slow_models/step3_vl_10b_bf16.yaml`
- `deploy/systemd/vla-nav-panel-strict-real.service`
- `deploy/systemd/internvla-real-mission-gateway.service`

### 12. 安全与数据策略

- secret 只能进入 `.env.local` 或受保护的 host service credential；
- 前端只应暴露在可信 LAN，或置于带认证的 TLS 后；
- 不保存或显示 raw chain-of-thought；
- 相机 preview 限频，默认不归档；
- 日志、传感器记录、标定数据、测试结果和报告都留在公共 Git 外；
- 资源租约与精确进程所有权避免一个任务杀死或控制另一个任务的服务。

### 13. 升级策略

- 每次只修改一个 driver/model/control layer；
- 固定并记录 model/config/source identity；
- 任何 ROS topic、QoS、TF、driver、model 或 frontend adapter 变化后重跑静止验收；
- velocity/acceleration/TTL、arm、E-stop 与 motion bridge 变化属于安全策略变更；
- 未经专门 strict review，禁止把仿真 PASS 配置继承到真机。
