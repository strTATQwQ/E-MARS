# E-MARS Strict Real-Go2 Deployment Guide

## English

### 1. Safety statement

This guide covers installation, stationary integration, control-chain
validation, and service operation. Physical motion must follow an approved
commissioning procedure with a clear area, an available E-stop, calibrated
frames, fresh state, and explicit human control.

Never copy `completion_sim` configuration into this branch. In particular, do
not enable GT pose, simplified dynamics, `use_sim_time`, raw-wire bypass,
WARN-only freshness/collision policies, or simulation command topics.

### 2. Target topology

The final edge topology is one complete co-located stack:

| Host/device | Responsibility |
| --- | --- |
| DGX edge computer | InternVLA, Step3-VL, ROS 2, sensor bridge, localization/map, Nav2, recovery, watchdog, control mux, operator backend |
| Unitree Go2 | Physical state, LiDAR/IMU/camera transport, and bounded motion execution |
| Intel RealSense D435 | RGB, depth, CameraInfo; IMU only if the actual hardware/driver exposes and validates it |
| Four semantic USB cameras | Front-left, front, front-right, rear visual context |
| Operator browser | HTTPS/LAN access to the panel; high-level mission/cancel/arm/E-stop requests only |

The DGX may use Wi-Fi for operator/upstream access while a separate wired NIC
connects to the Go2 network. Bind routes and ROS interfaces explicitly; do not
let DHCP changes silently move the Go2 transport to the wrong NIC.

### 3. Clone with dependencies

```bash
git clone --branch real-go2 --recurse-submodules \
  https://github.com/strTATQwQ/E-MARS.git
cd E-MARS
git submodule update --init --recursive
cp .env.example .env.local
chmod 600 .env.local
```

The submodules are:

- `frontend`: pinned `strTATQwQ/vla-nav-panel`;
- `hardware`: pinned `railgunqaq/unitree-go2-edge-ai-hardware`.

Install the panel only from the pinned submodule so simulation and real-Go2 use
the same reviewed frontend source:

```bash
python3 -m venv .venv-panel
source .venv-panel/bin/activate
python -m pip install -e ./frontend
```

Credentials, host names, NIC names, model paths, ROS domain, camera device
bindings, and run roots belong in `.env.local` or host-local systemd environment
files. Never commit them.

### 4. Host prerequisites

Install and verify:

- supported NVIDIA driver/CUDA and sufficient GPU memory;
- Ubuntu/Linux, ROS 2 Jazzy, Nav2, TF2, image transport, and required messages;
- Unitree/Go2 ROS transport compatible with the robot firmware;
- RealSense kernel/udev support and `librealsense`/ROS driver;
- V4L2/FFmpeg support for the four USB cameras and Go2 front stream;
- isolated Python environments for InternVLA, Step3-VL, and the panel;
- `flock`, `systemd`, `iproute2`, `ethtool`, `v4l2-ctl`, `jq`, and Git.

Read `dependencies.lock.yaml`; it records upstream revisions and local patch
facts that must be reviewed for the target host.

### 5. Network setup

Use placeholders and confirm interfaces on the actual host:

```bash
ip -brief link
ip -brief address
ip route
nmcli device status
```

Requirements:

- the wired Go2 NIC remains up and owns the robot subnet route;
- Wi-Fi/LAN remains the operator and package-access path;
- ROS 2 DDS uses the intended interface and `ROS_DOMAIN_ID`;
- model (8200) and panel (8300) ports are owned by expected processes only;
- no simulator `/clock` or simulation DDS peer is present;
- host firewall rules expose only the required LAN endpoints.

Do not infer device identity from a reused IP or SSH host key. Bind hardware by
stable interface/device path, serial where appropriate, and runtime topic data.

### 6. Hardware discovery without motion

Keep control disabled while performing discovery.

**Go2 and ROS graph**

```bash
source /opt/ros/jazzy/setup.bash
ros2 node list
ros2 topic list -t
ros2 topic info --verbose /lf/sportmodestate
ros2 topic info --verbose /utlidar/cloud
```

**D435**

```bash
rs-enumerate-devices
ros2 topic info --verbose /check/d435/color/image_raw
ros2 topic info --verbose /check/d435/depth/image_rect_raw
ros2 topic info --verbose /check/d435/color/camera_info
```

**USB cameras**

```bash
v4l2-ctl --list-devices
find /dev/v4l/by-path -maxdepth 1 -type l -print
```

Bind the four semantic cameras by stable `/dev/v4l/by-path` entries, not
enumeration order. Verify the physical view and only then assign front-left,
front, front-right, and rear. Preserve the configured horizontal correction for
these four streams; do not mirror Go2 or D435.

### 7. Expected sensor/topic contract

The checked-in strict config currently names the following logical sources.
Confirm actual type, QoS, frame, rate, and source before use:

| Source | Topic/config key | Required validation |
| --- | --- | --- |
| Go2 front camera | `/frontvideostream` or verified VideoHub source | decoder, orientation, timestamp/age |
| Go2 low state | `/lf/lowstate` | hardware source, rate, no simulation publisher |
| Go2 sport state | `/lf/sportmodestate` | mode/pose/velocity fields and freshness |
| LiDAR state | `/utlidar/lidar_state` | device health and timestamp |
| LiDAR IMU | `/utlidar/imu` | frame, units, covariance, rate |
| LiDAR cloud | `/utlidar/cloud` | frame, density, range, rate |
| Odometry | `/utlidar/robot_odom` | source, drift, frame chain, covariance |
| D435 RGB | `/check/d435/color/image_raw` | encoding, CameraInfo match, age |
| D435 RGB info | `/check/d435/color/camera_info` | measured intrinsics/frame |
| D435 depth | `/check/d435/depth/image_rect_raw` | encoding, scale, alignment, age |
| D435 depth info | `/check/d435/depth/camera_info` | intrinsics/frame |
| Static TF | `/tf_static` | measured, complete, no duplicate authority |
| Four semantic cameras | stable V4L2 bindings | view identity, orientation, frame, extrinsics |

If the D435 model does not expose IMU, mark it unsupported; do not synthesize an
IMU topic. Calibrate every required extrinsic before navigation use.

### 8. ROS and Python environments

Build only the required ROS packages into a host-local workspace:

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p "<ROS_WS>/src"
# Link the selected repository ROS packages into <ROS_WS>/src.
cd "<ROS_WS>"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Keep the Step3 model environment separate where its pinned Transformers and BF16
loader contract require it. Provision weights outside Git and record their
revision/hash in private run metadata.

### 9. Strict configuration

The primary files are:

- `configs/strict_real_go2.yaml` — frontend, sensors, topics, arm/E-stop,
  control bounds, required components;
- `configs/internnav_t5/strict_real_go2_mission_ingress.json` — language,
  identity, and raw-instruction boundary;
- `deploy/systemd/vla-nav-panel-strict-real.service` — panel/ROS adapter;
- `deploy/systemd/internvla-real-mission-gateway.service` — Step3-first mission
  gateway;
- `scripts/run_t5_strict_real_mission_gateway.sh` — gateway launcher.

Render environment placeholders locally and review the effective config. Do
not commit rendered values. Keep `use_sim_time=false` and review all arm,
E-stop, dispatch, motion-bridge, and control-bound settings as one strict policy.

### 10. Model and mission services

Start Step3-VL and verify its model/config identity before the mission gateway:

```bash
python -m slow_planner.serve \
  --config configs/slow_models/step3_vl_10b_bf16.yaml
```

Then start the strict mission gateway with the host-local environment expected
by its launcher/service. The gateway must:

- subscribe to raw `/user_instruction` only;
- call Step3 normalization on the local bounded endpoint;
- publish `/internvla/mission/canonical` only after schema and identity checks;
- discard timeout, abstain, stale/cross-reset, digest mismatch, or illegal output;
- never publish `cmd_vel`, physical motion, or terminal STOP.

Raw operator text must not reach InternVLA through a second IPC/topic path.

### 11. Frontend deployment

Install and run the pinned `frontend` submodule; do not copy its Python package
into this branch. Configure the service to listen on the approved LAN
interface/port and read the E-MARS `configs/strict_real_go2.yaml`. Updating the
panel means reviewing a new submodule commit and advancing the `frontend`
gitlink in both E-MARS branches.

The API surface includes:

- `GET /api/v1/health`
- `GET /api/v1/state`
- `GET /api/v1/cameras/{view_id}.jpg`
- `WS /api/v1/stream`
- high-level mission, cancel, arm, and E-stop request endpoints

The control endpoints remain governed by backend gates. UI controls cannot
bypass mission identity, sensor/TF freshness, arm, E-stop, watchdog, or command
bounds. Camera preview should be rate-limited; model consumers use ROS sensor
data, not HTTP JPEG.

### 12. Bring-up order

1. Confirm clear area, E-stop availability, robot support/stance, and zero
   non-zero publishers.
2. Verify wired Go2 and upstream network routes.
3. Start Go2 state/LiDAR/front-camera bridge in receive-only mode.
4. Start D435 and four semantic cameras; verify hardware bindings.
5. Start TF/static calibration sources only from measured configuration.
6. Start localization/map and verify its selected operating mode.
7. Start Step3-VL, InternVLA, ROS client, Nav2, recovery, and watchdog.
8. Start strict mission gateway and panel/ROS adapter.
9. Run stationary health/freshness/identity checks for at least 60 seconds.
10. Stop all task-owned processes and prove zero residual ports, sockets, locks,
    PID/PGID, and command publishers.

### 13. Stationary acceptance

Stationary integration passes only when:

- all required real topics have verified publisher/type/QoS/frame/source;
- sensor frequency, age, and gap are measured for 60 seconds;
- TF is complete for the intended consumers and tied to measured calibration;
- seven camera panels display the correct physical views and orientation;
- mission normalization rejects stale/cross-reset/invalid responses;
- Nav2/recovery/watchdog/control state is visible and internally consistent;
- E-stop latch and inactive arm are visible and effective;
- no non-zero motion command, goal, or terminal STOP is published; and
- cleanup proves zero task-owned residual state.

### 14. Calibration and motion commissioning

Calibration must measure camera intrinsics/extrinsics, robot/base frames,
LiDAR/IMU frames, time alignment, and repeatable device bindings. Store raw
measurements outside the public repository and commit only reviewed non-secret
configuration when appropriate.

Enabling arm or the motion bridge is a separate reviewed change. Before any
non-zero test, require explicit user authorization in the current task, clear
space, physical E-stop, conservative limits, current localization, healthy
Nav2/watchdog, command feedback, and a one-primitive test plan. Never infer
authorization from earlier sessions or from this document.

### 15. Stop, rollback, and cleanup

Stop only services owned by the current run. Use TERM, a bounded wait, then a
precise KILL for the known run root if necessary. Never kill unrelated model or
ROS processes by broad name matching. Verify zero task-owned PID/PGID, sockets,
ports, locks, and command publishers before release.

Rollback consists of stopping the mission/control services, restoring the
previous strict config SHA, placing E-stop in its safe state, and returning to
receive-only sensor monitoring.

### 16. Troubleshooting

| Symptom | Check first |
| --- | --- |
| Go2 topics vanish after Wi-Fi change | wired NIC route, DDS interface, robot subnet |
| Camera is green/corrupt | codec/stride/pixel format, decoder, source age |
| Camera labels are wrong | physical occlusion test and stable device path |
| D435 topics absent | USB bandwidth, device serial binding, driver namespace |
| TF unavailable | measured static transforms, frame IDs, duplicate authorities |
| Step3 rejects mission | service health, config SHA, request identity, JSON schema |
| InternVLA sees raw language | canonical-only latch and alternate IPC/topic bypass |
| Panel shows stale data | ROS adapter health, topic age, WebSocket state |
| Motion gate cannot arm | backend prerequisites, health, freshness, E-stop, arm policy |

---

## 中文

### 1. 安全声明

本文覆盖安装、静止接入、控制链验证和服务运行。物理运动必须遵循受审的
commissioning 流程，确保环境清空、急停可用、frame 已标定、状态新鲜并由人员
明确控制。

禁止从 `completion_sim` 复制配置。尤其禁止 GT pose、简化动力学、
`use_sim_time`、raw-wire 旁路、WARN-only freshness/collision 策略和仿真命令话题。

### 2. 目标拓扑

最终机载拓扑是一套完整共置栈：

| 主机/设备 | 职责 |
| --- | --- |
| DGX 机载计算机 | InternVLA、Step3-VL、ROS 2、sensor bridge、定位/地图、Nav2、recovery、watchdog、control mux、前端后端 |
| Unitree Go2 | 真实状态、LiDAR/IMU/相机传输与有界运动执行 |
| Intel RealSense D435 | RGB、depth、CameraInfo；只有硬件/驱动真实支持并通过验证时才使用 IMU |
| 四路语义 USB 相机 | 左前、正前、右前、正后视觉上下文 |
| 操作员浏览器 | 通过 LAN/HTTPS 访问面板；只能提交高层 mission/cancel/arm/E-stop 请求 |

DGX 可以用 Wi-Fi 访问操作员网络，同时由独立有线 NIC 连接 Go2。必须显式绑定
route 和 ROS interface，不能让 DHCP 变化把 Go2 transport 静默切到错误网卡。

### 3. 克隆与依赖

```bash
git clone --branch real-go2 --recurse-submodules \
  https://github.com/strTATQwQ/E-MARS.git
cd E-MARS
git submodule update --init --recursive
cp .env.example .env.local
chmod 600 .env.local
```

子模块包括：`frontend`（固定版本的 `vla-nav-panel`）和 `hardware`（固定版本的
Go2 边缘 AI 硬件描述）。凭证、主机名、NIC、模型路径、ROS domain、相机设备绑定
和 run root 只能放 `.env.local` 或主机本地 systemd environment file，绝不能提交。

前端只能从固定版本的子模块安装，从而让仿真与真机使用同一份经过审查的源码：

```bash
python3 -m venv .venv-panel
source .venv-panel/bin/activate
python -m pip install -e ./frontend
```

### 4. 主机前置条件

安装并验证：

- 兼容 NVIDIA driver/CUDA 和足够显存；
- Ubuntu/Linux、ROS 2 Jazzy、Nav2、TF2、image transport 与所需 message；
- 与 Go2 firmware 兼容的 Unitree/Go2 ROS transport；
- RealSense kernel/udev、`librealsense`/ROS driver；
- 四路 USB 与 Go2 front stream 所需 V4L2/FFmpeg；
- InternVLA、Step3-VL 和前端的隔离 Python 环境；
- `flock`、`systemd`、`iproute2`、`ethtool`、`v4l2-ctl`、`jq` 和 Git。

阅读 `dependencies.lock.yaml`，其中记录目标主机需要审查的上游 revision 与本地
patch 事实。

### 5. 网络配置

在实际主机上使用占位符并检查接口：

```bash
ip -brief link
ip -brief address
ip route
nmcli device status
```

要求：

- Go2 有线 NIC 持续 UP，并拥有机器人子网 route；
- Wi-Fi/LAN 承担操作员与软件访问；
- ROS 2 DDS 使用预期 interface 和 `ROS_DOMAIN_ID`；
- 模型 8200、前端 8300 端口只属于预期进程；
- 不存在 simulator `/clock` 或仿真 DDS peer；
- firewall 只开放必要 LAN endpoint。

不能用复用 IP 或 SSH host key 猜设备身份。应使用稳定 interface/device path、必要
时的 serial 和真实 topic 数据绑定硬件。

### 6. 禁止运动的硬件发现

发现阶段保持控制关闭。

```bash
source /opt/ros/jazzy/setup.bash
ros2 node list
ros2 topic list -t
ros2 topic info --verbose /lf/sportmodestate
ros2 topic info --verbose /utlidar/cloud
rs-enumerate-devices
v4l2-ctl --list-devices
find /dev/v4l/by-path -maxdepth 1 -type l -print
```

四路语义相机必须按稳定 `/dev/v4l/by-path` 绑定，而不是枚举编号。通过物理遮挡确认
实际视角后再分配左前、正前、右前、正后。四路相机保留配置的水平修正；Go2 和
D435 不做镜像。

### 7. 预期传感器/话题合同

严格配置当前命名如下。使用前必须核对实际 type、QoS、frame、rate 与来源：

| 来源 | Topic/config key | 必须验证 |
| --- | --- | --- |
| Go2 前视 | `/frontvideostream` 或已验证 VideoHub | decoder、方向、stamp/age |
| Go2 low state | `/lf/lowstate` | 真硬件来源、频率、无仿真 publisher |
| Go2 sport state | `/lf/sportmodestate` | mode/pose/velocity 和新鲜度 |
| LiDAR state | `/utlidar/lidar_state` | 设备健康与时间戳 |
| LiDAR IMU | `/utlidar/imu` | frame、单位、covariance、rate |
| LiDAR cloud | `/utlidar/cloud` | frame、密度、range、rate |
| Odometry | `/utlidar/robot_odom` | source、drift、frame chain、covariance |
| D435 RGB | `/check/d435/color/image_raw` | encoding、CameraInfo 匹配、age |
| D435 RGB info | `/check/d435/color/camera_info` | 实测 intrinsics/frame |
| D435 depth | `/check/d435/depth/image_rect_raw` | encoding、scale、alignment、age |
| D435 depth info | `/check/d435/depth/camera_info` | intrinsics/frame |
| Static TF | `/tf_static` | 实测、完整、无重复 authority |
| 四路语义相机 | 稳定 V4L2 binding | 视角身份、方向、frame、extrinsics |

D435 硬件不支持 IMU 时必须标记 unsupported，不能伪造 IMU。所有必需外参都应在
导航使用前完成标定。

### 8. ROS 与 Python 环境

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p "<ROS_WS>/src"
# 将所需 ROS package 链接到 <ROS_WS>/src。
cd "<ROS_WS>"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Step3 模型环境应保持隔离，以满足固定 Transformers/BF16 loader 合同。权重放在
Git 外部，并在私有 run metadata 记录 revision/hash。

### 9. 严格配置

主要文件：

- `configs/strict_real_go2.yaml`：前端、传感器、话题、arm/E-stop、控制上限；
- `configs/internnav_t5/strict_real_go2_mission_ingress.json`：语言、身份、原始文本边界；
- `deploy/systemd/vla-nav-panel-strict-real.service`：前端/ROS adapter；
- `deploy/systemd/internvla-real-mission-gateway.service`：Step3-first gateway；
- `scripts/run_t5_strict_real_mission_gateway.sh`：gateway 启动器。

本地渲染占位符并审查 effective config，不要提交渲染后的主机值。保持
`use_sim_time=false`，并把 arm、E-stop、dispatch、motion bridge 与控制边界作为
一套严格策略整体审查。

### 10. 模型与任务服务

先启动 Step3-VL 并验证 model/config identity：

```bash
python -m slow_planner.serve \
  --config configs/slow_models/step3_vl_10b_bf16.yaml
```

随后用主机本地环境启动 strict mission gateway。它必须只订阅原始
`/user_instruction`，调用 Step3 规范化，只有 schema/identity 通过后才发布
`/internvla/mission/canonical`；timeout、abstain、stale/cross-reset、digest mismatch
或非法输出一律丢弃；绝不能发布 `cmd_vel`、物理运动或最终 STOP。

必须检查不存在第二条 IPC/topic 路径把原始文本直接送给 InternVLA。

### 11. 前端部署

只安装并运行固定版本的 `frontend` 子模块，不再把其 Python package 复制到本分支。
根据 E-MARS 的 `configs/strict_real_go2.yaml` 配置 LAN 监听地址/端口。前端升级时
先审查新的子模块 commit，再同步推进两个 E-MARS 分支的 `frontend` gitlink。API
包含 health、state、camera、WebSocket，以及高层 mission、cancel、arm、E-stop
请求。

控制 endpoint 仍受后端 gate 约束。UI 控件不能绕过 mission identity、sensor/TF
freshness、arm、E-stop、watchdog 或 command bound。相机 preview 限频；模型直接
使用 ROS sensor，不使用 HTTP JPEG。

### 12. 启动顺序

1. 确认环境清空、急停可用、机器人安全姿态，并证明无非零 publisher；
2. 验证 Go2 有线与上游网络 route；
3. 以 receive-only 启动 Go2 state/LiDAR/front-camera bridge；
4. 启动 D435 与四路语义相机，验证硬件绑定；
5. 只从实测配置启动 TF/static calibration source；
6. 启动定位/地图并验证所选 operating mode；
7. 启动 Step3-VL、InternVLA、ROS client、Nav2、recovery、watchdog；
8. 启动 strict mission gateway 和前端 ROS adapter；
9. 至少做 60 秒静止 health/freshness/identity 检查；
10. 停止本任务进程并证明 port/socket/lock/PID/PGID/command publisher 零残留。

### 13. 静止验收

只有以下条件满足才可通过：

- 所有真实 topic 的 publisher/type/QoS/frame/source 已核对；
- 60 秒测得频率、age 和 gap；
- TF 对预期消费者完整，并绑定实测标定；
- 七路相机显示正确物理视角和方向；
- 任务规范化能拒绝 stale/cross-reset/非法响应；
- Nav2/recovery/watchdog/control 状态可见且内部一致；
- E-stop latch 和 inactive arm 可见且有效；
- 没有发布非零 motion、goal 或最终 STOP；
- 清理后本任务零残留。

### 14. 标定与运动 commissioning

标定必须测量相机内外参、robot/base frame、LiDAR/IMU frame、时间对齐和稳定设备
绑定。原始测量保存在公共仓库外，只在合适时提交经过审查且不含秘密的配置。

启用 arm 或 motion bridge 是单独的受审变更。任何非零测试前都需要当前任务中的
明确用户授权、清空空间、物理急停、保守限制、当前定位、健康 Nav2/watchdog、
命令 feedback 和一次一个 primitive 的计划。不能从旧会话或本文推断授权。

### 15. 停止、回滚与清理

只停止当前 run 所拥有的服务。TERM 后有限等待，必要时只对已知 run root 精确
KILL；不能按进程名广泛杀死其他模型或 ROS 任务。释放前验证 task-owned PID/PGID、
socket、port、lock 和 command publisher 为零。

回滚方式是停止 mission/control service，恢复上一 strict config SHA，把 E-stop
置于安全状态，并回到 receive-only sensor monitoring。

### 16. 常见故障

| 现象 | 优先检查 |
| --- | --- |
| Wi-Fi 变化后 Go2 话题消失 | 有线 NIC route、DDS interface、机器人子网 |
| 绿屏/图像损坏 | codec、stride、pixel format、decoder、source age |
| 相机标签错误 | 物理遮挡测试与稳定 device path |
| D435 话题缺失 | USB 带宽、serial binding、driver namespace |
| TF 不可用 | 实测 static transform、frame ID、重复 authority |
| Step3 拒绝 mission | service health、config SHA、request identity、JSON schema |
| InternVLA 看到原始语言 | canonical-only latch 与其他 IPC/topic 旁路 |
| 前端数据 stale | ROS adapter health、topic age、WebSocket state |
| motion gate 无法 arm | 后端前置条件、health、freshness、E-stop、arm policy |
