# E-MARS Real-Go2 Project Overview

## English

### 1. Purpose

The `real-go2` branch turns the E-MARS navigation research stack into a strict
physical-robot integration target. Its intended end state is one DGX-class edge
computer running InternVLA, Step3-VL, ROS 2, localization/mapping, Nav2,
recovery, watchdogs, the operator backend, and a single bounded control mux,
while the Unitree Go2 provides physical state and motion execution.

Physical motion stages require explicit human approval, a clear area, an
available E-stop, calibrated frames, fresh sensors/localization, and a healthy
command chain.

### 2. End-to-end mission path

```text
Operator enters a multilingual high-level task
  -> frontend creates mission_id + reset_generation + sequence_id + deadline
  -> strict mission gateway checks E-stop, arm, health, identity, and freshness
  -> Step3-VL normalizes raw text into bounded English canonical mission
  -> InternVLA consumes only the canonical mission and current sensor state
  -> optional Step3 timeout advisor may select among legal high-level candidates
  -> typed resolver validates identity, target/action bounds, and freshness
  -> Nav2 plans and controls; recovery handles bounded no-progress conditions
  -> watchdog/control mux enforces command age, speed, acceleration, and E-stop
  -> Go2 motion bridge publishes only when explicitly enabled and healthy
```

The frontend never publishes raw `cmd_vel` or terminal STOP. Step3 and InternVLA
also have no direct velocity authority. There is one final motion authority: the
strict control mux plus Go2 bridge.

### 3. Natural-language contract

Raw operator text is not an InternVLA input in this branch. Step3-VL must first
return a JSON object containing source language, canonical English instruction,
target description, constraints, confidence, and abstention state. The gateway
binds that object to the source-instruction digest, mission identity, and config
SHA. InternVLA accepts only the canonical topic.

Failure behavior is fail-closed:

- timeout, transport failure, invalid JSON, unexpected prose, or duplicate keys;
- abstention or unsafe/ambiguous request;
- response from a previous reset or sequence; or
- mismatched instruction/config digest

all result in safe hold. Step3 may later advise a navigation timeout, but only
within a current legal candidate set; it cannot publish velocity or authorize
arrival/STOP by itself.

### 4. Sensor and state inputs

The physical integration target exposes seven visual panels/inputs:

1. Go2 front camera stream;
2. D435 RGB;
3. D435 depth visualization/data;
4. semantic camera front-left;
5. semantic camera front;
6. semantic camera front-right; and
7. semantic camera rear.

Navigation also consumes Go2 low/sport state, Unitree LiDAR state/cloud/IMU,
odometry, D435 CameraInfo, and TF. Browser JPEG previews are for operators only;
navigation consumers use original ROS messages. The four semantic cameras may
apply the checked-in horizontal image correction, while Go2 and D435 streams
must preserve their hardware orientation.

Extrinsics must come from measured calibration. Topic names or camera labels
are not calibration evidence.

### 5. Localization and mapping

The strict branch prohibits GT pose and simplified simulation motion. A usable
physical navigation configuration requires a measured pose source and complete
TF chain. Candidate sources include LiDAR/IMU odometry and cuVSLAM, evaluated
for drift, TF age, tracking loss, restart, and reset behavior.

Mapping may use a surveyed/static global map, LiDAR local costmap, and Nvblox
only after each source is validated. Active Nvblox is not allowed to silently
replace a missing localization or calibration contract. A stale mapping layer
must be disabled or cause safe hold according to the strict configuration.

### 6. Motion and safety authority

The control path enforces:

- explicit human arm with a deliberate phrase/action;
- latched E-stop and controlled reset;
- one command source through the strict mux;
- configured linear/angular speed and acceleration limits;
- command-age watchdog and zero-command fallback;
- safe stop on stale sensors, localization, model, Nav2, network, or identity;
- rejection of cross-reset and duplicate missions; and
- no motion during stationary integration or calibration.

The exact numeric limits live in `configs/strict_real_go2.yaml`. Changing them
is a control-policy change and requires review and renewed qualification.

### 7. Operator panel

The `frontend` submodule is the canonical VLA navigation operator panel shared
with the simulation branch; E-MARS does not maintain a branch-local frontend
dependency. It shows camera previews, mission identity, canonical instruction,
structured decision, ROS/Go2 health, model/Nav2/recovery state, latency, and runtime
diagnostics. The high-level API supports mission submission, cancel, arm, and
E-stop requests, but backend gates remain authoritative. A visible button does
not bypass those gates.

Hidden chain-of-thought and raw model reasoning are not exposed. Only validated
structured decisions and bounded summaries are shown.

### 8. Hardware description

The `hardware` submodule points to the Unitree Go2 edge-AI hardware repository,
which owns CAD, mounting, power, camera placement, and physical-integration
documentation. Software must reference a measured hardware revision and
extrinsic/config SHA rather than copying provisional simulation coordinates.

### 9. Validation stages

Recommended promotion stages are:

1. **STATIONARY_INTEGRATION** — topics, types, QoS, frames, frequencies, health,
   UI, identity, and zero non-zero motion publication.
2. **CALIBRATION** — measured intrinsics/extrinsics, TF, time alignment, and
   repeatable sensor binding.
3. **INACTIVE_DRY_RUN** — mission through Nav2/control mux while physical output
   remains disabled; stale/cancel/E-stop cases verified.
4. **MANUAL_LOW_SPEED** — explicit user authorization, clear area, available
   E-stop, conservative limits, one primitive at a time.
5. **BOUNDED_AUTONOMY** — only after localization, mapping, recovery, watchdog,
   and command feedback meet the accepted contract.

No lower stage may be described as a higher one.

### 10. Public repository boundary

The public branch contains source, templates, documentation, and offline tests.
It excludes credentials, machine addresses, calibration measurements, recorded
sensor data, ROS bags, run logs, test results, reports, screenshots, videos, and
evidence bundles. Operational records belong in ignored host-local run roots.

---

## 中文

### 1. 项目目标

`real-go2` 将 E-MARS 导航研究栈推进为严格的物理机器人集成目标。最终希望在一台
DGX 级机载计算机上共置 InternVLA、Step3-VL、ROS 2、定位/地图、Nav2、
recovery、watchdog、前端后端和唯一有界 control mux，由 Unitree Go2 提供真实
状态和运动执行。

每个物理运动阶段都需要用户明确授权、清空环境、急停可用、frame 已标定、
传感器/定位新鲜且命令链健康。

### 2. 端到端任务链

```text
操作员输入多语言高层任务
  -> 前端生成 mission_id + reset_generation + sequence_id + deadline
  -> strict mission gateway 检查 E-stop、arm、health、identity 与 freshness
  -> Step3-VL 将原始文本规范化为受限英文 canonical mission
  -> InternVLA 只读取 canonical mission 与当前传感器状态
  -> 可选 Step3 timeout advisor 只能在合法高层候选中选择
  -> typed resolver 校验身份、目标/动作边界与新鲜度
  -> Nav2 规划和控制；recovery 处理有界无进展
  -> watchdog/control mux 限制 command age、速度、加速度和 E-stop
  -> 只有明确启用且健康时，Go2 motion bridge 才可发布
```

前端不能发布裸 `cmd_vel` 或最终 STOP；Step3 和 InternVLA 也没有直接速度权限。
唯一最终运动权限属于 strict control mux 与 Go2 bridge。

### 3. 自然语言合同

本分支禁止把操作员原始文本直接输入 InternVLA。Step3-VL 必须先返回 JSON，包含
源语言、英文 canonical instruction、目标描述、约束、confidence 和 abstain。
gateway 把它绑定到源指令 digest、mission identity 与 config SHA；InternVLA 只
接受 canonical topic。

以下情况全部 fail-closed 并进入 safe hold：

- timeout、传输失败、非法 JSON、额外 prose 或重复 key；
- abstain，或任务不安全/歧义过大；
- 响应来自旧 reset/sequence；
- instruction/config digest 不匹配。

Step3 后续可以在导航 timeout 时提供辅助，但只能在当前合法候选中选择，不能发布
速度，也不能独立决定 arrival/STOP。

### 4. 传感器与状态输入

物理接入目标包含七路视觉输入/面板：

1. Go2 前视相机；
2. D435 RGB；
3. D435 depth 数据/可视化；
4. 左前语义相机；
5. 正前语义相机；
6. 右前语义相机；
7. 正后语义相机。

导航还读取 Go2 low/sport state、Unitree LiDAR state/cloud/IMU、odometry、D435
CameraInfo 与 TF。浏览器 JPEG 仅供操作员查看，导航消费者使用原始 ROS 消息。
四路语义相机可以保留配置中的水平图像修正；Go2 与 D435 必须保持硬件方向。

外参必须来自实测标定；话题名或相机标签不构成标定证据。

### 5. 定位与地图

严格分支禁止 GT pose 和仿真简化运动。可用真机导航必须具备实测位姿源和完整 TF。
候选包括 LiDAR/IMU odometry 与 cuVSLAM，并需验收 drift、TF age、tracking loss、
重启与 reset 行为。

地图可以使用测量/static global map、LiDAR local costmap，并在独立验收后加入
Nvblox。Active Nvblox 不能掩盖定位或标定合同缺失；地图层 stale 时必须按严格
配置禁用或进入 safe hold。

### 6. 运动与安全权限

控制链必须保证：

- 显式 human arm，并使用有意设计的 phrase/action；
- 锁存式 E-stop 与受控 reset；
- 唯一命令源经过 strict mux；
- 配置化线/角速度和加速度上限；
- command-age watchdog 与零命令回退；
- 传感器、定位、模型、Nav2、网络或身份 stale 时 safe-stop；
- 拒绝跨 reset 与重复 mission；
- 静止接入和标定期间绝不运动。

准确数值位于 `configs/strict_real_go2.yaml`。修改它们属于控制策略变更，需要重新
审查和资格验证。

### 7. 操作员前端

`frontend` 子模块是与仿真分支共用的正式 VLA 导航操作员面板；E-MARS 不再维护
分支私有的前端依赖。它显示相机、mission identity、
canonical instruction、结构化决策、ROS/Go2 health、模型/Nav2/recovery、时延和
运行诊断。高层 API 支持 mission、cancel、arm 与 E-stop 请求，但后端 gate 始终
具有最终权限。按钮可见不代表可以绕过这些 gate。

前端不显示隐藏思维链或模型原始推理，只显示经过校验的结构化决策与受限摘要。

### 8. 硬件描述

`hardware` 子模块指向 Unitree Go2 边缘 AI 硬件仓库，由其维护 CAD、安装、电源、
相机位置和物理集成文档。软件必须引用已测量硬件 revision 和 extrinsic/config SHA，
不能把临时仿真坐标当作真机外参。

### 9. 验证阶段

推荐晋升顺序：

1. **STATIONARY_INTEGRATION**：验证 topic/type/QoS/frame/frequency/health/UI/
   identity，并证明没有非零运动发布；
2. **CALIBRATION**：完成实测内外参、TF、时间对齐和可重复设备绑定；
3. **INACTIVE_DRY_RUN**：任务经过 Nav2/control mux，但物理输出保持关闭；验证
   stale/cancel/E-stop；
4. **MANUAL_LOW_SPEED**：用户明确授权、环境清空、急停可用、保守限制、一次一个
   primitive；
5. **BOUNDED_AUTONOMY**：只有定位、地图、recovery、watchdog 和命令 feedback
   全部满足合同后才允许。

低阶段不得被描述成高阶段。

### 10. 公共仓库边界

公共分支只包含源码、模板、文档和离线测试，不包含凭证、机器地址、标定测量、
传感器录制、ROS bag、run log、测试结果、报告、截图、视频或 evidence bundle。
运行记录必须放在忽略的主机本地 run root。
