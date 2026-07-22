# E-MARS Simulation Technology Stack

## English

### 1. Stack overview

E-MARS combines robotics middleware, accelerated simulation and perception,
visual-language models, deterministic typed protocols, and explicit safety
boundaries. Checked-in configuration and `dependencies.lock.yaml` are the
authoritative machine contract; the versions below describe the public
integration baseline and may be isolated in separate environments.

| Layer | Technology | Role |
| --- | --- | --- |
| Simulation | NVIDIA Isaac Sim 6.0.0.1, Isaac Lab 6.1.14 | Go2 world, physics or simplified navigation motion, cameras, range sensors, `/clock`, reset |
| Scene/runtime | Omniverse Kit, USD, PhysX, RTX sensors | Scene graph, rendering, collisions, sensor generation |
| Accelerated ROS | Isaac ROS 4.5, Nvblox, Visual SLAM/cuVSLAM | Optional mapping, costmap, localization, GPU perception |
| Middleware | ROS 2 Jazzy, DDS, Nav2 | Typed nodes, discovery, TF, lifecycle, planning, control, recovery |
| Fast navigation | InternVLA / InternNav | RGB-D language-conditioned local action/trajectory policy |
| Semantic models | Step3-VL-10B BF16, Step-3.7-Flash, Cosmos Reason2 32B BF16 | Instruction normalization, semantic frontier/viewpoint comparison, research alternatives |
| Model runtime | PyTorch 2.12.1+cu130, Transformers 4.57.6 | Local BF16 loading and inference for pinned components |
| Protocols | ROS actions/services/messages, TCP/ZeroMQ, JSON, shared memory | Identity-bound model and navigation requests |
| Operator UI | FastAPI/Uvicorn, WebSocket, HTML/CSS/JS | Cameras, structured decisions, health, runtime diagnostics |
| Verification | Pytest, shell syntax checks, ROS introspection | Offline contracts and online smoke checks |

### 2. Hardware roles

| Platform | Responsibility |
| --- | --- |
| DGX-class edge computer | Complete co-located model + ROS 2 + localization/map + Nav2 + recovery/watchdog stack |
| x86 NVIDIA workstation | Isaac only: scene, Go2, simulated sensors, clock, episode/reset/evaluator |
| Unitree Go2 | Simulation target in this branch; physical integration belongs to `real-go2` |
| D435/semantic cameras | Simulated observations here; real devices require the strict branch and calibration |

A dual-Lane system repeats the complete DGX stack and pairs each copy with one
isolated Isaac worker. The x86 GPUs can be separate while CPU, RAM, SSD, and
network remain shared resources that require measurement and bounded use.

### 3. Model identities

| Model/dependency | Identity | Contract |
| --- | --- | --- |
| Step3-VL-10B | `5026053b0c2f5dfaa08fc2d149384162c3c8bca1` | `configs/slow_models/step3_vl_10b_bf16.yaml`; clean load, BF16, pinned checkpoint mapping |
| Step-3.7-Flash | Hosted provider model name `step-3.7-flash` | OpenAI-compatible text-only normalization adapter; API key from environment |
| Cosmos Reason2 32B | `4ed9828334c4397ace8b0c62134961adbe5aed0e` | Research slow-planner alternative |
| InternNav upstream | `7a5c62400ac45b313d9b709c740b64191556a242` | Recorded in `dependencies.lock.yaml`; local patch state must be reviewed |
| LongCLIP subdependency | `3966af9ae9331666309a22128468b734db4672a7` | Recorded upstream submodule identity |
| go2_ros2_sdk | `4e186b5f89bfec1f32c85676cbe22d4958e4f0fa` | External Go2 ROS integration dependency |
| unitree_ros2 | `668d1ec5a05d1c38d3306bdca7d59f2ba3581a88` | External Unitree ROS 2 dependency |

The hosted Step model is a provider-managed alias rather than a local immutable
checkpoint. Record the returned model identity and request timestamp in private
run metadata when reproducibility matters; do not commit API payloads or keys.

### 4. Model responsibilities and authority

**InternVLA** is the high-frequency policy. It consumes current observations
and a matching instruction/token representation, then proposes local behavior.
Its output enters a typed resolver and never bypasses Nav2/watchdog controls.

**Step3-VL-10B** supports two bounded protocols:

- text-only mission normalization; and
- multi-view semantic selection among current legal candidates.

**Step-3.7-Flash** is integrated in `sim` as a text-only normalization provider.
It receives no images, pose, frontier list, or velocity authority. The adapter
uses deterministic JSON mode and validates the same canonical mission schema.

**Cosmos Reason2** remains a research slow-planner route. Select one slow model
per runtime; do not keep multiple large advisors resident merely to hide a
selection decision.

### 5. ROS 2 and Nav2 architecture

The ROS layer uses:

- namespaced topics and a unique `ROS_DOMAIN_ID` per Lane;
- TF/TF2 for frame relationships;
- lifecycle-managed Nav2 planner, controller, behavior, and navigator servers;
- ROS actions/services for typed model steps, recovery, and command resolution;
- QoS selected per sensor/control stream; and
- `/clock` plus `use_sim_time=true` for all simulation-semantic deadlines.

The map stack can combine a static global map, LiDAR local costmap, Nvblox
shadow/active layers, and a selected odometry source. GT-derived odometry is a
simulation baseline, not a physical-robot localization result.

### 6. Data and wire contracts

Model and navigation requests carry episode ID, reset generation, sequence ID,
request ID, timestamps, deadline, and config identity. Sensor messages retain
source timestamps and frame IDs. A response is executable only while all
identity and freshness checks still match.

Transport choices include:

- ROS 2 DDS for sensors, TF, health, and navigation control;
- ZeroMQ/TCP for bounded slow-model requests;
- local shared memory or bounded compressed TCP for RGB-D model observations;
- JSON/YAML for configuration and structured decisions; and
- WebSocket/HTTP JPEG previews for the operator UI only—not as model input.

### 7. Time model

Isaac is the only `/clock` authority. Simulation time controls observation age,
action completion, replan deadlines, and episode semantics. Wall monotonic time
controls process liveness and external transport timeout. Mixing these clocks
can create false recovery failures when RTF is below one, so each contract
names its clock explicitly.

### 8. Mapping and localization options

| Component | Baseline/extension | Notes |
| --- | --- | --- |
| GT-derived odometry | Functional simulation baseline | Must be labeled as simulated pose |
| Static global map | Functional baseline | Stable global planning source |
| LiDAR local costmap | Functional baseline | Real simulated LiDAR observations |
| LiDAR/IMU odometry | Strict extension | Requires drift, TF age, loss, and reset evaluation |
| cuVSLAM | Strict extension | Uses its own stereo/IMU contract, not semantic camera aliases |
| Nvblox shadow | Strict extension | Produces map/slice data without navigation authority |
| Nvblox active-local | Strict extension | Must fail back to static+LiDAR when stale |

### 9. Safety and recovery components

The simulation stack keeps bounded linear/angular velocity and acceleration,
stale-command safe-stop, E-stop, episode reset isolation, and resource leases.
Completion-simulation deviations may relax collision/freshness evidence gates,
but must never erase command bounds or allow stale motion.

Recovery components monitor progress and short loops, cancel the current Nav2
goal, clear bounded trajectory state, optionally scan, and request a fresh plan.
All transitions are tied to the current episode/reset/sequence.

### 10. Frontend stack

The operator panel uses FastAPI/Uvicorn for HTTP and WebSocket endpoints plus a
browser UI for camera previews, structured decisions, mission identity, health,
GPU/latency telemetry, Nav2/recovery state, and command feedback. Camera preview
is rate-limited; raw ROS sensor data flows directly to navigation consumers.
Hidden chain-of-thought and unvalidated raw model text are not exposed.

### 11. Version and upgrade policy

- Pin model revisions and loader-critical library versions.
- Keep model, ROS, and Isaac environments isolated.
- Upgrade one layer at a time and rerun its focused contracts.
- Record code/config/model/scene identity in private run metadata.
- Do not silently replace a hosted model alias during a frozen evaluation.
- Never solve an OOM by changing precision or splitting the final edge stack
  without recording that the deployment contract changed.

---

## 中文

### 1. 技术栈总览

E-MARS 组合机器人中间件、GPU 加速仿真与感知、视觉语言模型、确定性类型化协议
和显式安全边界。仓库配置与 `dependencies.lock.yaml` 是机器可执行合同；下表描述
公共集成基线，不同组件可以使用隔离环境。

| 层级 | 技术 | 作用 |
| --- | --- | --- |
| 仿真 | NVIDIA Isaac Sim 6.0.0.1、Isaac Lab 6.1.14 | Go2 世界、物理或简化导航运动、相机、距离传感器、`/clock`、reset |
| 场景/运行时 | Omniverse Kit、USD、PhysX、RTX sensors | 场景图、渲染、碰撞与传感器生成 |
| 加速 ROS | Isaac ROS 4.5、Nvblox、Visual SLAM/cuVSLAM | 可选地图、costmap、定位与 GPU 感知 |
| 中间件 | ROS 2 Jazzy、DDS、Nav2 | 类型化节点、发现、TF、lifecycle、规划、控制、恢复 |
| 快速导航 | InternVLA / InternNav | RGB-D 与语言条件下的局部动作/轨迹策略 |
| 语义模型 | Step3-VL-10B BF16、Step-3.7-Flash、Cosmos Reason2 32B BF16 | 指令规范化、frontier/viewpoint 语义比较和研究替代路线 |
| 模型运行时 | PyTorch 2.12.1+cu130、Transformers 4.57.6 | 固定组件的本地 BF16 加载与推理 |
| 协议 | ROS action/service/message、TCP/ZeroMQ、JSON、shared memory | 绑定身份的模型和导航请求 |
| 操作员 UI | FastAPI/Uvicorn、WebSocket、HTML/CSS/JS | 相机、结构化决策、健康与运行诊断 |
| 验证 | Pytest、shell 语法检查、ROS introspection | 离线合同和在线 smoke |

### 2. 硬件角色

| 平台 | 职责 |
| --- | --- |
| DGX 级边缘计算机 | 共置完整模型 + ROS 2 + 定位/地图 + Nav2 + recovery/watchdog |
| x86 NVIDIA 工作站 | 只运行 Isaac：场景、Go2、模拟传感器、时钟、episode/reset/evaluator |
| Unitree Go2 | 本分支中的仿真目标；物理接入属于 `real-go2` |
| D435/语义相机 | 本分支使用模拟观测；真实设备必须进入严格分支并完成标定 |

双 Lane 会复制完整 DGX 栈，并让每套栈对应一个隔离 Isaac worker。即使 x86 GPU
分离，CPU、RAM、SSD 和网络仍是共享资源，必须测量并限制占用。

### 3. 模型与依赖身份

| 模型/依赖 | 身份 | 合同 |
| --- | --- | --- |
| Step3-VL-10B | `5026053b0c2f5dfaa08fc2d149384162c3c8bca1` | `configs/slow_models/step3_vl_10b_bf16.yaml`；clean load、BF16、固定 checkpoint mapping |
| Step-3.7-Flash | 托管模型名 `step-3.7-flash` | OpenAI-compatible 文本规范化 adapter；API key 来自环境变量 |
| Cosmos Reason2 32B | `4ed9828334c4397ace8b0c62134961adbe5aed0e` | 研究型慢规划替代路线 |
| InternNav upstream | `7a5c62400ac45b313d9b709c740b64191556a242` | 记录于 `dependencies.lock.yaml`；必须检查本地 patch 状态 |
| LongCLIP | `3966af9ae9331666309a22128468b734db4672a7` | 上游 submodule 身份 |
| go2_ros2_sdk | `4e186b5f89bfec1f32c85676cbe22d4958e4f0fa` | 外部 Go2 ROS 集成依赖 |
| unitree_ros2 | `668d1ec5a05d1c38d3306bdca7d59f2ba3581a88` | 外部 Unitree ROS 2 依赖 |

托管 Step 模型是供应商管理的 alias，不是本地不可变 checkpoint。需要复现时，应在
私有 run metadata 中记录返回的模型身份和请求时间，但不能提交 API payload 或 key。

### 4. 模型职责与权限

**InternVLA** 是高频策略，读取当前观测以及匹配的指令/token，提出局部行为。
输出必须进入 typed resolver，不能绕过 Nav2/watchdog。

**Step3-VL-10B** 支持两个受限协议：文本任务规范化，以及在当前合法候选中的
多视角语义选择。

**Step-3.7-Flash** 在 `sim` 中只作为文本规范化 provider，不接收图像、位姿、
frontier 或速度权限。adapter 使用确定性 JSON 模式并校验同一 canonical mission
schema。

**Cosmos Reason2** 保留为研究型慢规划路线。每个运行时只选择一个慢模型，不应
为了回避选择而让多个大模型长期同时驻留。

### 5. ROS 2 与 Nav2 架构

ROS 层使用：

- 每个 Lane 独立 namespace 和 `ROS_DOMAIN_ID`；
- TF/TF2 表达 frame 关系；
- Nav2 lifecycle planner、controller、behavior 与 navigator；
- ROS action/service 承载模型 step、recovery 与命令解析；
- 按传感器/控制流选择 QoS；
- `/clock` 和 `use_sim_time=true` 定义所有仿真语义 deadline。

地图栈可以组合 static global map、LiDAR local costmap、Nvblox shadow/active layer
与所选 odometry。GT 派生 odometry 只是仿真基线，不能当成真机定位结果。

### 6. 数据与线协议

模型和导航请求携带 episode ID、reset generation、sequence ID、request ID、
timestamp、deadline 与 config identity。传感器消息保留 source stamp 和 frame ID。
只有身份和新鲜度仍一致的响应才可执行。

传输方式包括：

- ROS 2 DDS：传感器、TF、health 与导航控制；
- ZeroMQ/TCP：有界慢模型请求；
- 本地 shared memory 或受限压缩 TCP：RGB-D 模型观测；
- JSON/YAML：配置和结构化决策；
- WebSocket/HTTP JPEG：只用于操作员预览，不能回灌为模型输入。

### 7. 时间模型

Isaac 是唯一 `/clock` authority。sim time 控制观测年龄、动作完成、replan deadline
和 episode 语义；wall monotonic time 控制进程失联和外部传输 timeout。RTF 小于 1
时混用两种时钟会制造假 recovery failure，因此每个合同都必须明确使用哪种时钟。

### 8. 地图与定位选项

| 组件 | 基线/扩展 | 说明 |
| --- | --- | --- |
| GT 派生 odometry | 仿真功能基线 | 必须标记为 simulated pose |
| Static global map | 功能基线 | 稳定全局规划源 |
| LiDAR local costmap | 功能基线 | 使用真实模拟 LiDAR |
| LiDAR/IMU odometry | 严格扩展 | 需要 drift、TF age、tracking loss、reset 验收 |
| cuVSLAM | 严格扩展 | 使用独立 stereo/IMU 合同，不能把语义相机冒充双目 |
| Nvblox shadow | 严格扩展 | 生成地图/slice，但没有导航权限 |
| Nvblox active-local | 严格扩展 | stale 时必须回退 static+LiDAR |

### 9. 安全与恢复组件

仿真栈仍保留线/角速度与加速度限制、stale-command safe-stop、E-stop、episode
reset 隔离和资源租约。completion-sim 可以放宽 collision/freshness evidence gate，
但不能删除命令边界或允许 stale motion。

Recovery 监控无进展和短回环，取消当前 Nav2 goal、清除有界轨迹状态、按需扫描并
请求新规划。所有状态转换都绑定当前 episode/reset/sequence。

### 10. 前端技术栈

操作员前端使用 FastAPI/Uvicorn 提供 HTTP 与 WebSocket，并用浏览器 UI 展示相机
预览、结构化决策、mission identity、health、GPU/延迟、Nav2/recovery 和命令
feedback。相机预览必须限频；原始 ROS 传感器直接进入导航消费者。前端不显示隐藏
思维链或未经校验的模型原文。

### 11. 版本与升级策略

- 固定模型 revision 和 loader 关键库版本；
- 模型、ROS 与 Isaac 环境隔离；
- 每次只升级一层，并重跑对应聚焦合同；
- 在私有 run metadata 中记录 code/config/model/scene identity；
- 冻结评测中不得静默替换托管模型 alias；
- 不能通过未记录的精度变化或拆分最终边缘栈来掩盖 OOM。
