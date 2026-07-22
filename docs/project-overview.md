# E-MARS Simulation Project Overview

## English

### 1. Purpose

E-MARS (Edge-deployed Multimodal Agent for Robotic Search-and-rescue) is a
research and engineering stack for language-guided mobile-robot navigation. It
is designed around a practical separation of concerns:

- a visual-language policy proposes local navigation behavior;
- a semantic model can normalize multilingual tasks and advise on ambiguous or
  long-horizon choices;
- ROS 2 carries typed observations, identity, health, and commands;
- Nav2 owns path planning, costmaps, controllers, and recovery integration;
- a watchdog and command adapter bound motion and reject stale state; and
- Isaac Sim owns the simulated world, sensors, `/clock`, episode reset, and
  evaluation lifecycle.

The project focuses on navigation, semantic understanding, planning, recovery,
and distributed edge deployment. It does not claim high-fidelity Go2 dynamics,
physical-robot safety certification, or real-world motion performance from
simulation results.

### 2. Simulation branch scope

The `sim` branch is the functional-development and evaluation branch. It may
use deviations that are explicitly scoped to `completion_sim`, including a
simplified planar motion implementation, tolerant freshness windows, shadow
mapping, or non-fatal evidence recorders. Every deviation must remain visible
in configuration and must not flow into `real-go2`.

The baseline functional path is:

```text
Isaac RGB-D/LiDAR/IMU/odometry + /clock
  -> ROS 2 bridge and observation identity
  -> optional multilingual mission normalization
  -> InternVLA fast navigation
  -> optional bounded semantic advisor
  -> typed command resolver
  -> Nav2 + recovery + costmaps
  -> bounded simulation command adapter
  -> Isaac Go2
```

### 3. Language and model roles

InternVLA is the fast navigation policy. It consumes a current observation,
the active instruction, pose/history context, and returns a local action or
trajectory candidate. It does not directly own an unrestricted velocity topic.

The Step layer has two distinct optional roles:

1. **Mission normalization** — once at mission ingress, local Step3-VL-10B or
   hosted `step-3.7-flash` converts multilingual operator text into a bounded,
   schema-validated English mission. The feature is disabled by default, and
   exactly one provider may be enabled.
2. **Semantic slow advice** — a selected slow planner compares only current
   legal frontiers, viewpoints, or relative targets. It may select or abstain;
   it does not publish `cmd_vel` or terminal STOP.

Normalization and semantic advice are independent switches. A deployment may
normalize the instruction without enabling online semantic reranking, or use a
local Step3 service for both while retaining separate request identities.

### 4. State and authority model

Every online request is bound to an episode/reset/sequence identity. Responses
that arrive after reset, reference a stale observation, or do not echo the
expected identity are discarded. Simulation time governs sensor freshness,
action completion, and navigation deadlines; wall monotonic time is reserved
for process and transport liveness.

Motion authority is intentionally narrow:

- models propose typed actions, frontiers, or relative targets;
- the resolver validates bounds and current identity;
- Nav2 produces a navigation command;
- the command adapter applies speed, acceleration, age, and E-stop limits; and
- stale model, localization, sensor, reset, or network state produces safe
  stop rather than replaying an old command.

### 5. Mapping, localization, and recovery

The functional simulation baseline can use GT-derived odometry, a static global
map, and a LiDAR local costmap. Sensor odometry, cuVSLAM, and active Nvblox are
strict extensions and must be evaluated separately before combination.

Recovery is bounded and identity-aware. No-progress or short-loop detection may
cancel the active goal, clear trajectory state, rotate/scan, and request a new
plan. It must not repeat an old trajectory forever or preserve state across an
episode reset.

### 6. Dual-Lane development

The repository supports two isolated development lanes. Each lane has its own
DGX, Isaac GPU, CPU set, ROS domain, namespace, ports, caches, run root, and
resource leases. Lanes may perform different research tasks in parallel, while
shared asset conversion, shader warm-up, and large archival operations remain
serialized. A candidate becomes portable only after it can start from the same
code/config bundle without cross-lane topic, reset, port, or cache pollution.

### 7. Optional simulation operator panel

The pinned `frontend` submodule is the single operator-panel source shared by
the `sim` and `real-go2` branches. In simulation it is optional: navigation and
evaluation continue without the browser UI, while enabling it adds a live view
of episode/reset/sequence identity, simulated cameras, structured decisions,
ROS/Nav2/recovery/watchdog health, command feedback, latency, and resource
telemetry. The panel consumes rate-limited projections from its ROS adapter;
navigation models continue to consume the original ROS sensor streams.

The simulation panel is observational at the deployment boundary. Starting or
stopping it must not change the frozen episode manifest, `/clock` authority,
model inputs, motion authority, or evaluator lifecycle. Pinning it as a Git
submodule lets both E-MARS branches advance to the same reviewed panel commit
without copying frontend source into each branch.

### 8. Public repository boundary

This repository publishes source code, configuration templates, contracts,
documentation, and offline tests. It intentionally excludes:

- credentials and `.env.local`;
- model checkpoints and provider API responses;
- licensed/private scenes and datasets;
- recorded camera, depth, LiDAR, IMU, or ROS bag data;
- run roots, logs, reports, evidence, benchmark results, and videos; and
- machine-specific IP addresses, usernames, and filesystem paths.

The dependency lock records known upstream identities, but a local checkout may
still require private patches or external assets. Read `dependencies.lock.yaml`
before interpreting a source checkout as a complete runtime installation.

### 9. Completion criteria

A simulation configuration is functionally useful when it can run the complete
language-to-motion loop, use real simulated sensor streams, recover without
stale motion, reset cleanly, and preserve raw machine-local logs outside the
public repository. Functional completion is not equivalent to strict timing,
active Nvblox, sensor odometry, long soak, or real-hardware qualification.

---

## 中文

### 1. 项目目标

E-MARS（Edge-deployed Multimodal Agent for Robotic Search-and-rescue）是面向
自然语言引导移动机器人导航的研究与工程栈。系统将职责拆分为：

- 视觉语言策略提出局部导航行为；
- 语义模型规范化多语言任务，并辅助处理歧义或长时程决策；
- ROS 2 传递类型化观测、身份、健康状态和命令；
- Nav2 负责路径规划、costmap、控制器和 recovery 集成；
- watchdog 与命令适配器限制运动并拒绝过期状态；
- Isaac Sim 负责仿真世界、传感器、`/clock`、episode reset 和评测生命周期。

项目重点是导航、语义理解、规划、恢复和端侧分布式部署。仿真结果不等同于
高保真 Go2 动力学验证、真机安全认证或真实运动性能证明。

### 2. sim 分支范围

`sim` 是功能研发和评测分支。它允许使用明确限定在 `completion_sim` 的偏差，
例如简化平面运动、较宽松的新鲜度窗口、shadow mapping 或非致命 evidence
recorder。所有放宽都必须在配置中可见，并且不得流入 `real-go2`。

基础功能链为：

```text
Isaac RGB-D/LiDAR/IMU/odometry + /clock
  -> ROS 2 bridge 与观测身份
  -> 可选多语言任务规范化
  -> InternVLA 快速导航
  -> 可选受限语义慢规划
  -> 类型化命令解析
  -> Nav2 + recovery + costmaps
  -> 有界仿真命令适配器
  -> Isaac Go2
```

### 3. 语言与模型职责

InternVLA 是快速导航策略。它读取当前观测、活动指令以及位姿/历史上下文，输出
局部动作或轨迹候选；它不直接拥有无限制的速度话题。

Step 层有两个彼此独立的可选职责：

1. **任务规范化**：在任务入口仅执行一次。本地 Step3-VL-10B 或托管
   `step-3.7-flash` 将多语言文本转换为受限、经过 schema 校验的英文任务。
   此功能默认关闭，启用时只能选择一个 provider。
2. **语义慢规划**：只比较当前合法的 frontier、viewpoint 或相对目标；可以选择
   或 abstain，但不能发布 `cmd_vel` 或最终 STOP。

任务规范化和语义慢规划是两个独立开关。部署可以只做语言规范化而不启用在线
语义 rerank，也可以让本地 Step3 服务承担两种职责，但两类请求必须保留独立身份。

### 4. 状态与权限模型

每个在线请求都绑定 episode/reset/sequence 身份。跨 reset、观测过期或未正确
回显身份的响应必须丢弃。传感器新鲜度、动作完成和导航 deadline 使用 sim time；
wall monotonic time 仅用于进程和传输失联检测。

运动权限被严格收窄：

- 模型只提出类型化动作、frontier 或相对目标；
- resolver 校验边界和当前身份；
- Nav2 生成导航命令；
- 命令适配器施加速度、加速度、命令年龄和急停限制；
- 模型、定位、传感器、reset 或网络状态过期时执行 safe-stop，不能重放旧命令。

### 5. 地图、定位与恢复

仿真功能基线可以使用 GT 派生 odometry、static global map 和 LiDAR local
costmap。sensor odometry、cuVSLAM 和 active Nvblox 属于严格扩展，必须先独立
验收，再尝试组合。

Recovery 必须有界并绑定身份。无进展或短回环检测可以取消活动 goal、清除轨迹
缓存、原地扫描并申请新规划，但不能无限重复旧轨迹，也不能把状态带入下一 episode。

### 6. 双 Lane 并行研发

仓库支持两个隔离研发 Lane。每个 Lane 拥有独立 DGX、Isaac GPU、CPU 集、
ROS domain、namespace、端口、cache、run root 和资源租约。两条 Lane 可以并行
推进不同研发功能；共享资产转换、shader 预热和大型归档仍需串行。候选方案只有
在相同代码/配置 bundle 下启动，并且无跨 Lane topic、reset、端口或缓存污染时，
才可认为具备可移植性。

### 7. 可选仿真操作员前端

固定版本的 `frontend` 子模块是 `sim` 与 `real-go2` 两个分支共用的唯一正式前端
来源。在仿真中前端是可选组件：不启动浏览器 UI 时导航与评测仍可继续；启动后可
实时查看 episode/reset/sequence 身份、模拟相机、结构化决策、ROS/Nav2/recovery/
watchdog 健康、命令反馈、延迟和资源遥测。前端只读取 ROS adapter 生成的限频投影，
导航模型仍直接消费原始 ROS 传感器流。

仿真前端在部署边界上只负责观测。启动或停止它不得改变冻结 episode manifest、
`/clock` authority、模型输入、运动权限或 evaluator 生命周期。使用 Git submodule
固定前端后，两个 E-MARS 分支可以共同升级到同一个经过审查的前端 commit，无需在
各分支复制前端源码。

### 8. 公共仓库边界

本仓库发布源码、配置模板、合同、文档和离线测试，明确不发布：

- 凭证与 `.env.local`；
- 模型 checkpoint 和供应商 API 响应；
- 受许可限制或私有的场景与数据集；
- 相机、depth、LiDAR、IMU 或 ROS bag 录制；
- run root、日志、报告、证据、评测结果和视频；
- 机器专用 IP、用户名和文件系统路径。

依赖锁记录已知上游身份，但本地运行仍可能需要外部资产或私有补丁。不要仅凭源码
checkout 就认定运行环境完整，应先阅读 `dependencies.lock.yaml`。

### 9. 功能完成边界

当一个仿真配置能够运行完整的语言到运动闭环、使用真实模拟传感器、恢复时不产生
stale motion、episode reset 后无污染，并把原始机器日志保存在公共仓库之外时，
即可认为具备功能价值。功能完成不等于 strict timing、active Nvblox、sensor
odometry、长时间 soak 或真实硬件资格已经通过。
