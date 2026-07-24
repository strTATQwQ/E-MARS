# InternNav 并行资源策略

状态：物理资源锁冻结；运行按 `strict_evidence` / `completion_sim fast path` 分轨（Codex 00 所有）
适用任务：01R、02–05，以及 S1 后的 10/20/30/40/50

## 运行策略分轨

- `strict_evidence` 保留既有 0.35 s、exact-time、Collision Monitor、active Nvblox、600 s soak 等证据合同，作为后置严格扩展。
- `completion_sim` 只允许 Isaac 仿真，目标是先完成可运行 T4 闭环；使用非致命 consumer-group shadow recorder、nearest/latest TF（容忍 2.5 s）、5 s source/watchdog、Collision Monitor warn-only、static global map + LiDAR local costmap、Nvblox shadow。
- `completion_sim` 仍强制唯一资源锁、有界速度、仿真急停、原始日志、配置差异和最终 PID/PGID/socket=0。
- 真实 Go2、硬件运动和任何默认硬件入口禁止选择 `completion_sim`。配置隔离见 `configs/runtime/{strict_evidence,completion_sim}.yaml`。

## 唯一资源与锁

| 资源 | 默认 SSH 目标 | 远端锁文件 | 可执行在线重任务的任务 |
|---|---|---|---|
| DGX_A | `railgun@10.100.100.128` | `/tmp/internnav_dgx.lock` | T5 Lane A 完整机载栈；fast path 绑定 clean SHA/run_id，正式 evidence 另用单次授权 |
| DGX_B | `rail@10.100.120.122` | `/tmp/internnav_dgx.lock` | T5 Lane B 完整机载栈；fast path 绑定 clean SHA/run_id，正式 evidence 另用单次授权 |
| Isaac legacy/global | `song@10.100.120.111` | `/tmp/internnav_isaac.lock` | 仅保留给旧单实例任务；不得与 T5 Lane grant 并存 |
| Isaac GPU0 / GPU1 | `song@10.100.120.123` | `/tmp/internnav_isaac_gpu0.lock` / `/tmp/internnav_isaac_gpu1.lock` | 当前 T5 fast path，分别配对 Lane A / B |

T4.6 吞吐续跑启用冻结的双 lane 拓扑；仅 `dgx_onboard_ablation20_parallel_resume`
授权可使用。旧入口继续使用上表的全局 Isaac 锁，不得与双 lane 同时获得授权。

| lane | 推理/导航主机锁 | Isaac GPU 锁 | ROS domain | TCP |
|---|---|---|---:|---:|
| A | `railgun@10.100.100.128:/tmp/internnav_dgx.lock` | `song@10.100.120.111:/tmp/internnav_isaac_gpu0.lock` | 71 | 24137 |
| B | `rail@10.100.120.116:/tmp/internnav_dgx.lock` | `song@10.100.120.111:/tmp/internnav_isaac_gpu1.lock` | 72 | 24138 |

上表的 71/72 只属于冻结 T4.6 证据。T5 使用新的对称 Lane 合同：

| T5 lane | 完整 DGX 栈 | Isaac worker | ROS domain | namespace | TCP controller |
|---|---|---|---:|---|---:|
| A | DGX_A：模型、ROS 2、定位、地图、Nav2、watchdog、速度控制 | `10.100.120.123` GPU0 | 75 | `/t5/lane_a` | 25137 |
| B | DGX_B：同一完整栈 | `10.100.120.123` GPU1 | 76 | `/t5/lane_b` | 25138 |

当前 completion_sim bring-up 为 A/B 同时 canary、随后 A/B 同时 fixed-five，不表示
任何一条 Lane 是 primary。两条 Lane 各持自己的锁并行研发；`all-lanes` 只保留给
最终正式 evidence 或确需一次修改两 Lane 共享状态的动作，其固定获取顺序仅用于避免死锁。
后置 `all-lanes` 的冻结顺序仍是 **DGX-A → DGX-B → Isaac-GPU0 → Isaac-GPU1**。
legacy/T4 的默认 `isaac` / `both` 仍指向旧主机；当前 T5 fast coordinator 显式把
`ISAAC_HOST=10.100.120.123` 传给 lease wrapper，使 GPU 锁与实际 worker 位于同一主机。
T5 Lane 之间不取得旧全局锁，仍可并行。

远端锁只证明“当前没有 holder”，不能证明上一次进程、容器或 socket 已清理。每条
Lane 因此另有与物理 GPU 对称的持久隔离标记：GPU0 使用
`/tmp/internnav_isaac_gpu0.quarantine`，GPU1 使用
`/tmp/internnav_isaac_gpu1.quarantine`；两者互不阻塞。共享资产或 legacy/global
污染才使用 `/tmp/internnav_isaac.quarantine`，任一 GPU holder 都必须同时检查该
global 标记与自己的 per-GPU 标记。DGX_A/B 位于不同主机，各自使用本机
`/tmp/internnav_dgx.quarantine`。在线 coordinator 在首个重任务前原子 arm，且只有
绑定同一 `run_tag`、role、deployment scope 的清理收据全部 PASS 时，才能在仍持有
flock 的情况下由 guard 删除；SSH 或审计不确定时标记保留。

双 lane 必须使用不同 deployment root、stage、结果目录和 `CUDA_VISIBLE_DEVICES`。
任一 lane 的 cleanup 只允许匹配该 lane 的 deployment root。fast prepare 以三个并发
任务分别使用 `dgx-a`、`dgx-b` 与仅限 x86 准备的 `isaac` profile；正常在线运行只
使用 `lane-a` / `lane-b`，不得申请 `all-lanes`。不得在仍有旧全局 Isaac grant 时启用。

锁文件必须位于上表所列共享主机。worktree 内的文件、PID 文件或约定文字均不构成租约。

## DGX 机载迁移边界

T5 的两台 DGX 各自负责一份完整且同构的模型、标准 ROS 传感器消费、定位、
static global map/Nvblox、LiDAR local costmap、Nav2、adapter/recovery、watchdog
与最终有界速度 relay；x86 只负责两个隔离的仿真、PhysX、传感器渲染、
evaluator 和本机 Unix IPC。Isaac phase 必须证明没有启动本机模型、Nav2、地图、
controller 或速度 relay。不得用跨 DGX MODEL/EDGE 拆分掩盖迁移问题。

Isaac runtime 到 DGX controller 使用 lane 对应的固定 IPv4 与 bounded port，并只接受
源地址 `10.100.120.123`。每条 JSON 消息继续受 512 KiB 上限与不超过 5 s 的
completion timeout 约束。该 TCP 通道不是第二份资源锁；legacy/T4 双机运行使用
`both`，T5 使用对应的 `lane-a` / `lane-b`，固定 DGX → Isaac。

“重任务”包括启动/停止 Isaac、在线 ROS 2 soak/smoke、GPU 模型服务、模型加载/推理、大文件模型下载以及会显著占用共享 CPU/GPU/内存/网络的任务。没有成功持有对应远端锁时，禁止执行这些操作。离线代码、单元测试、日志分析和 replay 不需要租约。

## 唯一入口

所有在线重任务必须由仓库中的 `scripts/with_resource_lease.sh` 包裹：

```bash
bash scripts/with_resource_lease.sh isaac \
  --owner codex-01 --task sensor-frequency-soak \
  --log-dir results/parallel/sensor_producer/online-soak \
  -- bash scripts/example_online_command.sh
```

脚本在启动命令前以非阻塞 `flock -n` 获取远端锁，并在锁中记录 resource、state、owner、task、远端 holder PID、开始时间、命令、日志目录和本地调用主机。SSH 断开、holder 消失、`flock` 不可用、认证失败、锁冲突或任一锁申请失败时都 fail closed；待执行命令不得开始，已开始的命令会被终止。终止时先给包裹命令一个可配置且有界的 graceful-cleanup 窗口，再按需升级到 `KILL`；所有仍存活的 holder 必须持续持锁到包裹进程组确认消失，并生成 `lease_cleanup_receipt.json`。D0 Lane 使用 600 s/60 s 的外层 TERM/KILL 预算，覆盖远端 runtime 的逐组清理上界及 SSH 审计余量。

legacy/T4 同时需要 DGX 与 Isaac 时使用 `both`；T5 正常研发使用 `lane-a` 或
`lane-b`，一次性 x86 双 worker 准备可使用 `isaac`，`all-lanes` 仅保留给最终正式
evidence。所有组合固定按 **DGX → Isaac** 申请，按反序释放。Isaac 申请失败时必须
先释放已经持有的 DGX 锁。禁止在脚本外嵌套两个单资源租约。

```bash
bash scripts/with_resource_lease.sh both \
  --owner codex-00 --task approved-two-host-smoke \
  --log-dir results/parallel/integration/two-host-smoke \
  -- bash scripts/example_two_host_command.sh
```

持久标记禁止手工 `rm`。只能使用 recovery-only 入口；它不读取授权板、模型凭证，
也不能替换为任意 workload。单资源恢复只取得对应 DGX 或 GPU 锁；global/all 恢复
按 DGX_A → DGX_B → GPU0 → GPU1 → legacy/global 的固定顺序取得全部相关锁：

```bash
bash scripts/recover_t5_resource_quarantine.sh isaac-gpu0 \
  "$PWD/results/internnav_t5/quarantine-recovery-$(date -u +%Y%m%dT%H%M%SZ)"
```

恢复 guard 只允许终止标记 scope 中仍可证明关联的 PGID、停止 scope 所有的固定
Isaac container，并验证对应端口、health socket 与 runtime lock 为零后删除标记；
错误 tag、复用 PGID、容器所有权不匹配或任何残留都返回 75 并保留标记。

## 凭证与下载

- 凭证只允许通过 SSH agent、交互认证或进程环境提供；不得写入 Git、handoff、结果、日志、命令行参数或锁元数据。
- 脚本支持 `ISAAC_PASSWORD` / `DGX_PASSWORD`，也支持更优先的 `ISAAC_PASSWORD_FILE` / `DGX_PASSWORD_FILE`。密码文件必须位于仓库和所有 worktree 之外并设为仅当前用户可读；优先使用 SSH key/agent。
- 上一条只描述冻结的 legacy/T4 入口。T5 D0 固定使用 `BatchMode=yes` 的 SSH key/agent，
  不读取密码字段；三台主机的公钥预检失败即在取锁/部署前 fail closed。
- Windows PowerShell 调用 WSL `bash` 时，临时环境中的密码变量不会自动跨入 WSL。单次调用必须在不打印值的前提下把所用变量名追加到进程级 `WSLENV`（例如 `ISAAC_PASSWORD`），并在调用结束后恢复原 `WSLENV`、删除密码环境；否则 SSH_ASKPASS 会等待不存在的凭证并在获取锁前超时。不得用 argv 或仓库文件绕过这一点。
- Hugging Face 大文件下载优先设置 `HF_ENDPOINT=https://hf-mirror.com`。T5 协调器从忽略的
  `.env.local` 读取 token，经匿名 FD 交给 DGX Lane，再只导出到模型子树；Nav2、bridge、
  evaluator 和 x86 不继承。归档前对收集结果做 exact-secret 扫描；禁止写进 URL、argv、
  脚本、配置或结果。
- 模型下载、加载和推理只能在相应 Lane 的 DGX 上进行，并需要该 DGX 租约；
  两台 DGX 都必须能独立加载同一个 Golden checkpoint。
- 不得把 secret 放进被包裹命令的 argv；argv 会作为审计字段记录。

## 在线调度与准入

旧 T4 在线阶段仍由 `TASK_BOARD.md` 管理。T5 的正常 completion_sim 研发使用
engineering fast path：入口必须绑定 clean exact code SHA、唯一 run_id/结果目录、
兼容的同 SHA prepare receipt 与实际 live lane physical lease；不读取 board-only
grant，不要求 predecessor、另一 Lane 静止或 `all-lanes`。当前顺序是：

```text
FAST-PREP：exact-SHA 双 DGX 并行 build + x86 串行 shared-I/O prepare
→ FAST-CANARY：A+B 同时约 60 s
→ FAST-5：A+B 同时各自 fixed-five
→ D1：A/B 独立并行研发
→ FINAL：合并后两 Lane 各一次短 canary；20 episodes 可先拆 A10+B10
```

最终正式 evidence run 才读取 `coordination/T5_DUAL_LANE_BOARD.md` 中唯一的
`INTERNAV_T5_DUAL_LANE_ONLINE_GRANT_V1`；旧 `T5_TASK_BOARD.md` 是不可授权的
tombstone。formal coordinator 必须 fail closed 拒绝
`authorization_mode=FAST_EXACT_REF_NO_BOARD` 的 prepare receipt，fast 结果不能被
静默转换为 formal predecessor。旧 D0.0→D0.1→D0.2→D0.3、逐 candidate 交换 Lane、
paired report 与每 Lane 完整 20 pilot 作为后置正式证据，不阻塞 completion_sim D1
或 T5 功能完成裁决。

以下到“租约操作纪律”前仅保留为冻结 T4/旧 01–05 的 legacy 准入说明，不能为
T5 授权。05 默认不得使用 Isaac 或 DGX；连接真实 Go2 还需要用户另行明确批准。

Worker worktree 内的 `coordination/TASK_BOARD.md` 是共同基线快照，不会为每次授权制造额外 worker commit。在线准入必须只读检查共享 Git ref 中的权威任务板：

```bash
git show refs/heads/codex/parallel-integration:coordination/TASK_BOARD.md
```

该 ref 必须包含唯一一段合法的 `INTERNAV_ONLINE_GRANT_V1` JSON，且同时满足：`status` 精确为 `GRANTED`；`worker`、`resource`、`profile`、`result_dir`、`grant_id` 均为非空并与本次动作逐项精确匹配；授权提交没有被后续提交撤销。`result_dir` 必须是协调器预先写入 grant 的全新独占路径，因此目录的原子创建同时消费这项单次许可。禁止使用 `"GRANTED" in row` 之类子串判断；`NOT GRANTED`、错误 profile、错误结果目录、重复 grant 区块或字段缺失都必须 fail closed。本地快照、聊天文字或环境变量均不能单独替代此检查。这样既保持 worker 相对基线恰好一个提交，也让所有 worktree 观察到同一授权状态。

01R 的 `completion_sim`、`bootstrap` 与 `soak` 是独立授权。当前先执行 60 s `completion_sim`；600 s `soak` 后置且不阻塞功能阶段。前一次 completion、deviation、零残留与锁释放证据冻结并将 grant 改回 `NO_GRANT` 后，才能签发下一次 grant。任何 grant 只允许对应的单一结果目录，既有或不完整目录均不可复用。

`completion_sim` bootstrap 必须证明真实模拟传感器至少到达 sidecar/bridge、Nav2 与 InternVLA consumer group 可用、日志/deviation 完整且最终残留为零；不要求 exact atomic batch 或 600 s soak。严格 10 Hz、p95、max gap、TF extrapolation 与 600 s soak 继续作为 `STRICT_EXTENSION_PENDING`，不得回滚已通过的功能阶段。

旧 Worker 01 `b94f308f24c3af04d709cffeb19a0d74db40c36c` 已退役，不得获得新租约或作为 02/03 在线绑定目标。

上一条“不得放宽 0.35 s/Collision Monitor”的限制仅适用于 `strict_evidence`。`completion_sim` 可按冻结 overlay 放宽，但在 `FUNCTIONAL_DONE` 前仍禁止 Step-3.7、NVFP4、模型微调和真实 Go2 运动。

## 租约操作纪律

1. 每次只申请一个短、已命名的在线动作；先跑最短 smoke/soak。
2. `--log-dir` 必须是该 worker 独占的 `results/parallel/<task>/...`。
3. 任务失败或达到时限立即退出包裹命令；退出即释放锁。
4. 不能通过删除锁文件“解锁”。仅远端 `flock` 的存活文件描述符代表所有权。
5. 锁冲突时只读取 owner/task/PID 等元数据并报告协调器，不得杀死其他 holder。
6. 怀疑陈旧状态时可尝试一次正常的非阻塞租约；不得凭锁文件中的旧文本判断锁已占用或可用。
7. 结果目录必须在启动前确认不存在；命令完成后必须存在机器可读 completion/validation，并禁止后续 attempt 复用。
8. cleanup 成功必须来自 ledger 中所有 PID/PGID 的实测零存活与 socket=0，不得由 ready 文件存在性或 trap 已执行推断。
9. T5 D0 coordinator 必须为 DGX 与 x86 远端命令创建独立 `setsid` supervisor ledger；异常退出按 `TERM → bounded wait → KILL` 处理 supervisor 和 runtime ledger 中的子 PGID，并在释放物理租约前写出 coordinator cleanup receipt 与容器 PID=0 审计。
