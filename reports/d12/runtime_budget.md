# D12 运行预算（第1天 上午 step 3）

**记录范围：** 本文件在远端 agent 容器内写成。容器是一次全新 clone，`data/`、`models/`
与 qlib provider 目录均按 `.gitignore` 未进入仓库，科学计算依赖也未安装。
**因此本文件不含任何实测耗时，也不构成对你本机的验收。** 下面第 2 节是需要你在本机执行、
并把输出回传的最小测量协议。

---

## 1. 已经确定的事实

| 项目 | 结论 | 证据 |
|---|---|---|
| 历史耗时记录 | **不存在** | `reports/d11/d11_v1/started.json`、`reports/d11/d11_diagnostic_v1/started.json` 及两个 `summary.json` 均无时间戳或时长字段；`D11运行说明.md`、`D11诊断运行说明.md`、`D9运行说明.md` 无耗时描述 |
| 需要重放的规模 | 392 个决策日、183,240 条订单记录、181,880 条成交记录 | `d11_diagnostic_v1/summary.json` `execution.*` |
| 需要重训的规模（仅 R2） | 10 折（2017–2026），训练行数 69,597 → 296,489 | `reports/d11/d11_v1/fold_*.json` |
| 求解器 | CLARABEL，`d7.solver`；D7 禁止 fallback，`fallback_used=false` | `config/config.yaml`、`decision_*.json` |
| 输入哈希开销 | 每次 preflight 对 **62,598 个文件 / 约 3.90 GB** 做 SHA256，运行前后各一次 | `reports/d11/d11_v1/inputs_before.json` |
| 容器资源 | 4 vCPU / 15 GB RAM / 30 GB 可写盘 | `nproc`、`free`、`df` |

**这里最容易被低估的一项是输入哈希。** `snapshot_inputs` 在运行前后各遍历一次全部
provider bin 文件；约 3.90 GB × 2 次纯 I/O，与每期优化开销无关，必须单独计时，
否则短样本外推会把这部分固定成本重复放大到全期估算里。

## 2. 本机最小测量协议（限时 60 分钟）

按顺序执行，每步记录 wall-clock。**不要为了测量去改任何已登记配置。**

```bash
cd <你的仓库>
git fetch --tags origin                      # 见 INF-03：全新 clone 无 tag 会导致 preflight 失败

# 1) 依赖盘点（秒级，无需科学栈）
python3 scripts/d12_startup_check.py --scope local
python3 scripts/d12_p0_monthly_recheck.py

# 2) preflight：单独测出"启动 + 哈希"固定开销
/usr/bin/time -v python3 scripts/d11_diagnose.py --preflight 2>&1 | tee /tmp/d12_preflight.log

# 3) 短样本：只有在 preflight 通过后才有意义
#    d11_diagnose.py 没有 --limit/--dates 选项，因此"短样本"必须由 D12 自己的入口提供。
#    在写出 D12 入口之前，用 preflight 的固定开销 + 单期求解耗时分别估算。
```

需要回传：`reports/d12/startup_check.tsv`、`/tmp/d12_preflight.log` 的 `Elapsed (wall clock)`
与 `Maximum resident set size`、以及 `python3 -c "import qlib,lightgbm,cvxpy,clarabel"` 的结果。

## 3. 估算口径（在拿到实测前不填数）

全期重放耗时按下式估算，三项分别计时、**不合并**：

```
T_全期 ≈ T_启动与哈希(固定，约2×3.90GB I/O)
       + 392 × T_单期(组合优化 + 撮合记账)
       + T_汇总与写盘(37个输出，约19MB)
```

R2 额外增加：`T_R2 ≈ Σ_折 T_训练(折) + T_预测 + T_全期重放`。训练行数逐年增长约 4.3 倍，
**不能用 2017 折的耗时线性外推到 2026 折**；至少测首尾两折再插值。

## 4. 决策点

| 时点 | 判据 | 结果 |
|---|---|---|
| 9/8 当日 | 本机 `--scope local` 是否显示源预测齐全 | 齐全 → 正常/加速路径；缺失 → 先恢复，恢复预算 2 小时（计划 §3.4） |
| 9/9 晚 | k=0 全期重放是否按原容差通过 | 通过才允许启动非零 R1（计划 §4.7） |
| **9/11 中午** | R0 可信 + R1 名义与压力完成 + P2 无未修复数据问题 + 实测预计 9/13 前留出汇总时间 | 全部满足才启动唯一 R2；否则顺延，**不因 R1 结果好坏改 R2 参数** |
| 9/13 晚 | 冻结点 | 未完成项标记未完成，保留逐折日志；不把部分年份当全期胜负 |

## 5. 已知的运行前阻塞（与 startup_check.tsv 对应）

1. `data/` 与 qlib provider 缺失 → `preflight` 抛 `DiagnosticError`（INF-01）。
2. 科学栈未安装；`d11_io.preflight` 还会**逐一比对版本号**，
   要求 qlib 0.9.7 / lightgbm 4.7.0 / cvxpy 1.9.2 / clarabel 0.11.1，版本不符直接拒绝运行。
3. `scripts/d11_diagnose.py` 要求分支为 `d11`，当前为 `d12`（INF-02）。
   计划 §12 已预留：D12 用独立入口，不删除 D11 的分支校验。
