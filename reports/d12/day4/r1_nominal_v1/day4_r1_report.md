# D12 第四天：R1 名义运行结果

状态：已通过修复 R0 的零系数全期门；R1 只新增冻结的事前冲击惩罚，复用修复 R0 预测。

| 指标 | R1 | R1 − 修复 R0 |
|---|---:|---:|
| 年化超额 | 3.739629% | -0.114709% |
| IR | 0.535717 | +0.007737 |
| 年化跟踪误差 | 6.980605% | -0.319558% |
| NW-t 最小值 | 1.571810 | +0.073367 |
| 最大相对回撤 | 13.442363% | -0.723262% |

- 原 D11、修复 R0 与 R1 的完整三列表见 `three_way_results.tsv`；修复差异单列，未记为策略改进。
- 买卖两侧参与率、5% 超限、执行端实际截断与未成交金额见 `participation_summary.tsv`。
- 本日未做归因、IC 或执行端冲击 ×2；它们按排期属于第五天。
- P0-12 成分生效日仍未修复，因此这是同一缺陷口径下的受控比较，不恢复历史 Go/No-Go；历史结论仍为 No-Go。
- 2019–2026 历史已被反复查看，本次不是新的样本外验证。

监测摘要：[{'side': 'buy', 'order_rows': 86627, 'requested_signal_cny': 1181248984.6658337, 'a_est_missing_rows': 0, 'ex_ante_above_5pct_rows': 0, 'maximum_ex_ante_participation': 0.006512286977109507, 'expected_impact_fraction': 0.024110598661325935, 'actual_capacity_limited_rows': 0, 'capacity_limited_unfilled_signal_cny': 0.0, 'all_unfilled_signal_cny': 48065723.98800246, 'cash_scaled_rows': 62505}, {'side': 'sell', 'order_rows': 103667, 'requested_signal_cny': 1161773553.912426, 'a_est_missing_rows': 1330, 'ex_ante_above_5pct_rows': 0, 'maximum_ex_ante_participation': 0.008754245206420555, 'expected_impact_fraction': 0.022589334572672575, 'actual_capacity_limited_rows': 1, 'capacity_limited_unfilled_signal_cny': 179725.88838225225, 'all_unfilled_signal_cny': 36959571.370924555, 'cash_scaled_rows': 0}, {'side': 'all', 'order_rows': 190294, 'requested_signal_cny': 2343022538.57826, 'a_est_missing_rows': 1330, 'ex_ante_above_5pct_rows': 0, 'maximum_ex_ante_participation': 0.008754245206420555, 'expected_impact_fraction': 0.04669993323399851, 'actual_capacity_limited_rows': 1, 'capacity_limited_unfilled_signal_cny': 179725.88838225225, 'all_unfilled_signal_cny': 85025295.35892701, 'cash_scaled_rows': 62505}]
