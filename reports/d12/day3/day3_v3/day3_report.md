# D12 第三天：本机阶段证据

原结果参照：d11_v1；未建立修复 R0。历史 No-Go 不变。

- 全期价格候选：6936 行；处于项目股票池：2718 行。
- 独立原始收盘价覆盖率：0.62%；独立因子相邻变化覆盖率：0.00%。
- 实际持仓/现金账本重建已通过；原 R0 全期重放通过：True。
- P1 分组四项费用已与全部成交对账；初始建仓、主动交易、强制退出单列。
- 成分证据状态：blocked_no_official_full_set_evidence。proxy 表是后见篮子对照，不是官方名单或权重。
- 正式 R1 仍阻断；新 R1 路径的零系数全期门尚未执行。

请先审阅 price_propagation_focus.tsv、alias_price_focus.tsv、price_candidates_top.tsv、price_coverage.tsv、membership_timeline.tsv 与 membership_proxy_windows.tsv。缺价与零涨跌停占位不等于无涨跌幅限制，复权异常候选也不等于已经确认的数据错误。P1 的 expected_trade_contribution 是事前信号量，不能当作已实现盈利。

只在独立证据定位根因后另建修复输入：价格问题需核查特征/标签/校准/权重/风险依赖；成分问题需核查生效日与公告可得时间，并补足合法权重来源。不得拼接别名价格、截断收益或倒推权重来缩小桥接。修复后重新决定预测缓存可否使用，不覆盖原 D11。
