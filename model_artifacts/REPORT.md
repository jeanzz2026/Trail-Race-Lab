# ITRA Race Score 离线估算原型 v1

生成时间：2026-09-23T03:35:59.513272+00:00  
定位：**非官方估算器**；用于验证公开数据能否支持 Race Score 近似计算。

## 数据审计

- TRAP：15,195 场有效比赛、2,878,496 条完赛记录。公开 CSV **不含 Race Score 标签**，因此只用于学习赛程长度/爬升与精英完赛时间的尺度先验。
- 现代标签：48 条公开的“时间 + Race Score”，覆盖 8 场比赛。
- 现代标签明显偏向精英（最低 763 分），普通跑者区间仍缺少直接校准数据。

## 最重要的经验结果

同一场比赛中，`Race Score × 完赛小时数` 几乎是常数。各场标签的一致性如下：

| 年份 | 比赛 | 标签数 | 比赛常数 | 常数相对极差 |
|---:|---|---:|---:|---:|
| 2023 | WMTRC Innsbruck-Stubai Long Trail | 20 | 9260.1 | 0.11% |
| 2023 | WMTRC Innsbruck-Stubai Short Trail | 20 | 4066.0 | 1.32% |
| 2025 | Chianti Ultra Trail by UTMB - Ultra Trail Chianti Castles | 1 | 9676.8 | 0.00% |
| 2025 | Javelina Jundred 100 Mile | 1 | 11310.8 | 0.00% |
| 2025 | OCC UTMB Mont-Blanc alternate course | 2 | 4765.5 | 0.01% |
| 2025 | UTMB Mont-Blanc | 1 | 18659.4 | 0.00% |
| 2025 | WMTRC Canfranc Long Trail | 2 | 8330.3 | 0.04% |
| 2025 | WMTRC Canfranc Short Trail | 1 | 4448.3 | 0.00% |

因此有同场一个官方锚点时，推荐：

`估算分 = 锚点分 × 锚点时间 ÷ 目标时间`

## 验证结果

### 同场有一个锚点

- 反比例时间模型：MAE 0.96 分，RMSE 1.95 分，最大绝对误差 10.33 分。
- GitHub `yama-asobi-lab/trail-race-planner` 的 UTMB 对数曲线复现基线：MAE 23.85 分，RMSE 28.19 分。
- 结论：当前官方公开样本更支持反比例关系；社区项目的对数拟合可作为参考，但不应替代逐场验证。

### 完全没有锚点，仅有距离和爬升

- 按比赛分组的留一法：MAE 34.84 分，RMSE 38.50 分，最大绝对误差 95.86 分。
- 80% 经验绝对误差：±36.0 分；95%：±67.7 分。
- 这个模式没有 GPX 海拔、技术难度、天气和 ITRA 的参赛者历史修正，必须显示为低置信度。

## 当前结论

1. **有同场 Race Score 锚点时已经可用**，适合把一个公开/用户自有分数扩展到整场选手。
2. **无锚点时只是粗估**；距离和 D+ 无法恢复 ITRA 的逐场环境修正系数。
3. 下一轮最有价值的数据不是更多精英冠军，而是 2021 年以后、覆盖 400–750 分的同场成组 Race Score。

## 运行

```powershell
.\.venv\Scripts\python.exe scripts\train_offline_model.py --trap-dir ..\TRAP-data\ITRA
.\.venv\Scripts\python.exe scripts\predict_race_score.py --distance 61 --elevation 3400 --time 07:30:00
.\.venv\Scripts\python.exe scripts\predict_race_score.py --distance 61 --elevation 3400 --time 07:30:00 --anchor-time 05:35:13 --anchor-score 853
```

## 数据和方法限制

- ITRA 未公开完整公式，本模型不声称重现官方算法。
- TRAP 仓库没有明确许可证；当前仅在本地用于研究验证，不随模型分发原始 CSV。
- 社区参考项目未声明许可证，本项目只独立复现其公开描述的数学基线，不复制其代码。

## 主要来源

- ITRA Race Score 说明：https://itra.run/FAQ/ItraScore
- ITRA 2021 算法更新说明：https://itra.run/content/news/EN-TRAIL_RUNNING_REPORT_2021.pdf
- TRAP 数据：https://github.com/ricfog/TRAP-data
- ScrapITRA：https://github.com/ricfog/ScrapITRA
- 社区对数曲线参考：https://github.com/yama-asobi-lab/trail-race-planner
- 逐条现代标签的来源保存在 `data/modern_race_score_labels.csv`。
