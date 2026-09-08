# Learned SNR Estimator — 实验结果

生成日期：2026-09-08

## 0. 一句话结论

用轻量 TinyCNN-SNR 把 SNR 估计的 test MAE 从 Ridge 的 **6.03 dB 降到 3.08 dB**
（约减半，Pearson 0.716 → 0.865），但冻结分类器的下游精度**几乎不因此恢复**：

- M3：oracle 66.80 → ridge 56.16 → tinycnn 56.57（tinycnn − ridge = +0.40 pp，95% CI [−0.42, +1.23]，不显著）
- M6：oracle 66.73 → ridge 53.85 → tinycnn 55.51（tinycnn − ridge = +1.67 pp，95% CI [+0.89, +2.44]，显著）
- 二者仍大幅低于 oracle（−10.23 / −11.21 pp）且**低于无条件基线 M0**（−5.25 / −6.31 pp）

即：**更好的 SNR 回归质量 ≠ 更好的条件化分类**。部署缺口主要不是 estimator 不够准，而是
**estimator–classifier 兼容性 / 条件分布匹配**问题。

## 1. Estimator 指标（test 集，`Table_LSE_1`）

| seed | estimator | MAE | RMSE | Pearson | Bias | ClipLow | ClipHigh |
|---|---:|---:|---:|---:|---:|---:|---:|
| — | Ridge | 6.03 | 8.05 | 0.716 | −0.008 | 0.0054 | 0.0003 |
| 2022 | TinyCNN | 3.035 | 5.930 | 0.865 | −0.764 | 0.000 | 0.0035 |
| 2023 | TinyCNN | 3.019 | 5.899 | 0.865 | −0.698 | 0.000 | 0.0083 |
| 2024 | TinyCNN | 3.082 | 6.094 | 0.862 | −1.322 | 0.000 | 0.0002 |
| 2025 | TinyCNN | 3.216 | 5.901 | 0.862 | −0.742 | 0.000 | 0.0018 |
| 2026 | TinyCNN | 3.065 | 6.049 | 0.863 | −1.197 | 0.000 | 0.0008 |

**汇总**：TinyCNN MAE 均值 ≈ 3.08 dB（SD≈0.08），相对 Ridge 改进约 **2.95 dB / 49%**；
RMSE 均值 ≈ 5.97 dB。TinyCNN 有轻微负偏（约 −0.7 至 −1.3 dB），推断略低估 SNR，
但几乎不触底（ClipLow=0），高段有极少量触顶。

## 2. 阈值敏感 MAE（`Table_LSE_2` / `Fig_LSE_1`）

TinyCNN 在每个 SNR bin 几乎都优于 Ridge，尤其两个极端：

- −20 dB：Ridge 11.69 → TinyCNN 6.19
- −14 dB：Ridge 6.47 → TinyCNN 1.74
- 0 dB：Ridge 2.98 → TinyCNN 2.22
- +18 dB：Ridge 11.58 → TinyCNN 4.75

TinyCNN 仅在极窄区间（≈ −10 dB 附近）与 Ridge 接近，其余均显著更优；在低 SNR 段改善最明显。

## 3. Frozen-classifier substitution（`Table_LSE_3` / `Table_LSE_4`）

冻结 M3/M6/M0 checkpoint，仅替换 condition source：

| Model | Condition | Acc mean | SD | Gain vs M0 | Drop vs Oracle |
|---|---:|---:|---:|---:|---:|
| M3 | Oracle | 66.80 | 0.23 | +4.98 | 0.00 |
| M3 | Ridge | 56.16 | 0.20 | −5.66 | −10.63 |
| M3 | TinyCNN | 56.57 | 0.72 | −5.25 | −10.23 |
| M6 | Oracle | 66.73 | 0.14 | +4.91 | 0.00 |
| M6 | Ridge | 53.85 | 0.33 | −7.97 | −12.88 |
| M6 | TinyCNN | 55.51 | 0.40 | −6.31 | −11.21 |
| M0 | None | 61.80 | 0.10 | — | — |

> 复现核对：M3 oracle 66.80 / M6 oracle 66.73 / M0 61.80 与 Phase 3 主结果一致；
> M3-Ridge 56.16 / M6-Ridge 53.85 与 Phase 8 E2 一致。冻结回放忠实复现权威数值。

## 4. 配对统计（`Table_LSE_5`，5 配对 seed，df=4，95% t 区间）

| 比较 | Model | Mean diff (pp) | SD | CI 下 | CI 上 |
|---|---:|---:|---:|---:|---:|
| TinyCNN − Ridge | M3 | +0.40 | 0.67 | −0.42 | +1.23 |
| TinyCNN − Ridge | M6 | +1.67 | 0.62 | +0.89 | +2.44 |
| TinyCNN − Oracle | M3 | −10.23 | 0.75 | −11.16 | −9.30 |
| TinyCNN − Oracle | M6 | −11.21 | 0.40 | −11.71 | −10.72 |
| TinyCNN − M0 | M3 | −5.25 | 0.73 | −6.16 | −4.35 |
| TinyCNN − M0 | M6 | −6.31 | 0.43 | −6.84 | −5.78 |

解释：TinyCNN 相对 Ridge 的**改善有限**（M3 不显著、M6 显著但仅 +1.67 pp）；
相对 oracle 的**缺口巨大且显著**；相对 M0 基线**仍为显著负收益**。

## 5. Estimator 误差 vs 下游精度（`Fig_LSE_2`）

5 个 estimator seed 的点集中在 MAE ≈ 3.0–3.2 dB、下游 55.5–57.4%（M3）/55.1–56.2%（M6）；
Ridge 单点位于 MAE 6.03 dB、56.1%（M3）/53.9%（M6）。
MAE 减半只带来约 +1 pp 级别的下游变化，二者无明显单调关系——**描述性**，不做因果外推。

## 6. 产物索引

- 表格：`tables/Table_LSE_1_estimator_metrics.csv` … `Table_LSE_5_paired_CI.csv`
- 图：`figures/Fig_LSE_1_estimator_mae_by_snr.{png,pdf}`、`figures/Fig_LSE_2_estimator_error_vs_classification.{png,pdf}`
- checkpoint：`checkpoints/tinycnn_snr_seed{2022..2026}.pt`
- 估计输出：`raw/estimator_predictions_seed{2022..2026}.npz`
- 冻结证据：`raw/frozen_rows.json`、`raw/frozen_m0_rows.json`
- 审计：`audit/learned_snr_estimator_audit.md`
