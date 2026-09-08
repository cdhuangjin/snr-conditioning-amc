# Learned SNR Estimator — 解释与论文含义

## 1. 属于哪个 Scenario

**Scenario A**：estimator quality alone does not close the deployment gap.

判定依据：

- TinyCNN MAE 明显低于 Ridge（6.03 → 3.08 dB，约减半；Pearson 0.716 → 0.865）。→ 满足“estimator 明显更强”。
- 但 M3/M6 的 TinyCNN 精度仍明显低于 oracle（−10.23 / −11.21 pp），且**低于 M0 基线**
  （−5.25 / −6.31 pp）。→ 满足“仍无法恢复 oracle gain，且低于/接近 M0”。
- TinyCNN 相对 Ridge 的改进很小：M3 +0.40 pp（不显著），M6 +1.67 pp（显著但量级小）。

结论按 Scenario A 撰写：**更好的 SNR 回归 ≠ 保证更好的条件化分类**。

## 2. 关键论证链

### 2.1 estimator 确实更强

这是“条件质量”的直接度量：MAE 几乎减半，且每个 SNR bin 都更好（除 −10 dB 附近几乎持平）。
尤其极端 SNR（−20 / +18 dB）改善巨大。因此不能把下游缺口归咎于“估计器不够好”。

### 2.2 但下游几乎不随之改善

frozen substitution 显示，即便把估计误差从 6 dB 降到 3 dB，M3/M6 的 accuracy 只提升约
0.4–1.7 pp，仍与 oracle 相差 10–11 pp，且低于不条件化（M0）的 61.8%。

这说明：**标量 SNR 估计误差（MAE）不足以刻画条件质量**。决定条件化收益的是
“estimator 输出的条件分布与分类器训练时所见条件分布是否匹配 / 兼容”，而非单纯回归精度。

### 2.3 潜在机制（描述性）

- M3 使用离散 bin（最近偶数 bin）；估计误差在 bin 边界附近会造成“跳 bin”，引入与训练分布不一致的条件。
- M6 使用归一化标量 gate；小误差经 `2*sigmoid(Wz+b)` 后可能被放大/偏移，且负偏（约 −1 dB）
  使条件整体偏低。
- 估计条件在低 SNR 段的**条件分布形状**（误差方差、偏置、类内不齐）与 oracle 不同，
  这比均值误差更能解释分类退化。

## 3. 论文建议

### 3.1 主文写法

- 结论措辞避免“estimated conditioning inherently fails”。应写成：
  “**在给定 estimator–classifier 兼容性与条件分布匹配的前提下**，更准确的 SNR 估计并未闭合部署缺口；
  标量估计误差不能单独解释条件化收益的丢失。”
- 强调这是一个区分性证据：**不是 estimator 不够准，而是 condition 的可用形式/分布不匹配**。

### 3.2 是否值得加入主文

**建议加入主文（作为一把收敛的“补充对照”表 + 一张图）**，因为：

- 直接回应该领域常见问题“是不是估计器太差导致的？”——用低成本设计明确否定。
- 强化论文的核心命题（oracle 收益依赖估计可靠性与分布匹配），且是**负结果/受控对照**，可信度高。

### 3.3 是否只放 SI

不建议只放 SI。它属于主线论证的一环（回答“部署缺口是否由 estimator 强度决定”），
放主文能显著提升说服力；完整 5 张表 + 2 图可放 SI。

### 3.4 abstract / conclusion

- **不必重写 abstract**，但可在 conclusion 加一句限定：
  “改用显著更准确的轻量学习估计器（MAE 6.0→3.1 dB）仍无法恢复 oracle 条件化收益，
  说明部署缺口主要由 estimator–classifier 兼容性与条件分布匹配决定，而非估计精度本身。”
- **conclusion 不必改变方向**，只需把“估计 SNR 敏感”的表述升级为“估计器质量与兼容性共同决定”。

### 3.5 是否建议继续 TinyCNN matched training

**不建议作为本次主实验的继续**。原因：

- 主结果已落入 Scenario A：问题在条件分布兼容性，不在估计器回归质量。
- 若继续，最有价值的是 **matched-training / residual-conditioned retraining**（让分类器适应
  TinyCNN 真实误差分布），但这属于可选扩展（plan 第十五节 P2），会显著增加工作量，且
  在当前证据链下更像“补救”而非“解释”。建议先完成投稿主实验，若有必要再作为 future work 提及。

## 4. 边界与限制（必须写进 Limitations）

- 仅评估一个轻量学习估计器（TinyCNN-SNR），不是 SNR 估计 SOTA。
- 单一数据集（RML2016.10a）+ 固定划分；结论条件于该数据集与估计器族。
- TinyCNN 存在约 −1 dB 的轻微负偏；未做 bias-correction。
- 5 个网络种子只刻画网络初始化波动，不表示跨数据集不确定性。
- 下游改进幅度小且 M3 不显著，不宜过度解读“TinyCNN 提升”。
