# Learned SNR Estimator — 项目审计

生成日期：2026-09-08

## 1. 目的

在原论文结论（oracle SNR conditioning 有收益，但换用 Ridge frame estimator 后 M3/M6 明显下降）
之上，新增一个轻量 learned estimator（TinyCNN-SNR），用于区分：

- conditioning 本身不可部署（deployment gap），还是
- 现有 Ridge estimator 太弱（estimator-limited）。

本审计只盘点现有项目：文件路径、函数、checkpoint、split、estimator 接入点、可复用结果与需新跑项。
确认无泄漏后，再进入 `train → eval → frozen substitution → summary`。

## 2. 关键路径（工作树 `AWN/.worktrees/phase2-v2`，分支 `codex/phase2-v2`）

| 作用 | 路径（相对 worktree root） |
|---|---|
| 数据 | `data/RML2016.10a_dict.pkl`（640 MB，`data_sha256=b29ccc25...8e6c`） |
| 固定 split | `splits/v2/RML2016.10a_seed2022.json` + `.npz`（60/20/20，`split_hash=42450053...54a9`） |
| 数据加载 | `scripts/v2/phase1_reproduce.py::_load_rml_dataset` |
| split 加载 | `scripts/v2/run_phase3.py::load_split` → `v2/splits.load_split` |
| 条件化模型 | `models/model_conditioning.py::AWNConditioned` |
| AWN 骨干 | `models/model.py`，`models/lifting.py` |
| Phase 3 契约 | `scripts/v2/run_phase3.py`（`_model`，`_load_state`，`_load_ledger`，`_ledger_has_result`） |
| Phase 3 run 注册 | `v2/phase3_analysis.py::_load_registered_runs`,`_load_fixed_data`,`_load_phase3_contracts` |
| Phase 3 配置 | `configs/v2/phase3.yaml` |
| Phase 3 结果 | `results/v2/phase3/`（40 个 completed run：M0–M7 × seed 2022–2026） |
| Phase 6（Ridge 估计器） | `results/v2/phase6_preprocessing/attempts/20260906T004123Z_a6acf299/legacy_identity/` |
| Phase 8（冻结替换参照实现） | `v2/phase8/core.py`（`conditions`,`head_logits`,`grouped_metrics`），`v2/phase8/evidence.py`（`load_estimator`） |
| Phase 8 设计 | `docs/phase8_design.md` |
| 新结果目录 | `results/learned_snr_estimator/`（新增，不污染 canonical） |

## 3. 数据与固定划分

- RML2016.10a：11 类 × 20 SNR bin（−20…+18，步进 2），每 cell 1000，共 220,000 帧。
- 帧形状：`[220000, 2, 128]`，`float32`，NCT 布局（通道=0 为 I/Q，1 为时间），保持原始幅值，无预处理。
- 固定划分：train 132,000 / val 44,000 / test 44,000，`class-SNR` 分层 60/20/20。
- 5 个网络种子：2022 / 2023 / 2024 / 2025 / 2026。
- 划分唯一（seed 2022），5 个种子只刻画**网络初始化波动**，不是跨数据集复制。
- SNR 归一化（分类器内部）：`z=(snr_db-(-1))/19`，输入 [−20,18]，输出 [−1,1]。
- SNR 条带：low ≤ −8 dB；mid −6…−2 dB；high ≥ 0 dB。

**复用结论**：estimator 的训练/验证/测试直接复用同一固定划分，不重新切分。

## 4. 数据加载细节

`_load_rml_dataset(data_path, 'RML2016.10a', repository_root=root)` 返回：

```python
{
  "signals": np.ndarray [220000, 2, 128] float32,   # 原始幅值
  "labels":  np.ndarray [220000,]    int64,          # 类索引 0..10
  "snrs":    np.ndarray [220000,]    float32,        # 真值 SNR dB
}
```

`dataset['signals'][idx]` 直接可作为 estimator 输入（`[2,128]`），无需额外归一化；estimator 内部用
BatchNorm 做 train-derived 归一化，不触碰主数据定义。

## 5. 现有 Ridge SNR estimator

- 实现：`v2/phase11/estimator.py::fit_ridge` / `predict_ridge`。
- 特征：7 个 frame statistics，`v2/phase6/audit.py::frame_features`：

```text
energy, l2_norm, variance_i, variance_q, centered_complex_variance,
peak_to_average, log_energy
```

- 拟合：train 上 `StandardScaler`（population variance），Ridge 网格按 validation MAE 选 α；**不** train+val 重拟合。
- 已冻结证据：`results/v2/phase6_preprocessing/attempts/20260906T004123Z_a6acf299/legacy_identity/`
  `snr_ridge_state.json`, `snr_ridge_selection.json`, `all_partition_predictions.npz`, `features.npz`。
- 复用入口：`v2/phase8/evidence.py::load_estimator(phase6, dataset, split)` 返回全量 `raw` 估计（含 test 估计）+ 溯源。

**当前 Ridge 指标**：`legacy_identity/metrics.csv`（train/val/test 的 MAE/RMSE）。本次实验会用
`load_estimator` 重新取全量 `raw` 估计，并在 `Table_LSE_1` 里固化 test 上的 MAE/RMSE/bias 等，作为
Ridge 与 TinyCNN 的同口径对照。

## 6. 条件化分类器 M0/M3/M6

- 实现：`models/model_conditioning.py::AWNConditioned`，共享 `AWN` 骨干（conv + lifting + pooling + SE）。
- M0：无条件（基线）。
- M3：`snr_embedding`（20×8 学习嵌入）+ `fc` 拼接，注入点=attended pooled features；条件表示=**离散 bin**。
- M6：`gate`（`2*sigmoid(W z + b)`），注入点=attended pooled features；条件表示=**归一化标量 z**。
- `forward_batch` 同时接受 `snr_db` 与 `snr_bin`；`encode_condition`/`apply_feature_conditioning` 按机制取用。
- `classify_pooled(features, snr_bin=…, snr_db=…)` 是**冻结替换**关键：只喂已提取的 pooled features + 条件，
  不重跑 backbone。

## 7. Frozen-checkpoint substitution（复用 Phase 8 机制）

`v2/phase8/core.py`：

- `conditions(estimates)`：`clip` 到 [−20,18]，并按最近偶数 bin 映射为 `snr_bin`。
- `head_logits(model, features, source, batch_size)`：对已提取 features 按 batch 调 `classify_pooled`，
  `source` 为估计 SNR（自动 `conditions`）。
- `grouped_metrics(y, logits, snr)`：返回 `overall` + `snr`（逐 SNR）+ `class` + `bands`（low/mid/high），
  含 accuracy / balanced_accuracy / concentration / prediction_entropy。

E2（frozen frame estimate）本质上就是：`head_logits(model, cached_features, raw[test])`。
新实验仅把 `raw[test]` 替换为 learned estimator 的 `predicted[test]`，并把 condition_source 从 `E1/E2/E3*`
改名为 `oracle/ridge/tinycnn`。

## 8. 现有 M3/M6/M0 checkpoint

- 来源：Phase 3 注册表（`results/v2/phase3/phase3_ledger.json` + `manifest.json`）。
- 通过 `_load_registered_runs(root, (root/'manifest.json').resolve())` 取 `runs[(model_id, seed)]`，
  其 `paths['checkpoint']` / `paths['predictions']` / `result_path` 均为解析后的绝对路径。
- 每 run 目录含 `checkpoint.pt`、`predictions.npz`、`result.json`、`run_spec.json`。
- 本次**不重新训练** M3/M6/M0，仅冻结加载做替换评估。

## 9. 可复用 / 需新跑

| 项 | 来源 | 是否复用 |
|---|---|---|
| 固定 split + 数据 | Phase 3 loader | 直接复用 |
| Ridge 估计 | Phase 6 `load_estimator` | 直接复用（含溯源校验） |
| M3/M6/M0 checkpoint | Phase 3 ledger | 直接复用（冻结） |
| `conditions`/`head_logits`/`grouped_metrics` | Phase 8 `core` | 直接复用 |
| paired t 区间（df=4） | `v2/phase8/runner.py::paired`（t.ppf .975, df=4） | 复刻语义 |
| TinyCNN-SNR estimator | 新 | 需新写（`models/tinycnn_snr.py`） |
| estimator 训练/评估 | 新 | 需新跑（5 seeds） |
| M3/M6 frozen × {oracle,ridge,tinycnn} | 新 | 需新跑（10 checkpoint × 3 条件） |
| 表格 LSE_1..5 + 图 LSE_1..2 + 报告 | 新 | 需新生成 |

## 10. 防泄漏确认

- estimator 仅用 train 训练，validation 选模型/早停，test 只最后评估。
- Ridge/CNN 的 scaler（Ridge 的 scaler；CNN 的 BatchNorm running stats）只由 train 统计得到。
- frozen substitution 不训练 classifier；estimator 的 test 预测不进入任何训练/调参。
- 配对口径：classifier seed `i` ↔ estimator seed `i`（2022→2022 … 2026→2026），5 组配对差。
