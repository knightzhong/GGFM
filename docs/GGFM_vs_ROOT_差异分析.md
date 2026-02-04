# GGFM vs ROOT 结果差异全面分析

目标：ROOT 在 Dkitty 上得到 **0.97 / 0.94 / 0.92**（100th/80th/50th 归一化分位），GGFM 得到约 **0.93 / 0.87 / 0.83**，且出现 **Optimized Mean 远低于 Original Mean**（模型把设计变差）。本文档说明根因与修复。

---

## 1. 训练/推理 y 顺序必须与 ROOT 完全一致（已修复）

- **ROOT 训练**：`p_losses` 中 `denoise_fn(x_t, t, y_high, y_low)` → MLP 收到 **(pair 高分, pair 低分)**。
- **ROOT 推理**：`p_sample` 中 `denoise_fn(x_t, t, y_low, y_high)` → MLP 收到 **(当前分数, 目标分数)**；当 type=highest 时即 (高, target)。

若训练改成 `(y_low, y_high)` 而推理用 `(y_low, y_high)`，则训练时 MLP 学的是「第一维=低、第二维=高」，推理时从 highest 出发会得到「第一维=高、第二维=target」，与训练分布不一致，易导致输出变差。

**修复**：GGFM 已与 ROOT 对齐：
- **训练**：`denoise_fn(x_t, t, y_high, y_low)`（与 ROOT 一致）。
- **推理**：`denoise_fn(x_t, t, y_low, y_high)`（与 ROOT 一致）。
- **必须重新训练**：旧 checkpoint 是按错误顺序训的，不能沿用。

---

## 2. 推理目标分数 y_high_cond（已修复）

- **ROOT**：  
  `high_cond_scores = (oracle_y_max - mean_y)/std_y * alpha`（alpha=0.8）。
- **GGFM 原实现**：  
  `y_high_cond = np.percentile(y_train_norm, 99)`。

已改为与 ROOT 一致：使用 `(oracle_y_max - mean_y)/std_y * TEST_ALPHA`，并在 config 中支持 `testing.alpha: 0.8`。

---

## 3. p_sample 方差与更新公式（已对齐）

- **ROOT**：  
  `sigma2_t = (var_t - var_nt*(1.-m_t)**2/(1.-m_nt)**2)*var_nt/var_t`，  
  `x_tminus_mean` 中分母为 `var_t`（无 1e-8）。
- **GGFM 原实现**：分母使用 `(1.0 - m_nt + 1e-8)` 和 `(var_t + 1e-8)`。

已改为与 ROOT 完全一致（仅保留 1e-10 防止 sqrt 除零）。

---

## 4. 训练数据来源与构造（未改，可能影响）

- **ROOT**：  
  - Dkitty 未开 `use_pairing_instead_of_gp`，因此用 **GP 采样**。  
  - 从 `best_x`（top 1024）出发，一次梯度得到 (low_x, high_x) 配对，格式与 `sampling_data_from_GP` 一致。
- **GGFM**：  
  - 同样用 GP 采样，但每 epoch 的起点是 **50% 来自 top 2000 + 50% 全局随机**（`mixed_indices`），再在同一批点上做梯度得到轨迹，只取轨迹 (start, end) 作为 (x_low, x_high)。

差异：ROOT 固定从 top 1024 出发；GGFM 混合 top 2000 与随机，低分起点更多，配对分布不同，可能影响学到的 bridge 质量。

---

## 5. EMA（未实现）

- **ROOT**：Dkitty 配置中 `use_ema: true`，测试时用 **EMA 权重**。
- **GGFM**：无 EMA，测试用当前模型权重。

若 ROOT 的 0.97 是用 EMA 测的，GGFM 不加 EMA 会系统性略低。

---

## 6. 其他配置差异（供对照）

| 项目           | ROOT (Dkitty) | GGFM (Dkitty_BB) |
|----------------|---------------|-------------------|
| batch_size     | 64            | 256 (FM_BATCH_SIZE) |
| accumulate_grad_batches | 2   | 无               |
| val_frac       | 0.1           | 无验证集          |
| clip_denoised  | false         | false             |
| CFG test       | false, weight -1.5 | 0.0           |
| learning_rate  | 0.001         | 0.001             |

batch 与梯度累积不同可能带来训练动态差异；CFG 在 ROOT 为关且 weight=-1.5，GGFM 为 0，当前已对齐为不用 CFG。

---

## 7. 已做修改汇总

1. **brownian_bridge.py**  
   - `p_sample` 与 `p_losses` 中 `denoise_fn` 统一为 `(x_t, t, y_low, y_high)`，与 ROOT 推理一致并保证训练/推理语义一致。  
   - `p_sample` 中方差与 `x_tminus_mean` 公式与 ROOT 对齐。

2. **main.py**  
   - 评估时 `y_high_cond` 改为 `(oracle_y_max - mean_y)/std_y * alpha`。

3. **config**  
   - 增加 `TEST_ALPHA`，从 `testing.alpha` 读取，Dkitty_BB 中为 0.8。

---

## 8. 若仍达不到 0.97 的后续建议

1. **加 EMA**：与 ROOT 一致做 EMA，测试时用 EMA 权重。  
2. **训练数据对齐**：改为从固定 top 1024 出发做 GP 采样（与 ROOT 一致），或做 ablation 对比当前混合策略。  
3. **batch/优化**：batch_size=64、accumulate_grad_batches=2 与 ROOT 对齐。  
4. **复现 ROOT 的 seed**：确认 ROOT 的 0.97 所用 seed，在 GGFM 中用同 seed 做数据与评估。

---

---

## 9. 评估起点：highest vs low（新增选项）

- **type_sampling=highest**（与 ROOT 默认一致）：从得分最高的 128 个样本出发，条件为 (高, target)，与训练时的 (高, 低) 不完全同分布，部分 run 可能表现不稳。
- **type_sampling=low**：从得分最低的 128 个样本出发，条件为 (低, target)，与训练 (低→高) 一致，更容易看到「优化提升」。

在 config 的 `testing.type_sampling` 中可设为 `highest` 或 `low`；Dkitty 想复现 ROOT 的 0.97 时用 `highest`，想先验证模型是否学会「低→高」时用 `low`。

**重要**：当前修复恢复了与 ROOT 一致的「训练 (y_high, y_low) / 推理 (y_low, y_high)」约定。**必须从头重新训练**；用旧 checkpoint 会得到错误结果。
