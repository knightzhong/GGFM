# GGFM vs ROOT：代码级差异清单（逐项可核对）

本文档只列**代码级别**的差异，不讨论 y_low/y_high 顺序。用于定位结果差距原因，或重做时做逐项对齐。

---

## 1. 训练数据来源（差异最大）

| 项目 | ROOT | GGFM |
|------|------|------|
| GP 采样的输入 `x_train` | **固定** `self.best_x`（top 1024 样本），每 epoch 相同 | **每 epoch 不同**：`current_epoch_x` = 50% 从 top 2000 随机 512 + 50% 从全量随机 512 |
| 代码位置 | `BaseRunner.py` 第 417 行：`data_from_GP = sampling_data_from_GP(x_train=self.best_x, ...)` | `main.py` 第 223–234 行：`mixed_indices` → `current_epoch_x`；第 267 行：`x_train=current_epoch_x` |
| 含义 | 训练对 (x_low, x_high) 始终来自「同一批 1024 个高分设计」的梯度下降/上升 | 训练对来自「每轮不同的 1024 点」，其中约一半来自全量随机，包含大量低分设计 |

**结论**：ROOT 的 (x_low, x_high) 分布始终围绕「高分区域」；GGFM 的分布包含大量「低分→高分」的远距离对。测试时从 highest 128 出发，ROOT 更贴近训练分布，GGFM 易出现分布偏移。

---

## 2. GP 采样返回格式与使用方式

| 项目 | ROOT | GGFM |
|------|------|------|
| 返回格式 | 仅**端点对**：`[(high_x, high_y), (low_x, low_y)]`，无轨迹 | **轨迹 + 端点分数**：`(trajectory, y_low, y_high)`，trajectory 为 (steps+1, dim) |
| 代码位置 | `runners/utils.py` 第 92 行：`sample = [(high_x[i].detach(), high_y[i].detach()), (low_x[i].detach(), low_y[i].detach())]` | `generator.py` 第 150 行：`sample = (full_trajectories_batch[i], y_low[idx], y_high[idx])` |
| 训练用到的数据 | 直接用这对 (x_high, x_low) | 用 `generate_trajectories_from_GP_samples` 得到 traj，再取 `traj[:, 0]` 和 `traj[:, -1]` 作为 (x_low, x_high) |

**结论**：端点 (x_low, x_high) 在两边都来自同一次 GP 的 descent/ascent，但 GGFM 多了一步轨迹构造与重采样；主要差异仍在第 1 条的「输入池」不同。

---

## 3. Dataloader 与 batch 构造

| 项目 | ROOT | GGFM |
|------|------|------|
| 是否有 DataLoader | 有：`create_train_dataloader(data_from_GP, val_frac=0.2, batch_size=64, shuffle=True)` | 无：直接对当轮所有 trajectory 做 `randperm` 后按 batch 切 |
| 每 epoch 用于训练的数据量 | 每个 function 只取 `samples[int(len*val_frac):]`（后 80%），再拼成 train_data | 当轮 GP 生成的所有 trajectory 全部参与训练 |
| batch_size | 64（Dkitty 配置） | 256（Config.FM_BATCH_SIZE） |
| 梯度累积 | `accumulate_grad_batches=2` | 无 |

**结论**：ROOT 有 val_frac 截断、固定 batch_size 和梯度累积；GGFM 全量训练、batch 更大、无累积。训练动态不同。

---

## 4. 优化器与学习率

| 项目 | ROOT | GGFM |
|------|------|------|
| 优化器 | 从 config 读（如 Adam, beta1=0.9, weight_decay=0） | `optim.Adam(net.parameters(), lr=Config.FM_LR)` |
| 学习率调度 | 有：`lr_scheduler`（如 ReduceLROnPlateau，patience=200 等） | 注释掉：`# scheduler = optim.lr_scheduler.CosineAnnealingLR(...)` |
| BB 训练时梯度裁剪 | 未在 BBDMRunner 中显式写 | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` |

**结论**：ROOT 有 scheduler 和可能不同的 optimizer 配置；GGFM 固定 lr、额外 grad clip。可能影响收敛与稳定性。

---

## 5. EMA

| 项目 | ROOT | GGFM |
|------|------|------|
| 是否使用 EMA | 是：`use_ema: true`，测试前 `apply_ema()` | 无 |
| 代码位置 | `BaseRunner.py` 第 655–656 行：`if self.use_ema: self.apply_ema()` | — |

**结论**：ROOT 测试用的是 EMA 权重；GGFM 用当前权重。若 ROOT 的 0.97 是用 EMA 测的，这会直接拉大差距。

---

## 6. 离散任务的数据编码（仅离散任务）

| 项目 | ROOT | GGFM |
|------|------|------|
| 编码方式 | `offline_x = task.to_logits(offline_x).reshape(...)`，不修改 `task.x` | `task.map_to_logits()` 后 `offline_x = task.x`，修改 task 内部状态 |

Dkitty 为连续任务，此项不影响当前 Dkitty 结果；若做 TFBind 等离散任务需对齐。

---

## 7. 布朗桥模型与采样公式

| 项目 | ROOT | GGFM |
|------|------|------|
| m_t / variance / steps 计算 | 一致（linear schedule, skip_sample, linear sample_type） | 已对齐 |
| q_sample / predict_x0_from_objective | 一致（objective='grad'） | 一致；GGFM 在 noise 分支分母有 `+1e-8`，ROOT 无 |
| p_sample 方差项 | `(1. - m_nt) ** 2` 等，无 1e-8 | 已改为与 ROOT 一致（仅保留 1e-10 防除零） |
| 训练时 predict_x0_from_objective 的第二个参数 | ROOT 误传 `y_high`（应为 x_low）；仅用于 log_dict，不参与 loss | GGFM 正确传 `x_low` |

**结论**：前向/采样公式已基本一致；ROOT 的 predict 误传只影响 log，不解释结果差距。

---

## 8. 评估阶段

| 项目 | ROOT | GGFM |
|------|------|------|
| 候选起点 | `sampling_from_offline_data(..., type=type_sampling, percentile_sampling=0.2)`；type_sampling=highest 时取排序后最后 128 个 | 同：按 y_train_norm 排序取最高 128（或 config 的 low 路径） |
| 归一化与 task.predict | 一致（oracle_y_min/max 归一化、离散 reshape 等） | 已对齐 |

**结论**：评估流程与 ROOT 对齐良好，主要差距来自训练与 EMA，而非评估逻辑。

---

## 9. 小结：最可能导致「结果差很多」的代码级原因

1. **训练输入池不同（第 1 条）**  
   ROOT 始终用 **best_x（top 1024）** 做 GP 采样；GGFM 用 **50% top2000 + 50% 全量随机**。这会直接改变 (x_low, x_high) 的分布，并影响「从 highest 128 出发」的测试表现。

2. **无 EMA（第 5 条）**  
   ROOT 测试用 EMA 权重，GGFM 用当前权重，会带来稳定的一截差距。

3. **batch / 梯度累积 / 数据量（第 3 条）**  
   val_frac、batch_size=64、accumulate_grad_batches=2 与 GGFM 的「全量 + batch 256、无累积」不同，训练动态不一致。

4. **学习率调度（第 4 条）**  
   ROOT 有 lr_scheduler，GGFM 固定 lr，可能影响收敛曲线和最终点。

---

## 10. 建议

- **若希望在不重做的前提下尽量对齐**：  
  - 把 GGFM 的 GP 输入改为**固定 best_x（top 1024）**，与 ROOT 一致；  
  - 加上 EMA，测试时用 EMA 权重；  
  - 将 batch_size 改为 64，并加上 val_frac 与 ROOT 的 create_train_dataloader 逻辑（或直接复用 ROOT 的 dataloader/数据构造）；  
  - 加上与 ROOT 相同的 lr_scheduler 和（若可能）accumulate_grad_batches。

- **若仍无法复现 ROOT 的结果**：  
  差异点较多且耦合（数据池、dataloader、EMA、scheduler、batch），建议**在仓库内直接复用 ROOT 的数据管线与训练循环**，只替换/保留你需要的模型部分，这样更容易做到「同一套数据、同一套训练流程」下的公平对比，或在此基础上重做一版 GGFM。
