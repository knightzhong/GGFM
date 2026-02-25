import argparse
import os
import time

import design_bench
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.config import Config, load_config
from src.generator import (
    GP,
    sampling_data_from_GP
)
from src.models import DriftNet
from src.flow import train_sde_drift_step, inference_sde
from src.utils import set_seed, Normalizer


def parse_args():
    parser = argparse.ArgumentParser(description="GGFM training (SDE Drift Matching)")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/TfBind8_FlowMatching.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cuda, cpu)",
    )
    return parser.parse_args()


def resolve_config_path(config_path: str) -> str:
    if os.path.isabs(config_path) or os.path.exists(config_path):
        return config_path
    return os.path.join(os.path.dirname(__file__), config_path)


def get_design_bench_data(task_name):
    """
    加载并标准化 Design-Bench 数据，支持离散任务转换。
    返回:
        task: Design-Bench 任务对象
        offline_x_norm: 归一化后的 x，形状 [N, D]
        offline_y_norm: 归一化后的 y，形状 [N]
        mean_x, std_x, mean_y, std_y: 标准化统计量
        logits_shape: 若为离散任务，则为原始 logits 形状 (N, L, V-1)，否则为 None
    """
    print(f"Loading task: {task_name}...")
    if task_name != "TFBind10-Exact-v0":
        task = design_bench.make(task_name)
    else:
        # 显存优化（与 ROOT 一致）
        task = design_bench.make(task_name, dataset_kwargs={"max_samples": 10000})

    offline_x = task.x
    logits_shape = None  # 保存 logits 形状信息

    if task.is_discrete:
        # ROOT 风格：使用 map_to_logits 修改 task 内部状态
        task.map_to_logits()
        offline_x = task.x  # 现在 task.x 已经是 logits 格式 (N, L, V-1)
        logits_shape = offline_x.shape
        offline_x = offline_x.reshape(offline_x.shape[0], -1)  # 展平为 (N, L*(V-1))
        print(
            f"[数据编码] 离散任务：已调用 map_to_logits，"
            f"Logits {logits_shape} -> 展平 {offline_x.shape}"
        )
    else:
        print("[数据编码] 连续任务：直接使用原始数据")

    # 计算统计量（与 ROOT 一致）
    mean_x = np.mean(offline_x, axis=0)
    std_x = np.std(offline_x, axis=0)
    std_x = np.where(std_x == 0, 1.0, std_x)
    offline_x_norm = (offline_x - mean_x) / std_x

    # 处理 Y（与 ROOT 一致）
    offline_y = task.y.reshape(-1)
    mean_y = np.mean(offline_y, axis=0)
    std_y = np.std(offline_y, axis=0)

    # 洗牌数据
    shuffle_idx = np.random.permutation(offline_x.shape[0])
    offline_x_norm = offline_x_norm[shuffle_idx]
    offline_y = offline_y[shuffle_idx]

    # 标准化 Y
    offline_y_norm = (offline_y - mean_y) / std_y

    return task, offline_x_norm, offline_y_norm, mean_x, std_x, mean_y, std_y, logits_shape


class DriftDataset(Dataset):
    """
    监督式 SDE 漂移数据集（K-SB-SDE 粒度）。

    每个样本对应一个二元组 (seed_id, k)，即一整条轨迹：
      - x:           轨迹状态序列 x_0...x_{T-2}      [T-1, D]
      - t:           归一化时间序列                 [T-1, 1]
      - cond:        concat([x_seed, s])            [D+1]
      - drift_label: 漂移标签序列                  [T-1, D]
    """

    def __init__(self, seed_x_array, traj_array, drift_label_array, score_array):
        super().__init__()

        seed_x_array = seed_x_array.astype(np.float32)
        traj_array = traj_array.astype(np.float32)
        drift_label_array = drift_label_array.astype(np.float32)
        score_array = score_array.astype(np.float32)

        self.N_seed, self.K, self.T, self.D = traj_array.shape
        self.M = self.T - 1

        # 每个样本是一整条轨迹
        self.x = torch.from_numpy(traj_array[:, :, : self.M, :])  # [N_seed, K, M, D]
        self.drift_label = torch.from_numpy(drift_label_array)  # [N_seed, K, M, D]

        t_values = np.arange(self.M, dtype=np.float32) / float(self.T - 1)
        self.t = torch.from_numpy(t_values.reshape(1, 1, self.M, 1)).repeat(self.N_seed, self.K, 1, 1)

        # cond = concat([x_seed, s])，其中 s = y_high - y_low
        y_low = score_array[:, :, 0]
        y_high = score_array[:, :, -1]
        delta_y = (y_high - y_low).astype(np.float32)  # [N_seed, K]

        x_seed = seed_x_array[:, None, :].repeat(self.K, axis=1)  # [N_seed, K, D]
        self.cond = torch.from_numpy(np.concatenate([x_seed, delta_y[..., None]], axis=-1))  # [N_seed, K, D+1]

        # 轨迹级权重
        flat_delta_y = delta_y.reshape(-1)
        if flat_delta_y.size > 0:
            delta_tensor = torch.from_numpy(flat_delta_y)
            n_traj = delta_tensor.shape[0]
            sorted_idx = torch.argsort(delta_tensor)
            ranks = torch.zeros_like(sorted_idx, dtype=torch.float32)
            ranks[sorted_idx] = torch.arange(n_traj, dtype=torch.float32)
            norm_ranks = ranks / max(1, n_traj - 1)
            self.weight = torch.softmax(norm_ranks * 3.0, dim=0) * n_traj
        else:
            self.weight = torch.zeros((0,), dtype=torch.float32)

    def __len__(self):
        return self.N_seed * self.K

    def __getitem__(self, idx):
        seed_id = idx // self.K
        k_id = idx % self.K

        # 完整轨迹 [M, D] / [M, 1]
        x_traj = self.x[seed_id, k_id]
        t_traj = self.t[seed_id, k_id]
        drift_traj = self.drift_label[seed_id, k_id]

        # 时间步子采样
        M = self.M
        m_sub = min(Config.DRIFT_SUBSAMPLE_STEPS, M)

        if m_sub <= 0:
            indices = torch.arange(0, M, dtype=torch.long)
        elif Config.DRIFT_SUBSAMPLE_MODE == "strided":
            # 均匀步长采样（不放回）
            step = max(M // m_sub, 1)
            indices = torch.arange(0, M, step, dtype=torch.long)[:m_sub]
        else:
            # 默认随机采样（不放回）
            perm = torch.randperm(M)
            indices = perm[:m_sub]

        x_sub = x_traj[indices]              # [m_sub, D]
        t_sub = t_traj[indices]              # [m_sub, 1]
        drift_label_sub = drift_traj[indices]  # [m_sub, D]

        sample = {
            "x": x_sub,
            "t": t_sub,
            "cond": self.cond[seed_id, k_id],          # [D+1]
            "drift_label": drift_label_sub,
        }
        if self.weight.numel() > 0:
            sample["weight"] = self.weight[idx]
        return sample


def main():
    args = parse_args()
    config_path = resolve_config_path(args.config)
    load_config(config_path)
    if args.seed is not None:
        Config.SEED = args.seed
    if args.device is not None:
        Config.DEVICE = args.device

    # 0. 初始化环境
    print(f"=== GGFM-SDE (Drift Matching) | Task: {Config.TASK_NAME} ===")
    print(f"[Config] Using: {config_path}")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")

    # 1. 加载并编码数据
    (
        task,
        X_train_norm,
        y_train_norm,
        mean_x,
        std_x,
        mean_y,
        std_y,
        logits_shape,
    ) = get_design_bench_data(Config.TASK_NAME)

    # 同步 Normalizer 状态（与原逻辑保持一致）
    x_normalizer = Normalizer(np.zeros((1, X_train_norm.shape[1])))
    x_normalizer.mean, x_normalizer.std, x_normalizer.device = mean_x, std_x, device

    # 2. 转换为 Tensor 供 GP 使用
    X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
    y_train_tensor = torch.FloatTensor(y_train_norm).reshape(-1, 1).to(device)

    # 3. 初始化 GP 超参数
    lengthscale = torch.tensor(Config.GP_INITIAL_LENGTHSCALE, device=device)
    variance = torch.tensor(Config.GP_INITIAL_OUTPUTSCALE, device=device)
    noise = torch.tensor(Config.GP_NOISE, device=device)
    mean_prior = torch.tensor(0.0, device=device)

    # 4. 预先计算采样池索引
    if Config.GP_TYPE_INITIAL_POINTS == "highest":
        best_indices = torch.argsort(y_train_tensor.view(-1))[-1024:]
        best_x = X_train_tensor[best_indices]
        print(
            "[GP Init] Using top 1024 samples for GP sampling "
            "(ROOT style: same samples, different order each epoch)"
        )
    elif Config.GP_TYPE_INITIAL_POINTS == "lowest":
        best_indices = torch.argsort(y_train_tensor.view(-1))[:1024]
        best_x = X_train_tensor[best_indices]
        print("[GP Init] Using bottom 1024 samples for GP sampling")
    else:
        best_x = X_train_tensor
        print("[GP Init] Using all samples for GP sampling")

    all_indices = torch.arange(X_train_tensor.shape[0], device=device)
    top_k_indices = torch.argsort(y_train_tensor.view(-1), descending=True)[:2000]

    # 5. 初始化 DriftNet（SDE 漂移网络）
    input_dim = X_train_norm.shape[1]
    cond_dim = input_dim + 1  # cond = concat([x_seed, s])，其中 s 为标量 Δy
    drift_model = DriftNet(
        input_dim=input_dim,
        cond_dim=cond_dim,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
    ).to(device)
    optimizer = optim.Adam(drift_model.parameters(), lr=Config.FM_LR)

    print(
        f"=== Training: SDE Drift Regression with GP Trajectories "
        f"({Config.FM_EPOCHS} Epochs) ==="
    )
    print(f"每个 Epoch 采样 n_e = {Config.GP_NUM_FUNCTIONS} 个 GP 函数")
    print(f"每个 GP 函数采样 {Config.GP_NUM_POINTS} 个起点")
    print(f"每个 seed 使用 K = {Config.K_TRAJ_PER_SEED} 条轨迹进行监督")

    for epoch in range(Config.FM_EPOCHS):
        epoch_start = time.time()
        print(f"\n=== Epoch {epoch + 1}/{Config.FM_EPOCHS} ===")

        # 构建混合采样池 (Top 高分 + 全局随机)
        num_high = int(Config.GP_NUM_POINTS // 2)
        idx_high = top_k_indices[torch.randperm(len(top_k_indices))[:num_high]]

        num_rand = Config.GP_NUM_POINTS - num_high
        idx_rand = all_indices[torch.randperm(len(all_indices))[:num_rand]]

        mixed_indices = torch.cat([idx_high, idx_rand])
        current_epoch_x = X_train_tensor[mixed_indices]

        # 构建 GP 模型
        gp_init_start = time.time()
        if Config.TASK_NAME == "TFBind8-Exact-v0":
            selected_fit_samples = torch.randperm(X_train_tensor.shape[0])[
                : Config.GP_NUM_FIT_SAMPLES
            ]
            GP_Model = GP(
                device=device,
                x_train=X_train_tensor[selected_fit_samples],
                y_train=y_train_tensor[selected_fit_samples].view(-1),
                lengthscale=lengthscale,
                variance=variance,
                noise=noise,
                mean_prior=mean_prior,
            )
        else:
            GP_Model = GP(
                device=device,
                x_train=X_train_tensor,
                y_train=y_train_tensor.view(-1),
                lengthscale=lengthscale,
                variance=variance,
                noise=noise,
                mean_prior=mean_prior,
            )
        gp_init_time = time.time() - gp_init_start

        # 从 GP 采样若干函数及其轨迹
        sampling_start = time.time()
        (
            seed_x_array,
            traj_array,
            drift_label_array,
            score_array,
        ) = sampling_data_from_GP(
            x_train=current_epoch_x,
            device=device,
            GP_Model=GP_Model,
            num_functions=Config.GP_NUM_FUNCTIONS,
            num_gradient_steps=Config.GP_NUM_GRADIENT_STEPS,
            num_points=Config.GP_NUM_POINTS,
            learning_rate=Config.GP_LEARNING_RATE,
            delta_lengthscale=Config.GP_DELTA_LENGTHSCALE,
            delta_variance=Config.GP_DELTA_VARIANCE,
            seed=epoch,
            threshold_diff=Config.GP_THRESHOLD_DIFF,
            verbose=(epoch == 0),
            return_seed_grouped=True,
            num_steps=Config.GP_TRAJ_STEPS,
            k_traj_per_seed=Config.K_TRAJ_PER_SEED,
        )
        sampling_time = time.time() - sampling_start
        traj_build_time = 0.0

        if seed_x_array.shape[0] == 0:
            print(
                f"Warning: No seed with at least {Config.K_TRAJ_PER_SEED} "
                f"trajectories in epoch {epoch + 1}"
            )
            continue

        print(
            f"  有效 seed 数量: {seed_x_array.shape[0]} | "
            f"K={Config.K_TRAJ_PER_SEED} | 轨迹步数 T={traj_array.shape[2]}"
        )
        print(
            f"  [⏱️ Time] GP初始化: {gp_init_time:.2f}s | "
            f"GP采样: {sampling_time:.2f}s | 轨迹构建: {traj_build_time:.2f}s"
        )

        # 基于 (seed_id, k) 构建训练样本（整条轨迹）
        dataset = DriftDataset(
            seed_x_array=seed_x_array,
            traj_array=traj_array,
            drift_label_array=drift_label_array,
            score_array=score_array,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=Config.FM_BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
            persistent_workers=True,
        )

        # 单 Epoch 训练：drift 回归
        train_start = time.time()
        running_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            batch = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            loss = train_sde_drift_step(drift_model, optimizer, batch, Config)
            running_loss += loss
            num_batches += 1

        train_time = time.time() - train_start
        epoch_total_time = time.time() - epoch_start

        avg_loss = running_loss / max(1, num_batches)
        print(
            f"  [⏱️ Time] 训练: {train_time:.2f}s | Epoch总计: {epoch_total_time:.2f}s | "
            f"平均 Loss: {avg_loss:.4f}"
        )

        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/drift_model_epoch_{epoch + 1}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": drift_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "input_dim": input_dim,
                    "cond_dim": cond_dim,
                },
                checkpoint_path,
            )
            print(f"  [💾 Checkpoint] Saved to {checkpoint_path}")

    # 保存最终模型
    final_model_path = "checkpoints/drift_model_final.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "epoch": Config.FM_EPOCHS,
            "model_state_dict": drift_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_dim": input_dim,
            "cond_dim": cond_dim,
            "hidden_dim": Config.HIDDEN_DIM,
        },
        final_model_path,
    )
    print(f"\n[💾 Final Model] Saved to {final_model_path}")

    # 6. 推理与 SOTA 评估 (Q=128)
    print(f"\n=== SOTA Evaluation (Highest-point, Q={Config.NUM_TEST_SAMPLES}) ===")

    test_q = Config.NUM_TEST_SAMPLES
    sorted_indices = np.argsort(y_train_norm)
    high_indices = sorted_indices[-test_q:]

    X_test_norm = X_train_norm[high_indices]
    y_test_start = y_train_norm[high_indices]

    print(f"Selected {test_q} highest samples as starting points")
    print(
        f"Starting scores (normalized): "
        f"mean={np.mean(y_test_start):.4f}, max={np.max(y_test_start):.4f}"
    )

    # 使用 SDE 进行多样本推理：对每个 seed 采样 NUM_SDE_SAMPLES 条候选
    num_sde_samples = 1 #Config.NUM_SDE_SAMPLES
    steps = Config.SDE_INFERENCE_STEPS

    x0_test = torch.FloatTensor(X_test_norm).to(device)  # [Q, D]
    Q, D = x0_test.shape

    # 为每个 seed 重复 NUM_SDE_SAMPLES 次
    x0_tiled = x0_test.unsqueeze(1).repeat(1, num_sde_samples, 1).reshape(
        Q * num_sde_samples, D
    )  # [Q*K, D]

    # cond = concat([x_seed, s])，这里 s 使用常数 1.0 作为目标提升尺度
    x_seed_tiled = x0_tiled.clone()
    s = torch.ones(Q * num_sde_samples, 1, device=device)
    cond_eval = torch.cat([x_seed_tiled, s], dim=-1)  # [Q*K, D+1]

    with torch.no_grad():
        xT = inference_sde(
            drift_model,
            x0_tiled,
            cond_eval,
            steps=steps,
            sigma_max=Config.SDE_SIGMA_MAX,
            sigma_min=Config.SDE_SIGMA_MIN,
            device=device,
        )  # [Q*K, D]

    opt_X_norm_all = xT.cpu().numpy().reshape(Q, num_sde_samples, D)  # [Q, K, D]

    # 反标准化
    opt_X_denorm_all = opt_X_norm_all * std_x[None, None, :] + mean_x[None, None, :]

    # 还原形状供 task.predict 打分
    if task.is_discrete and logits_shape is not None:
        # 候选 reshape 为 (Q*K, L, V-1)
        candidates_for_predict = opt_X_denorm_all.reshape(
            Q * num_sde_samples, logits_shape[1], logits_shape[2]
        )
        original_X_denorm = X_test_norm * std_x + mean_x
        original_X_for_predict = original_X_denorm.reshape(
            test_q, logits_shape[1], logits_shape[2]
        )
        print(
            f"[Discrete Task] Reshaped candidates to Logits format: "
            f"{candidates_for_predict.shape}"
        )
    else:
        candidates_for_predict = opt_X_denorm_all.reshape(Q * num_sde_samples, D)
        original_X_for_predict = X_test_norm * std_x + mean_x

    # 使用 Oracle 对所有候选打分，并为每个 seed 选出最佳样本
    all_candidate_scores = task.predict(candidates_for_predict).flatten()  # [Q*K]
    all_candidate_scores = all_candidate_scores.reshape(Q, num_sde_samples)  # [Q, K]
    best_idx = np.argmax(all_candidate_scores, axis=1)  # [Q]
    final_scores = all_candidate_scores[np.arange(Q), best_idx]  # [Q]

    # 同时保留原始起点分数
    original_scores = task.predict(original_X_for_predict).flatten()

    # 与 ROOT 一致的归一化评估
    task_to_min = {
        "TFBind8-Exact-v0": 0.0,
        "TFBind10-Exact-v0": -1.8585268,
        "AntMorphology-Exact-v0": -386.90036,
        "DKittyMorphology-Exact-v0": -880.4585,
    }
    task_to_max = {
        "TFBind8-Exact-v0": 1.0,
        "TFBind10-Exact-v0": 2.1287067,
        "AntMorphology-Exact-v0": 590.24445,
        "DKittyMorphology-Exact-v0": 340.90985,
    }

    oracle_y_min = task_to_min[Config.TASK_NAME]
    oracle_y_max = task_to_max[Config.TASK_NAME]

    final_score_normalized = (
        torch.from_numpy(final_scores) - oracle_y_min
    ) / (oracle_y_max - oracle_y_min)
    original_score_normalized = (
        torch.from_numpy(original_scores) - oracle_y_min
    ) / (oracle_y_max - oracle_y_min)

    percentiles = torch.quantile(
        final_score_normalized, torch.tensor([1.0, 0.8, 0.5]), interpolation="higher"
    )
    p100_score = percentiles[0].item()
    p80_score = percentiles[1].item()
    p50_score = percentiles[2].item()

    final_scores_sorted = np.sort(final_scores)
    print("\n[Result] Final scores distribution (raw):")
    print(f"  Min: {final_scores_sorted[0]:.4f}")
    print(f"  Max: {final_scores_sorted[-1]:.4f}")
    print(f"  Mean: {np.mean(final_scores):.4f}")
    print(f"  Std: {np.std(final_scores):.4f}")

    final_score_normalized_np = final_score_normalized.numpy()
    print("\n[Result] Normalized scores distribution (comparable with ROOT):")
    print(f"  Min: {np.min(final_score_normalized_np):.4f}")
    print(f"  Max: {np.max(final_score_normalized_np):.4f}")
    print(f"  Mean: {np.mean(final_score_normalized_np):.4f}")
    print(f"  Std: {np.std(final_score_normalized_np):.4f}")

    print("-" * 60)
    print(f"Original Mean (Top {test_q}, raw): {np.mean(original_scores):.4f}")
    print(f"Optimized Mean (Top {test_q}, raw): {np.mean(final_scores):.4f}")
    print(
        f"Improvement (raw):                   "
        f"{np.mean(final_scores) - np.mean(original_scores):.4f}"
    )
    print("-" * 60)
    print(f"Normalized 100th Percentile (Max):      {p100_score:.6f}")
    print(f"Normalized 80th Percentile:             {p80_score:.6f}")
    print(f"Normalized 50th Percentile (Median):    {p50_score:.6f}")
    print("-" * 60)
    print("[ROOT Comparable] These normalized percentiles are directly comparable with ROOT results")


if __name__ == "__main__":
    main()

