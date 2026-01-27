# main.py
import argparse
import design_bench
import torch
import torch.optim as optim
import numpy as np

from src.config import Config, load_config
from src.utils import set_seed, Normalizer
from src.oracle import NTKOracle
from src.generator import GP, sampling_data_from_GP, generate_trajectories_from_GP_samples,RFFGP
from src.models import VectorFieldNet
from src.flow import train_cfm_step,train_cfm, inference_ode
import time
import os


def parse_args():
    parser = argparse.ArgumentParser(description="GGFM training")
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


def resolve_config_path(config_path):
    if os.path.isabs(config_path) or os.path.exists(config_path):
        return config_path
    return os.path.join(os.path.dirname(__file__), config_path)

def get_design_bench_data(task_name):
    """
    加载并标准化 Design-Bench 数据，支持离散任务转换
    完全对齐 ROOT 的处理方式
    """
    print(f"Loading task: {task_name}...")
    if task_name != 'TFBind10-Exact-v0':
        task = design_bench.make(task_name)
    else:
        # 显存优化（与 ROOT 一致）
        task = design_bench.make(task_name, dataset_kwargs={"max_samples": 10000})
    
    offline_x = task.x
    logits_shape = None  # 保存 logits 形状信息
    
    if task.is_discrete:
        # ROOT 风格：使用 map_to_logits 修改 task 内部状态
        # 这样 task.predict() 才能正确处理 logits 格式的数据
        task.map_to_logits()
        offline_x = task.x  # 现在 task.x 已经是 logits 格式 (N, L, V-1)
        logits_shape = offline_x.shape  # 保存形状 (N, L, V-1)
        offline_x = offline_x.reshape(offline_x.shape[0], -1)  # 展平为 (N, L*(V-1))
        print(f"[数据编码] 离散任务：已调用 map_to_logits，Logits {logits_shape} -> 展平 {offline_x.shape}")
    else:
        print("[数据编码] 连续任务：直接使用原始数据")
    
    # 计算统计量（与 ROOT 完全一致）
    mean_x = np.mean(offline_x, axis=0)
    std_x = np.std(offline_x, axis=0)
    std_x = np.where(std_x == 0, 1.0, std_x)  # ROOT 使用 == 0，不是 < 1e-6
    offline_x_norm = (offline_x - mean_x) / std_x
    
    # 处理 Y（与 ROOT 一致）
    offline_y = task.y.reshape(-1)  # ROOT 使用 reshape(-1)，不是 reshape(-1, 1)
    mean_y = np.mean(offline_y, axis=0)
    std_y = np.std(offline_y, axis=0)
    
    # 洗牌数据（与 ROOT 一致）
    shuffle_idx = np.random.permutation(offline_x.shape[0])
    offline_x_norm = offline_x_norm[shuffle_idx]
    offline_y = offline_y[shuffle_idx]
    
    # 标准化 Y
    offline_y_norm = (offline_y - mean_y) / std_y
    
    return task, offline_x_norm, offline_y_norm, mean_x, std_x, mean_y, std_y, logits_shape

# main.py 预处理逻辑
# def preprocess_trajectories(oracle, X_train_norm):
#     device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
#     if os.path.exists(Config.TRAJECTORY_PATH):
#         print(f"Loading cached trajectories from {Config.TRAJECTORY_PATH}")
#         return np.load(Config.TRAJECTORY_PATH)['trajs']

#     print("=== Generating ALL long trajectories (GD reverse + GA) ===")
#     all_indices = np.arange(len(X_train_norm))
#     # 按照你的要求：用全量数据，或者至少 10000 条
#     sample_size = len(all_indices)
#     selected_idx = all_indices
#     # selected_idx = np.random.choice(all_indices, sample_size, replace=False)
    
#     all_valid = []
#     batch_size = 256
#     for i in range(0, sample_size, batch_size):
#         batch_x = X_train_norm[selected_idx[i : i + batch_size]]
#         trajs = generate_long_trajectories(oracle, batch_x, device)
#         all_valid.append(trajs)
#         print(f"Progress: {i + batch_size}/{sample_size}")
        
#     pool = np.concatenate(all_valid, axis=0)
#     np.savez_compressed(Config.TRAJECTORY_PATH, trajs=pool)
#     return pool

def main():
    args = parse_args()
    config_path = resolve_config_path(args.config)
    load_config(config_path)
    if args.seed is not None:
        Config.SEED = args.seed
    if args.device is not None:
        Config.DEVICE = args.device

    # 0. 初始化环境
    print(f"=== GGFM with ROOT GP Sampling: {Config.TASK_NAME} ===")
    print(f"[Config] Using: {config_path}")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    
    # 1. 加载并编码数据（完全对齐 ROOT 的处理方式）
    task, X_train_norm, y_train_norm, mean_x, std_x, mean_y, std_y, logits_shape = get_design_bench_data(Config.TASK_NAME)
    
    # 同步 Normalizer 状态
    x_normalizer = Normalizer(np.zeros((1, X_train_norm.shape[1])))
    x_normalizer.mean, x_normalizer.std, x_normalizer.device = mean_x, std_x, device
    
    # 2. 转换为 Tensor 供 GP 使用（与 ROOT 一致）
    X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
    # y_train_norm 现在是 (N,) 形状，需要转换为 (N, 1) 供 Oracle 使用
    y_train_tensor = torch.FloatTensor(y_train_norm).reshape(-1, 1).to(device)
    
    # 保存原始统计量（用于反标准化）
    mean_x_torch = torch.FloatTensor(mean_x).to(device)
    std_x_torch = torch.FloatTensor(std_x).to(device)
    mean_y_torch = torch.FloatTensor([mean_y]).to(device)
    std_y_torch = torch.FloatTensor([std_y]).to(device)
    
    # 3. 初始化 GP 超参数
    lengthscale = torch.tensor(Config.GP_INITIAL_LENGTHSCALE, device=device)
    variance = torch.tensor(Config.GP_INITIAL_OUTPUTSCALE, device=device)
    noise = torch.tensor(Config.GP_NOISE, device=device)
    mean_prior = torch.tensor(0.0, device=device)
    
    # 4. 选择用于 GP 拟合的初始点（完全对齐 ROOT）
    if Config.GP_TYPE_INITIAL_POINTS == 'highest':
        # ROOT: 固定 1024 个样本，每次全选但顺序不同
        best_indices = torch.argsort(y_train_tensor.view(-1))[-1024:]
        best_x = X_train_tensor[best_indices]
        print(f"[GP Init] Using top 1024 samples for GP sampling (ROOT style: same samples, different order each epoch)")
    elif Config.GP_TYPE_INITIAL_POINTS == 'lowest':
        best_indices = torch.argsort(y_train_tensor.view(-1))[:1024]
        best_x = X_train_tensor[best_indices]
        print(f"[GP Init] Using bottom 1024 samples for GP sampling")
    else:
        best_x = X_train_tensor
        print(f"[GP Init] Using all samples for GP sampling")
    # 1. 预先准备好全量索引 (在循环外)
    all_indices = torch.arange(X_train_tensor.shape[0], device=device)
    # 预先计算好高分索引 (Top 2000 作为一个池子，防止只盯着 Top 1024 过拟合)
    top_k_indices = torch.argsort(y_train_tensor.view(-1), descending=True)[:2000]
    
    # 5. 初始化 Flow Matching 网络
    input_dim = X_train_norm.shape[1]
    cfm_model = VectorFieldNet(
        input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
    ).to(device)
    optimizer = optim.Adam(cfm_model.parameters(), lr=Config.FM_LR)
    
    # 🔧 添加学习率调度器，防止后期学习率过大
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.FM_EPOCHS, eta_min=1e-5)

    # --- 核心修改：每个 Epoch 动态采样 GP 函数生成轨迹 ---
    print(f"=== Training: Dynamic GP Sampling ({Config.FM_EPOCHS} Epochs) ===")
    print(f"每个 Epoch 采样 n_e = {Config.GP_NUM_FUNCTIONS} 个 GP 函数")
    print(f"每个 GP 函数采样 {Config.GP_NUM_POINTS} 个配对")
    print(f"总计将生成约 {Config.GP_NUM_FUNCTIONS} × {Config.FM_EPOCHS} = {Config.GP_NUM_FUNCTIONS * Config.FM_EPOCHS} 个 GP 函数")

    # y_train_norm 已经是 (N,) 形状了，不需要 flatten
    y_scores_flat = y_train_norm

    for epoch in range(Config.FM_EPOCHS):
        # 每个 Epoch 重新采样具有不同超参数的 GP
        epoch_start = time.time()  # 记录 epoch 开始时间
        print(f"\n=== Epoch {epoch+1}/{Config.FM_EPOCHS} ===")

        # === 核心修改：构建混合采样池 (Mix 50/50) ===
        # A. 50% 来自 Top 高分 (保证上限)
        num_high = int(Config.GP_NUM_POINTS // 2)  # 例如 512
        # 从 Top 2000 里随机选 512 个，增加一点变异性
        idx_high = top_k_indices[torch.randperm(len(top_k_indices))[:num_high]]
        
        # B. 50% 来自全局随机 (保证中位数和泛化)
        num_rand = Config.GP_NUM_POINTS - num_high
        idx_rand = all_indices[torch.randperm(len(all_indices))[:num_rand]]
        
        # C. 合并
        mixed_indices = torch.cat([idx_high, idx_rand])
        current_epoch_x = X_train_tensor[mixed_indices]
        
        # -------------------------------------------------
        
        # 构建 GP 模型（TFBind8 使用部分样本，与 ROOT 一致）
        gp_init_start = time.time()
        # if Config.TASK_NAME == 'TFBind8-Exact-v0':
        #     selected_fit_samples = torch.randperm(X_train_tensor.shape[0])[:Config.GP_NUM_FIT_SAMPLES]
        #     GP_Model = GP(
        #         device=device,
        #         x_train=X_train_tensor[selected_fit_samples],
        #         y_train=y_train_tensor[selected_fit_samples].view(-1),  # 确保是 (N,) 形状
        #         lengthscale=lengthscale,
        #         variance=variance,
        #         noise=noise,
        #         mean_prior=mean_prior
        #     )
        # else:
        # GP_Model = GP(
        #     device=device,
        #     x_train=X_train_tensor,
        #     y_train=y_train_tensor.view(-1),  # 确保是 (N,) 形状
        #     lengthscale=lengthscale,
        #     variance=variance,
        #     noise=noise,
        #     mean_prior=mean_prior
        # )
        GP_Model = RFFGP(
                device=device,
                x_train=X_train_tensor, 
                y_train=y_train_tensor.view(-1),  # 确保是 (N,) 形状, 
                lengthscale=lengthscale, 
                variance=variance, 
                noise=noise,
                num_features=2048 
            )
        gp_init_time = time.time() - gp_init_start
        
        # 从 GP 采样 n_e = 8 个函数，每个函数生成 num_points 个配对
        # best_x = torch.randperm(X_train_tensor.shape[0])[:1024]
        sampling_start = time.time()
        data_from_GP = sampling_data_from_GP(
            x_train=current_epoch_x,
            device=device,
            GP_Model=GP_Model,
            num_functions=Config.GP_NUM_FUNCTIONS,
            num_gradient_steps=Config.GP_NUM_GRADIENT_STEPS,
            num_points=Config.GP_NUM_POINTS,
            learning_rate=Config.GP_LEARNING_RATE,
            delta_lengthscale=Config.GP_DELTA_LENGTHSCALE,
            delta_variance=Config.GP_DELTA_VARIANCE,
            seed=epoch,  # 使用 epoch 作为随机种子，确保每个 epoch 不同
            threshold_diff=Config.GP_THRESHOLD_DIFF,
            uncertainty_penalty=Config.GP_UNCERTAINTY_PENALTY,
            uncertainty_interval=Config.GP_UNCERTAINTY_INTERVAL,
            max_end_uncertainty=Config.GP_MAX_END_UNCERTAINTY,
            verbose=(epoch == 0)  # 第一个 epoch 显示详细计时
        )
        sampling_time = time.time() - sampling_start
        
        # 从 GP 采样结果生成轨迹
        traj_gen_start = time.time()
        trajs_array, scores_array = generate_trajectories_from_GP_samples(
            data_from_GP,
            device=device,
            num_steps=Config.GP_TRAJ_STEPS
        )
        # # 2. 计算动态 Softmax 权重 (Dynamic Softmax Weighting)
        # if len(scores_array) > 0:
        #     # 转为 Tensor
        #     batch_scores = torch.FloatTensor(scores_array).to(device)
            
        #     # === 关键步骤 1: Z-Score 标准化 (让分数分布对齐，不依赖绝对值) ===
        #     # 这样无论是 [0.2, 0.4] 还是 [0.8, 1.0]，相对差距都能被正确捕捉
        #     scores_mean = batch_scores.mean()
        #     scores_std = batch_scores.std() + 1e-6  # 防止除以 0
        #     z_scores = (batch_scores - scores_mean) / scores_std
            
        #     # === 关键步骤 2: 温度系数 k (Temperature) ===
        #     # k = 1.0 : 温和加权，保留低分样本的学习信号 (推荐起点)
        #     # k = 2.0 : 激进加权，强力冲击 SOTA
        #     # k > 5.0 : 极度贪婪，可能导致 Median 下降 (近似 Argmax)
        #     # 建议先设为 2.0，既能拉开差距，又不会把低分样本权重杀到 0
        #     k = 2.0 
        #     if Config.TASK_NAME == 'TFBind10-Exact-v0':
        #         k = 0.5
            
        #     # 计算 Softmax
        #     weights_softmax = torch.softmax(z_scores * k, dim=0)
            
        #     # === 关键步骤 3: 重缩放 (Rescaling) ===
        #     # Softmax 的和是 1，我们需要 和 = N (即平均权重为 1)
        #     # 否则梯度会消失
        #     batch_size = len(batch_scores)
        #     weights = weights_softmax * batch_size
            
        #     # 转回 numpy 传给训练函数
        #     weights_np = weights.cpu().numpy()
            
        #     # (可选) 打印一下权重的极值，看看是不是太极端
        #     if epoch % 10 == 0:
        #         print(f"  [Weight Debug] Min: {weights.min().item():.4f} | Max: {weights.max().item():.4f} | Mean: {weights.mean().item():.4f}")

        # else:
        #     weights_np = None
        if len(scores_array) > 0:
            batch_scores = torch.FloatTensor(scores_array).to(device)
            N = len(batch_scores)
            
            # === Rank-Based Weighting (基于排名的加权) ===
            # 1. 获取排名 (argsort 两次可以得到每个元素的排名索引)
            sorted_indices = torch.argsort(batch_scores)
            ranks = torch.zeros_like(sorted_indices, dtype=torch.float, device=device)
            ranks[sorted_indices] = torch.arange(N, device=device, dtype=torch.float)
            
            # 2. 归一化排名到 [0, 1] 区间
            normalized_ranks = ranks / (N - 1)  # Range: [0.0, 1.0]
            
            # 3. 带温度的 Softmax (k 控制贪婪程度)
            k = 3.0 
            weights_softmax = torch.softmax(normalized_ranks * k, dim=0)
            
            # 4. 重缩放 (保持 Sum = N)
            weights = weights_softmax * N
            weights_np = weights.cpu().numpy()
            
            if epoch % 10 == 0:
                print(f"  [Rank Weight] Min: {weights.min().item():.4f} | Max: {weights.max().item():.4f}")

        else:
            weights_np = None
        traj_gen_time = time.time() - traj_gen_start
        
        if len(trajs_array) == 0:
            print(f"Warning: No valid trajectories generated in epoch {epoch+1}")
            continue
        
        print(f"Generated {len(trajs_array)} trajectories from GP samples")
        print(f"  [⏱️ Time] GP初始化: {gp_init_time:.2f}s | GP采样: {sampling_time:.2f}s | 轨迹生成: {traj_gen_time:.2f}s")
        
        # 对这批轨迹进行流匹配训练更新 (Rank-Based Weighting)
        train_start = time.time()
        avg_loss = train_cfm_step(cfm_model, trajs_array, optimizer, device, weights=weights_np)
        train_time = time.time() - train_start
        
        epoch_total_time = time.time() - epoch_start
        
        # 🔧 更新学习率
        # scheduler.step()
        
        print(f"  [⏱️ Time] 训练: {train_time:.2f}s | Epoch总计: {epoch_total_time:.2f}s")
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{Config.FM_EPOCHS} | Loss: {avg_loss:.4f} | Trajs: {len(trajs_array)}")
            # 保存检查点
            checkpoint_path = f"checkpoints/cfm_model_epoch_{epoch+1}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': cfm_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  [💾 Checkpoint] Saved to {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = "checkpoints/cfm_model_final.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'epoch': Config.FM_EPOCHS,
        'model_state_dict': cfm_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': Config.HIDDEN_DIM,
    }, final_model_path)
    print(f"\n[💾 Final Model] Saved to {final_model_path}")

    # 4. 推理与 SOTA 评估 (Q=128)（完全对齐 ROOT 的测试逻辑）
    print(f"\n=== SOTA Evaluation (Highest-point, Q=128) ===")
    
    # 对齐 ROOT：从得分最高的 128 个标准化样本出发
    test_q = Config.NUM_TEST_SAMPLES
    
    # 使用标准化后的 y 来选择高分样本（与 ROOT 一致）
    # y_train_norm 是 numpy array，所以这里的索引都是 numpy
    sorted_indices = np.argsort(y_train_norm)
    high_indices = sorted_indices[-test_q:]
    
    # 获取标准化的高分样本作为起点
    X_test_norm = X_train_norm[high_indices]
    y_test_start = y_train_norm[high_indices]
    
    print(f"Selected {test_q} highest samples as starting points")
    print(f"Starting scores (normalized): mean={np.mean(y_test_start):.4f}, max={np.max(y_test_start):.4f}")
    
    # ODE 推理 (使用 Velocity Scaling 进行外推加速)
    # velocity_scale > 1.0: 利用 Rank Weighting 训练的高质量方向进行外推
    opt_X_norm = inference_ode(cfm_model, X_test_norm, device, velocity_scale=1.5)
    
    # 反标准化（与 ROOT 一致）
    opt_X_denorm = opt_X_norm * std_x + mean_x
    
    # 还原形状供 task.predict 打分（与 ROOT 一致）
    if task.is_discrete and logits_shape is not None:
        # 离散任务：需要 reshape 回 (N, L, V-1) 的形状
        # 使用数据加载时保存的 logits_shape 信息
        opt_X_for_predict = opt_X_denorm.reshape(test_q, logits_shape[1], logits_shape[2])
        # 原始样本也需要相同处理
        original_X_denorm = X_test_norm * std_x + mean_x
        original_X_for_predict = original_X_denorm.reshape(test_q, logits_shape[1], logits_shape[2])
        
        print(f"[Discrete Task] Reshaped to Logits format: {opt_X_for_predict.shape}")
    else:
        # 连续任务：直接使用
        opt_X_for_predict = opt_X_denorm
        original_X_for_predict = X_test_norm * std_x + mean_x
    
    # 使用 Oracle 评估（与 ROOT 一致，直接传入 numpy array）
    final_scores = task.predict(opt_X_for_predict).flatten()
    original_scores = task.predict(original_X_for_predict).flatten()
    
    # 与 ROOT 完全对齐：使用相同的 task_to_min, task_to_max, task_to_best 字典
    task_to_min = {'TFBind8-Exact-v0': 0.0, 'TFBind10-Exact-v0': -1.8585268, 'AntMorphology-Exact-v0': -386.90036, 'DKittyMorphology-Exact-v0': -880.4585}
    task_to_max = {'TFBind8-Exact-v0': 1.0, 'TFBind10-Exact-v0': 2.1287067, 'AntMorphology-Exact-v0': 590.24445, 'DKittyMorphology-Exact-v0': 340.90985}
    task_to_best = {'TFBind8-Exact-v0': 0.43929616, 'TFBind10-Exact-v0': 0.005328223, 'AntMorphology-Exact-v0': 165.32648, 'DKittyMorphology-Exact-v0': 199.36252}
    
    # 获取任务的 min 和 max 值（与 ROOT 一致）
    oracle_y_min = task_to_min[Config.TASK_NAME]
    oracle_y_max = task_to_max[Config.TASK_NAME]
    
    # 计算标准化分数（与 ROOT 完全一致）
    # ROOT 使用公式: final_score = (high_true_scores - oracle_y_min) / (oracle_y_max - oracle_y_min)
    final_score_normalized = (torch.from_numpy(final_scores) - oracle_y_min) / (oracle_y_max - oracle_y_min)
    original_score_normalized = (torch.from_numpy(original_scores) - oracle_y_min) / (oracle_y_max - oracle_y_min)
    
    # 计算百分位数（与 ROOT 完全一致：使用标准化后的分数）
    percentiles = torch.quantile(final_score_normalized, torch.tensor([1.0, 0.8, 0.5]), interpolation='higher')
    p100_score = percentiles[0].item()
    p80_score = percentiles[1].item()
    p50_score = percentiles[2].item()
    
    # 打印原始分数分布（用于调试）
    final_scores_sorted = np.sort(final_scores)
    print(f"\n[Result] Final scores distribution (raw):")
    print(f"  Min: {final_scores_sorted[0]:.4f}")
    print(f"  Max: {final_scores_sorted[-1]:.4f}")
    print(f"  Mean: {np.mean(final_scores):.4f}")
    print(f"  Std: {np.std(final_scores):.4f}")
    
    # 打印标准化后的分数分布
    final_score_normalized_np = final_score_normalized.numpy()
    print(f"\n[Result] Normalized scores distribution (comparable with ROOT):")
    print(f"  Min: {np.min(final_score_normalized_np):.4f}")
    print(f"  Max: {np.max(final_score_normalized_np):.4f}")
    print(f"  Mean: {np.mean(final_score_normalized_np):.4f}")
    print(f"  Std: {np.std(final_score_normalized_np):.4f}")
    
    print("-" * 60)
    print(f"Original Mean (Top {test_q}, raw): {np.mean(original_scores):.4f}")
    print(f"Optimized Mean (Top {test_q}, raw): {np.mean(final_scores):.4f}")
    print(f"Improvement (raw):                   {np.mean(final_scores) - np.mean(original_scores):.4f}")
    print("-" * 60)
    print(f"Normalized 100th Percentile (Max):      {p100_score:.6f}")
    print(f"Normalized 80th Percentile:             {p80_score:.6f}")
    print(f"Normalized 50th Percentile (Median):    {p50_score:.6f}")
    print("-" * 60)
    print(f"[ROOT Comparable] These normalized percentiles are directly comparable with ROOT results")

if __name__ == "__main__":
    main()


#TF10 0.685  0.526  0.473
#Ant 0.965 0.847 0.712
#Dkitty 0.972 0.938 0.919
#TF8  0.986   0.839 0.675