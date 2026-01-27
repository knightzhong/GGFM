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
    
    # 1. 加载并编码数据
    task, X_train_norm, y_train_norm, mean_x, std_x, mean_y, std_y, logits_shape = get_design_bench_data(Config.TASK_NAME)
    
    # 同步 Normalizer 状态
    x_normalizer = Normalizer(np.zeros((1, X_train_norm.shape[1])))
    x_normalizer.mean, x_normalizer.std, x_normalizer.device = mean_x, std_x, device
    
    # 2. 转换为 Tensor
    X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
    y_train_tensor = torch.FloatTensor(y_train_norm).reshape(-1, 1).to(device)
    
    # 3. 初始化 GP 超参数
    lengthscale = torch.tensor(Config.GP_INITIAL_LENGTHSCALE, device=device)
    variance = torch.tensor(Config.GP_INITIAL_OUTPUTSCALE, device=device)
    noise = torch.tensor(Config.GP_NOISE, device=device)
    
    # 预先准备好全量索引
    all_indices = torch.arange(X_train_tensor.shape[0], device=device)
    top_k_indices = torch.argsort(y_train_tensor.view(-1), descending=True)[:2000]
    
    # 4. 初始化 GP 模型 (固定参数，仅初始化一次)
    print(f"[GP Init] Fitting GP once with fixed parameters...")
    GP_Model = GP(
        device=device,
        x_train=X_train_tensor, 
        y_train=y_train_tensor.view(-1),
        lengthscale=lengthscale, 
        variance=variance, 
        noise=noise,
        mean_prior=torch.tensor(0.0, device=device)
    )
    # 计算 Cholesky 分解并固定参数
    GP_Model.set_hyper(lengthscale, variance)

    # --- [新增] 在训练开始前，一次性采样多个 GP functions ---
    K = Config.GP_NUM_FUNCTIONS
    fixed_gp_functions = GP_Model.sample_functions(
        num_functions=K,
        seed=Config.SEED
    )
    print(f"[GP] Sampled {K} fixed GP functions for FM training.")

    # 5. 初始化 Flow Matching 网络
    input_dim = X_train_norm.shape[1]
    cfm_model = VectorFieldNet(
        input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
    ).to(device)
    optimizer = optim.Adam(cfm_model.parameters(), lr=Config.FM_LR)
    
    # --- 核心修改：使用 Langevin 动力学采样生成轨迹 ---
    print(f"=== Training: Langevin GP Sampling ({Config.FM_EPOCHS} Epochs) ===")
    print(f"每个 Epoch 采样 n_e = 1 个固定 GP 函数集合 (K={Config.GP_NUM_FUNCTIONS})")
    print(f"每个 Epoch 尝试生成 {Config.GP_NUM_POINTS} 个 Langevin 轨迹")

    for epoch in range(Config.FM_EPOCHS):
        epoch_start = time.time()
        print(f"\n=== Epoch {epoch+1}/{Config.FM_EPOCHS} ===")

        # 构建混合采样池
        num_high = int(Config.GP_NUM_POINTS // 2)
        idx_high = top_k_indices[torch.randperm(len(top_k_indices))[:num_high]]
        num_rand = Config.GP_NUM_POINTS - num_high
        idx_rand = all_indices[torch.randperm(len(all_indices))[:num_rand]]
        mixed_indices = torch.cat([idx_high, idx_rand])
        current_epoch_x = X_train_tensor[mixed_indices]
        
        # Langevin 采样
        sampling_start = time.time()
        data_from_GP = sampling_data_from_GP(
            x_train=current_epoch_x,
            device=device,
            GP_Model=GP_Model,
            gp_functions=fixed_gp_functions, # 使用固定函数集合
            num_gradient_steps=Config.GP_NUM_GRADIENT_STEPS,
            num_points=Config.GP_NUM_POINTS,
            eta_min=Config.GP_ETA_MIN,
            eta_max=Config.GP_ETA_MAX,
            sigma_max=Config.GP_SIGMA_MAX,
            threshold_diff=Config.GP_THRESHOLD_DIFF,
            uncertainty_penalty=Config.GP_UNCERTAINTY_PENALTY,
            uncertainty_interval=Config.GP_UNCERTAINTY_INTERVAL,
            max_end_uncertainty=Config.GP_MAX_END_UNCERTAINTY,
            verbose=True # 开启详细诊断信息
        )
        sampling_time = time.time() - sampling_start
        
        # 从 GP 采样结果生成轨迹
        traj_gen_start = time.time()
        trajs_array, scores_array = generate_trajectories_from_GP_samples(
            data_from_GP,
            device=device,
            num_steps=Config.GP_TRAJ_STEPS
        )
        
        # Rank-Based Weighting
        if len(scores_array) > 0:
            batch_scores = torch.FloatTensor(scores_array).to(device)
            N = len(batch_scores)
            sorted_indices = torch.argsort(batch_scores)
            ranks = torch.zeros_like(sorted_indices, dtype=torch.float, device=device)
            ranks[sorted_indices] = torch.arange(N, device=device, dtype=torch.float)
            normalized_ranks = ranks / (N - 1)
            k = 3.0 
            weights_softmax = torch.softmax(normalized_ranks * k, dim=0)
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
        print(f"  [⏱️ Time] GP采样: {sampling_time:.2f}s | 轨迹生成: {traj_gen_time:.2f}s")
        
        # 训练更新
        train_start = time.time()
        avg_loss = train_cfm_step(cfm_model, trajs_array, optimizer, device, weights=weights_np)
        train_time = time.time() - train_start
        epoch_total_time = time.time() - epoch_start
        
        print(f"  [⏱️ Time] 训练: {train_time:.2f}s | Epoch总计: {epoch_total_time:.2f}s")
        print(f"Epoch {epoch+1}/{Config.FM_EPOCHS} | Loss: {avg_loss:.4f} | Trajs: {len(trajs_array)}")

        
        if (epoch + 1) % 10 == 0:
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

    # 推理与 SOTA 评估
    print(f"\n=== SOTA Evaluation (Highest-point, Q=128) ===")
    test_q = Config.NUM_TEST_SAMPLES
    sorted_indices = np.argsort(y_train_norm)
    high_indices = sorted_indices[-test_q:]
    X_test_norm = X_train_norm[high_indices]
    y_test_start = y_train_norm[high_indices]
    
    print(f"Selected {test_q} highest samples as starting points")
    print(f"Starting scores (normalized): mean={np.mean(y_test_start):.4f}, max={np.max(y_test_start):.4f}")
    
    opt_X_norm = inference_ode(cfm_model, X_test_norm, device, velocity_scale=1.5)
    opt_X_denorm = opt_X_norm * std_x + mean_x
    
    if task.is_discrete and logits_shape is not None:
        opt_X_for_predict = opt_X_denorm.reshape(test_q, logits_shape[1], logits_shape[2])
        original_X_denorm = X_test_norm * std_x + mean_x
        original_X_for_predict = original_X_denorm.reshape(test_q, logits_shape[1], logits_shape[2])
    else:
        opt_X_for_predict = opt_X_denorm
        original_X_for_predict = X_test_norm * std_x + mean_x
    
    final_scores = task.predict(opt_X_for_predict).flatten()
    original_scores = task.predict(original_X_for_predict).flatten()
    
    task_to_min = {'TFBind8-Exact-v0': 0.0, 'TFBind10-Exact-v0': -1.8585268, 'AntMorphology-Exact-v0': -386.90036, 'DKittyMorphology-Exact-v0': -880.4585}
    task_to_max = {'TFBind8-Exact-v0': 1.0, 'TFBind10-Exact-v0': 2.1287067, 'AntMorphology-Exact-v0': 590.24445, 'DKittyMorphology-Exact-v0': 340.90985}
    
    oracle_y_min = task_to_min[Config.TASK_NAME]
    oracle_y_max = task_to_max[Config.TASK_NAME]
    
    final_score_normalized = (torch.from_numpy(final_scores) - oracle_y_min) / (oracle_y_max - oracle_y_min)
    percentiles = torch.quantile(final_score_normalized, torch.tensor([1.0, 0.8, 0.5]), interpolation='higher')
    
    print(f"\n[Result] Final scores distribution (raw):")
    print(f"  Max: {np.max(final_scores):.4f} | Mean: {np.mean(final_scores):.4f}")
    print("-" * 60)
    print(f"Normalized 100th Percentile (Max):      {percentiles[0].item():.6f}")
    print(f"Normalized 80th Percentile:             {percentiles[1].item():.6f}")
    print(f"Normalized 50th Percentile (Median):    {percentiles[2].item():.6f}")
    print("-" * 60)

if __name__ == "__main__":
    main()
