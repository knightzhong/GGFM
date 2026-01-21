# main.py
import design_bench
import torch
import torch.optim as optim
import numpy as np

from src.config import Config
from src.utils import set_seed, Normalizer
from src.oracle import NTKOracle
from src.generator import generate_long_trajectories
from src.models import VectorFieldNet
from src.flow import train_cfm_step,train_cfm, inference_ode

import os

def get_design_bench_data(task_name):
    """
    加载并标准化 Design-Bench 数据，支持离散任务转换
    """
    print(f"Loading task: {task_name}...")
    if task_name != 'TFBind10-Exact-v0':
        task = design_bench.make(task_name)
    else:
        # 显存优化
        task = design_bench.make(task_name, dataset_kwargs={"max_samples": 10000})
    
    if task.is_discrete:
        # 1. 处理离散编码：转换为 (N, L, V-1) 的 Logits
        task.map_to_logits()
        print("[数据编码] 使用 ROOT 的 map_to_logits 方式")
        # 2. 展平为一维向量供网络处理: (N, L*(V-1))
        offline_x = task.x.reshape(task.x.shape[0], -1)
    else:
        offline_x = task.x
    
    # 3. 计算统计量 (Z-Score)
    mean_x = np.mean(offline_x, axis=0)
    std_x = np.std(offline_x, axis=0)
    std_x = np.where(std_x < 1e-6, 1.0, std_x)
    offline_x_norm = (offline_x - mean_x) / std_x
    
    # 4. 处理 Y
    offline_y = task.y.reshape(-1, 1) # 保持 (N, 1) 形状给 Oracle
    mean_y = np.mean(offline_y)
    std_y = np.std(offline_y)
    if std_y == 0: std_y = 1.0
    offline_y_norm = (offline_y - mean_y) / std_y
    
    return task, offline_x_norm, offline_y_norm, mean_x, std_x

# main.py 预处理逻辑
def preprocess_trajectories(oracle, X_train_norm):
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    if os.path.exists(Config.TRAJECTORY_PATH):
        print(f"Loading cached trajectories from {Config.TRAJECTORY_PATH}")
        return np.load(Config.TRAJECTORY_PATH)['trajs']

    print("=== Generating ALL long trajectories (GD reverse + GA) ===")
    all_indices = np.arange(len(X_train_norm))
    # 按照你的要求：用全量数据，或者至少 10000 条
    sample_size = len(all_indices)
    selected_idx = all_indices
    # selected_idx = np.random.choice(all_indices, sample_size, replace=False)
    
    all_valid = []
    batch_size = 256
    for i in range(0, sample_size, batch_size):
        batch_x = X_train_norm[selected_idx[i : i + batch_size]]
        trajs = generate_long_trajectories(oracle, batch_x, device)
        all_valid.append(trajs)
        print(f"Progress: {i + batch_size}/{sample_size}")
        
    pool = np.concatenate(all_valid, axis=0)
    np.savez_compressed(Config.TRAJECTORY_PATH, trajs=pool)
    return pool

def main():
    # 0. 初始化环境
    print(f"=== UAB-TD Starting: {Config.TASK_NAME} ===")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    
    # 1. 加载并编码数据 (使用 ROOT 的 map_to_logits 逻辑)
    task, X_train_norm, y_train_norm, mean_x, std_x = get_design_bench_data(Config.TASK_NAME)
    
    # 同步 Normalizer 状态
    x_normalizer = Normalizer(np.zeros((1, X_train_norm.shape[1])))
    x_normalizer.mean, x_normalizer.std, x_normalizer.device = mean_x, std_x, device
    
    # 2. 构建安全教师 (NTK Oracle)
    oracle = NTKOracle(X_train_norm, y_train_norm, device)
    
    # 3. 初始化 Flow Matching 网络
    input_dim = X_train_norm.shape[1]
    cfm_model = VectorFieldNet(input_dim, hidden_dim=Config.HIDDEN_DIM).to(device)
    optimizer = optim.Adam(cfm_model.parameters(), lr=Config.FM_LR)

    # --- 核心修改：全量动态轨迹训练 ---
    print(f"=== Training: Online Trajectory Distillation ({Config.FM_EPOCHS} Epochs) ===")
    trajs_pool = preprocess_trajectories(oracle, X_train_norm)
    print(f"Loaded {len(trajs_pool)} trajectories from {Config.TRAJECTORY_PATH}")
    print(f"Trajectories shape: {trajs_pool.shape}")

    all_indices = np.arange(len(X_train_norm))
    y_scores_flat = y_train_norm.flatten()

    for epoch in range(Config.FM_EPOCHS):
        # 动态采样：混合一半随机（全量覆盖）+ 一半高分（重点优化）
        rand_idx = np.random.choice(all_indices, size=Config.NUM_SEEDS // 2, replace=False)
        top_idx = np.argsort(y_scores_flat)[-(Config.NUM_SEEDS // 2):]
        current_seeds_idx = np.concatenate([rand_idx, top_idx])
        
        # X_seeds = X_train_norm[current_seeds_idx]
        
        # 阶段一：在线生成/更新轨迹
        valid_trajs = trajs_pool[current_seeds_idx]
        
        if len(valid_trajs) == 0:
            continue

        # 阶段三：对这批轨迹进行流匹配训练更新
        avg_loss = train_cfm_step(cfm_model, valid_trajs, optimizer, device)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{Config.FM_EPOCHS} | Loss: {avg_loss:.4f} | Valid Trajs: {len(valid_trajs)}")

    # 4. 推理与 SOTA 评估 (Q=128)
    print(f"\n=== SOTA Evaluation (Highest-point, Q=128) ===")
    
    # 对齐 SOTA：从得分最高的 128 个原始样本出发
    test_q = 128
    sota_test_indices = np.argsort(y_scores_flat)[-test_q:]
    
    if task.is_discrete:
        raw_x_test = task.x[sota_test_indices].reshape(test_q, -1)
    else:
        raw_x_test = task.x[sota_test_indices]
        
    X_test_norm = (raw_x_test - mean_x) / std_x
    
    # 阶段四：ODE 推理 (Solver)
    opt_X_norm = inference_ode(cfm_model, X_test_norm, device)
    opt_X_flat = x_normalizer.denormalize(opt_X_norm)
    
    # 还原形状供 task.predict 打分
    if task.is_discrete:
        opt_X = opt_X_flat.reshape(test_q, *task.x.shape[1:])
        original_X = task.x[sota_test_indices]
    else:
        opt_X = opt_X_flat
        original_X = task.x[sota_test_indices]
        
    final_scores = task.predict(opt_X).flatten()
    original_scores = task.predict(original_X).flatten()
    
    # 计算百分位数
    final_scores_sorted = np.sort(final_scores)
    print(f"final_scores_sorted: {final_scores_sorted}")
    max_score = final_scores_sorted[-1]
    p80_score = final_scores_sorted[int(0.8 * (test_q - 1))]
    p50_score = final_scores_sorted[int(0.5 * (test_q - 1))]
    
    print("-" * 40)
    print(f"Original Mean (Top 128): {np.mean(original_scores):.4f}")
    print(f"Optimized Mean (Top 128): {np.mean(final_scores):.4f}")
    print(f"Improvement:              {np.mean(final_scores) - np.mean(original_scores):.4f}")
    print("-" * 40)
    print(f"SOTA Max (100th):        {max_score:.4f} (Target: ~0.986)")
    print(f"SOTA 80th Percentile:    {p80_score:.4f} (Target: ~0.86)")
    print(f"SOTA Median (50th):      {p50_score:.4f} (Target: ~0.75)")
    print("-" * 40)

if __name__ == "__main__":
    main()