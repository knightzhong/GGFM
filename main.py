# main.py
import design_bench
import torch
import torch.optim as optim
import numpy as np

from src.config import Config
from src.utils import set_seed, Normalizer
from src.oracle import NTKOracle
from src.generator import generate_trajectories
from src.models import VectorFieldNet
from src.flow import train_cfm, inference_ode

def main():
    # 0. 初始化
    print(f"=== UAB-TD Starting: {Config.TASK_NAME} ===")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    
    task = design_bench.make(Config.TASK_NAME)
    x_normalizer = Normalizer(task.x)
    y_normalizer = Normalizer(task.y)
    
    X_train_norm = x_normalizer.normalize(task.x)
    y_train_norm = y_normalizer.normalize(task.y)
    
    # 1. 构建 NTK Oracle (Phase 1)
    oracle = NTKOracle(X_train_norm, y_train_norm, device)
    
    # 2. 生成并筛选轨迹 (Phase 1)
    # 随机采样一部分数据作为起点
    seed_indices = np.random.choice(len(task.x), size=Config.NUM_SEEDS, replace=False)
    X_seeds = X_train_norm[seed_indices]
    
    valid_trajs = generate_trajectories(oracle, X_seeds, device)
    
    if len(valid_trajs) == 0:
        print("Error: No valid trajectories found. Try relaxing KAPPA in config.")
        return

    # 3. 训练 Flow Matching 模型 (Phase 2 & 3)
    input_dim = task.x.shape[1]
    cfm_model = VectorFieldNet(input_dim, hidden_dim=Config.HIDDEN_DIM).to(device)
    optimizer = optim.Adam(cfm_model.parameters(), lr=Config.FM_LR)
    
    train_cfm(cfm_model, valid_trajs, optimizer, device)
    
    # 4. 推理与评估 (Phase 4)
    print("=== Evaluation ===")
    # 模拟测试：取一部分原始数据进行优化
    X_test = task.x[:128]
    X_test_norm = x_normalizer.normalize(X_test)
    
    # 执行 ODE 推理
    opt_X_norm = inference_ode(cfm_model, X_test_norm, device)
    opt_X = x_normalizer.denormalize(opt_X_norm)
    
    # 评估
    original_scores = task.predict(X_test)
    final_scores = task.predict(opt_X)
    
    print(f"Original Mean: {np.mean(original_scores):.4f}")
    print(f"Optimized Mean: {np.mean(final_scores):.4f}")
    print(f"Improvement:   {np.mean(final_scores) - np.mean(original_scores):.4f}")
    print(f"Max Score:     {np.max(final_scores):.4f}")

if __name__ == "__main__":
    main()