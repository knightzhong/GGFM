# src/generator.py
import torch
import torch.optim as optim
import numpy as np
from src.config import Config

def generate_trajectories(oracle, X_start_numpy, device):
    """
    Phase 1: 数据造轨迹过程
    返回: 筛选后的轨迹 [N_valid, Steps+1, Dim]
    """
    print(f"[Generator] Generating trajectories for {len(X_start_numpy)} seeds...")
    
    X_curr = torch.FloatTensor(X_start_numpy).to(device).requires_grad_(True)
    X_start = torch.FloatTensor(X_start_numpy).to(device)
    
    # 设置目标：简单设为当前预测值 + 2.0 (标准化后的分数提升)
    with torch.no_grad():
        y_target = oracle.predict_mean(X_start) + 2.0
    
    traj_batch = [X_curr.detach().cpu()]
    optimizer = optim.Adam([X_curr], lr=Config.TRAJ_LR)

    # --- 梯度演化 ---
    for t in range(Config.TRAJ_STEPS):
        optimizer.zero_grad()
        
        # 1. Forward Loss (Score Maximization)
        mu = oracle.predict_mean(X_curr)
        loss_fwd = torch.mean((mu - y_target) ** 2)
        
        # 2. Backward Loss (Consistency)
        loss_bwd = torch.mean((X_curr - X_start) ** 2)
        
        # 3. Uncertainty Penalty (Safety)
        sigma_sq = oracle.predict_uncertainty(X_curr)
        loss_unc = torch.mean(sigma_sq)
        
        # Total Loss
        loss = (loss_fwd 
                + Config.LAMBDA_BACKWARD * loss_bwd 
                + Config.LAMBDA_UNCERTAINTY * loss_unc)
        
        loss.backward()
        optimizer.step()
        traj_batch.append(X_curr.detach().cpu())

    # 堆叠轨迹 [Batch, Time, Dim]
    traj_tensor = torch.stack(traj_batch, dim=1).to(device)
    
    # --- 单调性验证 (Certified Filter) ---
    valid_trajs = _filter_trajectories(oracle, traj_tensor)
    
    return valid_trajs.cpu().numpy()

def _filter_trajectories(oracle, traj_tensor):
    """验证: mu(XT) - k*sigma(XT) > mu(X0) + k*sigma(X0)"""
    with torch.no_grad():
        mu_0 = oracle.predict_mean(traj_tensor[:, 0, :])
        sig_0 = torch.sqrt(oracle.predict_uncertainty(traj_tensor[:, 0, :]))
        
        mu_T = oracle.predict_mean(traj_tensor[:, -1, :])
        sig_T = torch.sqrt(oracle.predict_uncertainty(traj_tensor[:, -1, :]))
    
    lower_bound_T = mu_T - Config.KAPPA * sig_T
    upper_bound_0 = mu_0 + Config.KAPPA * sig_0
    
    valid_mask = (lower_bound_T > upper_bound_0).squeeze()
    num_valid = valid_mask.sum().item()
    print(f"[Generator] Filtered {traj_tensor.shape[0]} -> {num_valid} valid trajectories.")
    
    return traj_tensor[valid_mask]