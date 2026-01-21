# src/generator.py
import torch
import torch.optim as optim
import numpy as np
from src.config import Config

def generate_long_trajectories(oracle, X_init_numpy, device):
    """
    ROOT 风格长轨迹：GD 反转(500步) + GA 冲顶(800步)
    严格执行三部分 Loss: Forward (目标驱动) + Backward (一致性) + Uncertainty (安全)
    """
    X_curr = torch.FloatTensor(X_init_numpy).to(device).requires_grad_(True)
    X_start = torch.FloatTensor(X_init_numpy).to(device)
    
    # --- 阶段 A: 下探 (GD) 500步 ---
    # 目标：寻找低分区域作为起点
    y_target_low = torch.full((X_start.shape[0], 1), -2.0).to(device) # 探底目标
    desc_steps = []
    opt_desc = optim.Adam([X_curr], lr=Config.TRAJ_LR)
    
    for _ in range(Config.TRAJ_STEPS_DESC):
        opt_desc.zero_grad()
        mu = oracle.predict_mean(X_curr)
        sigma_sq = oracle.predict_uncertainty(X_curr)
        
        loss_fwd = torch.mean((mu - y_target_low) ** 2)
        loss_bwd = torch.mean((X_curr - X_start) ** 2)
        loss_unc = torch.mean(sigma_sq)
        
        # 严格执行你的复合 Loss 公式
        loss = (loss_fwd 
                + Config.LAMBDA_BACKWARD * loss_bwd 
                + Config.LAMBDA_UNCERTAINTY * loss_unc)
        
        loss.backward()
        opt_desc.step()
        desc_steps.append(X_curr.detach().cpu().clone())
    
    # 反转下探路径，构造 x_low -> x_init 的升分过程
    traj_part1 = desc_steps[::-1] 

    # --- 阶段 B: 冲顶 (GA) 800步 ---
    # 目标：从 x_init 冲向 0.986
    X_curr.data = X_start.data.clone() # 重置到原始点开始 GA
    y_target_high = torch.full((X_start.shape[0], 1), 4.0).to(device) # 冲顶目标
    traj_part2 = [X_curr.detach().cpu().clone()]
    
    opt_asc = optim.Adam([X_curr], lr=Config.TRAJ_LR)
    for _ in range(Config.TRAJ_STEPS_ASC):
        opt_asc.zero_grad()
        mu = oracle.predict_mean(X_curr)
        sigma_sq = oracle.predict_uncertainty(X_curr)
        
        loss_fwd = torch.mean((mu - y_target_high) ** 2)
        loss_bwd = torch.mean((X_curr - X_start) ** 2)
        loss_unc = torch.mean(sigma_sq)
        
        # 严格执行你的复合 Loss 公式
        loss = (loss_fwd 
                + Config.LAMBDA_BACKWARD * loss_bwd 
                + Config.LAMBDA_UNCERTAINTY * loss_unc)
        
        loss.backward()
        opt_asc.step()
        traj_part2.append(X_curr.detach().cpu().clone())

    # 合并为完整单调轨迹：x_low -> x_init -> x_high
    full_traj = traj_part1 + traj_part2
    traj_tensor = torch.stack(full_traj, dim=1).to(device)
    
    # 阶段一的核心：单调性验证 (Certified Filter)
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