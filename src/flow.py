# src/flow.py
import torch
import torch.nn.functional as F
import numpy as np
from src.config import Config

def train_cfm_step(model, trajectories, optimizer, device, gp_model=None, weights=None):
    """
    对一批在线生成的轨迹执行一次训练更新
    trajectories: numpy array [N, Steps+1, Dim]
    """
    model.train()
    trajs = torch.FloatTensor(trajectories).to(device)

    # [新增] 处理权重
    if weights is not None:
        weights = torch.FloatTensor(weights).to(device).view(-1) # (N,)


    N, T, Dim = trajs.shape
    M = T - 1 
    
    perm = torch.randperm(N)
    total_loss = 0
    total_cos_sim = 0
    total_l_grad = 0
    total_l_sigma = 0
    num_batches = 0
    
    for i in range(0, N, Config.FM_BATCH_SIZE):
        indices = perm[i:i+Config.FM_BATCH_SIZE]
        batch_traj = trajs[indices]
        batch_x0 = batch_traj[:, 0, :]

        # [新增] 获取当前 batch 的权重
        if weights is not None:
            batch_weights = weights[indices]
        
        # 采样时间段 k
        k = torch.randint(0, M, (len(indices),)).to(device)
        idx_range = torch.arange(len(indices))
        x_k = batch_traj[idx_range, k, :]
        x_k_next = batch_traj[idx_range, k+1, :]
        
        # 线性插值与目标速度计算
        alpha = torch.rand(len(indices), 1).to(device)
        t_global = (k.unsqueeze(1) + alpha) / M
        x_t = (1 - alpha) * x_k + alpha * x_k_next
        mu_target = M * (x_k_next - x_k)
        
        # 预测与优化
        mu_pred, log_sigma_pred = model(x_t, t_global, batch_x0)
        sigma_pred = torch.exp(log_sigma_pred)
        
        # 1. Drift Matching Loss (MSE)
        loss_drift_elementwise = torch.mean((mu_pred - mu_target) ** 2, dim=-1) # (B,)

        # 2. Diffusion Regularization Loss
        sigma_target = Config.SIGMA_MAX * (1.0 - t_global)
        loss_sigma_elementwise = (sigma_pred - sigma_target).pow(2).squeeze(-1) # (B,)

        # 3. GP Reward Gradient Alignment Loss (只作用于 drift)
        loss_grad_elementwise = torch.zeros_like(loss_drift_elementwise)
        batch_cos_sim = 0.0
        if gp_model is not None and Config.LAMBDA_GRAD > 0:
            x_t_for_grad = x_t.detach().requires_grad_(True)
            with torch.enable_grad():
                mu_t = gp_model.mean_posterior(x_t_for_grad)
                grad_r_t = torch.autograd.grad(mu_t.sum(), x_t_for_grad)[0]
            
            mu_unit = F.normalize(mu_pred, p=2, dim=-1, eps=1e-6)
            v_grad_unit = F.normalize(grad_r_t, p=2, dim=-1, eps=1e-6)
            
            cos_sim_elementwise = torch.clamp((mu_unit * v_grad_unit).sum(dim=-1), -1.0, 1.0)
            loss_grad_elementwise = 1.0 - cos_sim_elementwise
            batch_cos_sim = cos_sim_elementwise.mean().item()

        # 合并总 Loss
        loss_elementwise = (
            loss_drift_elementwise + 
            Config.LAMBDA_GRAD * loss_grad_elementwise + 
            Config.LAMBDA_SIGMA * loss_sigma_elementwise
        )
        
        if weights is not None:
            loss = torch.mean(loss_elementwise * batch_weights)
        else:
            loss = torch.mean(loss_elementwise)
        
        optimizer.zero_grad()
        loss.backward()
        # 🔑 松绑：提高裁剪阈值到 5.0，允许更强的学习信号，同时防止 nan
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_cos_sim += batch_cos_sim
        total_l_grad += loss_grad_elementwise.mean().item()
        total_l_sigma += loss_sigma_elementwise.mean().item()
        num_batches += 1
    
    avg_cos_sim = total_cos_sim / num_batches if num_batches > 0 else 0
    avg_l_grad = total_l_grad / num_batches if num_batches > 0 else 0
    avg_l_sigma = total_l_sigma / num_batches if num_batches > 0 else 0
    
    return total_loss / num_batches, avg_cos_sim, avg_l_grad, avg_l_sigma
# def train_cfm(model, trajectories, optimizer, device):
#     """
#     Phase 3: 训练条件流匹配
#     trajectories: numpy array [N, Steps+1, Dim]
#     """
#     model.train()
#     trajs = torch.FloatTensor(trajectories).to(device)
#     N, T, Dim = trajs.shape
#     M = T - 1 # 段数
    
#     print(f"[Flow] Start training on {N} trajectories...")
    
#     for epoch in range(Config.FM_EPOCHS):
#         perm = torch.randperm(N)
#         epoch_loss = 0
        
#         for i in range(0, N, Config.FM_BATCH_SIZE):
#             indices = perm[i:i+Config.FM_BATCH_SIZE]
#             batch_traj = trajs[indices]
#             batch_x0 = batch_traj[:, 0, :]
            
#             # 随机采样时间段 k
#             k = torch.randint(0, M, (len(indices),)).to(device)
            
#             # 获取 x_k, x_{k+1}
#             idx_range = torch.arange(len(indices))
#             x_k = batch_traj[idx_range, k, :]
#             x_k_next = batch_traj[idx_range, k+1, :]
            
#             # 线性插值
#             alpha = torch.rand(len(indices), 1).to(device)
#             t_global = (k.unsqueeze(1) + alpha) / M
#             x_t = (1 - alpha) * x_k + alpha * x_k_next
            
#             # 目标速度
#             v_target = M * (x_k_next - x_k)
            
#             # 预测与 Loss
#             v_pred = model(x_t, t_global, batch_x0)
#             loss = torch.mean((v_pred - v_target) ** 2)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
            
#         if (epoch + 1) % 20 == 0:
#             print(f"Epoch {epoch+1} | Loss: {epoch_loss / (N/Config.FM_BATCH_SIZE):.4f}")

def inference_ode(model, x_query, device):
    """
    Phase 4: 使用 Euler 法解 ODE
    
    Args:
        model: Flow Matching 模型
        x_query: 起始点 [N, D]
        device: torch.device
    
    Returns:
        numpy array [N, D]
    """
    model.eval()
    x_curr = torch.FloatTensor(x_query).to(device)
    x_0 = x_curr.clone() # Condition
    
    steps = Config.INFERENCE_STEPS
    dt = 1.0 / steps
    
    with torch.no_grad():
        for i in range(steps):
            t_val = i / steps
            t_tensor = torch.full((x_curr.shape[0], 1), t_val, device=device)
            
            mu_pred, log_sigma_pred = model(x_curr, t_tensor, x_0)
            sigma_pred = torch.exp(log_sigma_pred)
            
            # SDE update: x_{t+1} = x_t + mu*dt + sigma*sqrt(dt)*epsilon
            epsilon = torch.randn_like(x_curr)
            x_curr = x_curr + mu_pred * dt + sigma_pred * np.sqrt(dt) * epsilon
            
            # 边界裁剪，防止 MuJoCo 报错
            x_curr = torch.clamp(x_curr, -5.0, 5.0)
            
    return x_curr.cpu().numpy()