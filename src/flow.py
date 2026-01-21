# src/flow.py
import torch
import numpy as np
from src.config import Config

def train_cfm_step(model, trajectories, optimizer, device):
    """
    对一批在线生成的轨迹执行一次训练更新
    """
    model.train()
    trajs = torch.FloatTensor(trajectories).to(device)
    N, T, Dim = trajs.shape
    M = T - 1 
    
    perm = torch.randperm(N)
    total_loss = 0
    num_batches = 0
    
    for i in range(0, N, Config.FM_BATCH_SIZE):
        indices = perm[i:i+Config.FM_BATCH_SIZE]
        batch_traj = trajs[indices]
        batch_x0 = batch_traj[:, 0, :]
        
        # 采样时间段 k
        k = torch.randint(0, M, (len(indices),)).to(device)
        idx_range = torch.arange(len(indices))
        x_k = batch_traj[idx_range, k, :]
        x_k_next = batch_traj[idx_range, k+1, :]
        
        # 线性插值与目标速度计算
        alpha = torch.rand(len(indices), 1).to(device)
        t_global = (k.unsqueeze(1) + alpha) / M
        x_t = (1 - alpha) * x_k + alpha * x_k_next
        v_target = M * (x_k_next - x_k)
        
        # 预测与优化
        v_pred = model(x_t, t_global, batch_x0)
        loss = torch.mean((v_pred - v_target) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / num_batches
def train_cfm(model, trajectories, optimizer, device):
    """
    Phase 3: 训练条件流匹配
    trajectories: numpy array [N, Steps+1, Dim]
    """
    model.train()
    trajs = torch.FloatTensor(trajectories).to(device)
    N, T, Dim = trajs.shape
    M = T - 1 # 段数
    
    print(f"[Flow] Start training on {N} trajectories...")
    
    for epoch in range(Config.FM_EPOCHS):
        perm = torch.randperm(N)
        epoch_loss = 0
        
        for i in range(0, N, Config.FM_BATCH_SIZE):
            indices = perm[i:i+Config.FM_BATCH_SIZE]
            batch_traj = trajs[indices]
            batch_x0 = batch_traj[:, 0, :]
            
            # 随机采样时间段 k
            k = torch.randint(0, M, (len(indices),)).to(device)
            
            # 获取 x_k, x_{k+1}
            idx_range = torch.arange(len(indices))
            x_k = batch_traj[idx_range, k, :]
            x_k_next = batch_traj[idx_range, k+1, :]
            
            # 线性插值
            alpha = torch.rand(len(indices), 1).to(device)
            t_global = (k.unsqueeze(1) + alpha) / M
            x_t = (1 - alpha) * x_k + alpha * x_k_next
            
            # 目标速度
            v_target = M * (x_k_next - x_k)
            
            # 预测与 Loss
            v_pred = model(x_t, t_global, batch_x0)
            loss = torch.mean((v_pred - v_target) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1} | Loss: {epoch_loss / (N/Config.FM_BATCH_SIZE):.4f}")

def inference_ode(model, x_query, device):
    """
    Phase 4: 使用 Euler 法解 ODE
    """
    model.eval()
    x_curr = torch.FloatTensor(x_query).to(device)
    x_0 = x_curr.clone() # Condition
    
    steps = Config.INFERENCE_STEPS
    dt = 1.0 / steps
    
    with torch.no_grad():
        for i in range(steps):
            t_val = i / steps
            t_tensor = torch.full((x_curr.shape[0], 1), t_val).to(device)
            
            velocity = model(x_curr, t_tensor, x_0)
            x_curr = x_curr + velocity * dt
            
    return x_curr.cpu().numpy()