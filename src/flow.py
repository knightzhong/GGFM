# src/flow.py
import torch
import numpy as np
from src.config import Config

def train_cfm_step(model, trajectories, x0_array, optimizer, device, weights=None):
    """
    对一批在线生成的轨迹执行一次训练更新
    trajectories: numpy array [N, Steps+1, Dim]
    x0_array: numpy array [N, Dim]，每条轨迹对应的原始起点 x_start（锚点）
    """
    model.train()
    trajs = torch.FloatTensor(trajectories).to(device)
    x0s = torch.FloatTensor(x0_array).to(device)

    # [新增] 处理权重
    if weights is not None:
        weights = torch.FloatTensor(weights).to(device).view(-1) # (N,)


    N, T, Dim = trajs.shape
    M = T - 1 
    
    perm = torch.randperm(N)
    total_loss = 0
    num_batches = 0
    
    for i in range(0, N, Config.FM_BATCH_SIZE):
        indices = perm[i:i+Config.FM_BATCH_SIZE]
        batch_traj = trajs[indices]
        # 使用对应的 x_start 作为条件 x0（而不是轨迹第 0 帧）
        batch_x0 = x0s[indices]

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
        v_target = M * (x_k_next - x_k)
        
        # 预测与优化
        v_pred = model(x_t, t_global, batch_x0)
        # [修改] 计算加权 Loss
        loss_elementwise = torch.mean((v_pred - v_target) ** 2, dim=-1) # (B,)
        
        if weights is not None:
            # 加权平均
            loss = torch.mean(loss_elementwise * batch_weights)
        else:
            # 原始平均
            loss = torch.mean(loss_elementwise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches
def train_cfm(model, trajectories, x0_array, optimizer, device):
    """
    Phase 3: 训练条件流匹配
    trajectories: numpy array [N, Steps+1, Dim]
    x0_array: numpy array [N, Dim]，每条轨迹对应的原始起点 x_start（锚点）
    """
    model.train()
    trajs = torch.FloatTensor(trajectories).to(device)
    x0s = torch.FloatTensor(x0_array).to(device)
    N, T, Dim = trajs.shape
    M = T - 1 # 段数
    
    print(f"[Flow] Start training on {N} trajectories...")
    
    for epoch in range(Config.FM_EPOCHS):
        perm = torch.randperm(N)
        epoch_loss = 0
        
        for i in range(0, N, Config.FM_BATCH_SIZE):
            indices = perm[i:i+Config.FM_BATCH_SIZE]
            batch_traj = trajs[indices]
            batch_x0 = x0s[indices]
            
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
            
            velocity = model(x_curr, t_tensor, x_0)
            x_curr = x_curr + velocity * dt
            
    return x_curr.cpu().numpy()


def train_sde_drift_step(model, optimizer, batch, config=None):
    """
    单步 SDE 漂移回归训练（K-SB-SDE 对齐）。

    batch 支持两种输入粒度：
      - 序列级: x/t/drift_label 为 [B, m_sub, D] / [B, m_sub, 1] / [B, m_sub, D]
      - 时间级: x/t/drift_label 为 [B, D] / [B, 1] / [B, D]

    cond 为 [B, C]，会自动 broadcast 到每个时间步。
    """
    model.train()

    x = batch["x"]
    t = batch["t"]
    cond = batch["cond"]
    drift_label = batch["drift_label"]
    weight = batch.get("weight", None)

    if x.dim() == 3:
        # 序列级输入: 展平时间维度，按点计算 MSE
        bsz, steps, dim = x.shape
        x_flat = x.reshape(bsz * steps, dim)               # [B*m_sub, D]
        t_flat = t.reshape(bsz * steps, 1)                 # [B*m_sub, 1]
        cond_flat = (
            cond.unsqueeze(1)
            .expand(-1, steps, -1)
            .reshape(bsz * steps, cond.shape[-1])
        )                                                  # [B*m_sub, C]
        drift_label_flat = drift_label.reshape(bsz * steps, dim)  # [B*m_sub, D]

        drift_pred_flat = model(x_flat, t_flat, cond_flat)        # [B*m_sub, D]
        loss_elementwise = torch.mean(
            (drift_pred_flat - drift_label_flat) ** 2, dim=-1
        )  # [B*m_sub]

        if weight is not None:
            w = weight.view(-1)
            if w.numel() == bsz:
                # 轨迹级权重：重复到每个时间点
                w_rep = w.repeat_interleave(steps)  # [B*m_sub]
                loss = torch.mean(loss_elementwise * w_rep)
            elif w.numel() == loss_elementwise.numel():
                # 已经是按点的权重
                loss = torch.mean(loss_elementwise * w)
            else:
                loss = torch.mean(loss_elementwise)
        else:
            loss = torch.mean(loss_elementwise)
    else:
        # 时间级输入（兼容）
        drift_pred = model(x, t, cond)
        loss_elementwise = torch.mean(
            (drift_pred - drift_label) ** 2, dim=-1
        )  # [B]

        if weight is not None:
            w = weight.view(-1)
            if w.numel() == loss_elementwise.numel():
                loss = torch.mean(loss_elementwise * w)
            else:
                loss = torch.mean(loss_elementwise)
        else:
            loss = torch.mean(loss_elementwise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def inference_sde(model, x0, cond, steps, sigma_max, sigma_min, device):
    """
    使用 Euler–Maruyama 进行 SDE 采样。

    x_{n+1} = x_n + f_θ(x_n, t_n, cond) * dt + g(t_n) * sqrt(dt) * N(0, I)
    其中 g(t) = sigma_min + (sigma_max - sigma_min) * (1 - t)

    Args:
        model: DriftNet 模型
        x0: 初始状态 [B, D]
        cond: 条件向量 [B, C]（每条样本对应一个 cond）
        steps: 推理步数（SDE_INFERENCE_STEPS）
        sigma_max: g(t) 的最大值
        sigma_min: g(t) 的最小值
        device: torch.device

    Returns:
        x_T: 终点状态 [B, D]
    """
    model.eval()

    x = x0.to(device)  # [B, D]
    cond = cond.to(device)  # [B, C]

    B, D = x.shape
    dt = 1.0 / float(steps)
    sqrt_dt = dt ** 0.5

    with torch.no_grad():
        for i in range(steps):
            t_val = i / float(steps)
            t = torch.full((B, 1), t_val, device=device)  # [B, 1]

            drift = model(x, t, cond)  # [B, D]

            g_t = sigma_min + (sigma_max - sigma_min) * (1.0 - t)  # [B, 1]
            noise = torch.randn_like(x)  # [B, D]

            x = x + drift * dt + g_t * sqrt_dt * noise

    return x