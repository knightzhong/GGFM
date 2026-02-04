# src/flow.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.autonotebook import tqdm
from src.config import Config


def _fm_schedule(t, device):
    """与 BB 一致的 m_t、sigma_t（连续 t∈[0,1]）：m_t 线性，variance_t=2*(m_t-m_t^2)*max_var。"""
    m_min, m_max = 0.001, 0.999
    max_var = getattr(Config, "FM_SCHEDULE_MAX_VAR", 1.0)
    m_t = m_min + (m_max - m_min) * t
    variance_t = 2.0 * (m_t - m_t ** 2).clamp(min=0.0) * max_var
    sigma_t = torch.sqrt(variance_t + 1e-10)
    return m_t, sigma_t


def _maybe_apply_cfg(y_high, y_low, step_seed, cfg_prob):
    if cfg_prob <= 0:
        return y_high, y_low
    torch.manual_seed(step_seed)
    rand_mask = torch.rand(y_high.size(), device=y_high.device)
    mask = rand_mask <= cfg_prob
    y_high = y_high.clone()
    y_low = y_low.clone()
    y_high[mask] = 0.0
    y_low[mask] = 0.0
    return y_high, y_low


def fm_loss_on_batch(model, x_high, x_low, y_high, y_low, device):
    """Compute FM drift matching loss on a batch (no optimizer step)."""
    x_high = x_high.to(device)
    x_low = x_low.to(device)
    y_high = y_high.to(device)
    y_low = y_low.to(device)

    t_global = torch.rand(x_low.shape[0], 1, device=device)
    x_t = (1 - t_global) * x_low + t_global * x_high
    v = x_high - x_low

    drift_target_mode = getattr(Config, "FM_DRIFT_TARGET", "constant")
    if drift_target_mode == "schedule":
        m_t, sigma_t = _fm_schedule(t_global, device)
        mu_target = m_t * v + sigma_t * torch.randn_like(v, device=device)
    else:
        mu_target = v
        drift_noise_scale = getattr(Config, "FM_DRIFT_NOISE_SCALE", 0.0)
        if drift_noise_scale > 0:
            sigma_t = drift_noise_scale * torch.sqrt(t_global * (1.0 - t_global) + 1e-8)
            mu_target = mu_target + sigma_t * torch.randn_like(mu_target, device=device)

    mu_pred, _ = model(x_t, t_global, y_low, y_high)
    loss = torch.mean(torch.mean((mu_pred - mu_target).abs(), dim=-1))
    return loss


def train_cfm_batch(
    model,
    x_high,
    x_low,
    y_high,
    y_low,
    optimizer,
    device,
    use_cfg=False,
    cfg_prob=0.1,
    step_seed=0,
):
    """Train FM on a single batch to align with BB training flow."""
    if use_cfg:
        y_high, y_low = _maybe_apply_cfg(y_high, y_low, step_seed, cfg_prob)
    loss = fm_loss_on_batch(model, x_high, x_low, y_high, y_low, device)
    loss.backward()
    return loss


# --------------- Brownian Bridge 训练与推理 ---------------
def train_bb_step(model, train_loader, optimizer, device, use_cfg=False, cfg_prob=0.1):
    """
    与 ROOT BBDMRunner.loss_fn 对齐：使用 DataLoader 的 batch
    train_loader: DataLoader，每个 batch 是 ((x_high, y_high), (x_low, y_low))
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        (x_high, y_high), (x_low, y_low) = batch
        
        # 与 ROOT 对齐：Classifier-Free Guidance（训练阶段）
        if use_cfg:
            torch.manual_seed(num_batches)  # 使用 step 作为 seed
            rand_mask = torch.rand(y_high.size())
            mask = (rand_mask <= cfg_prob)
            y_high[mask] = 0.0
            y_low[mask] = 0.0
        
        x_high = x_high.to(device)
        y_high = y_high.to(device)
        x_low = x_low.to(device)
        y_low = y_low.to(device)
        
        optimizer.zero_grad()
        loss, _ = model(x_high, y_high, x_low, y_low)
        loss.backward()
        # ROOT 没有梯度裁剪，移除
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches else 0.0
    return avg_loss, 0.0, 0.0, 0.0


def inference_bb(model, x_low, y_low, y_high, device, clip_denoised=False, cfg_weight=0.0):
    """
    布朗桥采样：从 x_low 出发，条件 y_low, y_high，得到 x_high
    x_low: [N, D], y_low, y_high: [N] 或 [N,1]
    """
    model.eval()
    x_low_t = torch.FloatTensor(x_low).to(device)
    y_low_t = torch.FloatTensor(y_low).to(device).view(-1, 1)
    y_high_t = torch.FloatTensor(y_high).to(device).view(-1, 1)
    with torch.no_grad():
        out = model.sample(
            x_low_t, y_low_t, y_high_t,
            clip_denoised=clip_denoised,
            classifier_free_guidance_weight=cfg_weight,
        )
    return out.cpu().numpy()

def train_cfm_step(model, trajectories, y_low, y_high, optimizer, device, gp_model=None, weights=None):
    """
    对一批在线生成的轨迹执行一次训练更新（与 BB 一致：条件 y_low, y_high）
    trajectories: numpy array [N, Steps+1, Dim]
    y_low, y_high: numpy array [N] 或 [N,1]，与 trajectories 一一对应
    """
    model.train()
    trajs = torch.FloatTensor(trajectories).to(device)
    y_low = np.asarray(y_low).reshape(-1, 1).astype(np.float32)
    y_high = np.asarray(y_high).reshape(-1, 1).astype(np.float32)
    y_low_t = torch.FloatTensor(y_low).to(device)
    y_high_t = torch.FloatTensor(y_high).to(device)

    if weights is not None:
        weights = torch.FloatTensor(weights).to(device).view(-1)

    N, T, Dim = trajs.shape
    M = T - 1

    perm = torch.randperm(N)
    total_loss = 0
    total_cos_sim = 0
    total_l_grad = 0
    total_l_sigma = 0
    num_batches = 0

    batch_indices = list(range(0, N, Config.FM_BATCH_SIZE))
    pbar = tqdm(batch_indices, desc="FM batch", smoothing=0.01, leave=True)

    for i in pbar:
        indices = perm[i:i+Config.FM_BATCH_SIZE]
        batch_traj = trajs[indices]
        batch_x0 = batch_traj[:, 0, :]
        batch_y_low = y_low_t[indices]
        batch_y_high = y_high_t[indices]

        if weights is not None:
            batch_weights = weights[indices]

        # 直线流匹配 (Straight Flow)，与 BB q_sample 一致：网络只吃 (x_t, t, y_low, y_high)，不吃 x_low
        x_start = batch_traj[:, 0, :]
        x_end = batch_traj[:, -1, :]

        t_global = torch.rand(len(indices), 1).to(device)
        x_t = (1 - t_global) * x_start + t_global * x_end
        v = (x_end - x_start)
        drift_target_mode = getattr(Config, "FM_DRIFT_TARGET", "constant")
        if drift_target_mode == "schedule":
            # 与 BB 一致：目标随 t 变，mu_target = m_t * v + sigma_t * noise，推理时 v = mu_pred / m_t
            m_t, sigma_t = _fm_schedule(t_global, device)
            mu_target = m_t * v + sigma_t * torch.randn_like(v, device=device)
        else:
            mu_target = v
            drift_noise_scale = getattr(Config, "FM_DRIFT_NOISE_SCALE", 0.0)
            if drift_noise_scale > 0:
                sigma_t = drift_noise_scale * torch.sqrt(t_global * (1.0 - t_global) + 1e-8)
                mu_target = mu_target + sigma_t * torch.randn_like(mu_target, device=device)

        mu_pred, log_sigma_pred = model(x_t, t_global, batch_y_low, batch_y_high)
        sigma_pred = torch.exp(log_sigma_pred)
        
        # 1. Drift Matching Loss (L1)
        loss_drift_elementwise = torch.mean((mu_pred - mu_target).abs(), dim=-1)  # (B,)

        # 2. Diffusion Regularization Loss
        # sigma_target = Config.SIGMA_MAX * (1.0 - t_global)
        # loss_sigma_elementwise = (sigma_pred - sigma_target).pow(2).squeeze(-1) # (B,)

        # # 3. GP Reward Gradient Alignment Loss (只作用于 drift)
        # loss_grad_elementwise = torch.zeros_like(loss_drift_elementwise)
        # batch_cos_sim = 0.0
        # if gp_model is not None and Config.LAMBDA_GRAD > 0:
        #     x_t_for_grad = x_t.detach().requires_grad_(True)
        #     with torch.enable_grad():
        #         mu_t = gp_model.mean_posterior(x_t_for_grad)
        #         grad_r_t = torch.autograd.grad(mu_t.sum(), x_t_for_grad)[0]
            
        #     mu_unit = F.normalize(mu_pred, p=2, dim=-1, eps=1e-6)
        #     v_grad_unit = F.normalize(grad_r_t, p=2, dim=-1, eps=1e-6)
            
        #     cos_sim_elementwise = torch.clamp((mu_unit * v_grad_unit).sum(dim=-1), -1.0, 1.0)
        #     loss_grad_elementwise = 1.0 - cos_sim_elementwise
        #     batch_cos_sim = cos_sim_elementwise.mean().item()

        # 合并总 Loss
        loss_elementwise = (
            loss_drift_elementwise #+ 
            # Config.LAMBDA_GRAD * loss_grad_elementwise + 
            # Config.LAMBDA_SIGMA * loss_sigma_elementwise
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

        batch_loss = loss.item()
        total_loss += batch_loss
        # total_cos_sim += batch_cos_sim
        # total_l_grad += loss_grad_elementwise.mean().item()
        # total_l_sigma += loss_sigma_elementwise.mean().item()
        num_batches += 1
        pbar.set_description(f"FM batch loss: {batch_loss:.4f}")

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

def inference_ode(model, x_query, y_low, y_high, device):
    """
    与 BB 一致：网络只吃 (x_t, t, y_low, y_high)；x_low 只用在更新公式里。
    直线流 x(t)=x_low+t*v，故用 x_curr = x_low + t * v_pred 每步在公式里锚定起点。
    """
    model.eval()
    x_low_t = torch.FloatTensor(x_query).to(device)  # 起点，只在更新公式里用，不喂给网络
    y_low = np.asarray(y_low).reshape(-1, 1).astype(np.float32)
    y_high = np.asarray(y_high).reshape(-1, 1).astype(np.float32)
    y_low_t = torch.FloatTensor(y_low).to(device)
    y_high_t = torch.FloatTensor(y_high).to(device)

    steps = Config.INFERENCE_STEPS
    x_curr = x_low_t.clone()  # 当前状态，用于喂给网络；每步后用公式更新

    use_schedule = getattr(Config, "FM_DRIFT_TARGET", "constant") == "schedule"
    with torch.no_grad():
        for i in range(steps):
            t_curr = i / steps
            t_next = (i + 1) / steps
            t_tensor = torch.full((x_low_t.shape[0], 1), t_curr, device=device)
            mu_pred, _ = model(x_curr, t_tensor, y_low_t, y_high_t)
            if use_schedule:
                m_t, _ = _fm_schedule(t_tensor, device)
                v_pred = mu_pred / m_t.clamp(min=1e-6)
            else:
                v_pred = mu_pred
            x_curr = x_low_t + t_next * v_pred
            x_curr = torch.clamp(x_curr, -5.0, 5.0)

    return x_curr.cpu().numpy()
