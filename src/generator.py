# src/generator.py
import torch
import torch.optim as optim
import numpy as np
from src.config import Config
from gpytorch.kernels import(
    RBFKernel, LinearKernel, MaternKernel, RQKernel, PeriodicKernel,
    CosineKernel, PolynomialKernel 
)

kernel_dict = {'rbf': RBFKernel,'matern': MaternKernel, 
                'rq' : RQKernel, 'period': PeriodicKernel, 'cosine': CosineKernel,
                'poly': PolynomialKernel}

class GP: 
    """ROOT 风格的 GP 类，用于生成动态轨迹"""
    def __init__(self, device, x_train, y_train, lengthscale, variance, noise, mean_prior, kernel='rbf'):
        self.device = device 
        self.x_train = x_train
        self.y_train = y_train 
        self.kernel = kernel_dict[kernel]().to(device)
        self.noise = noise
        self.variance = variance
        self.mean_prior = mean_prior
        self.kernel.lengthscale = lengthscale
        
    def set_hyper(self, lengthscale, variance): 
        self.variance = variance 
        self.kernel.lengthscale = lengthscale
        if hasattr(self, 'coef'):
            del self.coef
        with torch.no_grad():
            # 计算核矩阵
            K_train_train = self.variance * self.kernel.forward(self.x_train, self.x_train)
            K_train_train.diagonal().add_(self.noise)  # In-place modification
            
            # Cholesky 分解（这是最耗时的操作）
            L = torch.linalg.cholesky(K_train_train)
            b = (self.y_train - self.mean_prior).unsqueeze(-1)
            self.coef = torch.cholesky_solve(b, L).squeeze(-1).detach()
    
    def mean_posterior(self, x_test): 
        # Posterior mean
        K_train_test = self.variance * self.kernel.forward(self.x_train, x_test)
        mu_star = self.mean_prior + torch.matmul(K_train_test.T, self.coef)
        return mu_star


def sampling_data_from_GP(x_train, device, GP_Model, num_gradient_steps=50, num_functions=5, num_points=10, 
                          learning_rate=0.001, delta_lengthscale=0.1, delta_variance=0.1, seed=0, threshold_diff=0.1, verbose=False):
    """
    改进版 GP 采样：生成从 x_low 到 x_high 的连续上升轨迹
    
    流程：
    1. 选择起始点
    2. 不再分两个阶段（先下降再上升），而是模仿 ROOT，将起始点复制成两份，一份用于下降（找 $x_{low}$），一份用于上升（找 $x_{high}$）。
    3. 最终得到 x_high，验证分数差
    
    返回格式: datasets = {f'f{i}': [(trajectory, y_start, y_end), ...], ...}
    其中 trajectory 是 (num_gradient_steps+1, dim) 的完整上升轨迹
    """
    import time
    lengthscale = GP_Model.kernel.lengthscale
    variance = GP_Model.variance 
    torch.manual_seed(seed=seed)
    datasets = {}

    total_set_hyper_time = 0
    total_gradient_time = 0
    total_posterior_time = 0
    
    for iter in range(num_functions):
        datasets[f'f{iter}'] = []
        
        # 1. 为每个函数采样不同的超参数
        set_hyper_start = time.time()
        new_lengthscale = lengthscale + delta_lengthscale*(torch.rand(1, device=device)*2 -1)
        new_variance = variance + delta_variance*(torch.rand(1, device=device)*2 -1)
        GP_Model.set_hyper(lengthscale=new_lengthscale, variance=new_variance)
        total_set_hyper_time += time.time() - set_hyper_start
    
        # 2. 随机选择起始点并构造并行计算张量
        selected_indices = torch.randperm(x_train.shape[0])[:num_points]
        x_base = x_train[selected_indices].clone()
        
        # 模仿 ROOT: 前一半点下降，后一半点上升
        # 这样我们可以同时得到 x_low 和 x_high
        joint_x = torch.cat([x_base.clone(), x_base.clone()], dim=0).requires_grad_(True)
        
        # 构造统一的学习率向量: 前 num_points 个点 -lr (下降), 后 num_points 个点 +lr (上升)
        lr_vec = torch.cat([
            -learning_rate * torch.ones(num_points, x_train.shape[1], device=device),
            learning_rate * torch.ones(num_points, x_train.shape[1], device=device)
        ], dim=0)
        
        gradient_start = time.time()
        
        # 预分配内存记录轨迹 (记录 num_gradient_steps 步)
        # 维度: (Steps, 2*num_points, dim)
        traj_record = []
        
        # 3. 统一优化阶段
        with torch.enable_grad():
            for t in range(num_gradient_steps):
                mu_star = GP_Model.mean_posterior(joint_x)
                grad = torch.autograd.grad(mu_star.sum(), joint_x)[0]
                
                # 统一更新
                joint_x.data.add_(lr_vec * grad)
                
                # 记录当前所有点的位置
                traj_record.append(joint_x.data.clone())
        
        total_gradient_time += time.time() - gradient_start
        
        # 4. 提取结果并构造从 low 到 high 的轨迹
        # joint_x 的前一半是最终的 x_low, 后一半是最终的 x_high
        x_low_final = joint_x.data[:num_points]
        x_high_final = joint_x.data[num_points:]
        
        # 构造完整的"弧度"轨迹: 从 x_low 经过起始点到 x_high
        # 在 GGFM 逻辑中，我们通常需要的是从低分到高分的单向轨迹
        # 我们可以通过将 traj_record 中前一半点的轨迹反转，再接上后一半点的轨迹
        
        posterior_start = time.time()
        with torch.no_grad():
            y_low = GP_Model.mean_posterior(x_low_final)
            y_high = GP_Model.mean_posterior(x_high_final)
            valid_mask = (y_high - y_low) > threshold_diff
        total_posterior_time += time.time() - posterior_start
        
        # 5. 批量保存有效轨迹
        valid_indices = torch.where(valid_mask)[0]
        for i in valid_indices:
            idx = i.item()
            
            # 提取第 idx 个点的下降路径 (从 start 到 low)
            # 我们需要将其反转，变成从 low 到 start
            path_down = torch.stack([step[idx] for step in traj_record], dim=0)
            path_low_to_start = torch.flip(path_down, dims=[0])
            
            # 提取第 idx+num_points 个点的上升路径 (从 start 到 high)
            path_start_to_high = torch.stack([step[idx + num_points] for step in traj_record], dim=0)
            
            # 合并成完整轨迹: low -> start -> high
            full_trajectory = torch.cat([path_low_to_start, path_start_to_high], dim=0)
            
            sample = (full_trajectory, y_low[idx], y_high[idx])
            datasets[f'f{iter}'].append(sample)

    # 恢复原始超参数
    GP_Model.kernel.lengthscale = lengthscale
    GP_Model.variance = variance
    
    if verbose:
        print(f"    [GP内部] set_hyper: {total_set_hyper_time:.2f}s | 梯度采样: {total_gradient_time:.2f}s | 后验: {total_posterior_time:.2f}s")
    
    return datasets


def generate_trajectories_from_GP_samples(GP_samples, device, num_steps=50):
    """
    从 GP 采样得到的完整非线性轨迹生成训练数据（高效版本）
    使用 GP 梯度搜索生成的真实非线性路径，而不是简单的直线插值
    
    Args:
        GP_samples: dict, 格式为 {f'f{i}': [(trajectory, y_start, y_end), ...], ...}
                    其中 trajectory 是 (gp_steps+1, dim) 的完整 GP 梯度搜索路径
        device: torch.device
        num_steps: int, 目标轨迹步数（用于重采样）
    
    Returns:
        trajs_array: numpy array (num_trajs, num_steps+1, dim)
    """
    # 收集所有轨迹
    all_trajectories = []
    all_scores = []  # <--- [新增] 收集分数
    
    for func_name, samples in GP_samples.items():
        for sample in samples:
            trajectory, y_start, y_end = sample
            # trajectory: (gp_steps+1, dim) - GP 的完整非线性路径
            all_trajectories.append(trajectory)
            all_scores.append(y_end) # <--- [新增] 保存 y_end
    
    if len(all_trajectories) == 0:
        return np.array([]).reshape(0, num_steps + 1, 0), np.array([])
    
    # 批量堆叠所有轨迹: (N, gp_steps+1, dim)
    trajectories_batch = torch.stack(all_trajectories, dim=0).to(device)
    N, gp_steps, dim = trajectories_batch.shape

    # [新增] 转换分数
    # y_end 可能是 tensor 也可能是 float，统一转 numpy
    scores_array = np.array([s.item() if torch.is_tensor(s) else s for s in all_scores])
    
    # 如果 GP 步数和目标步数相同，直接返回
    if gp_steps == num_steps + 1:
        return trajectories_batch.cpu().numpy(),scores_array
    
    # ✅ 高效重采样：批量插值（完全向量化，无循环）
    # 使用 F.interpolate 批量处理所有轨迹
    # 输入: (N, dim, gp_steps) - batch, channels, length
    # 输出: (N, dim, num_steps+1)
    trajectories_transposed = trajectories_batch.permute(0, 2, 1)  # (N, dim, gp_steps)
    
    resampled_transposed = torch.nn.functional.interpolate(
        trajectories_transposed,
        size=num_steps + 1,
        mode='linear',
        align_corners=True
    )  # (N, dim, num_steps+1)
    
    # 转回 (N, num_steps+1, dim)
    final_trajs = resampled_transposed.permute(0, 2, 1)
    
    # 转换为 numpy
    trajs_array = final_trajs.cpu().numpy()

    
    
    return trajs_array,scores_array


def generate_long_trajectories(oracle, X_init_numpy, device):
    """
    保留原有函数以保持兼容性（如果其他地方还在使用）
    ROOT 风格长轨迹：GD 反转(500步) + GA 冲顶(800步)
    """
    X_curr = torch.FloatTensor(X_init_numpy).to(device).requires_grad_(True)
    X_start = torch.FloatTensor(X_init_numpy).to(device)
    
    # --- 阶段 B: 冲顶 (GA) 200步 ---
    X_curr.data = X_start.data.clone()
    y_target_high = torch.full((X_start.shape[0], 1), 4.0).to(device)
    traj_part2 = [X_curr.detach().cpu().clone()]
    
    opt_asc = optim.Adam([X_curr], lr=Config.TRAJ_LR)
    for _ in range(Config.TRAJ_STEPS_ASC):
        opt_asc.zero_grad()
        mu = oracle.predict_mean(X_curr)
        sigma_sq = oracle.predict_uncertainty(X_curr)
        
        loss_fwd = torch.mean((mu - y_target_high) ** 2)
        loss_bwd = torch.mean((X_curr - X_start) ** 2)
        loss_unc = torch.mean(sigma_sq)
        
        loss = (loss_fwd 
                + Config.LAMBDA_BACKWARD * loss_bwd 
                + Config.LAMBDA_UNCERTAINTY * loss_unc)
        
        loss.backward()
        opt_asc.step()
        traj_part2.append(X_curr.detach().cpu().clone())

    full_traj = traj_part2
    traj_tensor = torch.stack(full_traj, dim=1).to(device)
    
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