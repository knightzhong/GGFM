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
    2. 梯度下降找到 x_low（低分起点）
    3. 从 x_low 开始梯度上升，记录每一步，形成完整的"弧度轨迹"
    4. 最终得到 x_high，验证分数差
    
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
    
        # 2. 随机选择起始点
        selected_indices = torch.randperm(x_train.shape[0])[:num_points]
        x_start = x_train[selected_indices].clone()
        
        gradient_start = time.time()
        
        # 3. 第一阶段：梯度下降找到 x_low（低分起点）
        # ✅ 优化：使用 in-place 操作，避免重复创建 tensor
        x_low = x_start.clone().requires_grad_(True)
        with torch.enable_grad():
            for _ in range(num_gradient_steps // 2):
                mu_star = GP_Model.mean_posterior(x_low)
                grad = torch.autograd.grad(mu_star.sum(), x_low, create_graph=False)[0]
                x_low.data.sub_(learning_rate * grad)  # in-place: x_low -= lr * grad
        
        # 4. 第二阶段：从 x_low 开始梯度上升，记录每一步形成"弧度轨迹"
        # ✅ 预分配内存存储轨迹: (num_gradient_steps+1, num_points, dim)
        trajectory_storage = torch.zeros(num_gradient_steps + num_gradient_steps + 1, num_points, x_train.shape[1], 
                                        device=device, dtype=x_train.dtype)
        
        # ✅ 复用 x_low，避免 clone
        x_curr = x_low
        trajectory_storage[0] = x_curr.data.clone()  # 只在保存时 clone
        
        with torch.enable_grad():
            for t in range(num_gradient_steps+num_gradient_steps):
                mu_star = GP_Model.mean_posterior(x_curr)
                grad = torch.autograd.grad(mu_star.sum(), x_curr, create_graph=False)[0]
                x_curr.data.add_(learning_rate * grad)  # in-place: x_curr += lr * grad
                trajectory_storage[t + 1] = x_curr.data.clone()  # 只在保存时 clone
        
        total_gradient_time += time.time() - gradient_start
        
        # 5. 批量计算起点和终点的分数
        posterior_start = time.time()
        with torch.no_grad():
            y_start = GP_Model.mean_posterior(trajectory_storage[0])  # x_low 的分数
            y_end = GP_Model.mean_posterior(trajectory_storage[-1])   # x_high 的分数
            # ✅ 一次性计算所有有效样本的 mask
            valid_mask = (y_end - y_start) > threshold_diff
        total_posterior_time += time.time() - posterior_start
        
        # 6. 批量过滤和保存轨迹
        # ✅ 优化：使用向量化操作，避免循环
        valid_indices = torch.where(valid_mask)[0]
        for i in valid_indices:
            i_cpu = i.item()
            # trajectory_storage 已经是 detached 的（因为我们用 .data.clone()）
            sample = (trajectory_storage[:, i_cpu, :], y_start[i_cpu], y_end[i_cpu])
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
    
    for func_name, samples in GP_samples.items():
        for sample in samples:
            trajectory, y_start, y_end = sample
            # trajectory: (gp_steps+1, dim) - GP 的完整非线性路径
            all_trajectories.append(trajectory)
    
    if len(all_trajectories) == 0:
        return np.array([]).reshape(0, num_steps + 1, 0)
    
    # 批量堆叠所有轨迹: (N, gp_steps+1, dim)
    trajectories_batch = torch.stack(all_trajectories, dim=0).to(device)
    N, gp_steps, dim = trajectories_batch.shape
    
    # 如果 GP 步数和目标步数相同，直接返回
    if gp_steps == num_steps + 1:
        return trajectories_batch.cpu().numpy()
    
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
    
    return trajs_array


def generate_long_trajectories(oracle, X_init_numpy, device):
    """
    保留原有函数以保持兼容性（如果其他地方还在使用）
    ROOT 风格长轨迹：GD 反转(500步) + GA 冲顶(800步)
    """
    X_curr = torch.FloatTensor(X_init_numpy).to(device).requires_grad_(True)
    X_start = torch.FloatTensor(X_init_numpy).to(device)
    
    # --- 阶段 B: 冲顶 (GA) 800步 ---
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