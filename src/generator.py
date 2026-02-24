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
    改进版 GP 采样（ROOT 风格）：
    - 负学习率：从原始设计出发做梯度下降，得到低分端轨迹
    - 正学习率：从原始设计出发做梯度上升，得到高分端轨迹
    - 将两段轨迹在“原始设计点”处拼接，构成从低分到高分的完整轨迹，
      中间样本即为原始样本，逻辑上保证得分单调上升（理想情况下）。
    
    返回格式: datasets = {f'f{i}': [(trajectory, x_start, y_low, y_high), ...], ...}
    其中 trajectory 形状为 (2 * num_gradient_steps + 1, dim)，
    第一帧为最低分点，最后一帧为最高分点。
    """
    import time
    lengthscale = GP_Model.kernel.lengthscale
    variance = GP_Model.variance 
    torch.manual_seed(seed=seed)
    datasets = {}
    
    total_set_hyper_time = 0
    total_gradient_time = 0
    total_posterior_time = 0
    
    dim = x_train.shape[1]
    
    for iter in range(num_functions):
        datasets[f'f{iter}'] = []
        
        # 1. 为每个函数采样不同的超参数
        set_hyper_start = time.time()
        new_lengthscale = lengthscale + delta_lengthscale*(torch.rand(1, device=device)*2 -1)
        new_variance = variance + delta_variance*(torch.rand(1, device=device)*2 -1)
        GP_Model.set_hyper(lengthscale=new_lengthscale, variance=new_variance)
        total_set_hyper_time += time.time() - set_hyper_start
    
        # 2. 随机选择起始点（中间锚点）
        selected_indices = torch.randperm(x_train.shape[0])[:num_points]
        x_start = x_train[selected_indices].clone()
        
        gradient_start = time.time()
        
        # 3. 负学习率：从 x_start 出发做梯度下降（找到低分端），同时记录整条低分轨迹
        # 4. 正学习率：从 x_start 出发做梯度上升（找到高分端），同时记录整条高分轨迹
        #    两条轨迹共享同一条 GP，实现 ROOT/utils.py 中 joint_x + learning_rate_vec 的逻辑
        low_traj = torch.zeros(num_gradient_steps + 1, num_points, dim, device=device, dtype=x_train.dtype)
        high_traj = torch.zeros_like(low_traj)
        
        x_low = x_start.clone().requires_grad_(True)
        x_high = x_start.clone().requires_grad_(True)
        
        low_traj[0] = x_low.data.clone()
        high_traj[0] = x_high.data.clone()
        
        with torch.enable_grad():
            for t in range(num_gradient_steps):
                # joint_x = [低分分支; 高分分支]，一次前向 / 反向，加速搜索
                joint_x = torch.cat((x_low, x_high), dim=0)
                mu_star = GP_Model.mean_posterior(joint_x)
                grad = torch.autograd.grad(mu_star.sum(), joint_x, create_graph=False)[0]
                
                grad_low = grad[:num_points]
                grad_high = grad[num_points:]
                
                # 负学习率：降低 mu_star，逼近低分区域
                x_low.data.sub_(learning_rate * grad_low)
                # 正学习率：提高 mu_star，逼近高分区域
                x_high.data.add_(learning_rate * grad_high)
                
                low_traj[t + 1] = x_low.data.clone()
                high_traj[t + 1] = x_high.data.clone()
        
        total_gradient_time += time.time() - gradient_start
        
        # 5. 只在最低点 / 最高点上计算分数，并做阈值过滤
        posterior_start = time.time()
        with torch.no_grad():
            # 低分端：低分轨迹的最后一个点
            y_low = GP_Model.mean_posterior(low_traj[-1])
            # 高分端：高分轨迹的最后一个点
            y_high = GP_Model.mean_posterior(high_traj[-1])
            valid_mask = (y_high - y_low) > threshold_diff
        total_posterior_time += time.time() - posterior_start
        
        # 6. 构造从低分到高分的完整轨迹，并过滤有效样本
        #    低分轨迹方向：x_start -> x_low
        #    为保证从低到高的逻辑，我们反转低分轨迹：x_low -> ... -> x_start
        #    再接上高分轨迹（去掉重复的 x_start）：x_start -> ... -> x_high
        #    得到的 full_traj 形状为 (2*num_gradient_steps + 1, num_points, dim)
        full_traj = torch.cat(
            (low_traj.flip(0), high_traj[1:]),
            dim=0
        )
        
        valid_indices = torch.where(valid_mask)[0]
        for i in valid_indices:
            i_cpu = i.item()
            # 同步保存该轨迹对应的原始起点（锚点）x_start
            sample = (full_traj[:, i_cpu, :], x_start[i_cpu].detach(), y_low[i_cpu], y_high[i_cpu])
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
        GP_samples: dict, 格式为 {f'f{i}': [(trajectory, x_start, y_low, y_high), ...], ...}
                    其中 trajectory 是 (gp_steps+1, dim) 的完整 GP 梯度搜索路径；
                    x_start 是该轨迹对应的原始起点（锚点）。
        device: torch.device
        num_steps: int, 目标轨迹步数（用于重采样）
    
    Returns:
        trajs_array: numpy array (num_trajs, num_steps+1, dim)
        x0_array: numpy array (num_trajs, dim)  # 对应每条轨迹的 x_start
        scores_array: numpy array (num_trajs,)  # 默认使用 y_high 作为分数
    """
    # 收集所有轨迹
    all_trajectories = []
    all_scores = []  # <--- [新增] 收集分数
    all_x0 = []      # <--- [新增] 收集每条轨迹的 x_start
    
    for func_name, samples in GP_samples.items():
        for sample in samples:
            trajectory, x_start, y_low, y_high = sample
            # trajectory: (gp_steps+1, dim) - GP 的完整非线性路径
            all_trajectories.append(trajectory)
            all_x0.append(x_start)
            all_scores.append(y_high) # 使用 y_high 作为分数
    
    if len(all_trajectories) == 0:
        return (
            np.array([]).reshape(0, num_steps + 1, 0),
            np.array([]).reshape(0, 0),
            np.array([])
        )
    
    # 批量堆叠所有轨迹: (N, gp_steps+1, dim)
    trajectories_batch = torch.stack(all_trajectories, dim=0).to(device)
    x0_batch = torch.stack(all_x0, dim=0).to(device)
    N, gp_steps, dim = trajectories_batch.shape

    # [新增] 转换分数
    # y_end 可能是 tensor 也可能是 float，统一转 numpy
    scores_array = np.array([s.item() if torch.is_tensor(s) else s for s in all_scores])
    x0_array = x0_batch.cpu().numpy()
    
    # 如果 GP 步数和目标步数相同，直接返回
    if gp_steps == num_steps + 1:
        return trajectories_batch.cpu().numpy(), x0_array, scores_array
    
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

    
    
    return trajs_array, x0_array, scores_array


def generate_seed_grouped_trajectories_from_GP_samples(
    GP_samples,
    device,
    num_steps=50,
    k_traj_per_seed=4,
):
    """
    从 GP 采样结果构造“按 seed 分组”的多条轨迹与漂移标签。

    Args:
        GP_samples: dict, {f'f{i}': [(trajectory, x_start, y_low, y_high), ...], ...}
                    trajectory 形状为 (gp_steps+1, dim)
        device: torch.device
        num_steps: 目标步数（段数）；返回轨迹点数 T = num_steps+1，dt = 1/num_steps
        k_traj_per_seed: 每个 seed 选择的轨迹条数 K

    Returns:
        seed_x_array: [N_seed, D]，每个 seed 的起点 x_seed
        traj_array: [N_seed, K, T, D]，每个 seed 的 K 条轨迹
        drift_label_array: [N_seed, K, T-1, D]，局部漂移标签 (x_{i+1}-x_i)/dt
        score_array: [N_seed, K, T]，沿轨迹的“伪”score（线性插值，便于计算 y_low/y_high/Δy）
    """
    import torch.nn.functional as F

    # 先收集所有样本，统一重采样到 T = num_steps+1 个点
    all_trajectories = []
    all_x0 = []
    all_y_low = []
    all_y_high = []

    for _, samples in GP_samples.items():
        for trajectory, x_start, y_low, y_high in samples:
            all_trajectories.append(trajectory)  # (gp_steps+1, dim)
            all_x0.append(x_start)
            all_y_low.append(y_low)
            all_y_high.append(y_high)

    if len(all_trajectories) == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0, 0, 0), dtype=np.float32),
            np.zeros((0, 0, 0, 0), dtype=np.float32),
            np.zeros((0, 0, 0), dtype=np.float32),
        )

    trajectories_batch = torch.stack(all_trajectories, dim=0).to(device)  # [N, gp_steps, D]
    x0_batch = torch.stack(all_x0, dim=0).to(device)  # [N, D]

    # 统一整理 y_low / y_high 为 tensor
    y_low_tensor = torch.stack(
        [y if torch.is_tensor(y) else torch.tensor(y, device=device, dtype=torch.float32) for y in all_y_low],
        dim=0,
    )  # [N]
    y_high_tensor = torch.stack(
        [y if torch.is_tensor(y) else torch.tensor(y, device=device, dtype=torch.float32) for y in all_y_high],
        dim=0,
    )  # [N]

    N, gp_steps, dim = trajectories_batch.shape

    # 目标轨迹点数 T = num_steps+1（与 generate_trajectories_from_GP_samples 一致），dt = 1/(T-1) = 1/num_steps
    T_target = num_steps + 1
    if gp_steps == T_target:
        final_trajs = trajectories_batch  # [N, T, D]
    else:
        # 使用线性插值重采样到 T_target 个点
        trajectories_transposed = trajectories_batch.permute(0, 2, 1)  # [N, D, gp_steps]
        resampled_transposed = F.interpolate(
            trajectories_transposed,
            size=T_target,
            mode='linear',
            align_corners=True,
        )  # [N, D, T]
        final_trajs = resampled_transposed.permute(0, 2, 1)  # [N, T, D]

    trajs_array = final_trajs.detach().cpu().numpy()  # [N, T, D]
    x0_array = x0_batch.detach().cpu().numpy()  # [N, D]
    y_low_array = y_low_tensor.detach().cpu().numpy().reshape(-1)  # [N]
    y_high_array = y_high_tensor.detach().cpu().numpy().reshape(-1)  # [N]

    T = trajs_array.shape[1]
    dt = 1.0 / (T - 1)

    # -------------------------
    #  按 seed 分组并为每个 seed 选出 K 条轨迹
    # -------------------------
    from collections import defaultdict

    seed_to_indices = defaultdict(list)
    for idx in range(N):
        # 用 x_seed 的字节表示作为 key（同一个 seed 坐标完全一致）
        key = x0_array[idx].tobytes()
        seed_to_indices[key].append(idx)

    seed_x_list = []
    traj_group_list = []
    drift_group_list = []
    score_group_list = []

    for key, idx_list in seed_to_indices.items():
        if len(idx_list) < k_traj_per_seed:
            continue  # 此 seed 拥有的轨迹数不足 K，直接丢弃

        idx_array = np.array(idx_list, dtype=np.int64)
        # 随机选择 K 条轨迹（不放回），保证每个 seed 的轨迹数一致
        chosen = np.random.choice(idx_array, size=k_traj_per_seed, replace=False)

        trajs_for_seed = trajs_array[chosen]  # [K, T, D]
        y_low_sel = y_low_array[chosen]       # [K]
        y_high_sel = y_high_array[chosen]     # [K]

        # 线性构造沿轨迹的“伪 score”：y(t) = y_low + (y_high - y_low) * (t / 1.0)
        t_grid = np.linspace(0.0, 1.0, T, dtype=np.float32)  # [T]
        delta_y = (y_high_sel - y_low_sel).astype(np.float32)  # [K]
        scores = y_low_sel[:, None].astype(np.float32) + delta_y[:, None] * t_grid[None, :]  # [K, T]

        # 漂移标签: (x_{i+1} - x_i)/dt
        drift = (trajs_for_seed[:, 1:, :] - trajs_for_seed[:, :-1, :]) / dt  # [K, T-1, D]

        seed_x = np.frombuffer(key, dtype=x0_array.dtype).reshape(1, -1)  # [1, D]

        seed_x_list.append(seed_x[0])
        traj_group_list.append(trajs_for_seed)
        drift_group_list.append(drift)
        score_group_list.append(scores)

    if len(seed_x_list) == 0:
        # 没有任何 seed 拥有足够多的轨迹；T = num_steps+1
        T_out = num_steps + 1
        return (
            np.zeros((0, x0_array.shape[1]), dtype=np.float32),
            np.zeros((0, k_traj_per_seed, T_out, x0_array.shape[1]), dtype=np.float32),
            np.zeros((0, k_traj_per_seed, T_out - 1, x0_array.shape[1]), dtype=np.float32),
            np.zeros((0, k_traj_per_seed, T_out), dtype=np.float32),
        )

    seed_x_array = np.stack(seed_x_list, axis=0).astype(np.float32)          # [N_seed, D]
    traj_array = np.stack(traj_group_list, axis=0).astype(np.float32)        # [N_seed, K, T, D]
    drift_label_array = np.stack(drift_group_list, axis=0).astype(np.float32)  # [N_seed, K, T-1, D]
    score_array = np.stack(score_group_list, axis=0).astype(np.float32)      # [N_seed, K, T]

    return seed_x_array, traj_array, drift_label_array, score_array


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