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
    与 ROOT 完全一致：GP 采样，只返回端点对，不返回轨迹
    
    返回格式: datasets = {f'f{i}': [[(high_x, high_y), (low_x, low_y)], ...], ...}
    与 ROOT runners/utils.py 第 92-94 行完全一致
    """
    lengthscale = GP_Model.kernel.lengthscale
    variance = GP_Model.variance 
    torch.manual_seed(seed=seed)
    datasets = {}
    learning_rate_vec = torch.cat((-learning_rate*torch.ones(num_points, x_train.shape[1], device=device), 
                                    learning_rate*torch.ones(num_points, x_train.shape[1], device=device)))

    for iter in range(num_functions):
        datasets[f'f{iter}'] = []
        
        new_lengthscale = lengthscale + delta_lengthscale*(torch.rand(1, device=device)*2 -1)
        new_variance = variance + delta_variance*(torch.rand(1, device=device)*2 -1)
        GP_Model.set_hyper(lengthscale=new_lengthscale, variance=new_variance)
    
        selected_indices = torch.randperm(x_train.shape[0])[:num_points]
        low_x = x_train[selected_indices].clone().detach().requires_grad_(True)
        high_x = x_train[selected_indices].clone().detach().requires_grad_(True)
        joint_x = torch.cat((low_x, high_x)) 
        
        # Using gradient ascent and descent to find high and low designs 
        for t in range(num_gradient_steps): 
            mu_star = GP_Model.mean_posterior(joint_x)
            grad = torch.autograd.grad(mu_star.sum(), joint_x)[0]
            joint_x += learning_rate_vec * grad 

        joint_y = GP_Model.mean_posterior(joint_x)
        
        low_x = joint_x[:num_points, :]
        high_x = joint_x[num_points:, :]
        low_y = joint_y[:num_points]
        high_y = joint_y[num_points:]
        
        for i in range(num_points):
            if high_y[i] - low_y[i] <= threshold_diff:
                continue
            # 与 ROOT 完全一致：返回 [(high_x, high_y), (low_x, low_y)]
            sample = [(high_x[i].detach(), high_y[i].detach()), (low_x[i].detach(), low_y[i].detach())]
            datasets[f'f{iter}'].append(sample)

    # restore lengthscale and variance of GP
    GP_Model.kernel.lengthscale = lengthscale
    GP_Model.variance = variance
    
    return datasets


# ========== 与 ROOT 一致的 DataLoader 工具函数 ==========
class CustomDataset(torch.utils.data.Dataset):
    """与 ROOT runners/utils.py 第 102-111 行完全一致"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        [[x_high, y_high], [x_low, y_low]] = self.data[idx]
        return (x_high, y_high), (x_low, y_low)

def create_train_dataloader(data_from_GP, val_frac=0.2, batch_size=32, shuffle=True):
    """
    与 ROOT runners/utils.py 第 114-124 行完全一致
    每个 function 只取 samples[int(len*val_frac):]（后 80%）用于训练
    """
    train_data = []
    val_data = []
    for function, function_samples in data_from_GP.items():
        train_data = train_data + function_samples[int(len(function_samples)*val_frac):]
        val_data = val_data + function_samples[:int(len(function_samples)*val_frac)]
        
    train_dataset = CustomDataset(train_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_data


def create_val_dataloader(val_dataset, batch_size=32, shuffle=False):
    """与 ROOT runners/utils.py 第 126-131 行一致"""
    valid_dataset = CustomDataset(val_dataset)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    return valid_dataloader

# ========== 以下函数 ROOT 没有，暂时注释 ==========
# def generate_trajectories_from_GP_samples(GP_samples, device, num_steps=50):
    # """
    # 从 GP 采样得到的完整非线性轨迹生成训练数据（高效版本）
    # 使用 GP 梯度搜索生成的真实非线性路径，而不是简单的直线插值
    
    # Args:
    #     GP_samples: dict, 格式为 {f'f{i}': [(trajectory, y_start, y_end), ...], ...}
    #                 其中 trajectory 是 (gp_steps+1, dim) 的完整 GP 梯度搜索路径
    #     device: torch.device
    #     num_steps: int, 目标轨迹步数（用于重采样）
    
    # Returns:
    #     trajs_array: numpy array (num_trajs, num_steps+1, dim)
    # """
    # # 收集所有轨迹及起点/终点分数（BB 训练需要 y_low, y_high）
    # all_trajectories = []
    # all_scores_low = []
    # all_scores_high = []

    # for func_name, samples in GP_samples.items():
    #     for sample in samples:
    #         trajectory, y_start, y_end = sample
    #         all_trajectories.append(trajectory)
    #         all_scores_low.append(y_start)
    #         all_scores_high.append(y_end)

    # if len(all_trajectories) == 0:
    #     return np.array([]).reshape(0, num_steps + 1, 0), np.array([]), np.array([])
    
    # # 批量堆叠所有轨迹: (N, gp_steps+1, dim)
    # trajectories_batch = torch.stack(all_trajectories, dim=0).to(device)
    # N, gp_steps, dim = trajectories_batch.shape

    # def _to_np(s):
    #     return s.item() if torch.is_tensor(s) else float(s)
    # scores_low_array = np.array([_to_np(s) for s in all_scores_low])
    # scores_high_array = np.array([_to_np(s) for s in all_scores_high])

    # # 如果 GP 步数和目标步数相同，直接返回
    # if gp_steps == num_steps + 1:
    #     return trajectories_batch.cpu().numpy(), scores_low_array, scores_high_array
    
    # # ✅ 高效重采样：批量插值（完全向量化，无循环）
    # # 使用 F.interpolate 批量处理所有轨迹
    # # 输入: (N, dim, gp_steps) - batch, channels, length
    # # 输出: (N, dim, num_steps+1)
    # trajectories_transposed = trajectories_batch.permute(0, 2, 1)  # (N, dim, gp_steps)
    
    # resampled_transposed = torch.nn.functional.interpolate(
    #     trajectories_transposed,
    #     size=num_steps + 1,
    #     mode='linear',
    #     align_corners=True
    # )  # (N, dim, num_steps+1)
    
    # # 转回 (N, num_steps+1, dim)
    # final_trajs = resampled_transposed.permute(0, 2, 1)
    
    # trajs_array = final_trajs.cpu().numpy()
    # return trajs_array, scores_low_array, scores_high_array


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