# src/generator.py
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import math
from src.config import Config
from gpytorch.kernels import(
    RBFKernel, LinearKernel, MaternKernel, RQKernel, PeriodicKernel,
    CosineKernel, PolynomialKernel 
)

kernel_dict = {'rbf': RBFKernel,'matern': MaternKernel, 
                'rq' : RQKernel, 'period': PeriodicKernel, 'cosine': CosineKernel,
                'poly': PolynomialKernel}

class RFFGP(nn.Module):
    """
    RFF-GP: 使用随机傅里叶特征近似的高斯过程
    适用于大规模数据(N > 10k)和需要对输入x求导的场景(如梯度上升)
    """
    def __init__(self, device, x_train, y_train, lengthscale, variance, noise, num_features=1024):
        super().__init__()
        self.device = device
        self.num_features = num_features
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        
        # 确保输入数据在正确的设备上
        self.x_train = x_train.to(device)
        self.y_train = y_train.to(device)
        
        # 数据维度
        self.input_dim = x_train.shape[1]
        
        # --- 1. 初始化随机特征权重 (固定不更新) ---
        # 对应 RBF 核的谱密度是高斯分布
        # W ~ N(0, 1), b ~ U(0, 2pi)
        self.register_buffer('W', torch.randn(self.num_features, self.input_dim, device=device))
        self.register_buffer('b', torch.rand(self.num_features, device=device) * 2 * math.pi)
        
        # --- 2. 拟合模型 (一次性计算) ---
        self._fit()
        
    def _compute_features(self, x):
        """
        计算特征映射 phi(x)
        Formula: phi(x) = sqrt(2/D) * cos(Wx/l + b)
        """
        # x: (Batch, Input_Dim)
        # W: (Num_Features, Input_Dim)
        # term: (Batch, Num_Features)
        
        # 注意：除以 lengthscale
        projection = torch.matmul(x / self.lengthscale, self.W.t()) + self.b
        phi = math.sqrt(2.0 / self.num_features) * torch.cos(projection)
        
        # 乘上信号方差的平方根，使内积近似于 variance * kernel
        return phi * math.sqrt(self.variance)

    def _fit(self):
        """
        贝叶斯线性回归的闭式解
        将 N 个训练样本的信息压缩进权重 w_mean 和协方差矩阵 Sigma 中
        """
        with torch.no_grad():
            # 1. 计算训练集的特征矩阵 Phi: (N_train, Num_Features)
            Phi = self._compute_features(self.x_train)
            
            # 2. 计算精度矩阵 A = Phi^T @ Phi + sigma_n^2 * I
            # 这里假设权重先验 w ~ N(0, 1)
            # 对于数值稳定性，噪声项加在对角线
            A = torch.matmul(Phi.t(), Phi)
            identity = torch.eye(self.num_features, device=self.device)
            A.add_(identity * self.noise) 
            
            # 3. Cholesky 分解 A = L @ L.T
            # L: (Num_Features, Num_Features) -> (1024, 1024)
            # 这个矩阵比原始 GP 的 (10000, 10000) 小得多
            L = torch.linalg.cholesky(A)
            
            # 4. 计算后验均值的权重 w_mean
            # solve A @ w = Phi^T @ y
            rhs = torch.matmul(Phi.t(), self.y_train) # (Num_Features, )
            
            # 使用 cholesky_solve 求解
            # w_mean: (Num_Features, )
            self.w_mean = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze()
            
            # 5. 预计算 L 的逆，用于快速方差计算
            # L_inv: (Num_Features, Num_Features)
            self.L_inv = torch.linalg.solve_triangular(L, identity, upper=False)
            
            # 清理显存（不再需要 Phi 和 A）
            del Phi, A, L, rhs
            torch.cuda.empty_cache()

    def predict_with_uncertainty(self, x_test):
        """
        前向预测，支持 Autograd
        """
        # 1. 计算测试点特征
        # phi: (Batch, Num_Features)
        phi = self._compute_features(x_test)
        
        # 2. 均值预测: mu = phi @ w_mean
        mu = torch.matmul(phi, self.w_mean)
        
        # 3. 方差预测 (Epistemic Uncertainty)
        # Var(x) = phi @ Sigma @ phi.T
        # 其中 Sigma = inv(A) * noise (近似) 或直接是 inv(A) 视推导而定
        # 标准形式下，预测方差 = noise + phi @ A^{-1} @ phi.T * noise
        
        # 计算 v = L^{-1} @ phi.T -> (Num_Features, Batch)
        # L_inv 是预计算好的 (1024, 1024)
        v = torch.matmul(self.L_inv, phi.t())
        
        # 计算 ||v||^2，这是模型的不确定性部分
        # sum(v**2, dim=0) -> (Batch, )
        model_uncertainty = torch.sum(v * v, dim=0) * self.noise
        
        # 总方差 = 模型方差 + 数据噪声
        # 如果目的是为了探索（Gradient Ascent），通常只需要 model_uncertainty
        # 为了防止数值问题导致负数，用 clamp
        total_variance = torch.clamp(model_uncertainty, min=1e-6)
        std = torch.sqrt(total_variance)
        
        return mu, std
    # --- 请将以下两个方法添加到 RFFGP 类中 ---

    def set_hyper(self, lengthscale, variance):
        """
        动态更新超参数并重新拟合后验权重
        RFF的优势：这里只需要解一个 DxD (1024x1024) 的方程，而不是 NxN，速度极快
        """
        self.lengthscale = lengthscale
        self.variance = variance
        # 参数变了，必须重新计算 w_mean 和 L_inv
        self._fit()

    def predict_mean(self, x_test):
        """
        只计算均值，不计算方差。
        用于梯度上升中不需要方差惩罚的步骤，速度是光速。
        """
        # phi: (Batch, Num_Features)
        phi = self._compute_features(x_test)
        # mu = phi @ w_mean
        return torch.matmul(phi, self.w_mean)

class SampledGPFunction:
    """从 GP 后验中采样出的固定函数对象"""
    def __init__(self, gp_model, w_prior, W, b):
        self.gp_model = gp_model
        self.w_prior = w_prior  # (M,)
        self.W = W              # (M, D)
        self.b = b              # (M,)
        self.num_features = W.shape[0]
        
        # 预计算在训练集上的 RFF 特征，用于计算条件项
        with torch.no_grad():
            # phi_train: (N, M)
            phi_train = self._compute_prior_features(gp_model.x_train)
            # f_prior_train: (N,)
            f_prior_train = torch.matmul(phi_train, w_prior)
            # 计算条件权重: v = K_inv @ (f_prior_train)
            # 使用已有的 Cholesky 因子 L
            self.v_cond = torch.cholesky_solve(f_prior_train.unsqueeze(-1), gp_model.L).squeeze(-1)

    def _compute_prior_features(self, x):
        """计算 RFF 特征 phi(x)"""
        # x: (..., D)
        # projection: (..., M)
        projection = torch.matmul(x / self.gp_model.kernel.lengthscale, self.W.t()) + self.b
        phi = math.sqrt(2.0 * self.gp_model.variance / self.num_features) * torch.cos(projection)
        return phi

    def __call__(self, x):
        """计算 f(x) = mu(x) + f_prior(x) - E[f_prior(x) | data]"""
        # mu(x)
        mu = self.gp_model.mean_posterior(x)
        
        # f_prior(x)
        phi_x = self._compute_prior_features(x)
        f_prior = torch.matmul(phi_x, self.w_prior)
        
        # E[f_prior(x) | data] = k(x, x_train) @ K_inv @ f_prior_train
        k_x_train = self.gp_model.variance * self.gp_model.kernel.forward(x, self.gp_model.x_train)
        f_prior_cond = torch.matmul(k_x_train, self.v_cond)
        
        return mu + f_prior - f_prior_cond

    def grad(self, x):
        """计算 f(x) 对 x 的梯度"""
        x_req = x.detach().requires_grad_(True)
        y = self.__call__(x_req)
        grad = torch.autograd.grad(y.sum(), x_req)[0]
        return grad

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
        
    def sample_functions(self, num_functions=8, num_features=1024, seed=None):
        """从后验分布中采样 K 个函数"""
        if seed is not None:
            torch.manual_seed(seed)
        
        functions = []
        D = self.x_train.shape[1]
        
        for _ in range(num_functions):
            # 采样 RFF 权重和基向量
            W = torch.randn(num_features, D, device=self.device)
            b = torch.rand(num_features, device=self.device) * 2 * math.pi
            w_prior = torch.randn(num_features, device=self.device)
            
            f = SampledGPFunction(self, w_prior, W, b)
            functions.append(f)
            
        return functions

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
            # 存储 Cholesky 因子 L，供后续方差计算复用
            self.L = L.detach()
            # 预计算 L^{-1}，用于加速方差计算（避免每次 solve_triangular）
            # 使用 triangular_solve 求 L @ L_inv = I，即 L_inv = L^{-1}
            I = torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
            self.L_inv = torch.linalg.solve_triangular(L, I, upper=False).detach()
            b = (self.y_train - self.mean_prior).unsqueeze(-1)
            self.coef = torch.cholesky_solve(b, L).squeeze(-1).detach()
    
    def mean_posterior(self, x_test): 
        # Posterior mean
        K_train_test = self.variance * self.kernel.forward(self.x_train, x_test)
        mu_star = self.mean_prior + torch.matmul(K_train_test.T, self.coef)
        return mu_star
    
    def predict_with_uncertainty(self, x_test):
        """
        同时返回后验均值和标准差（可微分）
        
        Args:
            x_test: (N, D) 测试点
            
        Returns:
            mu: (N,) 后验均值
            std: (N,) 后验标准差
            
        方差计算公式：
            Var(x_*) = K(x_*, x_*) - k_*^T (L L^T)^{-1} k_*
                     = K(x_*, x_*) - || L^{-1} k_* ||^2
        
        优化：使用预计算的 L_inv 进行矩阵乘法，避免每次 solve_triangular
        """
        # 计算 K(x_train, x_test): (N_train, N_test)
        K_train_test = self.variance * self.kernel.forward(self.x_train, x_test)
        
        # 计算后验均值: mu = mean_prior + K_train_test^T @ coef
        mu = self.mean_prior + torch.matmul(K_train_test.T, self.coef)
        
        # 计算后验方差（可微分）
        # K(x_*, x_*) 对于 RBF 等平稳核，对角线元素等于 variance
        K_test_test_diag = self.variance * torch.ones(x_test.shape[0], device=x_test.device, dtype=x_test.dtype)
        
        # 使用预计算的 L_inv 进行矩阵乘法（比 solve_triangular 快很多）
        # v = L^{-1} @ K_train_test
        v = torch.matmul(self.L_inv, K_train_test)  # (N_train, N_test)
        
        # 方差 = K(x_*, x_*) - || v ||^2 (沿 N_train 维度求和)
        var = K_test_test_diag - torch.sum(v ** 2, dim=0)
        
        # 数值稳定性：确保方差非负
        var = torch.clamp(var, min=1e-8)
        std = torch.sqrt(var)
        
        return mu, std


def sampling_data_from_GP(x_train, device, GP_Model, gp_functions, num_gradient_steps=50, num_points=10, 
                        eta_min=0.01, eta_max=0.1, sigma_max=0.1, threshold_diff=0.1, 
                        uncertainty_penalty=1.0, uncertainty_interval=1, max_end_uncertainty=None, verbose=False):
    """
    适配固定 GP 函数集合的 Langevin 动力学采样版本
    """
    import time
    import random
    
    datasets = {}
    num_functions = len(gp_functions)
    
    # 初始化数据集字典
    for i in range(num_functions):
        datasets[f'f{i}'] = []

    total_gradient_time = 0
    
    # 轨迹生成逻辑
    # 为了效率，我们按 GP 函数分组处理，或者随机分配
    # 这里采用随机分配：每个点随机选一个函数
    
    # 1. 随机选择起始点
    selected_indices = torch.randperm(x_train.shape[0])[:num_points]
    x_start = x_train[selected_indices].clone()
    
    # 2. 固定动力学参数（简化版：固定步长、无噪声）
    etas = torch.full((num_points, 1), eta_min, device=device)
    sigmas = torch.zeros((num_points, 1), device=device)
    
    # 3. 为每个点随机分配一个 GP 函数
    func_indices = [random.randint(0, num_functions - 1) for _ in range(num_points)]
    
    # 4. Langevin Rollout
    t1 = time.time()
    
    # 预分配存储
    trajectory_storage = torch.zeros(num_gradient_steps + 1, num_points, x_train.shape[1], 
                                    device=device, dtype=x_train.dtype)
    
    x_curr = x_start.clone().requires_grad_(True)
    trajectory_storage[0] = x_curr.detach()

    for t in range(num_gradient_steps):
        # 计算梯度：由于每个点可能对应不同函数，我们需要循环或者分组
        # 这里简单起见使用循环计算梯度（如果 K 不大且 num_points 适中，性能可接受）
        # 优化：如果追求极致速度，可以按 func_idx 分组后批量计算
        
        grads = torch.zeros_like(x_curr)
        total_grad_norm = 0
        active_funcs = 0
        
        for f_idx in range(num_functions):
            mask = [i for i, idx in enumerate(func_indices) if idx == f_idx]
            if not mask:
                continue
            
            active_funcs += 1
            x_subset = x_curr[mask]
            f = gp_functions[f_idx]
            
            # 使用 GP 均值 + 采样函数 + 不确定性 共同决定目标：
            # - mu_sub: mean_posterior，代表 GP 对 oracle 的平均判断
            # - f_val:  对应采样函数 f_k(x) 的取值，引入探索
            # - std_subset: GP 不确定性，高不确定区域会被惩罚
            mu_sub, std_subset = GP_Model.predict_with_uncertainty(x_subset)
            f_val = f(x_subset)
            # alpha=1.0: 完全跟 mu_sub，alpha<1 增加 f_val 的探索成分
            alpha = 0.5
            obj = mu_sub + alpha * (f_val - mu_sub) - uncertainty_penalty * std_subset
            
            grads_subset = torch.autograd.grad(obj.sum(), x_subset)[0]
            total_grad_norm += grads_subset.norm(dim=-1).mean().item()
            grads[mask] = grads_subset

        # Langevin 更新（简化版：无噪声）
        with torch.no_grad():
            x_curr += etas * grads
        
        trajectory_storage[t + 1] = x_curr.detach()
        
        if verbose and t == 0:
            print(f"    [Diagnostic] Step 0 Avg Grad Norm: {total_grad_norm/active_funcs:.6f}")

    total_gradient_time += time.time() - t1
    
    # 5. 批量后处理与过滤
    with torch.no_grad():
        # 打分使用均值后验（Oracle 的代理）
        y_start = GP_Model.mean_posterior(trajectory_storage[0])
        y_end = GP_Model.mean_posterior(trajectory_storage[-1])
        
        diff = y_end - y_start
        
        # --- [诊断信息] ---
        if verbose:
            print(f"    [Diagnostic] Total Points: {num_points}")
            print(f"    [Diagnostic] y_start: mean={y_start.mean():.4f}, min={y_start.min():.4f}, max={y_start.max():.4f}")
            print(f"    [Diagnostic] y_end:   mean={y_end.mean():.4f}, min={y_end.min():.4f}, max={y_end.max():.4f}")
            print(f"    [Diagnostic] Score Diff: mean={diff.mean():.4f}, max={diff.max():.4f}")
            
            # 额外诊断：用每条轨迹对应的采样函数 f 评估起点/终点
            f_start = torch.zeros_like(y_start)
            f_end = torch.zeros_like(y_end)
            for f_idx in range(num_functions):
                mask = [i for i, idx in enumerate(func_indices) if idx == f_idx]
                if not mask:
                    continue
                f = gp_functions[f_idx]
                f_start[mask] = f(trajectory_storage[0][mask])
                f_end[mask] = f(trajectory_storage[-1][mask])
            f_diff = f_end - f_start
            print(f"    [Diagnostic] f_start: mean={f_start.mean():.4f}, min={f_start.min():.4f}, max={f_start.max():.4f}")
            print(f"    [Diagnostic] f_end:   mean={f_end.mean():.4f}, min={f_end.min():.4f}, max={f_end.max():.4f}")
            print(f"    [Diagnostic] f_diff:  mean={f_diff.mean():.4f}, max={f_diff.max():.4f}, pos={int((f_diff>0).sum().item())}")
        else:
            # 非 verbose 时也需要 f_diff 用于过滤
            f_start = torch.zeros_like(y_start)
            f_end = torch.zeros_like(y_end)
            for f_idx in range(num_functions):
                mask = [i for i, idx in enumerate(func_indices) if idx == f_idx]
                if not mask:
                    continue
                f = gp_functions[f_idx]
                f_start[mask] = f(trajectory_storage[0][mask])
                f_end[mask] = f(trajectory_storage[-1][mask])
            f_diff = f_end - f_start
        
        # 过滤逻辑基于采样函数 f_k 的提升（探索性）
        valid_mask = f_diff > threshold_diff
        print(f"    [Diagnostic] Passed threshold ({threshold_diff}): {valid_mask.sum().item()}")
        
        if max_end_uncertainty is not None:
            _, std_end = GP_Model.predict_with_uncertainty(trajectory_storage[-1])
            unc_mask = std_end < max_end_uncertainty
            print(f"    [Diagnostic] Passed Uncertainty ({max_end_uncertainty}): {unc_mask.sum().item()}")
            valid_mask = valid_mask & unc_mask
        # ------------------
        
        # 过滤并存入对应函数的列表
        # 注意：这里存的“打分分数”用于下游 rank-based weighting，
        #       应与 GP 均值 mean_posterior 对齐，而不是采样函数 f_k。
        for i in range(num_points):
            if valid_mask[i]:
                f_idx = func_indices[i]
                datasets[f'f{f_idx}'].append((
                    trajectory_storage[:, i, :].cpu(),
                    y_start[i].cpu(),
                    y_end[i].cpu()
                ))

    if verbose:
        print(f"GP Langevin Sampling Report (Fixed Functions):")
        print(f"  Rollout Time: {total_gradient_time:.4f}s")

    return datasets
# def sampling_data_from_GP(x_train, device, GP_Model, num_gradient_steps=50, num_functions=1, num_points=10, 
#                         eta_min=0.01, eta_max=0.1, sigma_max=0.1, seed=0, threshold_diff=0.1, 
#                         uncertainty_penalty=1.0, uncertainty_interval=1, max_end_uncertainty=None, verbose=False):
#     """
#     适配 RFF-GP 的 Langevin 动力学采样版本
#     """
#     import time
    
#     torch.manual_seed(seed=seed)
#     datasets = {}

#     total_gradient_time = 0
    
#     total_steps = num_gradient_steps 
#     trajectory_storage = torch.zeros(total_steps + 1, num_points, x_train.shape[1], 
#                                     device=device, dtype=x_train.dtype)

#     for iter in range(num_functions):
#         datasets[f'f{iter}'] = []
        
#         # --- 1. 随机选择起始点 ---
#         selected_indices = torch.randperm(x_train.shape[0])[:num_points]
#         x_start = x_train[selected_indices].clone()
        
#         # --- 2. 采样动力学参数 (eta 和 sigma) ---
#         # eta = 10 ** uniform(log10(eta_min), log10(eta_max))
#         log_eta_min = math.log10(eta_min)
#         log_eta_max = math.log10(eta_max)
#         random_log_eta = torch.rand(num_points, 1, device=device) * (log_eta_max - log_eta_min) + log_eta_min
#         etas = 10 ** random_log_eta
        
#         # sigma = uniform(0, sigma_max)
#         sigmas = torch.rand(num_points, 1, device=device) * sigma_max
        
#         # --- 3. 梯度上升/下降主循环 (Langevin Rollout) ---
#         t1 = time.time()
        
#         x_curr = x_start.clone().requires_grad_(True)
#         trajectory_storage[0] = x_curr.detach()

#         for t in range(total_steps):
#             # A. 计算均值和方差引导的梯度
#             # 始终使用均值和方差引导
#             mu, std = GP_Model.predict_with_uncertainty(x_curr)
#             objective = mu - uncertainty_penalty * std
            
#             # B. 计算对 x 的梯度
#             grads = torch.autograd.grad(objective.sum(), x_curr)[0]
            
#             # 梯度裁剪防止爆炸
#             torch.nn.utils.clip_grad_norm_([x_curr], max_norm=1.0)
            
#             # C. Langevin 更新: x = x + eta * grad + sigma * noise
#             with torch.no_grad():
#                 noise = torch.randn_like(x_curr)
#                 x_curr += etas * grads + sigmas * noise
            
#             # D. 存储轨迹
#             trajectory_storage[t + 1] = x_curr.detach()

#         total_gradient_time += time.time() - t1
        
#         # --- 4. 批量后处理与过滤 ---
#         with torch.no_grad():
#             y_start = GP_Model.predict_mean(trajectory_storage[0])
#             y_end = GP_Model.predict_mean(trajectory_storage[-1])
            
#             valid_mask = (y_end - y_start) > threshold_diff
            
#             if max_end_uncertainty is not None:
#                 _, std_end = GP_Model.predict_with_uncertainty(trajectory_storage[-1])
#                 unc_mask = std_end < max_end_uncertainty
#                 valid_mask = valid_mask & unc_mask
            
#             valid_indices = torch.where(valid_mask)[0]
            
#             if len(valid_indices) > 0:
#                 valid_trajs = trajectory_storage[:, valid_indices, :].permute(1, 0, 2).cpu()
#                 valid_y_start = y_start[valid_indices].cpu()
#                 valid_y_end = y_end[valid_indices].cpu()
                
#                 for k in range(len(valid_indices)):
#                     datasets[f'f{iter}'].append((
#                         valid_trajs[k], 
#                         valid_y_start[k], 
#                         valid_y_end[k]
#                     ))

#     if verbose:
#         print(f"RFF-GP Langevin Sampling Report:")
#         print(f"  Rollout Time:      {total_gradient_time:.4f}s")
#         print(f"  Avg Time per Func: {total_gradient_time/num_functions:.4f}s")

#     return datasets



def generate_trajectories_from_GP_samples(GP_samples, device, num_steps=50):
    """
    从 GP 采样得到的完整非线性轨迹生成训练数据（高效版本）
    使用 GP 梯度搜索生成的真实非线性路径，而不是简单的直线插值
    
    Args:
        GP_samples: dict, 格式为 {f'f{i}': [(trajectory, y_start, y_end), ...], ...}
                    其中 trajectory 是 (gp_steps+1, dim) 的完整 GP 梯度搜索路径
                    y_* 为 GP mean_posterior 在起点/终点的取值（用于 rank-based weighting）
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
            # 下游权重使用 GP 均值 y_end（与 oracle proxy 对齐）
            all_scores.append(y_end)
    
    if len(all_trajectories) == 0:
        return np.array([]).reshape(0, num_steps + 1, 0), np.array([])
    
    # 批量堆叠所有轨迹: (N, gp_steps+1, dim)
    trajectories_batch = torch.stack(all_trajectories, dim=0).to(device)
    N, gp_steps, dim = trajectories_batch.shape

    # [新增] 转换分数
    # y_end 可能是 tensor 也可能是 float，统一转 numpy
    scores_array = np.array([s.item() if torch.is_tensor(s) else s for s in all_scores])
    # scores_array=all_scores
    
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


# def generate_long_trajectories(oracle, X_init_numpy, device):
#     """
#     保留原有函数以保持兼容性（如果其他地方还在使用）
#     ROOT 风格长轨迹：GD 反转(500步) + GA 冲顶(800步)
#     """
#     X_curr = torch.FloatTensor(X_init_numpy).to(device).requires_grad_(True)
#     X_start = torch.FloatTensor(X_init_numpy).to(device)
    
#     # --- 阶段 B: 冲顶 (GA) 200步 ---
#     X_curr.data = X_start.data.clone()
#     y_target_high = torch.full((X_start.shape[0], 1), 4.0).to(device)
#     traj_part2 = [X_curr.detach().cpu().clone()]
    
#     opt_asc = optim.Adam([X_curr], lr=Config.TRAJ_LR)
#     for _ in range(Config.TRAJ_STEPS_ASC):
#         opt_asc.zero_grad()
#         mu = oracle.predict_mean(X_curr)
#         sigma_sq = oracle.predict_uncertainty(X_curr)
        
#         loss_fwd = torch.mean((mu - y_target_high) ** 2)
#         loss_bwd = torch.mean((X_curr - X_start) ** 2)
#         loss_unc = torch.mean(sigma_sq)
        
#         loss = (loss_fwd 
#                 + Config.LAMBDA_BACKWARD * loss_bwd 
#                 + Config.LAMBDA_UNCERTAINTY * loss_unc)
        
#         loss.backward()
#         opt_asc.step()
#         traj_part2.append(X_curr.detach().cpu().clone())

#     full_traj = traj_part2
#     traj_tensor = torch.stack(full_traj, dim=1).to(device)
    
#     valid_trajs = _filter_trajectories(oracle, traj_tensor)
    
#     return valid_trajs.cpu().numpy()


# def _filter_trajectories(oracle, traj_tensor):
#     """验证: mu(XT) - k*sigma(XT) > mu(X0) + k*sigma(X0)"""
#     with torch.no_grad():
#         mu_0 = oracle.predict_mean(traj_tensor[:, 0, :])
#         sig_0 = torch.sqrt(oracle.predict_uncertainty(traj_tensor[:, 0, :]))
        
#         mu_T = oracle.predict_mean(traj_tensor[:, -1, :])
#         sig_T = torch.sqrt(oracle.predict_uncertainty(traj_tensor[:, -1, :]))
    
#     lower_bound_T = mu_T - Config.KAPPA * sig_T
#     upper_bound_0 = mu_0 + Config.KAPPA * sig_0
    
#     valid_mask = (lower_bound_T > upper_bound_0).squeeze()
#     num_valid = valid_mask.sum().item()
#     print(f"[Generator] Filtered {traj_tensor.shape[0]} -> {num_valid} valid trajectories.")
    
#     return traj_tensor[valid_mask]