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


# def sampling_data_from_GP(x_train, device, GP_Model, num_gradient_steps=50, num_functions=5, num_points=10, 
#                           learning_rate=0.001, delta_lengthscale=0.1, delta_variance=0.1, seed=0, threshold_diff=0.1, 
#                           uncertainty_penalty=0.5, uncertainty_interval=5, max_end_uncertainty=None, verbose=False):
#     """
#     改进版 GP 采样：生成从 x_low 到 x_high 的连续上升轨迹（带不确定性感知）
    
#     流程：
#     1. 选择起始点
#     2. 梯度下降找到 x_low（低分起点）
#     3. 从 x_low 开始梯度上升，记录每一步，形成完整的"弧度轨迹"
#     4. 最终得到 x_high，验证分数差和不确定性
    
#     Args:
#         uncertainty_penalty: float, 不确定性惩罚系数 (lambda)。
#                              优化目标为 objective = mu - lambda * std。
#                              设为 0 则退化为仅基于均值的梯度上升。
#         uncertainty_interval: int, 每隔多少步计算一次方差惩罚（默认10步）。
#                               设为 1 则每步都计算（最慢但最精确）。
#         max_end_uncertainty: float or None, 终点最大允许的不确定性（标准差）。
#                              如果终点 std > max_end_uncertainty，则丢弃该轨迹。
#                              设为 None 则不进行此过滤。
    
#     返回格式: datasets = {f'f{i}': [(trajectory, y_start, y_end), ...], ...}
#     其中 trajectory 是 (num_gradient_steps+1, dim) 的完整上升轨迹
#     """
#     import time
#     lengthscale = GP_Model.kernel.lengthscale
#     variance = GP_Model.variance 
#     torch.manual_seed(seed=seed)
#     datasets = {}

#     total_set_hyper_time = 0
#     total_gradient_time = 0
#     total_posterior_time = 0
    
#     for iter in range(num_functions):
#         datasets[f'f{iter}'] = []
        
#         # 1. 为每个函数采样不同的超参数
#         set_hyper_start = time.time()
#         new_lengthscale = lengthscale + delta_lengthscale*(torch.rand(1, device=device)*2 -1)
#         new_variance = variance + delta_variance*(torch.rand(1, device=device)*2 -1)
#         GP_Model.set_hyper(lengthscale=new_lengthscale, variance=new_variance)
#         total_set_hyper_time += time.time() - set_hyper_start
    
#         # 2. 随机选择起始点
#         selected_indices = torch.randperm(x_train.shape[0])[:num_points]
#         x_start = x_train[selected_indices].clone()
        
#         gradient_start = time.time()
        
#         # 3. 第一阶段：梯度下降找到 x_low（低分起点）
#         # ✅ 优化：使用 in-place 操作，避免重复创建 tensor
#         x_low = x_start.clone().requires_grad_(True)
#         # x_low = x_start.clone().requires_grad_(True)
#         # with torch.enable_grad():
#         #     for _ in range(num_gradient_steps):
#         #         mu_star = GP_Model.mean_posterior(x_low)
#         #         grad = torch.autograd.grad(mu_star.sum(), x_low, create_graph=False)[0]
#         #         x_low.data.sub_(learning_rate * grad)  # in-place: x_low -= lr * grad
        
#         # 4. 第二阶段：从 x_low 开始梯度上升，记录每一步形成"弧度轨迹"
#         # ✅ 预分配内存存储轨迹: (num_gradient_steps+1, num_points, dim)
#         trajectory_storage = torch.zeros(num_gradient_steps + num_gradient_steps + 1, num_points, x_train.shape[1], 
#                                         device=device, dtype=x_train.dtype)
        
#         # ✅ 复用 x_low，避免 clone
#         x_curr = x_low
#         trajectory_storage[0] = x_curr.data.clone()  # 只在保存时 clone
        
#         with torch.enable_grad():
#             for t in range(num_gradient_steps+num_gradient_steps):
#                 # 不确定性感知梯度上升 (Uncertainty-Aware Gradient Ascent)
#                 # 每隔 uncertainty_interval 步计算一次方差惩罚
#                 if uncertainty_penalty > 0 and t % uncertainty_interval == 0:
#                     # 方差惩罚步：objective = mu - lambda * std
#                     mu, std = GP_Model.predict_with_uncertainty(x_curr)
#                     objective = mu - uncertainty_penalty * std
#                 else:
#                     # 普通步：只用均值，速度快
#                     objective = GP_Model.mean_posterior(x_curr)
                
#                 grad = torch.autograd.grad(objective.sum(), x_curr, create_graph=False)[0]
#                 x_curr.data.add_(learning_rate * grad)  # in-place: x_curr += lr * grad
#                 trajectory_storage[t + 1] = x_curr.data.clone()  # 只在保存时 clone
        
#         total_gradient_time += time.time() - gradient_start
        
#         # 5. 批量计算起点和终点的分数 + 最终安检
#         posterior_start = time.time()
#         with torch.no_grad():
#             y_start = GP_Model.mean_posterior(trajectory_storage[0])  # x_low 的分数
#             y_end = GP_Model.mean_posterior(trajectory_storage[-1])   # x_high 的分数
            
#             # ✅ 基础过滤：分数提升必须大于阈值
#             valid_mask = (y_end - y_start) > threshold_diff
            
#             # ✅ 最终安检：过滤掉终点不确定性过高的轨迹
#             if max_end_uncertainty is not None:
#                 _, std_end = GP_Model.predict_with_uncertainty(trajectory_storage[-1])
#                 uncertainty_mask = std_end < max_end_uncertainty
#                 valid_mask = valid_mask & uncertainty_mask
                
#         total_posterior_time += time.time() - posterior_start
        
#         # 6. 批量过滤和保存轨迹
#         # ✅ 优化：使用向量化操作，避免循环
#         valid_indices = torch.where(valid_mask)[0]
#         for i in valid_indices:
#             i_cpu = i.item()
#             # trajectory_storage 已经是 detached 的（因为我们用 .data.clone()）
#             sample = (trajectory_storage[:, i_cpu, :], y_start[i_cpu], y_end[i_cpu])
#             datasets[f'f{iter}'].append(sample)

#     # 恢复原始超参数
#     GP_Model.kernel.lengthscale = lengthscale
#     GP_Model.variance = variance
    
#     if verbose:
#         print(f"    [GP内部] set_hyper: {total_set_hyper_time:.2f}s | 梯度采样: {total_gradient_time:.2f}s | 后验: {total_posterior_time:.2f}s")
    
#     return datasets
def sampling_data_from_GP(x_train, device, GP_Model, num_gradient_steps=50, num_functions=5, num_points=10, 
                        learning_rate=0.01, delta_lengthscale=0.1, delta_variance=0.1, seed=0, threshold_diff=0.1, 
                        uncertainty_penalty=0.5, uncertainty_interval=5, max_end_uncertainty=None, verbose=False):
    """
    适配 RFF-GP 的极速采样版本
    """
    import time
    
    # 记录原始超参数，以便函数结束时恢复
    original_lengthscale = GP_Model.lengthscale
    original_variance = GP_Model.variance
    
    torch.manual_seed(seed=seed)
    datasets = {}

    total_set_hyper_time = 0
    total_gradient_time = 0
    
    # 预分配轨迹存储 tensor，避免在循环中重复申请显存
    # Shape: (total_steps + 1, num_points, dim)
    # 注意：这里假设总步数是 num_gradient_steps (如果像你之前代码是 2倍，请自行调整)
    total_steps = num_gradient_steps 
    trajectory_storage = torch.zeros(total_steps + 1, num_points, x_train.shape[1], 
                                    device=device, dtype=x_train.dtype)

    for iter in range(num_functions):
        datasets[f'f{iter}'] = []
        
        # --- 1. 极速超参数扰动 ---
        t0 = time.time()
        # 随机扰动
        rand_l = (torch.rand(1, device=device) * 2 - 1) * delta_lengthscale
        rand_v = (torch.rand(1, device=device) * 2 - 1) * delta_variance
        
        new_lengthscale = original_lengthscale + rand_l
        new_variance = original_variance + rand_v
        # 2. 【关键】重置随机特征 W, b (换一张新的“布”)
        # 这一步必须在 _fit 之前做
        GP_Model.W.normal_(0, 1)
        GP_Model.b.uniform_(0, 2 * math.pi)
        # RFF 这里只需解 1024x1024 矩阵，耗时约 10-20ms
        GP_Model.set_hyper(lengthscale=new_lengthscale, variance=new_variance)
        total_set_hyper_time += time.time() - t0
    
        # --- 2. 随机选择起始点 ---
        # 这里的 num_points 就是 Batch Size
        selected_indices = torch.randperm(x_train.shape[0])[:num_points]
        x_start = x_train[selected_indices].clone()
        
        # --- 3. 梯度上升/下降主循环 ---
        t1 = time.time()
        
        # 初始化当前点 x_curr
        x_curr = x_start.clone().requires_grad_(True)
        
        # 存入起点
        trajectory_storage[0] = x_curr.detach()

        # 开启梯度计算
        # RFF-GP 对 Autograd 非常友好
        for t in range(total_steps):
            # 这里的 optimizer 需要每次清零，或者直接用 torch.autograd.grad 手动更新（你原代码的做法更好）
            
            # A. 决策：是否计算方差？
            # RFF 计算均值极快，计算方差稍慢。
            if uncertainty_penalty > 0 and (t % uncertainty_interval == 0):
                # 带方差惩罚：Objective = Mean - lambda * Std
                mu, std = GP_Model.predict_with_uncertainty(x_curr)
                objective = mu - uncertainty_penalty * std
            else:
                # 纯均值步：Objective = Mean
                # 调用我们新加的 predict_mean，速度最快
                objective = GP_Model.predict_mean(x_curr)
            
            # B. 计算对 x 的梯度
            # sum() 是因为我们要对一个 batch 求导
            grads = torch.autograd.grad(objective.sum(), x_curr)[0]
            torch.nn.utils.clip_grad_norm_([x_curr], max_norm=1.0)
            # C. 更新 x (In-place)
            # 注意：在梯度上升中，我们是 += lr * grad
            with torch.no_grad():
                x_curr += learning_rate * grads
            
            # D. 存储轨迹
            trajectory_storage[t + 1] = x_curr.detach()

        total_gradient_time += time.time() - t1
        
        # --- 4. 批量后处理与过滤 ---
        with torch.no_grad():
            # 批量计算起点和终点的分数
            # 直接取 trajectory_storage 的首尾
            start_points = trajectory_storage[0]
            end_points = trajectory_storage[-1]
            
            # 使用 predict_mean 快速打分
            y_start = GP_Model.predict_mean(start_points)
            y_end = GP_Model.predict_mean(end_points)
            
            # 基础过滤：分数有提升
            valid_mask = (y_end - y_start) > threshold_diff
            
            # 不确定性过滤 (如果设置了)
            if max_end_uncertainty is not None:
                _, std_end = GP_Model.predict_with_uncertainty(end_points)
                unc_mask = std_end < max_end_uncertainty
                valid_mask = valid_mask & unc_mask
            
            # 保存有效轨迹
            valid_indices = torch.where(valid_mask)[0]
            
            # 将 Tensor 转回 CPU 存入列表 (为了节省显存，通常把结果挪出 GPU)
            # 如果后续还在 GPU 用，可以不转 .cpu()
            if len(valid_indices) > 0:
                # 提取 valid 的轨迹: (Steps, Valid_Num, Dim) -> 换轴 -> (Valid_Num, Steps, Dim)
                valid_trajs = trajectory_storage[:, valid_indices, :].permute(1, 0, 2).cpu()
                valid_y_start = y_start[valid_indices].cpu()
                valid_y_end = y_end[valid_indices].cpu()
                
                for k in range(len(valid_indices)):
                    datasets[f'f{iter}'].append((
                        valid_trajs[k], 
                        valid_y_start[k], 
                        valid_y_end[k]
                    ))

    # --- 恢复原始超参数 ---
    GP_Model.set_hyper(lengthscale=original_lengthscale, variance=original_variance)
    
    if verbose:
        print(f"RFF-GP Sampling Report:")
        print(f"  Hyperparam Update: {total_set_hyper_time:.4f}s")
        print(f"  Gradient Ascent:   {total_gradient_time:.4f}s")
        print(f"  Avg Time per Func: {(total_gradient_time + total_set_hyper_time)/num_functions:.4f}s")

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