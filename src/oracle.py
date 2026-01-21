# src/oracle.py
import torch
import numpy as np
from src.config import Config

class NTKOracle:
    """
    NTK-GP Oracle: 提供梯度指引 (Mean) 和安全边界 (Uncertainty/Variance)
    """
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, device):
        self.device = device
        self.length_scale = Config.NTK_LENGTH_SCALE
        self.beta = Config.NTK_BETA
        
        # 转换为 Tensor
        X_t = torch.FloatTensor(X_train).to(device)
        y_t = torch.FloatTensor(y_train).to(device)
        
        # Nyström 近似处理大数据集
        N = X_t.shape[0]
        if N > Config.NYSTROM_SAMPLES:
            perm = torch.randperm(N)[:Config.NYSTROM_SAMPLES]
            self.X_inducing = X_t[perm]
            self.y_inducing = y_t[perm]
        else:
            self.X_inducing = X_t
            self.y_inducing = y_t
            
        print(f"[NTK] Initialized with {self.X_inducing.shape[0]} inducing points.")
        self._fit()

    def _compute_kernel(self, X1, X2):
        # RBF Kernel
        dist_sq = torch.cdist(X1, X2, p=2) ** 2
        return torch.exp(-0.5 * dist_sq / (self.length_scale ** 2))

    def _fit(self):
        self.K_mm = self._compute_kernel(self.X_inducing, self.X_inducing)
        # 计算 A_inv = (K + beta * I)^-1
        eye = torch.eye(self.X_inducing.shape[0], device=self.device)
        self.A_inv = torch.linalg.solve(self.K_mm + self.beta * eye, eye)
        # 计算 alpha = A_inv * y
        self.alpha = self.A_inv @ self.y_inducing

    def predict_mean(self, X):
        K_xm = self._compute_kernel(X, self.X_inducing)
        return K_xm @ self.alpha

    def predict_uncertainty(self, X):
        K_xm = self._compute_kernel(X, self.X_inducing)
        k_xx = torch.ones(X.shape[0], 1, device=self.device) # RBF 自相关恒为 1
        
        # Variance = k(x,x) - k(x,m) A^-1 k(m,x)
        # 优化：只计算对角线
        variance_reduction = torch.sum((K_xm @ self.A_inv) * K_xm, dim=1, keepdim=True)
        return torch.clamp(k_xx - variance_reduction, min=1e-6)