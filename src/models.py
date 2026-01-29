# src/models.py
import torch
import torch.nn as nn
import numpy as np

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class VectorFieldNet(nn.Module):
    """
    条件流匹配网络 v_theta(x_t, t, x_0)
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 输入投影 (x_t, x_0 拼接)
        self.input_proj = nn.Linear(input_dim * 2, hidden_dim)
        
        # 主干网络 (Residual MLP)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout=dropout) for _ in range(6)])
        
        # SDE 解耦输出层
        self.mu_head = nn.Linear(hidden_dim, input_dim) # Drift mu
        self.sigma_head = nn.Linear(hidden_dim, 1)      # Log-sigma
        
        # 初始化 sigma head 使其初始值偏小 (bias < 0)
        nn.init.constant_(self.sigma_head.bias, -2.0)

    def forward(self, x, t, x_0):
        """
        Args:
            x: 当前状态 [B, D]
            t: 时间 [B, 1]
            x_0: 初始状态 [B, D]
        """
        t_emb = self.time_mlp(t)
        x_input = torch.cat([x, x_0], dim=-1)
        x_emb = self.input_proj(x_input)
        h = x_emb + t_emb
        for block in self.blocks:
            h = block(h)
        
        mu_pred = self.mu_head(h)
        log_sigma_pred = self.sigma_head(h)
        # 限制 log_sigma 范围，防止数值爆炸
        # 🔑 稍微放宽上限，允许模型在需要的地方有更大的探索性
        log_sigma_pred = torch.clamp(log_sigma_pred, min=-10.0, max=3.0)
        return mu_pred, log_sigma_pred


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))