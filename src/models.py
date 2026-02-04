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
    条件流匹配网络 v_theta(x_t, t, y_low, y_high)，与 BB denoise_fn 一致：网络不接收 x_low，
    x_low 只在推理的更新公式里使用（见 inference_ode 中 x_curr = x_low + t * v_pred）。
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
        # 输入投影（当前状态 x_t）
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # 条件嵌入 (y_low, y_high)，与 BB denoise_fn 一致
        self.cond_proj = nn.Linear(2, hidden_dim)

        # 主干网络 (Residual MLP)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout=dropout) for _ in range(6)])

        # 输出层
        self.mu_head = nn.Linear(hidden_dim, input_dim)
        self.sigma_head = nn.Linear(hidden_dim, 1)

        nn.init.constant_(self.sigma_head.bias, -2.0)

    def forward(self, x, t, y_low, y_high):
        """
        Args:
            x: 当前状态 [B, D]（即 x_t）
            t: 时间 [B, 1]
            y_low: 起点分数 [B, 1]
            y_high: 目标分数 [B, 1]
        """
        t_emb = self.time_mlp(t)
        x_emb = self.input_proj(x)
        y_cat = torch.cat([y_low, y_high], dim=-1)
        cond_emb = self.cond_proj(y_cat)
        h = x_emb + t_emb + cond_emb
        for block in self.blocks:
            h = block(h)

        mu_pred = self.mu_head(h)
        log_sigma_pred = self.sigma_head(h)
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