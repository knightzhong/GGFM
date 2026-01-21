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
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 输入投影 (x_t 和 x_0 拼接)
        self.input_proj = nn.Linear(input_dim * 2, hidden_dim)
        
        # 主干网络 (ResNet-like or MLP)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim) # 输出速度向量
        )

    def forward(self, x, t, x_0):
        t_emb = self.time_mlp(t)
        x_input = torch.cat([x, x_0], dim=-1)
        x_emb = self.input_proj(x_input)
        return self.net(x_emb + t_emb)