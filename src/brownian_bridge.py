# src/brownian_bridge.py
# 布朗桥模型 (Brownian Bridge)，与 ROOT 中 BBDM 对齐
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial


def extract(a, t, x_shape):
    """从 buffer a 中按索引 t 取值并 reshape 为 x_shape 的广播形状"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


class Swish(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x) * x


class BBMLP(nn.Module):
    """BBDM 的 MLP denoiser：输入 (x_t, t, y_high, y_low)，输出 objective 预测"""
    def __init__(self, input_dim, index_dim=1, hidden_dim=128, act=None):
        super().__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim
        self.hidden_dim = hidden_dim
        self.act = act if act is not None else Swish()
        self.y_dim = 1
        self.main = nn.Sequential(
            nn.Linear(input_dim + index_dim + 2 * self.y_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_t, t, y_high, y_low):
        sz = x_t.size()
        x_t = x_t.view(-1, self.input_dim)
        t = t.view(-1, self.index_dim).float()
        y_high = y_high.view(-1, self.y_dim).float()
        y_low = y_low.view(-1, self.y_dim).float()
        h = torch.cat([x_t, t, y_high, y_low], dim=1)
        return self.main(h).view(*sz)


class BrownianBridgeModel(nn.Module):
    """布朗桥扩散模型，与 ROOT BrownianBridgeModel 接口一致"""
    def __init__(self, image_size, hidden_size, num_timesteps=1000, mt_type="linear",
                 max_var=1.0, eta=1.0, loss_type="l1", objective="grad",
                 skip_sample=True, sample_type="linear", sample_step=200):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.mt_type = mt_type
        self.max_var = max_var
        self.eta = eta
        self.loss_type = loss_type
        self.objective = objective
        self.skip_sample = skip_sample
        self.sample_type = sample_type
        self.sample_step = sample_step
        self.image_size = image_size
        self.register_schedule()
        self.denoise_fn = BBMLP(
            input_dim=image_size,
            index_dim=1,
            hidden_dim=hidden_size,
        )

    def register_schedule(self):
        T = self.num_timesteps
        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError(self.mt_type)
        m_tminus = np.append(0, m_t[:-1])
        variance_t = 2.0 * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0.0, variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1.0 - m_t) / (1.0 - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("m_t", to_torch(m_t))
        self.register_buffer("m_tminus", to_torch(m_tminus))
        self.register_buffer("variance_t", to_torch(variance_t))
        self.register_buffer("variance_tminus", to_torch(variance_tminus))
        self.register_buffer("variance_t_tminus", to_torch(variance_t_tminus))
        self.register_buffer("posterior_variance_t", to_torch(posterior_variance_t))
        if self.skip_sample:
            if self.sample_type == "linear":
                midsteps = torch.arange(
                    self.num_timesteps - 1, 1,
                    step=-((self.num_timesteps - 1) / (self.sample_step - 2)),
                ).long()
                steps_t = torch.cat((midsteps, torch.tensor([1, 0]).long()), dim=0)
            elif self.sample_type == "cosine":
                steps = np.linspace(0, self.num_timesteps, self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.0) / 2.0 * self.num_timesteps
                steps_t = torch.from_numpy(steps).long()
            else:
                steps_t = torch.arange(self.num_timesteps - 1, -1, -1)
        else:
            steps_t = torch.arange(self.num_timesteps - 1, -1, -1).long()
        self.register_buffer("steps", steps_t)

    def forward(self, x_high, y_high, x_low, y_low):
        b, d, device = *x_high.shape, x_high.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x_high, y_high, x_low, y_low, t)

    def p_losses(self, x_high, y_high, x_low, y_low, t, noise=None):
        b, d = x_high.shape
        noise = default(noise, lambda: torch.randn_like(x_high))
        x_t, objective = self.q_sample(x_high, x_low, t, noise)
        # 与 ROOT 完全一致：训练时 denoise_fn(x_t, t, y_high, y_low) = (pair 高分, pair 低分)
        objective_recon = self.denoise_fn(x_t, t, y_high, y_low)
        objective_recon = objective_recon.reshape(objective_recon.shape[0], -1)
        if self.loss_type == "l1":
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == "l2":
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError(self.loss_type)
        x0_recon = self.predict_x0_from_objective(x_t, x_low, t, objective_recon)
        return recloss, {"loss": recloss, "x0_recon": x0_recon}

    def q_sample(self, x_high, x_low, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_high))
        m_t = extract(self.m_t, t, x_high.shape)
        var_t = extract(self.variance_t, t, x_high.shape)
        sigma_t = torch.sqrt(var_t)
        if self.objective == "grad":
            objective = m_t * (x_low - x_high) + sigma_t * noise
        elif self.objective == "noise":
            objective = noise
        elif self.objective == "ysubx":
            objective = x_low - x_high
        else:
            raise NotImplementedError(self.objective)
        x_t = (1.0 - m_t) * x_high + m_t * x_low + sigma_t * noise
        return x_t, objective

    def predict_x0_from_objective(self, x_t, x_low, t, objective_recon):
        if self.objective == "grad":
            x0_recon = x_t - objective_recon
        elif self.objective == "noise":
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * x_low - sigma_t * objective_recon) / (1.0 - m_t + 1e-8)
        elif self.objective == "ysubx":
            x0_recon = x_low - objective_recon
        else:
            raise NotImplementedError(self.objective)
        return x0_recon

    @torch.no_grad()
    def p_sample(self, x_t, x_low, y_low, y_high, i, clip_denoised=False, classifier_free_guidance_weight=0.0):
        # 与 ROOT 完全一致：推理时 denoise_fn(x_t, t, y_low, y_high) = (当前分数, 目标分数)，与 ROOT p_sample 一致
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), int(self.steps[i].item()), device=x_t.device, dtype=torch.long)
            objective_recon = (
                (1 + classifier_free_guidance_weight) * self.denoise_fn(x_t, t, y_low, y_high)
                - classifier_free_guidance_weight * self.denoise_fn(x_t, t, torch.zeros_like(y_low), torch.zeros_like(y_high))
            )
            x0_recon = self.predict_x0_from_objective(x_t, x_low, t, objective_recon)
            if clip_denoised:
                x0_recon = x0_recon.clamp(-1.0, 1.0)
            return x0_recon, x0_recon
        t = torch.full((x_t.shape[0],), int(self.steps[i].item()), device=x_t.device, dtype=torch.long)
        n_t = torch.full((x_t.shape[0],), int(self.steps[i + 1].item()), device=x_t.device, dtype=torch.long)
        objective_recon = (
            (1 + classifier_free_guidance_weight) * self.denoise_fn(x_t, t, y_low, y_high)
            - classifier_free_guidance_weight * self.denoise_fn(x_t, t, torch.zeros_like(y_low), torch.zeros_like(y_high))
        )
        x0_recon = self.predict_x0_from_objective(x_t, x_low, t, objective_recon)
        if clip_denoised:
            x0_recon = x0_recon.clamp(-1.0, 1.0)
        m_t = extract(self.m_t, t, x_t.shape)
        m_nt = extract(self.m_t, n_t, x_t.shape)
        var_t = extract(self.variance_t, t, x_t.shape)
        var_nt = extract(self.variance_t, n_t, x_t.shape)
        # 与 ROOT BrownianBridgeModel.p_sample 公式完全一致
        sigma2_t = (var_t - var_nt * (1.0 - m_t) ** 2 / (1.0 - m_nt) ** 2) * var_nt / var_t
        sigma_t = torch.sqrt(sigma2_t.clamp(min=0)) * self.eta
        noise = torch.randn_like(x_t)
        x_tminus_mean = (
            (1.0 - m_nt) * x0_recon + m_nt * x_low
            + torch.sqrt((var_nt - sigma2_t).clamp(min=0) / (var_t + 1e-10))
            * (x_t - (1.0 - m_t) * x0_recon - m_t * x_low)
        )
        return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, x_low, y_low, y_high, clip_denoised=True, classifier_free_guidance_weight=0.0):
        img = x_low
        for i in range(len(self.steps)):
            img, _ = self.p_sample(
                x_t=img, x_low=x_low, y_low=y_low, y_high=y_high,
                i=i, clip_denoised=clip_denoised,
                classifier_free_guidance_weight=classifier_free_guidance_weight,
            )
        return img

    @torch.no_grad()
    def sample(self, x_low, y_low, y_high, clip_denoised=True, classifier_free_guidance_weight=0.0):
        return self.p_sample_loop(
            x_low, y_low, y_high,
            clip_denoised=clip_denoised,
            classifier_free_guidance_weight=classifier_free_guidance_weight,
        )
