"""DreamWaQ 上下文估计器。"""

import torch
from torch import nn
from torch.nn import functional as F

class ContextEstimator(nn.Module):
    """上下文估计器网络。"""

    def __init__(
        self,
        obs_dim: int = 22,
        history_length: int = 5,
        latent_dim: int = 16,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.history_length = history_length
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim * history_length, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.velocity_head = nn.Linear(64, 3)      # 预测速度的头部
        self.mu_head = nn.Linear(64, latent_dim)            # 预测潜在变量的均值
        self.logvar_head = nn.Linear(64, latent_dim)        # 预测潜在变量的对数方差

        self.decoder = nn.Sequential(
            nn.Linear(3 + latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, obs_dim),
        )

    def encode(self, history: torch.Tensor):
        """编码历史观测以获得潜在表示。"""
        if history.ndim != 3:
            raise ValueError(f"Expected 3-D history, got {tuple(history.shape)}")
        if history.shape[1:] != (self.history_length, self.obs_dim):
            raise ValueError(f"Unexpected history shape: {tuple(history.shape)}")

        features = self.encoder(history.flatten(start_dim=1))
        velocity = self.velocity_head(features)
        mu = self.mu_head(features)
        logvar = self.logvar_head(features).clamp(-10.0, 5.0)  # 限制 logvar 的范围以避免数值不稳定

        return velocity, mu, logvar

    def forward(self, history: torch.Tensor, sample: bool = True):
        """前向传播以获得潜在表示和重建观测。"""
        velocity, mu, logvar = self.encode(history)

        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std  # 重参数化技巧
        else:
            z = mu

        next_observation = self.decoder(torch.cat([velocity, z], dim=-1))

        return {
            "velocity": velocity,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "next_observation": next_observation,
        }

    def compute_loss(
        self,
        history: torch.Tensor,
        true_velocity: torch.Tensor,
        next_observation: torch.Tensor,
        valid_next: torch.Tensor,
        beta: float = 0.01,
    ):
        """计算上下文估计器的损失，包括重建损失和 KL 散度。"""
        output = self.forward(history, sample=True)

        velocity_loss = F.mse_loss(output["velocity"], true_velocity.detach())
        prediction_error = (output["next_observation"] - next_observation.detach()).square().mean(dim=-1)
        mask = valid_next.reshape(-1).to(
            dtype=prediction_error.dtype, device=prediction_error.device
        )
        prediction_loss = (prediction_error * mask).sum() / mask.sum().clamp_min(1.0)
        kl_loss = -0.5 * (
            1.0
            + output["logvar"]
            - output["mu"].square()
            - output["logvar"].exp()
        ).sum(dim=-1).mean()
        total_loss = velocity_loss + prediction_loss + beta * kl_loss

        metrics = {
            "velocity_loss": velocity_loss.detach(),
            "prediction_loss": prediction_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }
        return total_loss, metrics