"""使用历史状态估计的 DreamWaQ Actor。"""

import copy

import torch
from torch import nn

from rsl_rl.models import MLPModel

from .estimator import ContextEstimator

class DreamWaQActor(MLPModel):
    """DreamWaQ Actor 网络。"""

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(256, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
        history_length=5,
        latent_dim=16,
        history_key="history",
    ):
        self.context_latent_dim = latent_dim

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=copy.deepcopy(distribution_cfg),
        )

        if self.obs_groups != ["policy"]:
            raise ValueError("DreamWaQActor expects actor groups ['policy'].")
        self.history_key = history_key
        self.history_length = history_length

        expected = (history_length, self.obs_dim)
        if tuple(obs[history_key].shape[1:]) != expected:
            raise ValueError(
                f"Expected history [N, {expected[0]}, {expected[1]}], "
                f"got {tuple(obs[history_key].shape)}"
            )

        self.estimator = ContextEstimator(
            obs_dim=self.obs_dim,
            history_length=history_length,
            latent_dim=latent_dim,
        )

    def _get_latent_dim(self):
        # 当前观测 + 估计速度 + 隐变量。
        return self.obs_dim + 3 + self.context_latent_dim

    def normalize_history(self, history):
        """对历史观测进行归一化。"""
        return self.obs_normalizer(history)

    def get_latent(self, obs, masks=None, hidden_state=None):
        """获取潜在表示，包括当前观测、估计速度和隐变量。"""
        current = torch.cat([obs[name] for name in self.obs_groups], dim=-1)
        current = self.obs_normalizer(current)

        # PPO 不通过动作损失更新估计器。
        # 后续由独立辅助损失训练估计器。
        with torch.no_grad():
            history = self.normalize_history(obs[self.history_key])
            velocity, mu, _ = self.estimator.encode(history)

        # 策略使用确定性的 mu，避免重复采样潜变量。
        return torch.cat([current, velocity, mu], dim=-1)

    def as_jit(self):
        return DreamWaQExport(self)

    def as_onnx(self, verbose=False):
        return DreamWaQExport(self,verbose=verbose)


class DreamWaQExport(nn.Module):
    """导出包含估计器的确定性策略，接收当前帧与历史两项输入。"""

    is_recurrent: bool = False

    def __init__(self, actor, verbose=False):
        super().__init__()

        self.verbose = verbose
        self.obs_dim = actor.obs_dim
        self.history_length = actor.history_length

        self.normalizer = copy.deepcopy(actor.obs_normalizer)
        self.encoder = copy.deepcopy(actor.estimator.encoder)
        self.velocity_head = copy.deepcopy(actor.estimator.velocity_head)
        self.mu_head = copy.deepcopy(actor.estimator.mu_head)
        self.mlp = copy.deepcopy(actor.mlp)

        if actor.distribution is None:
            self.output_module = nn.Identity()
        else:
            self.output_module = actor.distribution.as_deterministic_output_module()

    def forward(
        self,
        observation: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播以获得动作输出。

        Args:
            observation: 当前观测，形状为 [N, obs_dim]。
            history: 历史观测，形状为 [N, history_length, obs_dim]。
        Returns:
            动作输出，形状为 [N, output_dim]。
        """
        observation = self.normalizer(observation)
        history = self.normalizer(history)
        features = self.encoder(history.flatten(start_dim=1))
        velocity = self.velocity_head(features)
        mu = self.mu_head(features)
        latent = torch.cat([observation, velocity, mu], dim=-1)

        return self.output_module(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        """重置导出模型的状态。"""
        pass

    def get_dummy_inputs(self):
        """获取用于导出模型的虚拟输入。"""
        dummy_observation = torch.zeros(1, self.obs_dim)
        dummy_history = torch.zeros(1, self.history_length, self.obs_dim)
        return dummy_observation, dummy_history

    @property
    def input_names(self):
        return ["observation", "history"]

    @property
    def output_names(self):
        return ["action"]