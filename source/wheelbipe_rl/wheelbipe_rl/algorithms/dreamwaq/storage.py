"""估计器使用的单轮采样缓存。"""

import torch


class EstimatorStorage:
    """用于估计器的单轮采样缓存。

    该类用于在每个训练轮中存储观测、动作、奖励等信息，以便在训练过程中进行批量处理。
    """

    def __init__(
        self,
        num_steps,
        num_envs,
        history_length,
        obs_dim,
        device,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.step = 0
        self.history = torch.empty(
            num_steps, num_envs, history_length, obs_dim, device=device
        )
        self.velocity = torch.empty(num_steps, num_envs, 3, device=device)
        self.next_observation = torch.empty(num_steps, num_envs, obs_dim, device=device)
        self.valid_next = torch.empty(num_steps, num_envs, device=device, dtype=torch.bool)

    @torch.no_grad()
    def add(
        self,
        history,
        velocity,
        next_observation,
        dones,
    ):
        """将新的采样数据添加到缓存中。
        """
        if self.step >= self.num_steps:
            raise RuntimeError("Estimator storage is full")

        i = self.step
        self.history[i].copy_(history)
        self.velocity[i].copy_(velocity)
        self.next_observation[i].copy_(next_observation)
        self.valid_next[i].copy_(~dones.reshape(-1).bool())
        self.step += 1

    def batches(self, num_mini_batches, num_epochs):
        """生成用于训练的批次。

        该方法将缓存中的数据划分为多个小批次，以便在训练过程中进行迭代。
        """
        if self.step != self.num_steps:
            raise RuntimeError("Estimator rollout is incomplete")

        total = self.step * self.num_envs
        if not 1 <= num_mini_batches <= total:
            raise ValueError("Invalid estimator mini-batch count")
        # 将时间维、环境维合并为样本维。
        # 形状: [num_steps * num_envs, history_length, obs_dim]
        history = self.history.flatten(start_dim=0, end_dim=1)
        velocity = self.velocity.flatten(start_dim=0, end_dim=1)
        next_observation = self.next_observation.flatten(start_dim=0, end_dim=1)
        valid_next = self.valid_next.flatten(start_dim=0, end_dim=1)

        for _ in range(num_epochs):
            # 随机打乱样本索引
            indices = torch.randperm(total, device=history.device)
            for ids in torch.tensor_split(indices, num_mini_batches):
                yield (
                    history[ids],
                    velocity[ids],
                    next_observation[ids],
                    valid_next[ids],
                )

    def clear(self):
        """清空缓存以准备下一轮采样。"""
        self.step = 0