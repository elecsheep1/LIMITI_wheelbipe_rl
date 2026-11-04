"""在 PPO 中加入独立训练的上下文估计器。"""

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO

from .storage import EstimatorStorage


class DreamWaQPPO(PPO):
    """带有上下文估计器的 PPO 算法。

    该类在标准 PPO 的基础上，增加了一个独立训练的上下文估计器，用于从历史观测中提取潜在表示。
    """

    def __init__(
        self,
        actor,
        critic,
        storage,
        estimator_learning_rate=3.0e-4,
        estimator_epochs=1,
        estimator_num_mini_batches=4,
        estimator_beta=0.01,
        estimator_max_grad_norm=1.0,
        **kwargs,
    ):
        super().__init__(actor, critic, storage, **kwargs)

        # 第一版只处理单GPU、无额外增强的训练。
        if self.is_multi_gpu or self.rnd or self.symmetry:
            raise NotImplementedError(
                "Use one GPU without RND or symmetry "
                "for the initial DreamWaQ integration."
            )
        if estimator_epochs < 1 or estimator_num_mini_batches < 1:
            raise ValueError("Estimator update counts must be positive")

        # 从 PPO 优化器中排除估计器参数。
        estimator_ids = {id(p) for p in actor.estimator.parameters()}
        policy_parameters = [p for p in actor.parameters() if id(p) not in estimator_ids]
        policy_parameters += list(critic.parameters())
        optimizer_type = type(self.optimizer)
        self.optimizer = optimizer_type(
            policy_parameters,
            **self.optimizer.defaults,
        )

        # 估计器单独使用一个优化器。
        self.estimator_optimizer = torch.optim.Adam(
            actor.estimator.parameters(),
            lr=estimator_learning_rate,
        )
        self.estimator_epochs = estimator_epochs
        self.estimator_num_mini_batches = estimator_num_mini_batches
        self.estimator_beta = estimator_beta
        self.estimator_max_grad_norm = estimator_max_grad_norm
        self.estimator_storage = EstimatorStorage(
            num_steps=storage.num_transitions_per_env,
            num_envs=storage.num_envs,
            history_length=actor.history_length,
            obs_dim=actor.obs_dim,
            device=self.device,
        )

        self._pending_estimator_sample = None

    def act(self, obs):
        """在环境中执行动作，并存储估计器所需的历史观测。"""
        if self._pending_estimator_sample is not None:
            raise RuntimeError("process_env_step must follow act")
        # 在执行动作之前保存 history_t、velocity_t。
        with torch.no_grad():
            history = self.actor.normalize_history(obs[self.actor.history_key]).detach().clone()
            velocity = obs["privileged"][:, :3].detach().clone()
        if velocity.shape != (self.storage.num_envs, 3):
            raise ValueError("privileged must contain only three , raw body-frame linear velocities.")

        self._pending_estimator_sample = (history, velocity)
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        """在环境步之后处理估计器的缓存。"""
        if self._pending_estimator_sample is None:
            raise RuntimeError("act must be called before process_env_step")
        history, velocity = self._pending_estimator_sample

        with torch.no_grad():
            # 此处 obs 已经是执行动作后的下一帧。
            # 使用与刚才 history 相同的归一化状态。
            next_observation = self.actor.obs_normalizer(obs["policy"])
            self.estimator_storage.add(
                history=history,
                velocity=velocity,
                next_observation=next_observation,
                dones=dones,
            )
        self._pending_estimator_sample = None
        # 基类随后保存 PPO 转换，并更新归一化统计。
        super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        # 防止上一轮估计器梯度参与 PPO 的梯度裁剪。
        self.estimator_optimizer.zero_grad(set_to_none=True)
        # 更新 PPO。
        losses = super().update()
        # 更新估计器。
        sums = {}
        sample_count = 0
        batches = self.estimator_storage.batches(
            num_mini_batches=self.estimator_num_mini_batches,
            num_epochs=self.estimator_epochs,
        )

        for history, velocity, next_obs, valid in batches:
            loss, metrics = self.actor.estimator.compute_loss(
                history=history,
                true_velocity=velocity,
                next_observation=next_obs,
                valid_next=valid,
                beta=self.estimator_beta,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Estimator loss is not finite: {loss.item()}")
            self.estimator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.estimator.parameters(),
                self.estimator_max_grad_norm,
                error_if_nonfinite=True,
            )
            self.estimator_optimizer.step()
            count = history.shape[0]
            sample_count += count
            metrics = {
                **metrics,
                "total": loss.detach(),
            }
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + float(value) * count

        self.estimator_optimizer.zero_grad(set_to_none=True)
        self.estimator_storage.clear()

        losses.update({
            f"estimator_{name}": value / sample_count for name, value in sums.items()
        })

        return losses

    def save(self):
        state = super().save()
        # actor_state_dict 已包含估计器和归一化参数。
        state["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        return state

    def load(
        self,
        loaded_dict,
        load_cfg=None,
        strict=True,
    ):
        load_iteration = super().load(
            loaded_dict,
            load_cfg=load_cfg,
            strict=strict,
        )
        load_estimator_optimizer = (
            load_cfg is None
            or load_cfg.get(
                "estimator_optimizer",
                load_cfg.get("optimizer", False),
            )
        )

        if load_estimator_optimizer:
            self.estimator_optimizer.load_state_dict(
                loaded_dict["estimator_optimizer_state_dict"]
            )

        self.estimator_optimizer.zero_grad(set_to_none=True)
        self.estimator_storage.clear()
        self._pending_estimator_sample = None

        return load_iteration
