import torch

from isaaclab.envs import ManagerBasedRLEnv

from wheelbipe_rl.terrains.terrain_curriculum import (
    TerrainCurriculum,
)


class WheelbipeRoughEnv(ManagerBasedRLEnv):
    """使用动作对应的原始指令，记录重置前的真实状态。"""

    def __init__(
        self,
        cfg,
        render_mode=None,
        **kwargs,
    ):
        self.terrain_curriculum = None
        self._curriculum_in_step = False

        super().__init__(
            cfg,
            render_mode=render_mode,
            **kwargs,
        )

        if cfg.traversability.enabled:
            # 在 runner 的 inference_mode 外创建可变缓冲区。
            self.terrain_curriculum = (
                TerrainCurriculum(
                    self,
                    cfg.traversability,
                )
            )

            self._curriculum_recorded = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )

    def step(self, action):
        tracker = self.terrain_curriculum

        if tracker is None:
            return super().step(action)

        # 保存执行本次动作时使用的指令。
        self._curriculum_command = (
            self.command_manager.get_command(
                self.cfg.traversability.command_name
            ).clone()
        )

        self._curriculum_recorded.zero_()
        self._curriculum_in_step = True

        try:
            result = super().step(action)

            # 已终止环境在 _reset_idx 中记录。
            # 这里仅记录没有重置的环境。
            ids = (
                (~self._curriculum_recorded)
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )

            tracker.observe(
                ids,
                self._curriculum_command,
                self.reset_terminated,
            )

            tracker.update()

            return result

        finally:
            self._curriculum_in_step = False

    def _reset_idx(self, env_ids):
        tracker = self.terrain_curriculum

        if tracker is not None:
            ids = tracker.ids(env_ids)

            if self._curriculum_in_step:
                # 此时仍然是终止前的刚体状态。
                tracker.observe(
                    ids,
                    self._curriculum_command,
                    self.reset_terminated,
                )

                self._curriculum_recorded[ids] = True

                tracker.end_episode(
                    ids,
                    self.reset_terminated,
                )

            else:
                # 手动 reset 不当作一次运动或跌倒。
                terminated = torch.zeros(
                    self.num_envs,
                    dtype=torch.bool,
                    device=self.device,
                )

                tracker.end_episode(
                    ids,
                    terminated,
                )

        # 原实现先执行 CurriculumManager，
        # 再用更新后的 env_origins 重置机器人。
        super()._reset_idx(env_ids)
