import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass


class GasSpringForce(ActionTerm):
    """A custom action term that applies a gas spring force to the robot's knee joints."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        if cfg.force < 0.0 or cfg.damping < 0.0:
            raise ValueError("Gas spring force and damping must be non-negative.")
        if cfg.direction not in (-1.0, 1.0):
            raise ValueError("Gas spring direction must be either -1.0 or 1.0.")

        self.joint_ids, names = self._asset.find_joints(
            cfg.joint_names,
            preserve_order=True,
        )

        if len(self.joint_ids) != 2:
            raise ValueError(f"Expected exactly 2 knee joints, but found {len(self.joint_ids)}: {names}")

        # 不接收策略动作。
        self._actions = torch.zeros(
            env.num_envs,
            0,
            device=env.device,
        )

        # 保留请求力，方便检查。
        self.last_force = torch.zeros(
            env.num_envs,
            len(self.joint_ids),
            device=env.device,
        )

        self._export_IO_descriptor = False

    @property
    def action_dim(self):
        return 0

    @property
    def raw_actions(self):
        return self._actions

    @property
    def processed_actions(self):
        return self._actions

    def process_actions(self, actions):
        pass

    def apply_actions(self):
        """Apply the gas spring force to the knee joints."""
        velocity = self._asset.data.joint_vel.torch[:, self.joint_ids]
        force = self.cfg.direction * self.cfg.force - self.cfg.damping * velocity
        self.last_force.copy_(
            force.clamp(
                min=-self.cfg.max_force,
                max=self.cfg.max_force,
            )
        )
        self._asset.set_joint_effort_target_index(
            target=self.last_force,
            joint_ids=self.joint_ids,
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            self.last_force.zero_()
        else:
            self.last_force[env_ids] = 0.0


@configclass
class GasSpringForceCfg(ActionTermCfg):
    """Configuration for the GasSpringForce action term."""

    class_type: type = GasSpringForce
    asset_name: str = "robot"

    joint_names: list[str] = [
        "left_spring_slide_joint",
        "right_spring_slide_joint",
    ]
    force: float = 100.0
    damping: float = 0.0
    max_force: float = 250.0
    direction: float = -1.0