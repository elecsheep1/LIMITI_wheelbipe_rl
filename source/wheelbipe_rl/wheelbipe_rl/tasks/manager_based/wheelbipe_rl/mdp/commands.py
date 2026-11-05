import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand


class DeadzoneVelocityCommand(UniformVelocityCommand):
    """对速度指令应用死区，观测和奖励使用同一个指令。"""

    def _update_command(self):
        super()._update_command()

        linear_speed = torch.linalg.vector_norm(self.vel_command_b[:, :2], dim=-1)
        self.vel_command_b[linear_speed < self.cfg.linear_deadzone, :2] = 0.0
        self.vel_command_b[self.vel_command_b[:, 2].abs() < self.cfg.yaw_deadzone, 2] = 0.0

