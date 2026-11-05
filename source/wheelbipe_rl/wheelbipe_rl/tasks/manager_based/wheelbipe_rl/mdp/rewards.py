# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def planar_speed_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    velocity = robot.data.root_lin_vel_b.torch
    return torch.sum(torch.square(velocity[:, :2]), dim=1)


def low_speed_tracking_l1(
    env,
    command_name: str,
    threshold: float = 0.1,
) -> torch.Tensor:
    """低速指令下，惩罚实际平面速度与目标速度的 L1 误差。"""
    command = env.command_manager.get_command(command_name)
    target_linear_velocity = command[:, :2]
    actual_linear_velocity = env.scene["robot"].data.root_lin_vel_b.torch[:, :2]
    low_speed_mask = torch.linalg.vector_norm(target_linear_velocity, dim=-1) < threshold
    tracking_error = torch.abs(actual_linear_velocity - target_linear_velocity).sum(dim=-1)
    return tracking_error * low_speed_mask.float()
