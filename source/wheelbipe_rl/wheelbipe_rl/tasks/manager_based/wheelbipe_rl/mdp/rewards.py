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
