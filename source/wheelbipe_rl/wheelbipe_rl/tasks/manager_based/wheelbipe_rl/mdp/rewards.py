# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils.math import quat_apply_inverse

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


def wheel_pair_alignment_l2(
    env: ManagerBasedRLEnv,
    axis: str,
    tolerance: float,
    error_scale: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚机身坐标系中超过容差的左右轮位置差。

    axis:
        "x"：左右轮前后错位。
        "z"：左右轮高度差。

    tolerance、error_scale 的单位均为米。
    """
    if axis not in ["x", "z"]:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x' or 'z'.")
    if tolerance < 0.0:
        raise ValueError(f"Invalid tolerance: {tolerance}. Must be non-negative.")
    if error_scale <= 0.0:
        raise ValueError(f"Invalid error_scale: {error_scale}. Must be positive.")

    robot = env.scene[asset_cfg.name]
    wheel_positions_w = robot.data.body_pos_w.torch[:, asset_cfg.body_ids, :]

    if wheel_positions_w.shape[1] != 2:
        raise ValueError(f"Expected 2 wheel bodies, but got {wheel_positions_w.shape[1]}.")
    delta_w = wheel_positions_w[:, 0, :] - wheel_positions_w[:, 1, :]
    delta_b = quat_apply_inverse(robot.data.root_quat_w.torch, delta_w)
    axis_index = 0 if axis == "x" else 2
    difference = delta_b[:, axis_index].abs()
    # 容差内为零，只惩罚超出部分。
    excess = torch.clamp(difference - tolerance, min=0.0)
    return torch.square(excess / error_scale)


def undesired_body_contacts(
        env: ManagerBasedRLEnv,
        sensor_names: list[str],
        threshold: float = 3.0,
) -> torch.Tensor:
    """惩罚机器人非轮子部位的接触力超过阈值。

    sensor_names: 需要统计的接触传感器名称列表。
    threshold: 接触力阈值，单位牛顿。
    """
    if threshold < 0.0: raise ValueError("Contact threshold must be non-negative.")
    harmful_contact = torch.zeros(
        env.num_envs,
        device=env.device,
        dtype=torch.bool,
    )
    for name in sensor_names:
        sensor = env.scene.sensors[name]
        # [环境, 历史帧, 刚体, xyz]
        forces = sensor.data.net_forces_w_history.torch
        # 取历史帧最大接触力。
        peak_forces = torch.linalg.vector_norm(forces, dim=-1).amax(dim=(1, 2))
        harmful_contact |= peak_forces > threshold
    return harmful_contact.float()