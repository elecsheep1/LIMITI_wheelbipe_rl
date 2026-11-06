from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_patch_boundary(
    env: ManagerBasedRLEnv,
    margin: float = 0.5,
) -> torch.Tensor:
    """机器人接近出生地形块边界时结束回合。"""
    terrain = env.scene.terrain
    generator_cfg = terrain.cfg.terrain_generator

    if generator_cfg is None:
        raise ValueError("terrain_patch_boundary requires a generated terrain.")
    half_size = torch.tensor(
        generator_cfg.size,
        device=env.device,
        dtype=torch.float32,
    ) * 0.5
    limits = half_size - margin
    if torch.any(limits <= 0.0):
        raise ValueError("Terrain boundary margin must be smaller than half the terrain size.")
    root_position = env.scene["robot"].data.root_pos_w.torch
    relative_xy = (root_position[:, :2] - env.scene.env_origins[:, :2])

    return torch.any(
        relative_xy.abs() > limits,
        dim=-1,
    )