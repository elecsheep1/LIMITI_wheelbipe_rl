from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def traversability_sampling(env, env_ids):
    tracker = getattr(
        env,
        "terrain_curriculum",
        None,
    )

    if tracker is None:
        return None

    tracker.sample(env_ids)

    return tracker.metrics()