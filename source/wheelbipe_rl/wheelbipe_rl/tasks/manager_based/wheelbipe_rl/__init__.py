# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Wheelbipe-Rl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelbipe_rl_env_cfg:WheelbipeRlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Wheelbipe-DreamWaQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelbipe_rl_env_cfg:WheelbipeRlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.dreamwaq_ppo_cfg:DreamWaQRunnerCfg",
    },
)

gym.register(
    id="Template-Wheelbipe-DreamWaQ-Rough-v0",
    entry_point=f"{__name__}.wheelbipe_rough_env:WheelbipeRoughEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelbipe_rough_env_cfg:WheelbipeRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.dreamwaq_ppo_cfg:DreamWaQRunnerCfg",
    },
)

gym.register(
    id="Template-Wheelbipe-DreamWaQ-Rough-Play-v0",
    entry_point=f"{__name__}.wheelbipe_rough_env:WheelbipeRoughEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelbipe_rough_env_cfg:WheelbipeRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.dreamwaq_ppo_cfg:DreamWaQRunnerCfg",
    },
)

gym.register(
    id="Template-Wheelbipe-DreamWaQ-Jump-v0",
    entry_point=f"{__name__}.wheelbipe_jump_env:WheelbipeJumpEnv",

    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelbipe_jump_env_cfg:WheelbipeJumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.dreamwaq_ppo_cfg:DreamWaQRunnerCfg",
    },
)

gym.register(
    id="Template-Wheelbipe-DreamWaQ-Jump-Play-v0",
    entry_point=f"{__name__}.wheelbipe_jump_env:WheelbipeJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelbipe_jump_env_cfg:WheelbipeJumpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.dreamwaq_ppo_cfg:DreamWaQRunnerCfg",
    },
)