# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class DreamWaQActorCfg(RslRlMLPModelCfg):

    class_name: str = (
        "wheelbipe_rl.algorithms.dreamwaq.actor:"
        "DreamWaQActor"
    )

    history_length: int = 5
    latent_dim: int = 16
    history_key: str = "history"


@configclass
class DreamWaQPPOCfg(RslRlPpoAlgorithmCfg):

    class_name: str = (
        "wheelbipe_rl.algorithms.dreamwaq.ppo:"
        "DreamWaQPPO"
    )

    estimator_learning_rate: float = 3.0e-4
    estimator_epochs: int = 1
    estimator_num_mini_batches: int = 4
    estimator_beta: float = 0.01
    estimator_max_grad_norm: float = 1.0


@configclass
class DreamWaQRunnerCfg(RslRlOnPolicyRunnerCfg):

    class_name = "OnPolicyRunner"

    num_steps_per_env = 24
    max_iterations = 4096
    save_interval = 50
    experiment_name = "wheelbipe_dreamwaq"
    clip_actions = None
    obs_groups={
        "actor": ["policy"],
        "critic": ["policy", "privileged"],
    }
    actor = DreamWaQActorCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = DreamWaQPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,

        estimator_learning_rate=3.0e-4,
        estimator_epochs=1,
        estimator_num_mini_batches=4,
        estimator_beta=0.01,
        estimator_max_grad_norm=1.0,

        rnd_cfg=None,
        symmetry_cfg=None,
    )