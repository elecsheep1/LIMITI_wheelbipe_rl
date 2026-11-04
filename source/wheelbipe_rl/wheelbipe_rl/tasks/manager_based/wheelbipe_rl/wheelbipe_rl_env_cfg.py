# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from . import mdp

##
# Pre-defined configs
##

from wheelbipe_rl.assets import WHEELBIPE_V1_CFG
HIP_JOINTS = [
    "left_hip_joint",
    "right_hip_joint",
]
KNEE_DRIVE_JOINTS = [
    "left_knee_drive_joint",
    "right_knee_drive_joint",
]
LEG_JOINTS = HIP_JOINTS + KNEE_DRIVE_JOINTS
WHEEL_JOINTS = [
    "left_wheel_joint",
    "right_wheel_joint",
]
ACTIVE_JOINTS = LEG_JOINTS + WHEEL_JOINTS
LEG_RANGE = math.radians(15.0)

##
# Scene definition
##


@configclass
class WheelbipeRlSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
    )

    # robot
    robot: ArticulationCfg = WHEELBIPE_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    hips = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=HIP_JOINTS,
        preserve_order=True,
        scale=LEG_RANGE,
        use_default_offset=True,
    )
    knee_drives = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=KNEE_DRIVE_JOINTS,
        preserve_order=True,
        scale=LEG_RANGE,
        use_default_offset=True,
    )
    wheels = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINTS,
        preserve_order=True,
        scale=10.0,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        leg_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True),
            },
        )
        active_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True),
            },
        )
        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class HistoryCfg(PolicyCfg):
        """Observations for history group."""

        def __post_init__(self) -> None:
            super().__post_init__()
            self.history_length = 5
            self.flatten_history_dim = False
            self.concatenate_terms = True
            self.concatenate_dim = -1

    history: HistoryCfg = HistoryCfg()

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Observations for privileged group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    privileged :PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={
            "reset_joint_targets": True,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)

    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)

    planar_speed = RewTerm(func=mdp.planar_speed_l2, weight=-1.0)

    vertical_speed = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)

    roll_pitch_speed = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)})


##
# Environment configuration
##


@configclass
class WheelbipeRlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: WheelbipeRlSceneCfg = WheelbipeRlSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=False,
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 10.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)