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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from . import mdp

from wheelbipe_rl.assets.contact_reports import (
    spawn_with_contact_reports,
)

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
TARGET_BASE_HEIGHT = 0.30

UNDESIRED_CONTACT_SENSORS = [
    "base_contact",
    "left_thigh_contact",
    "right_thigh_contact",
    "left_shank_contact",
    "right_shank_contact",
    "left_knee_drive_contact",
    "right_knee_drive_contact",
    "left_passive_wheel_1_contact",
    "right_passive_wheel_1_contact",
]
def make_body_contact_sensor(relative_path: str,) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/Geometry/"
            + relative_path
        ),
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

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

    base_contact = make_body_contact_sensor("base_link")
    left_thigh_contact = make_body_contact_sensor("base_link/left_thigh_link")
    right_thigh_contact = make_body_contact_sensor("base_link/right_thigh_link")
    left_shank_contact = make_body_contact_sensor(
        "base_link/left_thigh_link/"
        "left_shank_link"
    )
    right_shank_contact = make_body_contact_sensor(
        "base_link/right_thigh_link/"
        "right_shank_link"
)
    left_knee_drive_contact = make_body_contact_sensor("base_link/left_knee_drive_link")
    right_knee_drive_contact = make_body_contact_sensor("base_link/right_knee_drive_link")
    left_passive_wheel_1_contact = make_body_contact_sensor("base_link/left_thigh_link/left_shank_link/left_passive_wheel_1_link")
    right_passive_wheel_1_contact = make_body_contact_sensor("base_link/right_thigh_link/right_shank_link/right_passive_wheel_1_link")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1500.0),
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
    gas_spring = mdp.GasSpringForceCfg(
        asset_name="robot",
        force=100.0,
        damping=0.5,
        max_force=150.0,
        direction=-1.0,
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.DeadzoneVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.2,
        heading_command=False,
        rel_heading_envs=0.0,
        debug_vis=False,
        linear_deadzone=0.04,
        yaw_deadzone=0.04,
        ranges=mdp.DeadzoneVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
        ),
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
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

    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-6.0)

    vertical_speed = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)

    roll_pitch_speed = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    height_tracking = RewTerm(
        func=mdp.base_height_l2,
        weight=-100.0,
        params={
            "target_height": TARGET_BASE_HEIGHT,
        },
    )

    track_linear_velocity = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": 0.25,
        },
    )

    track_yaw_velocity = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    low_speed_tracking = RewTerm(
        func=mdp.low_speed_tracking_l1,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.1,
        },
    )

    wheel_pair_x = RewTerm(
        func=mdp.wheel_pair_alignment_l2,
        weight=-1.0,
        params={
            "axis": "x",
            "tolerance": 0.05,
            "error_scale": 0.05,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_wheel_link",
                    "right_wheel_link",
                ],
                preserve_order=True,
            ),
        },
    )

    wheel_pair_z = RewTerm(
        func=mdp.wheel_pair_alignment_l2,
        weight=-0.5,
        params={
            "axis": "z",
            "tolerance": 0.05,
            "error_scale": 0.05,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_wheel_link",
                    "right_wheel_link",
                ],
                preserve_order=True,
            ),
        },
    )

    undesired_contact = RewTerm(
        func=mdp.undesired_body_contacts,
        weight=-2.0,
        params={
            "sensor_names": list(
                UNDESIRED_CONTACT_SENSORS
            ),
            "threshold": 3.0,
        },
    )


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
    commands: CommandsCfg = CommandsCfg()

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

        self.scene.robot.spawn.func = (spawn_with_contact_reports)
        # 自定义生成函数会逐个启用嵌套刚体报告。
        self.scene.robot.spawn.activate_contact_sensors = False
