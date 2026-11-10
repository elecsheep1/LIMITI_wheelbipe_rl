from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils.configclass import configclass

from . import mdp
from .wheelbipe_rl_env_cfg import (
    WheelbipeRlEnvCfg,
    WheelbipeRlSceneCfg,
    ObservationsCfg,
    make_body_contact_sensor,
)

from .mdp.jump.jump_state import JumpCfg
from .mdp.jump.jump_observations import jump_observation
from .mdp.jump.jump_terminations import jump_failed
from .mdp.jump import jump_rewards


@configclass
class JumpSceneCfg(WheelbipeRlSceneCfg):
    left_wheel_contact = make_body_contact_sensor(
        "base_link/left_thigh_link/"
        "left_shank_link/left_wheel_link"
    )
    right_wheel_contact = make_body_contact_sensor(
        "base_link/right_thigh_link/"
        "right_shank_link/right_wheel_link"
    )


@configclass
class JumpObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        jump = ObsTerm(func=jump_observation)

    @configclass
    class HistoryCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5
            self.flatten_history_dim = False
            self.concatenate_terms = True
            self.concatenate_dim = -1

    policy: PolicyCfg = PolicyCfg()
    history: HistoryCfg = HistoryCfg()


@configclass
class WheelbipeJumpEnvCfg(WheelbipeRlEnvCfg):
    scene: JumpSceneCfg = JumpSceneCfg(
        num_envs=1024,
        env_spacing=4.0,
        replicate_physics=True,
    )

    observations: JumpObservationsCfg = (
        JumpObservationsCfg()
    )

    jump: JumpCfg = JumpCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0

        # 第一版只训练原地跳跃。
        command = self.commands.base_velocity
        command.rel_standing_envs = 1.0
        command.ranges.lin_vel_x = (0.0, 0.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)

        # 关闭与跳跃冲突或暂时无关的基础奖励。
        self.rewards.height_tracking = None
        self.rewards.vertical_speed = None
        self.rewards.track_linear_velocity = None
        self.rewards.track_yaw_velocity = None
        self.rewards.low_speed_tracking = None

        # 保留姿态、非期望接触和左右轮对齐。
        self.rewards.action_rate.weight = -0.005

        # 原地跳跃限制水平漂移。
        self.rewards.planar_drift = RewTerm(
            func=mdp.planar_speed_l2,
            weight=-2.0,
        )

        self.rewards.jump_height = RewTerm(
            func=jump_rewards.jump_height_tracking,
            weight=2.0,
        )

        self.rewards.jump_push_velocity = RewTerm(
            func=jump_rewards.jump_push_velocity,
            weight=3.0,
        )

        self.rewards.jump_takeoff = RewTerm(
            func=jump_rewards.jump_takeoff_velocity,
            weight=0.5,
        )

        self.rewards.jump_peak = RewTerm(
            func=jump_rewards.jump_peak_tracking,
            weight=1.0,
        )

        self.rewards.jump_success = RewTerm(
            func=jump_rewards.jump_success,
            weight=2.0,
        )

        self.rewards.jump_retraction = RewTerm(
            func=jump_rewards.jump_leg_retraction,
            weight=0.3,
        )

        self.rewards.stand_vertical_velocity = RewTerm(
            func=jump_rewards.jump_stand_vertical_velocity,
            weight=-1.0,
        )

        self.rewards.landing_impact = RewTerm(
            func=jump_rewards.jump_landing_impact,
            weight=-0.1,
        )

        self.terminations.jump_failed = DoneTerm(
            func=jump_failed,
        )

        self.rewards.air_wheel_torque = RewTerm(
            func=jump_rewards.jump_air_wheel_torque_l2,
            weight=-0.05,
        )


@configclass
class WheelbipeJumpEnvCfg_PLAY(WheelbipeJumpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
