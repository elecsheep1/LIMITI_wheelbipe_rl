import torch

from .jump_state import (
    STAND,
    CROUCH,
    PUSH,
    FLIGHT,
    LAND,
    get_jump_state,
)


def jump_height_tracking(env):
    state = get_jump_state(env)

    enabled = (
        (state.reward_phase == STAND)
        | (state.reward_phase == CROUCH)
        | (state.reward_phase == LAND)
    )

    error = state.height() - state.reference_height

    return enabled.float() * torch.exp(
        -(error / 0.03).square()        # 0.03 是容差，约为 3 厘米。
    )


def jump_push_velocity(env):
    state = get_jump_state(env)
    enabled = state.reward_phase == PUSH

    vz = state.robot.data.root_lin_vel_w[:, 2]
    target_vz = torch.sqrt(2.0 * 9.81 * state.target_height)

    progress = (vz / target_vz).clamp(0.0, 1.0)
    release_scale = torch.where(
        state.push_release_ready,
        torch.ones_like(progress),
        torch.full_like(progress, 0.1),
    )

    return (
        enabled.float()
        * release_scale
        * progress
    )

def jump_release(env):
    state = get_jump_state(env)

    return (
        state.takeoff_event.float()
        / env.step_dt
    )


def jump_takeoff_velocity(env):
    state = get_jump_state(env)
    enabled = state.takeoff_event

    target = torch.sqrt(
        2.0
        * state.cfg.gravity
        * state.target_height
    )
    error = state.takeoff_vz - target

    return (
        enabled.float()
        * torch.exp(-(error / 0.40).square())       # 0.40 是容差，约为 40 厘米/秒。
        / env.step_dt   # 如果改变仿真步长，事件奖励的总贡献大致保持不变。
    )


def jump_peak_tracking(env):
    state = get_jump_state(env)
    enabled = state.landing_event       # 奖励在落地时结算

    rise = (state.peak_height - state.takeoff_height).clamp_min(0.0)

    error = (state.target_height - rise).clamp_min(0.0)

    # 真正离地后，在首次落地时结算一次。
    return (
        enabled.float()
        * torch.exp(-(error / 0.025).square())   # 0.025 是容差，约为 2.5 厘米。
        / env.step_dt
    )


def jump_success(env):
    state = get_jump_state(env)
    return (
        state.success_event.float()
        / env.step_dt
    )


def jump_stand_vertical_velocity(env):
    state = get_jump_state(env)

    # 不惩罚蹬地和腾空所必需的竖直速度。
    enabled = state.reward_phase == STAND
    vz = state.robot.data.root_lin_vel_w[:, 2]

    return enabled.float() * vz.square()


def jump_leg_retraction(env):
    """空中收腿奖励"""
    state = get_jump_state(env)
    vz = state.robot.data.root_lin_vel_w[:, 2]
    enabled = (
        state.airborne_sample
        & (vz > 0.05)
    )

    # z 向上；每侧轮心分别与离地瞬间比较。
    upward = (
        state.wheel_pos_b[:, :, 2]
        - state.takeoff_wheel_pos_b[:, :, 2]
    )
    # 分别评价两侧收腿，左右同步由高度差惩罚负责。
    retraction = (
        upward / state.cfg.leg_retraction_distance
    ).clamp(0.0, 1.0).mean(dim=-1)

    return retraction * enabled.float()


def jump_landing_extension(env):
    """下降阶段鼓励两条腿伸回离地时的位置。"""
    state = get_jump_state(env)

    vz = state.robot.data.root_lin_vel_w[:, 2]
    enabled = (
        state.airborne_sample
        & (vz < -0.05)
    )

    # 正值表示轮子仍高于起跳瞬间，即腿仍处于收缩状态。
    remaining_retraction = (
        state.wheel_pos_b[:, :, 2]
        - state.takeoff_wheel_pos_b[:, :, 2]
    ).clamp_min(0.0)

    # 分别评价两侧伸腿，允许每侧获得连续学习信号。
    extension = (
        1.0
        - remaining_retraction
        / state.cfg.leg_retraction_distance
    ).clamp(0.0, 1.0).mean(dim=-1)

    return enabled.float() * extension


def jump_wheel_forward_swing(env):
    state = get_jump_state(env)
    enabled = state.airborne_sample
    forward_shift = (
        state.wheel_pos_b[:, :, 0]
        - state.takeoff_wheel_pos_b[:, :, 0]
    )
    excess = (forward_shift - 0.04).clamp_min(0.0)
    return enabled.float() * (
        (excess / 0.05).square().mean(dim=-1)
    )


def jump_landing_impact(env):
    state = get_jump_state(env)

    peaks = torch.stack(
        [
            torch.linalg.vector_norm(
                sensor.data.net_forces_w_history.torch,
                dim=-1,
            ).amax(dim=(1, 2))
            for sensor in [*state.sensors, *state.support_sensors]
        ],
        dim=-1,
    ).amax(dim=-1)

    # 用整机重力归一化；留出正常支撑和落地容差。
    mass = state.robot.data.body_mass.torch.sum(dim=-1)
    excess = (
        peaks / (mass * 9.81).clamp_min(1.0) - 2.0
    ).clamp_min(0.0)

    enabled = (
        (state.reward_phase == LAND)
        | state.landing_event
    )

    return enabled.float() * excess.square()


def jump_air_wheel_torque_l2(env):
    """空中驱动轮转矩惩罚"""
    state = get_jump_state(env)
    robot = state.robot

    joint_ids = getattr(state, "_wheel_joint_ids", None)
    if joint_ids is None:
        joint_ids, _ = robot.find_joints(
            ["left_wheel_joint", "right_wheel_joint"],
            preserve_order=True,
        )
        state._wheel_joint_ids = joint_ids
    torque = robot.data.applied_torque[:, joint_ids]

    enabled = state.airborne_sample

    return (
        enabled.float()
        * torque.square().sum(dim=-1)
    )


def jump_air_wheel_z_diff(env):
    """约束蹬地和腾空期间的左右主轮高度差。"""
    state = get_jump_state(env)

    difference = (
        state.wheel_pos_b[:, 0, 2]
        - state.wheel_pos_b[:, 1, 2]
    ).abs()

    # 2 mm 以内不惩罚，超过后开始约束左右轮同步。
    excess = (
        difference - 0.002
    ).clamp_min(0.0)

    # 左右不同步通常从蹬地阶段开始形成，
    # 因此不能等确认腾空后才进行约束。
    enabled = (
        (state.reward_phase == PUSH)
        | state.airborne_sample
    )

    return (
        enabled.float()
        * (excess / 0.01).square()
    )


def jump_trajectory_height(env):
    """跟踪PUSH和FLIGHT阶段的机身高度轨迹。"""
    state = get_jump_state(env)

    enabled = (
        (state.reward_phase == PUSH)
        | (state.reward_phase == FLIGHT)
    )

    error = (
        state.height()
        - state.reference_height
    )

    return enabled.float() * torch.exp(
        -(
            error
            / state.cfg.trajectory_height_tolerance
        ).square()
    )


def jump_trajectory_velocity(env):
    """跟踪PUSH和FLIGHT阶段的机身竖直速度轨迹。"""
    state = get_jump_state(env)

    enabled = (
        (state.reward_phase == PUSH)
        | (state.reward_phase == FLIGHT)
    )

    vz = state.robot.data.root_lin_vel_w[:, 2]

    error = (
        vz - state.reference_vz
    )

    return enabled.float() * torch.exp(
        -(
            error
            / state.cfg.trajectory_velocity_tolerance
        ).square()
    )


def jump_wheel_z_trajectory(env):
    """跟踪左右相同的空中轮心高度轨迹。"""
    state = get_jump_state(env)

    enabled = (
        (state.reward_phase == FLIGHT)
        & state.has_taken_off
    )

    error = (
        state.wheel_pos_b[:, :, 2]
        - state.reference_wheel_z_b
    )

    per_leg_reward = torch.exp(
        -(
            error
            / state.cfg.wheel_z_trajectory_tolerance
        ).square()
    )

    return (
        enabled.float()
        * per_leg_reward.min(dim=-1).values
    )
