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
    error = vz - state.reference_vz

    return enabled.float() * torch.exp(
        -(error / 0.40).square()        # 0.40 是容差，约为 40 厘米/秒。
    )


def jump_takeoff_velocity(env):
    state = get_jump_state(env)
    enabled = state.takeoff_event

    target = torch.sqrt(
        2.0 * 9.81 * state.target_height
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

    error = rise - state.target_height

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
    enabled = state.airborne_sample

    shortening = (
        state.takeoff_leg_extension - state.leg_extension
    )

    return (
        enabled.float()
        * (
            shortening / state.cfg.leg_retraction_distance
        ).clamp(0.0, 1.0)
    )


def jump_landing_impact(env):
    state = get_jump_state(env)

    peaks = torch.stack(
        [
            torch.linalg.vector_norm(
                sensor.data.net_forces_w_history.torch,
                dim=-1,
            ).amax(dim=(1, 2))
            for sensor in state.sensors
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

