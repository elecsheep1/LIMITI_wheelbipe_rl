import math

import torch
from isaaclab.utils.configclass import configclass

STAND = 0       # 双轮接触地面并稳定站立
CROUCH = 1      # 下蹲，降低机体高度
PUSH = 2        # 蹬地起跳
FLIGHT = 3      # 双轮离地，处于空中
LAND = 4        # 重新接触地面并恢复稳定


@configclass
class JumpCfg:
    stand_height: float = 0.30                  # 正常站立时的参考高度
    crouch_height: float = 0.27                 # 下蹲阶段的参考高度。

    # 相对实际离地位置继续上升的机体高度。
    target_height: float = 0.04
    # 空中相对离地时缩短竖直腿长 2 cm，收腿奖励达到满分。
    leg_retraction_distance: float = 0.02

    stand_duration: float = 1.0                 # 每次跳跃前固定准备1秒。
    # stand_timeout: float = 3.0                  # 如果机器人在站立阶段超过 3 秒没有起跳，则判定为失败。
    crouch_duration: float = 0.35               # 下蹲阶段持续 0.35 秒。下蹲阶段持续 0.35 秒。
    push_timeout: float = 0.60                  # 进入蹬地阶段后，最多等待 0.60 秒。
    flight_timeout: float = 0.90                # 已经起跳后，最多允许飞行 0.90 秒
    land_timeout: float = 1.50                  # 进入落地阶段后，最多等待 1.50 秒恢复稳定。

    takeoff_confirm_time: float = 0.04          # 必须连续离地 0.04 秒。
    landing_stable_time: float = 0.20           # 连续稳定 0.20 秒

    contact_force: float = 3.0
    wheel_radius: float = 0.06
    takeoff_clearance: float = 0.003            # 轮子离开地面的最小间隙，单位为米。

    takeoff_min_vz: float = 0.10                # 起跳时机器人根节点的竖直速度必须大于 0.10 m/s。
    stable_vz: float = 0.15                     # 判定落地稳定时，竖直速度绝对值必须小于 0.15 m/s
    stable_ang_vel: float = 0.50                # 判定落地稳定时，根节点角速度绝对值必须小于 0.50 rad/s
    stable_height_error: float = 0.04           # 稳定站立时，当前高度与 stand_height 的误差必须小于 0.04 m。
    stable_tilt: float = math.radians(12.0)     # 机器人倾斜角不能超过 12°。

    # 低于此值不能判定为有效跳跃。
    min_valid_rise: float = 0.01
    # 容差
    height_tolerance: float = 0.015


class JumpState:
    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.jump
        self.robot = env.scene["robot"]
        self.wheel_ids, _ = self.robot.find_bodies(
            ["left_wheel_link", "right_wheel_link"],
            preserve_order=True,
        )
        self.sensors = [
            env.scene["left_wheel_contact"],
            env.scene["right_wheel_contact"],
        ]
        sensor_names = env.cfg.rewards.undesired_contact.params["sensor_names"]
        self.support_sensors = [env.scene[name] for name in sensor_names]

        n = env.num_envs
        device = env.device

        def zeros(dtype=torch.float32):
            return torch.zeros(n, dtype=dtype, device=device)

        self.phase = zeros(torch.long)          # 保存每个环境当前的跳跃阶段。
        self.reward_phase = zeros(torch.long)   # 保存本次仿真步开始时的阶段，主要用于奖励计算。
        self.time = zeros()                     # 记录每个环境当前阶段已经持续的时间。
        self.stable_time = zeros()              # 记录当前阶段中连续满足稳定条件的时间。
        self.air_confirm_time = zeros()         # 记录连续满足离地条件的时间

        self.target_height = torch.full(
            (n,),
            self.cfg.target_height,
            device=device,
        )

        self.takeoff_height = zeros()           # 起跳时的机器人高度。
        self.takeoff_vz = zeros()               # 起跳时的竖直速度。
        self.peak_height = zeros()              # 本次跳跃过程中的最高高度。
        self.air_time = zeros()                 # 本次跳跃的有效腾空时间。
        self.peak_clearance = zeros()           # 本次跳跃中，左右轮间隙较小值的最大值
        self.takeoff_leg_extension = zeros()    # 离地时的竖直腿长
        self.max_air_time = zeros()             # episode 内最大的腾空时间
        self.max_clearance = zeros()            # episode 内最大的共同离地间隙
        self.airborne_sample = zeros(torch.bool)# 当前采样是否有效腾空

        self.has_taken_off = zeros(torch.bool)          # 记录每个环境是否已经起跳。
        self.failed = zeros(torch.bool)                 # 记录每个环境是否失败。

        self.takeoff_event = zeros(torch.bool)          # 记录当前仿真步是否发生起跳事件。
        self.landing_event = zeros(torch.bool)          # 记录当前仿真步是否发生落地事件。
        self.success_event = zeros(torch.bool)          # 记录当前仿真步是否发生成功跳跃事件。

        # 用于记录整个 episode 中的统计。
        self.attempts = zeros()                 # 记录跳跃尝试次数。
        self.successes = zeros()                # 记录成功跳跃次数。
        self.max_rise = zeros()                 # 记录当前 episode 中达到的最高跳跃高度。

        self.reference_height = torch.full(
            (n,),
            self.cfg.stand_height,
            device=device,
        )
        self.reference_vz = zeros()

        self.phase_durations = torch.tensor(
            [
                self.cfg.stand_duration,
                self.cfg.crouch_duration,
                self.cfg.push_timeout,
                self.cfg.flight_timeout,
                self.cfg.land_timeout,
            ],
            device=device,
        )

        self.leg_extension = zeros()

    def reset(self,env_ids):
        self.phase[env_ids] = STAND
        self.reward_phase[env_ids] = STAND
        for value in (
            self.time,
            self.stable_time,
            self.air_confirm_time,
            self.takeoff_height,
            self.takeoff_vz,
            self.peak_height,
            self.air_time,
            self.peak_clearance,
            self.takeoff_leg_extension,
            self.max_air_time,
            self.max_clearance,
            self.attempts,
            self.successes,
            self.max_rise,
            self.reference_vz,
            self.leg_extension,
        ):
            value[env_ids] = 0.0
        for value in (
            self.airborne_sample,
            self.has_taken_off,
            self.failed,
            self.takeoff_event,
            self.landing_event,
            self.success_event,
        ):
            value[env_ids] = False
        self.target_height[env_ids] = self.cfg.target_height
        self.reference_height[env_ids] = self.cfg.stand_height

    def height(self):
        return (
            self.robot.data.root_pos_w[:, 2]
            - self.env.scene.env_origins[:, 2]
        )

    def wheel_forces(self):
        # 当前力用于状态切换；历史力用于冲击奖励。
        return torch.stack(
            [
                torch.linalg.vector_norm(
                    sensor.data.net_forces_w.torch, dim=-1
                ).amax(dim=-1)
                for sensor in self.sensors
            ],
            dim=-1,
        )

    def stable(self, height, vz, contact):
        gravity = self.robot.data.projected_gravity_b
        tilt_cos = -gravity[:, 2]   # 获取重力方向在机器人自身坐标系下的投影。如果机器人完全直立，重力投影对应的值接近1

        ang_vel = torch.linalg.vector_norm(
            self.robot.data.root_ang_vel_b[:, :2],
            dim=-1,
        )

        return (
            contact.all(dim=-1)
            & (vz.abs() < self.cfg.stable_vz)
            & (ang_vel < self.cfg.stable_ang_vel)
            & (tilt_cos > math.cos(self.cfg.stable_tilt))
            & ((height - self.cfg.stand_height).abs() < self.cfg.stable_height_error)
        )

    def enter(self, mask, phase):
        """让满足条件的环境进入指定阶段。"""
        self.phase[mask] = phase
        self.time[mask] = 0.0           # 进入新阶段时重新计时。
        self.stable_time[mask] = 0.0    # 进入新阶段时清除之前的稳定计时。

    def update(self):
        env = self.env
        cfg = self.cfg
        dt = env.step_dt

        self.takeoff_event.zero_()
        self.landing_event.zero_()
        self.success_event.zero_()

        # 本步奖励对应执行动作时所处的阶段。
        old_phase = self.phase.clone()
        self.reward_phase.copy_(old_phase)
        # 当前阶段时间
        self.time += dt

        height = self.height()
        self.leg_extension.copy_(
            self.robot.data.root_pos_w[:, 2]
            - self.robot.data.body_pos_w[
                :, self.wheel_ids, 2
            ].mean(dim=-1)
        )
        vz = self.robot.data.root_lin_vel_w[:, 2]
        forces = self.wheel_forces()
        contact = forces > cfg.contact_force
        no_contact = (~contact).all(dim=-1)
        for sensor in self.support_sensors:
            force = torch.linalg.vector_norm(
                sensor.data.net_forces_w.torch,
                dim=-1,
            ).amax(dim=-1)

            no_contact &= force <= cfg.contact_force

        wheel_height = (
            self.robot.data.body_pos_w[:, self.wheel_ids, 2]
            - env.scene.env_origins[:, 2:3]
        )
        clearance = wheel_height - cfg.wheel_radius

        unsupported =(
            no_contact
            & (clearance > 0.0).all(dim=-1)
        )

        stable = self.stable(height, vz, contact)

        # 设置默认参考值
        self.reference_height.fill_(cfg.stand_height)
        self.reference_vz.zero_()


        # 1. STAND：双轮支撑并站稳。
        stand = old_phase == STAND
        # self.stable_time[stand] = torch.where(
        #     stable[stand],
        #     self.stable_time[stand] + dt,
        #     0.0,
        # )

        begin = stand & (
            self.time >= cfg.stand_duration
        )
        # begin = stand & (self.stable_time >= cfg.stand_duration)
        # self.failed |= (
        #     stand
        #     & (~begin)
        #     & (self.time >= cfg.stand_timeout)
        # )

        self.has_taken_off[begin] = False
        self.air_confirm_time[begin] = 0.0
        self.takeoff_height[begin] = 0.0
        self.takeoff_vz[begin] = 0.0
        self.peak_height[begin] = 0.0
        self.air_time[begin] = 0.0
        self.peak_clearance[begin] = 0.0
        self.takeoff_leg_extension[begin] = 0.0
        self.attempts[begin] += 1.0
        self.enter(begin, CROUCH)

        # 2. CROUCH：平滑降低参考高度。
        crouch = old_phase == CROUCH
        ratio = (self.time / cfg.crouch_duration).clamp(0.0, 1.0)
        smooth = ratio.square() * (3.0 - 2.0 * ratio)
        # 在站立高度和下蹲高度之间进行插值。
        crouch_reference = (
            cfg.stand_height
            + (cfg.crouch_height - cfg.stand_height) * smooth
        )
        self.reference_height[crouch] = (
            crouch_reference[crouch]
        )

        self.enter(
            crouch & (self.time >= cfg.crouch_duration),
            PUSH,
        )

        # 3. PUSH：参考向上速度逐渐增大。
        push = old_phase == PUSH
        target_vz = torch.sqrt(
            2.0 * 9.81 * self.target_height
        )
        ramp = (self.time / 0.15).clamp(0.0, 1.0)
        # 在 0.15 秒内将蹬地参考速度从 0 平滑增加到目标速度。
        self.reference_vz[push] = (target_vz * ramp)[push]

        airborne_candidate = (
            push
            & unsupported
            & (clearance > cfg.takeoff_clearance).all(dim=-1)
            & (vz > cfg.takeoff_min_vz)
        )

        first_air = airborne_candidate & (self.air_confirm_time == 0.0)

        self.takeoff_height[first_air] = height[first_air]
        self.takeoff_vz[first_air] = vz[first_air]
        self.peak_height[first_air] = height[first_air]     # 将当前高度作为初始最高高度.
        self.takeoff_leg_extension[first_air] = self.leg_extension[first_air]
        self.air_time[first_air] = 0.0
        self.peak_clearance[first_air] = 0.0

        self.air_confirm_time[push] = torch.where(
            airborne_candidate[push],
            self.air_confirm_time[push] + dt,
            0.0,
        )
        takeoff = push & (self.air_confirm_time >= cfg.takeoff_confirm_time)
        self.takeoff_event[takeoff] = True
        self.has_taken_off[takeoff] = True
        self.enter(takeoff, FLIGHT)

        self.failed |= (
            push
            & (~takeoff)
            & (self.time >= cfg.push_timeout)
        )

        # 记录真正离地后的最高点。
        # 找到已经起跳，并且当前处于 PUSH 或 FLIGHT 的环境。
        airborne = unsupported & (
            (push & (self.air_confirm_time > 0.0))
            | (
                (old_phase == FLIGHT)
                & self.has_taken_off
            )
        )
        self.airborne_sample.copy_(airborne)
        self.air_time[airborne] += dt
        self.peak_clearance[airborne] = torch.maximum(
            self.peak_clearance[airborne],
            clearance[airborne].amin(dim=-1),
        )
        self.peak_height[airborne] = torch.maximum(
            self.peak_height[airborne],
            height[airborne],
        )
        confirmed = airborne & self.has_taken_off
        self.max_air_time[confirmed] = torch.maximum(
            self.max_air_time[confirmed],
            self.air_time[confirmed],
        )
        self.max_clearance[confirmed] = torch.maximum(
            self.max_clearance[confirmed],
            self.peak_clearance[confirmed],
        )

        # 4. FLIGHT：重新接触主轮则进入落地阶段。
        flight = old_phase == FLIGHT
        # 只要任意一个轮子重新检测到接触，就认为发生了触地。
        touchdown = flight & (~unsupported)

        self.landing_event[touchdown] = True
        self.enter(touchdown, LAND)

        self.failed |= (
            flight
            & (~touchdown)
            & (self.time >= cfg.flight_timeout)
        )

        # 5. LAND：双轮支撑、姿态和高度恢复稳定。
        land = old_phase == LAND
        self.stable_time[land] = torch.where(
            stable[land],
            self.stable_time[land] + dt,
            0.0,
        )

        rise = (self.peak_height - self.takeoff_height).clamp_min(0.0)
        height_reached = (
            (rise - self.target_height).abs()
            <= cfg.height_tolerance
        )
        valid_jump = (
            self.has_taken_off
            & (rise >= cfg.min_valid_rise)
            & height_reached
        )

        success = (
            land
            & valid_jump
            & (self.stable_time >= cfg.landing_stable_time)
        )

        self.success_event[success] = True
        self.successes[success] += 1.0

        self.max_rise.copy_(
            torch.where(
                self.has_taken_off,
                torch.maximum(self.max_rise, rise),
                self.max_rise,
            )
        )

        # 回到站立阶段，冷却后可以再次跳。
        self.enter(success, STAND)
        self.has_taken_off[success] = False

        self.failed |= (
            land
            & (~success)
            & (self.time >= cfg.land_timeout)
        )


def get_jump_state(env):
    state = getattr(env, "_jump_state", None)
    if state is None:
        state = JumpState(env)
        env._jump_state = state
    return state





