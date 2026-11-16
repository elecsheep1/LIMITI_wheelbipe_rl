import math

import torch
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse

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
    target_height: float = 0.08
    # 空中相对离地时缩短竖直腿长 2 cm，收腿奖励达到满分。
    leg_retraction_distance: float = 0.02
    # 整次腾空期间左右主轮最大高度差不超过 2 cm。
    max_wheel_z_diff_for_success: float = 0.02

    # 机身轨迹。
    gravity: float = 9.81
    # 计划离地时的机身高度。
    release_height: float = 0.30
    # 根据实际下蹲高度计算蹬地时间后，再限制到此范围。
    min_push_trajectory_time: float = 0.06
    max_push_trajectory_time: float = 0.15
    # 空中腿部轨迹。
    tuck_duration: float = 0.08         # 空中收腿参考动作的持续时间。
    extension_duration: float = 0.12    # 空中伸腿参考动作的持续时间

    # 轨迹奖励容差。
    trajectory_height_tolerance: float = 0.02       # 机身高度跟踪奖励的误差尺度
    trajectory_velocity_tolerance: float = 0.30     # 轨迹速度跟踪奖励的误差尺度
    wheel_z_trajectory_tolerance: float = 0.01      # 轮子竖直位置跟踪奖励的误差尺度

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
    height_tolerance: float = 0.01


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
        self.max_wheel_z_diff = zeros()          # 当前跳跃的最大左右轮高度差
        self.episode_max_wheel_z_diff = zeros()  # 本回合的最大左右轮高度差
        self.wheel_pos_b = torch.zeros(n, 2, 3, device=device)              # 当前轮子在机器人自身坐标系下的位置
        self.takeoff_wheel_pos_b = torch.zeros(n, 2, 3, device=device)      # 离地时的轮子在机器人自身坐标系下的位置
        self.max_air_time = zeros()             # episode 内最大的腾空时间
        self.max_clearance = zeros()            # episode 内最大的共同离地间隙
        self.airborne_sample = zeros(torch.bool)# 当前采样是否有效腾空

        self.push_max_vz = zeros()
        self.push_candidate_steps = zeros()
        self.push_wheel_free_steps = zeros()
        self.push_support_free_steps = zeros()
        self.push_clearance_steps = zeros()
        self.push_release_ready = zeros(torch.bool)

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

        # 从进入PUSH开始连续累计，进入FLIGHT时不清零。
        self.trajectory_time = zeros()
        # 蹬地轨迹的起始和释放参数。
        self.push_start_height = zeros()
        self.reference_release_height = zeros()
        self.reference_release_vz = zeros()
        self.reference_push_time = zeros()
        # 左右主轮共用同一条空中高度轨迹。
        self.takeoff_wheel_z_b = zeros()        # 起跳时轮子在机身坐标系下的竖直位置
        self.reference_wheel_z_b = torch.full(
            (n, 2),
            self.cfg.wheel_radius - self.cfg.stand_height,      # 站立时轮心在机身坐标系中位置
            device=device,
        )

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
            self.max_air_time,
            self.max_clearance,
            self.attempts,
            self.successes,
            self.max_rise,
            self.reference_vz,
            self.wheel_pos_b,
            self.takeoff_wheel_pos_b,
            self.max_wheel_z_diff,
            self.episode_max_wheel_z_diff,
            self.push_max_vz,
            self.push_candidate_steps,
            self.push_wheel_free_steps,
            self.push_support_free_steps,
            self.push_clearance_steps,
            self.trajectory_time,
            self.push_start_height,
            self.reference_release_height,
            self.reference_release_vz,
            self.reference_push_time,
            self.takeoff_wheel_z_b,
        ):
            value[env_ids] = 0.0
        for value in (
            self.airborne_sample,
            self.has_taken_off,
            self.failed,
            self.takeoff_event,
            self.landing_event,
            self.success_event,
            self.push_release_ready,
        ):
            value[env_ids] = False
        self.target_height[env_ids] = self.cfg.target_height
        self.reference_height[env_ids] = self.cfg.stand_height
        self.reference_vz[env_ids] = 0.0
        self.reference_wheel_z_b[env_ids] = (
            self.cfg.wheel_radius
            - self.cfg.stand_height
        )

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

        # 机身轨迹从PUSH开始连续计时，进入FLIGHT后不重置。
        trajectory_active = (
            (old_phase == PUSH)
            | (old_phase == FLIGHT)
        )
        self.trajectory_time[trajectory_active] += dt

        height = self.height()
        wheel_delta_w = (
            self.robot.data.body_pos_w[:, self.wheel_ids, :]
            - self.robot.data.root_pos_w[:, None, :]
        )
        root_quat = self.robot.data.root_quat_w[:, None, :].expand(-1, 2, -1)
        self.wheel_pos_b.copy_(
            quat_apply_inverse(root_quat, wheel_delta_w)
        )
        vz = self.robot.data.root_lin_vel_w[:, 2]
        forces = self.wheel_forces()
        contact = forces > cfg.contact_force
        no_contact = (~contact).all(dim=-1)
        main_wheels_free = no_contact.clone()
        support_free = torch.ones(
            self.env.num_envs,
            dtype=torch.bool,
            device=self.env.device,
        )
        for sensor in self.support_sensors:
            force = torch.linalg.vector_norm(
                sensor.data.net_forces_w.torch,
                dim=-1,
            ).amax(dim=-1)
            support_free &= force <= cfg.contact_force

        no_contact &= support_free

        wheel_height = (
            self.robot.data.body_pos_w[:, self.wheel_ids, 2]
            - env.scene.env_origins[:, 2:3]
        )
        clearance = wheel_height - cfg.wheel_radius
        clearance_ok = (
            clearance > cfg.takeoff_clearance
        ).all(dim=-1)

        self.push_release_ready.copy_(
            main_wheels_free
            & support_free
            & clearance_ok
        )

        unsupported =(
            no_contact
            & (clearance > 0.0).all(dim=-1)
        )

        stable = self.stable(height, vz, contact)

        # 设置默认参考值
        self.reference_height.fill_(cfg.stand_height)
        self.reference_vz.zero_()
        # 非腾空阶段默认使用左右轮当前位置的平均值，
        # 避免参考观测中出现无意义的零值。
        current_symmetric_wheel_z = (
            self.wheel_pos_b[:, :, 2]
            .mean(dim=-1, keepdim=True)
        )
        self.reference_wheel_z_b.copy_(
            current_symmetric_wheel_z.expand(-1, 2)     # 把平均值扩展成左右两份，并写入参考张量
        )

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
        self.max_wheel_z_diff[begin] = 0.0
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

        start_push = crouch & (self.time >= cfg.crouch_duration)
        # 保存实际开始蹬地时的机身高度。
        self.push_start_height[start_push] = (
            height[start_push]
        )
        # 为了在离地后继续上升target_height，
        # 计算所需的离地竖直速度。根号2gh
        release_vz = torch.sqrt(
            2.0
            * cfg.gravity
            * self.target_height[start_push]
        )
        self.reference_release_vz[start_push] = release_vz
        # 从实际下蹲高度运动到计划离地高度。
        push_distance = (
            cfg.release_height
            - self.push_start_height[start_push]
        ).clamp_min(0.0)
        # 初速度为零、匀加速到离地速度的假设，计算所需的蹬地时间。
        push_time = (
            2.0
            * push_distance
            / release_vz.clamp_min(1.0e-6)
        ).clamp(
            cfg.min_push_trajectory_time,
            cfg.max_push_trajectory_time,
        )
        self.reference_push_time[start_push] = push_time
        # 使用实际受限后的push_time重新计算释放高度，
        # 保证PUSH轨迹和FLIGHT轨迹连续。
        self.reference_release_height[start_push] = (
            self.push_start_height[start_push]
            + 0.5
            * release_vz            # 0.5 * vz ：平均速度
            * push_time
        )

        # 从PUSH起点开始连续计时。
        self.trajectory_time[start_push] = 0.0

        self.enter(start_push, PUSH)

        # 3. PUSH和FLIGHT共用一条连续机身轨迹。
        push = old_phase == PUSH
        flight = old_phase == FLIGHT
        trajectory_phase = push | flight

        push_time = self.reference_push_time.clamp_min(dt)
        elapsed = self.trajectory_time
        push_elapsed = torch.minimum(elapsed, push_time)

        push_acceleration = (
            self.reference_release_vz / push_time
        )

        # PUSH阶段恒加速度轨迹。
        push_height_ref = (
            self.push_start_height
            + 0.5
            * push_acceleration
            * push_elapsed.square()
        )
        push_vz_ref = push_acceleration * push_elapsed

        # 计划释放后的自由飞行时间。
        flight_elapsed = (elapsed - push_time).clamp_min(0.0)
        flight_height_ref = (
            self.reference_release_height
            + self.reference_release_vz * flight_elapsed
            - 0.5 * cfg.gravity * flight_elapsed.square()
        )
        flight_vz_ref = (
            self.reference_release_vz
            - cfg.gravity * flight_elapsed
        )

        still_pushing_reference = (
            elapsed < push_time
        )
        trajectory_height_ref = torch.where(
            still_pushing_reference,
            push_height_ref,
            flight_height_ref,
        )
        trajectory_vz_ref = torch.where(
            still_pushing_reference,
            push_vz_ref,
            flight_vz_ref,
        )

        self.reference_height[trajectory_phase] = (
            trajectory_height_ref[trajectory_phase]
        )
        self.reference_vz[trajectory_phase] = (
            trajectory_vz_ref[trajectory_phase]
        )

        # 判断是否处于空中。
        airborne_candidate = (
            push
            & self.push_release_ready
            & (vz > cfg.takeoff_min_vz)
        )

        self.push_max_vz[push] = torch.maximum(
            self.push_max_vz[push], vz[push]
        )
        self.push_wheel_free_steps += (
            push & main_wheels_free
        ).float()
        self.push_candidate_steps += airborne_candidate.float()
        self.push_support_free_steps += (
            push & support_free
        ).float()
        self.push_clearance_steps += (
            push & clearance_ok
        ).float()

        first_air = airborne_candidate & (self.air_confirm_time == 0.0)

        self.takeoff_height[first_air] = height[first_air]
        self.takeoff_vz[first_air] = vz[first_air]
        self.peak_height[first_air] = height[first_air]     # 将当前高度作为初始最高高度.
        self.takeoff_wheel_pos_b[first_air] = self.wheel_pos_b[first_air]
        # 左右轮使用共同的轨迹起点。
        # 在刚离地时，给左右轮设置相同的空中轨迹起点
        symmetric_takeoff_wheel_z = (
            self.wheel_pos_b[first_air, :, 2]
            .mean(dim=-1)
        )
        self.takeoff_wheel_z_b[first_air] = (
            symmetric_takeoff_wheel_z
        )
        self.reference_wheel_z_b[first_air] = (
            symmetric_takeoff_wheel_z
            .unsqueeze(-1)
            .expand(-1, 2)
        )
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

        # 根据计划离地速度计算最高点时间。
        apex_time = (
            self.reference_release_vz
            / cfg.gravity
        )
        # 上升阶段，从离地位置向上收腿。tuck:收腿
        tuck_ratio = (
            self.air_time
            /cfg.tuck_duration
        ).clamp(0.0, 1.0)
        tuck_smooth = (
            tuck_ratio.square()
            * (3.0 - 2.0 * tuck_ratio)
        )
        tucked_wheel_z_b = (
            self.takeoff_wheel_z_b
            + cfg.leg_retraction_distance
        )
        ascent_wheel_z_ref = (      #ascent:上升阶段
            self.takeoff_wheel_z_b
            + cfg.leg_retraction_distance
            * tuck_smooth
        )
        # 计划越过最高点后，开始伸腿准备落地。
        descent_time = (
            self.air_time - apex_time
        ).clamp_min(0.0)
        extension_ratio = (
            descent_time
            / cfg.extension_duration
        ).clamp(0.0, 1.0)
        extension_smooth = (
            extension_ratio.square()
            * (3.0 - 2.0 * extension_ratio)
        )
        descent_wheel_z_ref = (
            tucked_wheel_z_b
            + (
                self.takeoff_wheel_z_b
                - tucked_wheel_z_b
            )
            * extension_smooth
        )

        wheel_z_reference = torch.where(
            self.air_time <= apex_time,
            ascent_wheel_z_ref,
            descent_wheel_z_ref,
        )
        self.reference_wheel_z_b[flight] = (
            wheel_z_reference[flight]
            .unsqueeze(-1)
            .expand(-1, 2)
        )

        # 更新本次腾空的最高点和最大间隙。
        self.peak_clearance[airborne] = torch.maximum(
            self.peak_clearance[airborne],
            clearance[airborne].amin(dim=-1),
        )
        self.peak_height[airborne] = torch.maximum(
            self.peak_height[airborne],
            height[airborne],
        )
        wheel_z_diff = (
            self.wheel_pos_b[:, 0, 2]
            - self.wheel_pos_b[:, 1, 2]
        ).abs()
        self.max_wheel_z_diff[airborne] = torch.maximum(
            self.max_wheel_z_diff[airborne],
            wheel_z_diff[airborne],
        )
        self.episode_max_wheel_z_diff[airborne] = torch.maximum(
            self.episode_max_wheel_z_diff[airborne],
            wheel_z_diff[airborne],
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
        height_reached = rise >= torch.clamp_min(
            self.target_height - cfg.height_tolerance,
            cfg.min_valid_rise,
        )
        valid_jump = (
            self.has_taken_off
            & (rise >= cfg.min_valid_rise)
            & height_reached
            & (self.max_wheel_z_diff <= cfg.max_wheel_z_diff_for_success)
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





