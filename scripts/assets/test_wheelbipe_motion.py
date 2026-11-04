"""无重力测试：髋、膝驱动和轮子，检查闭链运动。"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--cycles", type=int, default=2, help="测试循环次数")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 仿真相关模块必须在启动应用之后导入。
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from wheelbipe_rl.assets import WHEELBIPE_CFG

def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=0.002,
            device=args_cli.device,
            gravity=(0.0, 0.0, 0.0),
        )
    )
    sim.set_camera_view(
        eye=(1.0, 1.0, 0.7),
        target=(0.0, 0.0, 0.35),
    )

    light = sim_utils.DomeLightCfg(intensity=1500.0)
    light.func("/World/Light", light)

    cfg = WHEELBIPE_CFG.copy()
    cfg.prim_path = "/World/Robot"
    cfg.spawn.rigid_props.disable_gravity = True

    robot = Articulation(cfg)
    sim.reset()

    # 按名称获取索引，不依赖 USD 内部关节排列。
    leg_names = [
        "left_hip_joint",
        "right_hip_joint",
        "left_knee_drive_joint",
        "right_knee_drive_joint",
    ]
    wheel_names = [
        "left_wheel_joint",
        "right_wheel_joint",
    ]
    leg_ids = [robot.joint_names.index(name) for name in leg_names]
    wheel_ids = [robot.joint_names.index(name) for name in wheel_names]

    device = robot.data.joint_pos.torch.device
    leg_target = torch.zeros((1, 4), device=device)
    wheel_target = torch.zeros((1, 2), device=device)

    # 每段 6 秒；腿部单向往返，避免跨过当前被动关节零位限位。
    phases = [
        ("零位保持", None, 0.0),
        ("左髋：0 → -15° → 0", 0, -math.radians(15.0)),
        ("右髋：0 → -15° → 0", 1, -math.radians(15.0)),
        ("左膝驱动：0 → +15° → 0", 2, math.radians(15.0)),
        ("右膝驱动：0 → +15° → 0", 3, math.radians(15.0)),
        ("左轮：正转、停止、反转、停止", "left_wheel", 0.0),
        ("右轮：正转、停止、反转、停止", "right_wheel", 0.0),
    ]

    dt = sim.get_physics_dt()
    phase_duration = 10.0
    phase_steps = round(phase_duration / dt)
    print_steps = round(1.0 / dt)

    for cycle in range(args_cli.cycles):
        for label, channel, amplitude in phases:
            print(f"\n循环 {cycle + 1}：{label}")

            for step in range(phase_steps):
                if not simulation_app.is_running():
                    return

                t = step * dt
                leg_target.zero_()
                wheel_target.zero_()

                if isinstance(channel, int):
                    # 平滑往返：起点、终点均为零，3 秒到达最大幅度。
                    blend = 0.5 * (
                        1.0 - math.cos(2.0 * math.pi * t / phase_duration)
                    )
                    leg_target[0, channel] = amplitude * blend

                elif channel in ("left_wheel", "right_wheel"):
                    idx = 0 if channel == "left_wheel" else 1

                    # 正、反转均带平滑加减速，峰值 3 rad/s。
                    if t < 2.0:
                        speed = 3.0 * math.sin(math.pi * t / 2.0) ** 2
                    elif t < 3.0:
                        speed = 0.0
                    elif t < 5.0:
                        speed = -3.0 * math.sin(math.pi * (t - 3.0) / 2.0) ** 2
                    else:
                        speed = 0.0

                    wheel_target[0, idx] = speed

                robot.set_joint_position_target_index(
                    target=leg_target,
                    joint_ids=leg_ids,
                )
                robot.set_joint_velocity_target_index(
                    target=wheel_target,
                    joint_ids=wheel_ids,
                )

                robot.write_data_to_sim()
                sim.step()
                robot.update(dt)

                q = robot.data.joint_pos.torch
                qd = robot.data.joint_vel.torch

                if not (
                    torch.isfinite(q).all().item()
                    and torch.isfinite(qd).all().item()
                ):
                    raise RuntimeError("关节状态出现 NaN/Inf，停止测试。")

                if step % print_steps == 0:
                    actual_deg = torch.rad2deg(q[0, leg_ids]).tolist()
                    target_deg = torch.rad2deg(leg_target[0]).tolist()
                    wheel_speed = qd[0, wheel_ids].tolist()

                    print(
                        f"t={t:.1f}s | "
                        f"腿目标°={[round(v, 2) for v in target_deg]} | "
                        f"腿实际°={[round(v, 2) for v in actual_deg]} | "
                        f"轮速rad/s={[round(v, 2) for v in wheel_speed]}"
                    )

    print("\n运动测试结束。请结合画面判断连接、方向和跟随情况。")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()