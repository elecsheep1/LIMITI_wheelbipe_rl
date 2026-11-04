"""仅检查机器人资产加载，不创建强化学习环境。"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 必须在启动应用后导入仿真相关模块。
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from wheelbipe_rl.assets import WHEELBIPE_CFG

def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.002,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=(1.2, 1.2, 0.8),
        target=(0.0, 0.0, 0.25),
    )

    light = sim_utils.DomeLightCfg(intensity=3000.0)
    light.func("/World/Light", light)

    robot_cfg = WHEELBIPE_CFG.replace(
        prim_path="/World/Robot"
    )

    # 首次只检查资产和闭链加载，暂时关闭重力。
    robot_cfg.spawn.rigid_props.disable_gravity = False

    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)

    robot = Articulation(robot_cfg)
    sim.reset()

    print("关节数量：", robot.num_joints)
    print("关节名称：", robot.joint_names)
    print("刚体数量：", robot.num_bodies)
    print("刚体名称：", robot.body_names)

    dt = sim.get_physics_dt()

    for _ in range(2500):
        if not simulation_app.is_running():
            break

        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
