from copy import deepcopy

import isaaclab.sim as sim_utils

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import (
    RayCasterCfg,
    patterns,
)
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from wheelbipe_rl.terrains.terrain_cfg import (
    WHEELBIPE_TERRAINS_CFG,
)
from wheelbipe_rl.terrains.terrain_curriculum import (
    TraversabilityCfg,
)

from . import mdp
from .wheelbipe_rl_env_cfg import (
    TARGET_BASE_HEIGHT,
    WheelbipeRlEnvCfg,
    WheelbipeRlSceneCfg,
)


@configclass
class WheelbipeRoughSceneCfg(WheelbipeRlSceneCfg):
    """Configuration for a cart-pole scene."""

    ground = None

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type = "generator",
        terrain_generator=deepcopy(WHEELBIPE_TERRAINS_CFG),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # 测量机身附近地面高度，仅用于奖励。
    height_scanner = RayCasterCfg(
        prim_path=("{ENV_REGEX_NS}/Robot/Geometry/base_link"),
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 2.0),
        ),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.01,
            size=(0.02, 0.02),
        ),
        mesh_prim_paths=["/World/ground"],
        max_distance=10.0,
        debug_vis=False,
    )


@configclass
class RoughCurriculumCfg:
    traversability = CurrTerm(
        func=mdp.traversability_sampling,
    )


@configclass
class WheelbipeRoughEnvCfg(WheelbipeRlEnvCfg):
    scene: WheelbipeRoughSceneCfg = WheelbipeRoughSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
        filter_collisions=True,
    )

    curriculum: RoughCurriculumCfg = RoughCurriculumCfg()
    traversability: TraversabilityCfg = TraversabilityCfg()

    def __post_init__(self):
        super().__post_init__()

        # 先由继承的 reset_scene 恢复关节与控制目标，
        # 再随机化整个机器人根部的位置。
        self.events.randomize_spawn = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                },
                "velocity_range": {},
            },
        )

        # 扫描器按控制频率更新。
        self.scene.height_scanner.update_period = (self.decimation * self.sim.dt)
        # 高度奖励改为机身相对局部地面的高度。
        self.rewards.height_tracking.params = {
            "target_height": TARGET_BASE_HEIGHT,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        }
        # 地形块边界终止按超时处理。
        self.terminations.terrain_boundary = DoneTerm(
            func=mdp.terrain_patch_boundary,
            time_out=True,
            params={
                "margin": 0.5,
            },
        )
        # 必须按难度行、类型列组织地图。
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class WheelbipeRoughEnvCfg_PLAY(WheelbipeRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16

        # 播放时不收集或更新课程。
        self.traversability.enabled = False
        self.curriculum.traversability = None
        # 默认检查最低等级。
        self.scene.terrain.max_init_terrain_level = 0
