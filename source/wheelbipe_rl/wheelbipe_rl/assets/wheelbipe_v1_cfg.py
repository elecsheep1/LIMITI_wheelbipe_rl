from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


ASSET_DIR = Path(__file__).resolve().parent

WHEELBIPE_USD = (
    ASSET_DIR / "robots" / "wheelbipe_v1" / "wheelbipe_import" / "wheelbipe_corrected.usda"
)

if not WHEELBIPE_USD.is_file():
    raise FileNotFoundError(f"Wheelbipe USD not found: {WHEELBIPE_USD}")

WHEELBIPE_V1_CFG = ArticulationCfg(
    articulation_root_prim_path="/Geometry/base_link",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(WHEELBIPE_USD),
        variants={"Physics": "physx"},
        activate_contact_sensors=True,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),

    soft_joint_pos_limit_factor=1.0,

    actuators={
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_joint",
                "right_hip_joint",
            ],
            effort_limit_sim=20.0,
            velocity_limit_sim=18.0,
            stiffness=30.0,
            damping=2.0,
            armature=0.004,
        ),
        "knee_drives": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_knee_drive_joint",
                "right_knee_drive_joint",
            ],
            effort_limit_sim=20.0,
            velocity_limit_sim=18.0,
            stiffness=30.0,
            damping=2.0,
            armature=0.004,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_wheel_joint",
                "right_wheel_joint",
            ],
            effort_limit_sim=2.0,
            velocity_limit_sim=55.0,
            stiffness=0.0,
            damping=0.2,
            armature=0.003,
        ),
        # 显式声明其余关节为被动。
        "passive": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_(knee|short|bellcrank|connecting|spring_mount|spring_slide)_joint",
                ".*_passive_wheel_[123]_joint",
            ],
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.0,
            armature=0.0,
        ),
    },
)