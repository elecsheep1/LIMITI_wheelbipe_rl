from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass


@configclass
class DeadzoneVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type | str = (
        "wheelbipe_rl.tasks.manager_based.wheelbipe_rl."
        "mdp.commands:DeadzoneVelocityCommand"
    )

    linear_deadzone: float = 0.04
    yaw_deadzone: float = 0.04
