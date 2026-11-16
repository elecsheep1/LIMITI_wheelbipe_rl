import torch
import torch.nn.functional as F

from .jump_state import get_jump_state


def jump_observation(env):
    state = get_jump_state(env)

    phase = F.one_hot(
        state.phase,
        num_classes=5,
    ).float()

    progress = (
        state.time / state.phase_durations[state.phase]
    ).clamp(0.0, 1.0)

    # 5维阶段 + 目标高度 + 阶段进度
    # + 参考机身高度 + 参考竖直速度
    # + 左右轮参考高度，共11维。
    return torch.cat(
        [
            phase,
            state.target_height.unsqueeze(-1),
            progress.unsqueeze(-1),
            state.reference_height.unsqueeze(-1),
            state.reference_vz.unsqueeze(-1),
            state.reference_wheel_z_b,
        ],
        dim=-1,
    )
