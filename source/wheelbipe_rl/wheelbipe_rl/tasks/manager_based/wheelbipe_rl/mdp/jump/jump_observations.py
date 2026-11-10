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

    # 新增 7 维观测。
    return torch.cat(
        [
            phase,
            state.target_height.unsqueeze(-1),
            progress.unsqueeze(-1),
        ],
        dim=-1,
    )