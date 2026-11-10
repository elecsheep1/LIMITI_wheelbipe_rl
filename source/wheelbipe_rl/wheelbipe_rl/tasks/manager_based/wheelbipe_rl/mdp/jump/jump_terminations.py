from .jump_state import get_jump_state


def jump_failed(env):
    return get_jump_state(env).failed