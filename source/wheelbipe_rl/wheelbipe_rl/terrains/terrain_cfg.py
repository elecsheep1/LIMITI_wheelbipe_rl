from copy import deepcopy

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils.configclass import configclass



def difficulty_rough(difficulty, cfg):
    """根据难度改变起伏幅度，运行时才导入网格生成函数。"""
    from isaaclab.terrains.height_field.hf_terrains import (
        random_uniform_terrain,
    )
    local_cfg = deepcopy(cfg)
    minimum, maximum = cfg.amplitude_range
    amplitude = minimum + float(difficulty) * (maximum - minimum)       # 随难度线性插值幅度
    local_cfg.noise_range = (-amplitude, amplitude)                     # 随机高度噪声范围：-amplitude 到 +amplitude
    return random_uniform_terrain(difficulty, local_cfg)


@configclass
class DifficultyRoughCfg(terrain_gen.HfRandomUniformTerrainCfg):
    """随机起伏地形，随难度调整噪声幅度。"""

    function: str = (
        "wheelbipe_rl.terrains.terrain_cfg:"
        "difficulty_rough"
    )
    amplitude_range: tuple[float, float] = (0.003, 0.03)

WHEELBIPE_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    curriculum=True,                        # 默认随机生成；开启后按行排列难度，环境配置可覆盖此开关。
    difficulty_range=(0.0, 1.0),            # 归一化难度范围；关闭课程时，每块地形独立随机采样难度。
    size=(8.0, 8.0),
    num_rows=10,
    num_cols=20,
    border_width=10.0,
    horizontal_scale=0.1,                   # 高度场在 x、y 方向的网格间距：10 厘米。
    vertical_scale=0.001,                   # 高度场在 z 方向的量化精度：1 毫米。
    slope_threshold=0.75,                   # 高度场转网格时，将超过此坡度阈值的表面修正为竖直面。
    use_cache=False,                        # 不读取或保存地形缓存，每次运行重新生成。

    sub_terrains={
        # 平坦地面。
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        ),
        # 随机起伏地面；此类型不使用 difficulty 调整噪声幅度。
        "random_rough": DifficultyRoughCfg(
            proportion=0.2,
            amplitude_range=(0.003, 0.03),
            # 实际生成时由 difficulty_rough 覆盖。
            noise_range=(-0.003, 0.003),    # 随机高度噪声范围
            noise_step=0.001,               # 随机采样高度的离散步长
            downsampled_scale=0.2,          # 每隔 20 厘米采样随机高度，再插值到高度场网格。
            border_width=0.5,               # 此子地形内部四周保留 0.5 米宽的边界带。
        ),
        # 正金字塔坡面：从四周向中心升高，中心为平坦平台。
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.03, 0.25),
            platform_width=2.0,
            border_width=0.5,
        ),
        # 倒金字塔坡面：从四周向中心降低，中心为平坦平台。
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.03, 0.25),
            platform_width=2.0,
            border_width=0.5,
        ),
        # 中央平台较低，离开出生平台时上台阶。
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.005, 0.035),
            step_width=0.25,
            platform_width=2.0,
            border_width=0.5,
            holes=False,
        ),
        # 中央平台较高，离开出生平台时下台阶。
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.005, 0.035),
            step_width=0.25,
            platform_width=2.0,
            border_width=0.5,
            holes=False,
        ),
    }
)
