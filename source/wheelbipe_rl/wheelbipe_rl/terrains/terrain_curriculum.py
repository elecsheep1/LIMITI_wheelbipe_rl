"""
- 有效运动段统计。
- 可通行性与失败率估计。
- 开放下一等级。
- 更新采样概率。
- 保存和恢复课程统计。
"""

from dataclasses import dataclass
from pathlib import Path
import math

import torch


@dataclass      # 数据类装饰器，用于创建数据类
class TraversabilityCfg:
    """课程统计配置。"""
    enabled: bool = True
    # 恢复已有 Rough 训练时，可以指定课程统计文件。
    state_path: str = ""
    command_name: str = "base_velocity"
    # 只有满足条件的平移指令才参与课程评价。
    min_speed: float = 0.1
    max_yaw: float = 0.2
    # 出生或明显切换指令后的等待时间。
    grace_s: float = 0.5
    # 按有效运动时间划分窗口。
    segment_s: float = 2.0              # 累计够 2 秒的有效运动步，就提交一段统计。
    min_segment_s: float = 1.0          # 允许短失败窗口，防止早期跌倒被忽略。果指令改变或回合结束，当前窗口还没满 2 秒，只要已有至少 1 秒有效运动，也可以提交。
    # 成功步的沿指令方向速度范围。
    progress_ratio: float = 0.5         # 0.5 * 指令速度 <= progress <= 1.5 * 指令速度 + 0.05
    max_progress_ratio: float = 1.5
    # rollout 为 24 步时，相当于约 100 轮 PPO。
    update_steps: int = 2400
    # 每个地形单元积累足够的新运动段后才更新估计。
    min_segments: int = 30
    ema_alpha: float = 0.2
    # 适宜的可通行性范围。
    lower: float = 0.5
    upper: float = 0.9
    # 开放更高等级时允许的最大失败率。
    max_fall_rate: float = 0.1
    min_episodes: int = 30
    # 用于排除明显错误的初始状态。
    min_spawn_clearance: float = 0.10


class TerrainCurriculum:
    """固定地形列，按可通行性调整难度行的采样概率。

    每个“难度行、地形列”单独保存统计。
    地形列不变，因此原有地形类型比例保持不变。
    """

    def __init__(self, env, cfg):
        # 保存环境和配置
        self.env = env
        self.cfg = cfg
        self.terrain = env.scene.terrain
        self.device = env.device
        self.dt = env.step_dt
        # 保存地形生成器配置
        generator_cfg = self.terrain.cfg.terrain_generator
        self.rows = generator_cfg.num_rows
        self.cols = generator_cfg.num_cols
        num_envs = env.num_envs
        # 每个环境当前运动段的统计。
        self.nvalid = torch.zeros(
            num_envs,
            device=self.device,
        )
        self.nsuccess = torch.zeros_like(self.nvalid)       # 成功步数
        self.age = torch.zeros_like(self.nvalid)            # 运动段的累计时间
        self.command_age = torch.zeros_like(self.nvalid)    # 当前指令的累计时间
        self.spawn_pending = torch.ones(
            num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        self.spawn_valid = torch.zeros_like(self.spawn_pending)
        self.episode_eligible = torch.zeros_like(self.spawn_pending)
        self.previous_command = torch.zeros(
            num_envs,
            3,
            device=self.device,
        )
        shape = (self.rows, self.cols)
        # 累计运动段数量。
        self.total = torch.zeros(
            shape,
            device=self.device,
        )
        self.batch_n = torch.zeros_like(self.total)         # 尚未用于更新估计的新运动段数量
        self.batch_score = torch.zeros_like(self.total)     # 这些运动段的得分之和
        self.batch_mid = torch.zeros_like(self.total)       # 得分落在 [lower, upper] 内的运动段数量
        self.batch_fall = torch.zeros_like(self.total)      # 以失败终止的运动段数量
        self.batch_episode_n = torch.zeros_like(self.total)
        self.fall_estimated = torch.zeros(
            shape,
            device=self.device,
            dtype=torch.bool,
        )
        # 各地形单元的平滑估计。
        self.score = torch.full(
            shape,
            0.5,
            device=self.device,
        )
        self.desirability = torch.zeros_like(self.total)    # 运动段可通行性处于目标区间的比例
        self.fall_rate = torch.zeros_like(self.total)       # 各地形单元的失败率估计
        self.estimated = torch.zeros(                       # 各地形单元的估计是否有效
            shape,
            device=self.device,
            dtype=torch.bool,
        )
        # 每列已经开放的最高等级。
        self.frontier = torch.zeros(
            self.cols,
            device=self.device,
            dtype=torch.long,
        )
        # 每一列内，各难度行的采样概率。
        """
        [
            [1, 1, 1, ..., 1],  # 第 0 级
            [0, 0, 0, ..., 0],  # 第 1 级
            [0, 0, 0, ..., 0],  # 第 2 级
            [0, 0, 0, ..., 0],  # 第 3 级
        ]
        """
        self.probability = torch.zeros_like(self.total)
        self.probability[0] = 1.0
        self.next_update = int(env.common_step_counter) + cfg.update_steps      # 下一次更新的全局步数
        # 将完整运动段时长换算成有效步数。
        self.full_steps = max(
            1,
            math.ceil(cfg.segment_s / self.dt),
        )
        # 将最短保留时长换算为步数。
        self.minimum_steps = max(
            1,
            math.ceil(cfg.min_segment_s / self.dt),
        )
        # 取得地形名称，顺序与配置字典一致
        keys = list(generator_cfg.sub_terrains)
        weights = torch.tensor(
            [
                item.proportion
                for item in generator_cfg.sub_terrains.values()
            ],
            device=self.device,
        )
        # 将权重变成区间边界
        cumulative = (weights / weights.sum()).cumsum(dim=0)        # 归一化再求和
        # 在 [0, 1) 区间内均匀采样，找到对应的地形类型（给每一列计算一个位置）
        column_choice = (
            torch.arange(
                self.cols,
                device=self.device,
            )
            / self.cols
            + 0.001
        )
        # 对每个位置，找到第一个严格大于它的区间边界，返回该边界的索引。
        type_ids = torch.searchsorted(
            cumulative,
            column_choice,
            right=True,
        )
        # 保存每列对应的地形类型名称
        self.column_names = [keys[int(index)] for index in type_ids.cpu()]
        self.flat_columns = torch.tensor(
            [
                name == "flat"
                for name in self.column_names
            ],
            device=self.device,
        )
        if cfg.state_path:
            self.load(cfg.state_path)

    def ids(self, env_ids):
        """把环境索引转换成设备上的 long Tensor。"""
        if env_ids is None or isinstance(env_ids, slice):
            all_ids = torch.arange(
                self.env.num_envs,
                device=self.device,
            )
            selection = (
                slice(None)
                if env_ids is None
                else env_ids
            )
            return all_ids[selection]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _flush(self, env_ids):
        """结束并提交指定环境的当前运动段。"""
        ids = self.ids(env_ids)
        if ids.numel() == 0: return
        # 取出这些环境当前积累的有效步数。
        count = self.nvalid[ids]
        # 普通窗口要求足够长。
        # 短失败窗口也保留，防止早期跌倒被忽略。
        keep = count >= self.minimum_steps
        selected = ids[keep]
        if selected.numel():
            score = self.nsuccess[selected] / self.nvalid[selected].clamp_min(1)
            mid = ((score >= self.cfg.lower) & (score <= self.cfg.upper)).float()
            # 别取得每个环境所在的： level：难度行。 column：地形列。
            level = (
                self.terrain
                .terrain_levels[selected]
                .long()
            )
            column = (
                self.terrain
                .terrain_types[selected]
                .long()
            )
            # 把二维坐标换成一维索引
            index = level * self.cols + column
            one = torch.ones_like(score)
            # 把每个环境的运动段结果，加到它所在的地形统计表里
            updates = (
                (self.total, one),
                (self.batch_n, one),
                (self.batch_score, score),
                (self.batch_mid, mid),
            )
            for target, values in updates:
                target.view(-1).scatter_add_(
                    0,
                    index,
                    values,
                )
        self.nvalid[ids] = 0
        self.nsuccess[ids] = 0

    def observe(self, env_ids, command, terminated):
        """记录一次真实物理步后的状态。"""
        ids = self.ids(env_ids)
        if ids.numel() == 0: return
        cmd = command[ids]
        delta = cmd - self.previous_command[ids]
        changed = (
            (delta[:, :2].norm(dim=-1) > 0.02)
            | (delta[:, 2].abs() > 0.05)
        )
        # 指令明显改变时，结束上一运动段。
        self._flush(ids[changed])
        self.command_age[ids[changed]] = 0
        self.previous_command[ids] = cmd
        self.age[ids] += self.dt
        self.command_age[ids] += self.dt
        speed = cmd[:, :2].norm(dim=-1)
        velocity = self.env.scene["robot"].data.root_lin_vel_b.torch[ids, :2]
        direction = cmd[:, :2] / speed.clamp_min(1e-6).unsqueeze(-1)
        progress = (velocity * direction).sum(dim=-1)
        command_ok = (
            self.spawn_valid[ids]
            & (speed >= self.cfg.min_speed)
            & (cmd[:, 2].abs() <= self.cfg.max_yaw)
            & torch.isfinite(cmd).all(dim=-1)
        )
        self.episode_eligible[ids] |= command_ok
        valid = (
            command_ok
            & (self.age[ids] >= self.cfg.grace_s)
            & (self.command_age[ids] >= self.cfg.grace_s)
            & torch.isfinite(velocity).all(dim=-1)
        )
        success = (
            (progress >= self.cfg.progress_ratio * speed)
            & (
                progress
                <= (
                    self.cfg.max_progress_ratio
                    * speed
                    + 0.05
                )
            )
            & (~terminated[ids])
        )
        self.nvalid[ids] += valid.float()
        self.nsuccess[ids] += (valid & success).float()
        full = (
            (self.nvalid[ids] >= self.full_steps)
            & (~terminated[ids])
        )
        self._flush(ids[full])

    def end_episode(
        self,
        env_ids,
        terminated,
        record_outcome=True,
    ):
        """回合结束，提交剩余运动段并清空环境统计。"""
        ids = self.ids(env_ids)
        self._flush(ids)
        if record_outcome:
            selected = ids[self.episode_eligible[ids]]

            if selected.numel() > 0:
                level = self.terrain.terrain_levels[selected].long()
                column = self.terrain.terrain_types[selected].long()
                index = level * self.cols + column

                self.batch_episode_n.view(-1).scatter_add_(
                    0,
                    index,
                    torch.ones(
                        selected.numel(),
                        device=self.device,
                    ),
                )

                self.batch_fall.view(-1).scatter_add_(
                    0,
                    index,
                    terminated[selected].float(),
                )

        self.spawn_pending[ids] = True
        self.spawn_valid[ids] = False
        self.episode_eligible[ids] = False
        self.age[ids] = 0
        self.command_age[ids] = 0
        self.previous_command[ids] = 0

    def update(self):
        """
        周期性更新统计估计和采样分布。
        等数据够了再评价
        每隔 update_steps 步检查一次。某块地形积累的运动段达到 min_segments，才更新它；不足的继续积累。

        更新对各地形的能力判断
        计算平均得分、适宜运动段比例和失败率，用移动平均平滑，避免一次表现好坏就大幅改变判断。

        练好了才开放下一级
        每列独立判断：当前最高开放等级得分够高、失败率够低，就再开放一级。已经开放的等级不会关闭。

        安排后续练习
        在已经开放的等级中，混合三种采样策略：
        - 70%：重点练适合当前能力、失败率较低的等级。
        - 20%：回顾较低等级，巩固已有能力。
        - 10%：练当前最高开放等级，继续探索。
        平地固定选第 0 级。

        清理并保存
        清空本次已经用于更新的批次数据，保留长期估计，并按配置保存课程状态。
        """
        if self.env.common_step_counter < self.next_update:
            return
        self.next_update = int(self.env.common_step_counter) + self.cfg.update_steps
        # 判断哪些地形数据足够
        ready = self.batch_n >= self.cfg.min_segments
        # 数据已经够了，而且以前还没有完成过估计。
        first = ready & (~self.estimated)
        # 更新三种表现估计
        estimates = (
            (self.score, self.batch_score),
            (self.desirability, self.batch_mid),
        )
        for estimate, summed in estimates:
            # 计算本批平均值
            value = (summed / self.batch_n.clamp_min(1))
            # ema平滑
            blended = torch.lerp(
                estimate,
                value,
                self.cfg.ema_alpha,
            )
            # 已有估计、数据足够：使用平滑结果。
            estimate[ready] = blended[ready]
            # 第一次估计：直接使用本批值
            estimate[first] = value[first]
        # 把本次完成估计的地形标记为 True，以前已经为 True 的保持不变
        self.estimated |= ready
        fall_ready = (
            self.batch_episode_n >= self.cfg.min_episodes
        )
        fall_first = fall_ready & (~self.fall_estimated)

        fall_value = (
            self.batch_fall
            / self.batch_episode_n.clamp_min(1.0)
        )

        fall_blended = torch.lerp(
            self.fall_rate,
            fall_value,
            self.cfg.ema_alpha,
        )

        self.fall_rate[fall_ready] = fall_blended[fall_ready]
        self.fall_rate[fall_first] = fall_value[fall_first]

        self.fall_estimated |= fall_ready
        # 更新每列的前沿等级：如果当前等级的可通行性足够高，且失败率足够低，就开放下一等级。
        columns = torch.arange(
            self.cols,
            device=self.device,
        )
        # 每列目前最高开放到哪一级
        level = self.frontier
        # 表示索引，# 第columns列的第level级
        mastered = (
            ready[level, columns]
            & self.fall_estimated[level, columns]
            & (self.score[level, columns] > self.cfg.upper)
            & (self.fall_rate[level, columns] < self.cfg.max_fall_rate)
            & (~self.flat_columns)
        )
        # 更新前沿等级，不能超过总行数减一（mastered布尔值转化为整数）
        self.frontier = (level + mastered.long()).clamp_max(self.rows - 1)
        # 把所有等级编号排成一列
        levels = torch.arange(
            self.rows,
            device=self.device,
        )[:, None]
        # 逐格判断这个等级是否已经开放
        """
        [
            [0 <= 1, 0 <= 2, 0 <= 2],
            [1 <= 1, 1 <= 2, 1 <= 2],
            [2 <= 1, 2 <= 2, 2 <= 2],
            [3 <= 1, 3 <= 2, 3 <= 2],
        ]
        allowed = tensor([
            [True,  True,  True ],
            [True,  True,  True ],
            [False, True,  True ],
            [False, False, False],
        ])
        """
        allowed = levels <= self.frontier[None, :]
        # 70%：适宜地形。
        # 给每个等级一个“值得重点练习”的初始权重
        focus = torch.where(
            self.estimated,
            self.desirability,
            0.1,
        )
        # 调整权重，排除未开放等级
        focus = (
            (focus + 0.02)
            * (1 - self.fall_rate)
            * allowed
        )
        # 归一化成概率分布
        focus /= (
            focus.sum(dim=0, keepdim=True)
            .clamp_min(1e-6)
        )

        # 20%：较低等级回顾。
        # 选出低于当前最高开放等级的那些等级。
        easy = (
            allowed
            & (levels < self.frontier[None, :])
        ).float()
        # 如果没有旧等级，就回顾第 0 级
        empty = easy.sum(dim=0) == 0
        easy[0, empty] = 1
        easy /= easy.sum(
            dim=0,
            keepdim=True,
        )
        # 10%：当前开放前沿。
        explore = (levels == self.frontier[None, :]).float()
        # 混合三套方案，得到最终概率
        self.probability.copy_(
            0.7 * focus
            + 0.2 * easy
            + 0.1 * explore
        )
        # 平地不需要重复练习不同难度行。
        self.probability[:, self.flat_columns] = 0
        self.probability[0, self.flat_columns] = 1
        for buffer in (
            self.batch_n,
            self.batch_score,
            self.batch_mid,
        ):
            buffer[ready] = 0

        self.batch_episode_n[fall_ready] = 0
        self.batch_fall[fall_ready] = 0
        log_dir = getattr(
            self.env.cfg,
            "log_dir",
            None,
        )
        if log_dir:
            self.save(
                Path(log_dir)
                / "terrain_curriculum.pt"
            )

    def sample(self, env_ids):
        """重置前采样新等级并同步环境原点。"""
        ids = self.ids(env_ids)
        if ids.numel() == 0: return
        column = (
            self.terrain
            .terrain_types[ids]
            .long()
        )
        probabilities = self.probability[:, column].T
        levels = torch.multinomial(
            probabilities,
            num_samples=1,
        ).squeeze(-1)
        self.terrain.terrain_levels[ids] = levels
        origins = self.terrain.terrain_origins[
            levels,
            column,
        ]
        self.terrain.env_origins[ids] = origins

    def metrics(self):
        """返回 CurriculumManager 可以记录的标量。"""
        known = (self.estimated & (~self.flat_columns[None, :]))

        def mean(value):
            return (
                (value * known).sum()
                / known.sum().clamp_min(1)
            )

        probability = self.probability
        fall_known = known & self.fall_estimated

        mean_fall_rate = (
            (self.fall_rate * fall_known).sum()
            / fall_known.sum().clamp_min(1)
        )

        return {
            "traversability": mean(self.score),
            "desirability": mean(self.desirability),
            "fall_rate": mean_fall_rate,
            "segments": self.total.sum(),
            "mean_level": (
                self.terrain
                .terrain_levels
                .float()
                .mean()
            ),
            "frontier": (self.frontier.float().mean()),
            "sampling_entropy": -(
                probability
                * probability.clamp_min(1e-8).log()
            ).sum(dim=0).mean(),
        }

    def save(self, path):
        """保存统计分布，不保存进行中的运动段。"""
        keys = (
            "total",
            "batch_n",
            "batch_score",
            "batch_mid",
            "batch_fall",
            "batch_episode_n",
            "fall_estimated",
            "score",
            "desirability",
            "fall_rate",
            "estimated",
            "frontier",
            "probability",
        )
        state = {
            key: getattr(self, key).cpu()
            for key in keys
        }
        state["column_names"] = self.column_names
        state["rows"] = self.rows
        state["version"] = 2

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path):
        state = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )
        if state.get("version", 1) != 2:
            raise ValueError(
                "课程统计格式已改变，请使用新的课程统计，"
                "不要加载旧 terrain_curriculum.pt。"
            )
        if (
            state["rows"] != self.rows
            or (
                state["column_names"]
                != self.column_names
            )
        ):
            raise ValueError(
                "Curriculum checkpoint does not "
                "match the terrain layout."
            )
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                getattr(self, key).copy_(value)

    def validate_spawn(self):
        ids = self.ids(
            self.spawn_pending.nonzero(
                as_tuple=False
            ).squeeze(-1)
        )
        if ids.numel() == 0:
            return

        robot = self.env.scene["robot"]

        root = robot.data.root_pos_w.torch[ids]
        gravity = robot.data.projected_gravity_b.torch[ids]

        hits = (
            self.env.scene["height_scanner"]
            .data.ray_hits_w.torch[ids, :, 2]
        )

        finite_hits = torch.where(
            torch.isfinite(hits),
            hits,
            torch.full_like(hits, float("nan")),
        )
        ground = finite_hits.nanmedian(dim=1).values

        self.spawn_valid[ids] = (
            torch.isfinite(root).all(dim=-1)
            & torch.isfinite(gravity).all(dim=-1)
            & torch.isfinite(ground)
            & (
                root[:, 2] - ground
                > self.cfg.min_spawn_clearance
            )
            & (-gravity[:, 2] > 0.5)
        )

        self.spawn_pending[ids] = False