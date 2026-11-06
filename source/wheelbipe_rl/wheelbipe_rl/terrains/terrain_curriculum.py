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
    segment_s: float = 2.0
    min_segment_s: float = 1.0
    # 成功步的沿指令方向速度范围。
    progress_ratio: float = 0.5
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


class TerrainCurriculum:
    """固定地形列，按可通行性调整难度行的采样概率。

    每个“难度行、地形列”单独保存统计。
    地形列不变，因此原有地形类型比例保持不变。
    """

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.terrain = env.scene.terrain
        self.device = env.device
        self.dt = env.step_dt
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
        self.batch_n = torch.zeros_like(self.total)         # 当前批次的运动段数量
        self.batch_score = torch.zeros_like(self.total)     # 当前批次的成功步数
        self.batch_mid = torch.zeros_like(self.total)       # 当前批次的中间步数
        self.batch_fall = torch.zeros_like(self.total)      # 当前批次的失败步数
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
        self.probability = torch.zeros_like(self.total)
        self.probability[0] = 1.0
        self.next_update = int(env.common_step_counter) + cfg.update_steps      # 下一次更新的全局步数
        self.full_steps = max(
            1,
            math.ceil(cfg.segment_s / self.dt),
        )
        self.minimum_steps = max(
            1,
            math.ceil(cfg.min_segment_s / self.dt),
        )

        keys = list(generator_cfg.sub_terrains)
        weights = torch.tensor(
            [
                item.proportion
                for item in generator_cfg.sub_terrains.values()
            ],
            device=self.device,
        )
        cumulative = (weights / weights.sum()).cumsum(dim=0)
        column_choice = (
            torch.arange(
                self.cols,
                device=self.device,
            )
            / self.cols
            + 0.001
        )
        type_ids = torch.searchsorted(
            cumulative,
            column_choice,
            right=True,
        )
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

    def _flush(self, env_ids, failed=None):
        """结束并提交指定环境的当前运动段。"""
        ids = self.ids(env_ids)
        if ids.numel() == 0: return
        if failed is None:
            failed = torch.zeros(
                ids.numel(),
                device=self.device,
                dtype=torch.bool,
            )
        count = self.nvalid[ids]
        # 普通窗口要求足够长。
        # 短失败窗口也保留，防止早期跌倒被忽略。
        keep = ((count >= self.minimum_steps) | (failed & (count > 0)))
        selected = ids[keep]
        if selected.numel():
            failures = failed[keep]
            score = self.nsuccess[selected] / self.nvalid[selected].clamp_min(1)
            # 跌倒终止的末窗口视为失败。
            score = torch.where(
                failures,
                torch.zeros_like(score),
                score,
            )
            mid = ((score >= self.cfg.lower) & (score <= self.cfg.upper)).float()
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
            index = level * self.cols + column
            one = torch.ones_like(score)
            updates = (
                (self.total, one),
                (self.batch_n, one),
                (self.batch_score, score),
                (self.batch_mid, mid),
                (self.batch_fall, failures.float()),
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
        valid = (
            (speed >= self.cfg.min_speed)
            & (cmd[:, 2].abs() <= self.cfg.max_yaw)
            & (self.age[ids] >= self.cfg.grace_s)
            & (self.command_age[ids] >= self.cfg.grace_s)
            & torch.isfinite(velocity).all(dim=-1)
            & torch.isfinite(cmd).all(dim=-1)
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

    def end_episode(self, env_ids, terminated):
        """回合结束，提交剩余运动段并清空环境统计。"""
        ids = self.ids(env_ids)
        self._flush(
            ids,
            failed=terminated[ids],
        )
        self.age[ids] = 0
        self.command_age[ids] = 0
        self.previous_command[ids] = 0

    def update(self):
        """周期性更新统计估计和采样分布。"""
        if self.env.common_step_counter < self.next_update:
            return
        self.next_update = int(self.env.common_step_counter) + self.cfg.update_steps
        # 数据不足的单元继续积累，不清空。
        ready = self.batch_n >= self.cfg.min_segments
        first = ready & (~self.estimated)

        estimates = (
            (self.score, self.batch_score),
            (self.desirability, self.batch_mid),
            (self.fall_rate, self.batch_fall),
        )
        for estimate, summed in estimates:
            value = (summed / self.batch_n.clamp_min(1))
            blended = torch.lerp(
                estimate,
                value,
                self.cfg.ema_alpha,
            )
            estimate[ready] = blended[ready]
            estimate[first] = value[first]
        self.estimated |= ready
        columns = torch.arange(
            self.cols,
            device=self.device,
        )
        level = self.frontier
        mastered = (
            ready[level, columns]
            & (self.score[level, columns] > self.cfg.upper)
            & (self.fall_rate[level, columns] < self.cfg.max_fall_rate)
            & (~self.flat_columns)
        )
        self.frontier = (level + mastered.long()).clamp_max(self.rows - 1)

        levels = torch.arange(
            self.rows,
            device=self.device,
        )[:, None]
        allowed = levels <= self.frontier[None, :]
        # 70%：适宜地形。
        # 未充分评价的已开放单元保留一个探索先验。
        focus = torch.where(
            self.estimated,
            self.desirability,
            0.1,
        )
        focus = (
            (focus + 0.02)
            * (1 - self.fall_rate)
            * allowed
        )
        focus /= (
            focus.sum(dim=0, keepdim=True)
            .clamp_min(1e-6)
        )

        # 20%：较低等级回顾。
        easy = (
            allowed
            & (levels < self.frontier[None, :])
        ).float()
        empty = easy.sum(dim=0) == 0
        easy[0, empty] = 1
        easy /= easy.sum(
            dim=0,
            keepdim=True,
        )
        # 10%：当前开放前沿。
        explore = (levels == self.frontier[None, :]).float()
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
            self.batch_fall,
        ):
            buffer[ready] = 0
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

        return {
            "traversability": mean(self.score),
            "desirability": mean(self.desirability),
            "fall_rate": mean(self.fall_rate),
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