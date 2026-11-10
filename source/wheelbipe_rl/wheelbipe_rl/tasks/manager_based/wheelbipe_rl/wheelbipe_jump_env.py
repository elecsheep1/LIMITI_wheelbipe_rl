from isaaclab.envs import ManagerBasedRLEnv

from .mdp.jump.jump_state import get_jump_state


class WheelbipeJumpEnv(ManagerBasedRLEnv):
    def step(self, action):
        self.action_manager.process_action(
            action.to(self.device)
        )
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.is_rendering

        if self._physics_handles_decimation:
            self._sim_step_counter += self.cfg.decimation
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()

            if (
                self._sim_step_counter
                % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render(
                    skip_app_pumping=not self.render_enabled
                )

            self.scene.update(dt=self.step_dt)

        else:
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1

                self.action_manager.apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.recorder_manager.record_post_physics_decimation_step()

                if (
                    self._sim_step_counter
                    % self.cfg.sim.render_interval == 0
                    and is_rendering
                ):
                    self.sim.render(
                        skip_app_pumping=not self.render_enabled
                    )

                self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # 本步物理结果已经更新，但尚未计算终止和奖励。
        get_jump_state(self).update()

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = (
            self.termination_manager.terminated
        )
        self.reset_time_outs = (
            self.termination_manager.time_outs
        )

        self.reward_buf = self.reward_manager.compute(
            dt=self.step_dt
        )

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = (
            self.reset_buf.nonzero(as_tuple=False)
            .squeeze(-1)
            .int()
        )

        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(
                reset_env_ids
            )

            self._reset_idx(reset_env_ids)

            if (
                self.render_enabled
                and is_rendering
                and self.has_rtx_sensors
                and self.cfg.num_rerenders_on_reset > 0
            ):
                for _ in range(
                    self.cfg.num_rerenders_on_reset
                ):
                    self.sim.render()

            self.recorder_manager.record_post_reset(
                reset_env_ids
            )

        self.command_manager.compute(dt=self.step_dt)

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="interval",
                dt=self.step_dt,
            )

        self.obs_buf = self.observation_manager.compute(
            update_history=True
        )

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def _reset_idx(self, env_ids):
        state = getattr(self, "_jump_state", None)

        metrics = {}
        if state is not None and len(env_ids) > 0:
            attempts = state.attempts[env_ids]
            successes = state.successes[env_ids]

            metrics = {
                "Jump/attempts": attempts.mean(),
                "Jump/successes": successes.mean(),
                "Jump/success_rate": (
                    successes.sum()
                    / attempts.sum().clamp_min(1.0)
                ),
                "Jump/max_rise": (
                    state.max_rise[env_ids].mean()
                ),
                "Jump/max_air_time": (
                    state.max_air_time[env_ids].mean()
                ),
                "Jump/max_clearance": (
                    state.max_clearance[env_ids].mean()
                ),
            }

        super()._reset_idx(env_ids)

        if state is not None:
            state.reset(env_ids)
            self.extras["log"].update(metrics)
