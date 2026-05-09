import gymnasium as gym


class OperationMujocoDualHsrHandoverReceive:
    """Rollout / eval in the same dual-HSR handover XML as ``collect_handover_demos.py``."""

    def setup_env(self, render_mode="human"):
        # RolloutPhase：达到 pick_eval_max_policy_action_steps（与 action 图横轴 policy 步一致）
        # 后调用 env.compute_pick_eval_outcome：baton 最低世界 z ≥ 0.05 m（离地约 5 cm），不比基线。
        # B 夹紧后 A 松手再后退：handover_rollout_coordinated_release（默认真）。
        self.env = gym.make(
            "robo_manip_baselines/MujocoDualHsrHandoverReceiveEnv-v0",
            render_mode=render_mode,
            robot_a_xy_reset_perturb_half_extent_m=0.0,
            robot_b_xy_reset_perturb_half_extent_m=0.0,
            pick_eval_max_policy_action_steps=70,
            pick_eval_baton_length_m=0.14,
            pick_eval_success_bottom_clearance_ratio=0.5,
            handover_pick_eval_min_baton_lowest_z_m=0.05,
        )

    def get_pre_motion_phases(self):
        return []
