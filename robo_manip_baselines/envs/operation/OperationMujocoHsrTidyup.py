import gymnasium as gym


class OperationMujocoHsrTidyup:
    def setup_env(self, render_mode="human"):
        # Rollout / Teleop：5cm×5cm XY 扰动；pick 在 action 图横轴 policy step==70 时结算（见 RolloutPhase）。
        self.env = gym.make(
            "robo_manip_baselines/MujocoHsrTidyupEnv-v0",
            render_mode=render_mode,
            bottle_xy_reset_perturb_half_extent_m=0.025,
            pick_eval_max_policy_action_steps=70,
            pick_eval_baton_length_m=0.14,
            pick_eval_success_bottom_clearance_ratio=0.5,
        )

    def get_pre_motion_phases(self):
        return []
