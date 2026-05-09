import gymnasium as gym


class OperationMujocoHsrGroundBatonGrasp:
    def setup_env(self, render_mode="human"):
        # 与 side_pick 对称：瓶位 XY 扰动；第 70 policy step 结算，离地 ≥3 cm 判成功。
        self.env = gym.make(
            "robo_manip_baselines/MujocoHsrGroundBatonGraspEnv-v0",
            render_mode=render_mode,
            bottle_xy_reset_perturb_half_extent_m=0.025,
            pick_eval_max_policy_action_steps=70,
            pick_eval_min_lowest_z_above_floor_m=0.03,
        )

    def get_pre_motion_phases(self):
        return []
