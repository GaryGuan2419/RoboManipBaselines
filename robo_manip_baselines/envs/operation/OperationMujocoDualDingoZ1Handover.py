import gymnasium as gym


class OperationMujocoDualDingoZ1Handover:
    """Operation class for the dual DingoZ1 handover task.

    Follows the same pattern as OperationMujocoDingoZ1Grasp
    and OperationMujocoAlohaHandover.
    """

    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoDualDingoZ1HandoverEnv-v0",
            render_mode=render_mode,
        )

    def get_pre_motion_phases(self):
        return []
