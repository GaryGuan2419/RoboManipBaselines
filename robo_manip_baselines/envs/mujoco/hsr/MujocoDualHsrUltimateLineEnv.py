"""Dual HSR on one world-x line: baton x=-2, handover x=0, goal x=+2 (`env_dual_hsr_ultimate_line.xml`).

Robot A starts at (-2.8, 0): same 0.8 m x-offset from baton as the original demo (-1 vs -0.2).
Robot B starts at (2.5, 0, pi), east of the goal, facing -X.
"""

import numpy as np

from robo_manip_baselines.envs.mujoco.hsr.MujocoDualHsrDemoEnv import MujocoDualHsrDemoEnv


class MujocoDualHsrUltimateLineEnv(MujocoDualHsrDemoEnv):
    default_camera_config = {
        "azimuth": -120.0,
        "elevation": -25.0,
        "distance": 4.5,
        "lookat": [0.0, 0.0, 0.15],
    }

    def __init__(self, **kwargs):
        # Robot A arm_lift: pre-pick vertical slide (m); tuned vs legacy 0.26 m.
        init_qpos = np.array(
            [
                -2.8,
                0.0,
                0.0,
                0.2300,
                -2.4844,
                0.0039,
                0.97,
                0.0021,
                0.8,
                0.8,
                0.8,
                2.5,
                0.0,
                3.1416,
                0.2600,
                -2.4844,
                0.0039,
                1.0132,
                0.0021,
                0.8,
                0.8,
                0.8,
                -2.0,
                0.0,
                0.01,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        super().__init__(
            xml_filename="env_dual_hsr_ultimate_line.xml",
            init_qpos=init_qpos,
            **kwargs,
        )
