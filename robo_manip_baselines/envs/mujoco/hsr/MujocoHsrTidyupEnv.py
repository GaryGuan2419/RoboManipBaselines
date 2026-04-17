from os import path

import mujoco
import numpy as np

from .MujocoHsrEnvBase import MujocoHsrEnvBase


class MujocoHsrTidyupEnv(MujocoHsrEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoHsrEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/hsr/env_hsr_tidyup.xml",
            ),
            # arm_lift 抬高：初始末端更高，便于演示/采样「下探→夹紧→抬起」
            np.array(
                [
                    -0.1760,
                    0.0000,
                    0.0000,
                    0.2600,
                    -2.4844,
                    0.0039,
                    1.0132,
                    0.0021,
                    0.8000,
                ]
            ),
            **kwargs,
        )

        self.bottle_names = []
        for name in ["bottle1", "bottle2"]:
            try:
                self.model.body(name)
                self.bottle_names.append(name)
            except KeyError:
                pass

        if "bottle1" in self.bottle_names:
            self.original_bottle1_pos = self.model.body("bottle1").pos.copy()
        if "bottle2" in self.bottle_names:
            self.original_bottle2_pos = self.model.body("bottle2").pos.copy()

        self.bottle_pos_offsets = np.zeros((1, 3))  # [m] No offset

        self.target_task = None  # One of [None, "either", "both"]

    def _get_reward(self):
        reward = 0.0
        container_half_extents = np.array([0.1, 0.15, 0.08])  # [m]

        try:
            bottle1_pos = self.data.body("bottle1").xpos.copy()
            container1_pos = self.data.body("container1").xpos.copy()
            if np.all(np.abs(bottle1_pos - container1_pos) <= container_half_extents):
                reward += 0.5
        except KeyError:
            pass

        try:
            bottle2_pos = self.data.body("bottle2").xpos.copy()
            container2_pos = self.data.body("container2").xpos.copy()
            if np.all(np.abs(bottle2_pos - container2_pos) <= container_half_extents):
                reward += 0.5
        except KeyError:
            pass

        if self.target_task == "either":
            reward = np.min([2.0 * reward, 1.0])

        return reward

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.bottle_pos_offsets)

        delta_pos = self.bottle_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            delta_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )

        for bottle_name in self.bottle_names:
            bottle_joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{bottle_name}_freejoint"
            )
            bottle_qpos_addr = self.model.jnt_qposadr[bottle_joint_id]
            if bottle_name == "bottle1":
                original_bottle_pos = self.original_bottle1_pos
            else:  # if bottle_name == "bottle2"
                original_bottle_pos = self.original_bottle2_pos
            self.init_qpos[bottle_qpos_addr : bottle_qpos_addr + 3] = (
                original_bottle_pos + delta_pos
            )

        return world_idx
