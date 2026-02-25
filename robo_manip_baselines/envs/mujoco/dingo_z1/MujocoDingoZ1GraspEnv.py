from os import path

import mujoco
import numpy as np

from .MujocoDingoZ1EnvBase import MujocoDingoZ1EnvBase


class MujocoDingoZ1GraspEnv(MujocoDingoZ1EnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoDingoZ1EnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/dingo_z1/env_dingo_z1_grasp.xml",
            ),
            # Initial qpos: [x, y, z, qw, qx, qy, qz, left_wheel, right_wheel, j1-6, gripper]
            # Base at origin, wheels at 0, arm in home pose, gripper open
            np.array([0.0, 0.0, 0.015, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.785, -0.261, -0.523, 0.0, 0.0, 0.0]),
            **kwargs,
        )

        # Store original object position for randomization
        self.original_object_pos = self.model.body("object").pos.copy()
        
        # Define position offsets for world randomization
        self.object_pos_offsets = np.array(
            [
                [0.0, -0.1, 0.0],
                [0.0, -0.05, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.1, 0.0],
                [0.05, -0.05, 0.0],
                [0.05, 0.0, 0.0],
                [0.05, 0.05, 0.0],
            ]
        )  # [m]

    def _get_reward(self):
        """
        Reward function for grasping task.
        - 0.5 points if object is grasped (close to gripper and gripper is closed)
        - 0.5 points if object is at target location
        """
        object_pos = self.data.body("object").xpos.copy()
        target_pos = self.data.body("target").xpos.copy()
        gripper_pos = self.data.body("link06").xpos.copy()  # End-effector position
        
        # Get gripper state (joint position)
        gripper_joint_pos = self.data.joint("jointGripper").qpos[0]
        
        reward = 0.0
        
        # Check if object is grasped (close to gripper and gripper is closed)
        gripper_to_object_dist = np.linalg.norm(object_pos - gripper_pos)
        gripper_closed = gripper_joint_pos < -0.5  # Gripper range: -1.51844 to 0
        
        if gripper_to_object_dist < 0.08 and gripper_closed:
            reward += 0.5
        
        # Check if object is at target location
        object_to_target_dist = np.linalg.norm(object_pos - target_pos)
        if object_to_target_dist < 0.1:  # Within 10cm of target
            reward += 0.5
        
        return reward

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """
        Modify simulation world by randomizing object position.
        """
        if world_idx is None:
            world_idx = cumulative_idx % len(self.object_pos_offsets)

        delta_pos = self.object_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            delta_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )

        # Update object position in init_qpos
        object_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint"
        )
        object_qpos_addr = self.model.jnt_qposadr[object_joint_id]
        self.init_qpos[object_qpos_addr : object_qpos_addr + 3] = (
            self.original_object_pos + delta_pos
        )

        return world_idx
