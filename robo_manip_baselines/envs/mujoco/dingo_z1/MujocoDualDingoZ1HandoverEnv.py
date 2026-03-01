"""
MujocoDualDingoZ1HandoverEnv: Dual-robot handover task environment.

Task: Robot A picks an object from Table A, both robots move to the handover
zone, Robot A passes the object to Robot B, and Robot B places it on Table B.

Reward function (progressive):
    0.0  — Nothing useful
    0.2  — Robot A gripper is near the object
    0.5  — Object is lifted off Table A (grasped)
    0.7  — Object is in the handover zone with both grippers nearby
    1.0  — Object is placed on Table B (at target location)
"""

from os import path

import mujoco
import numpy as np

from .MujocoDualDingoZ1EnvBase import MujocoDualDingoZ1EnvBase


class MujocoDualDingoZ1HandoverEnv(MujocoDualDingoZ1EnvBase):
    def __init__(self, **kwargs):
        # Initial qpos for both robots:
        #   Each robot: [x, y, z, qw, qx, qy, qz, left_wheel, right_wheel, j1-j6, gripper]
        #   Robot A: at (-0.55, 0, 0.015), facing +Y (quat: 0.7071, 0, 0, 0.7071)
        #   Robot B: at (0.55, 0, 0.015), facing +Y (quat: 0.7071, 0, 0, 0.7071)
        robot_a_qpos = np.array([
            -0.55, 0.0, 0.015, 0.7071, 0.0, 0.0, 0.7071,  # freejoint pos+quat
            0.0, 0.0,                                  # wheel joints
            0.0, 0.785, -0.261, -0.523, 0.0, 0.0,     # arm joints (home pose)
            0.0,                                        # gripper (open)
        ])
        robot_b_qpos = np.array([
            0.55, 0.0, 0.015, 0.7071, 0.0, 0.0, 0.7071,   # freejoint pos+quat
            0.0, 0.0,                                  # wheel joints
            0.0, 0.785, -0.261, -0.523, 0.0, 0.0,     # arm joints (home pose)
            0.0,                                        # gripper (open)
        ])

        MujocoDualDingoZ1EnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/dingo_z1/env_dingo_z1_handover.xml",
            ),
            np.concatenate([robot_a_qpos, robot_b_qpos]),
            **kwargs,
        )

        # Store original positions for randomization
        self.original_object_pos = self.model.body("object").pos.copy()

        # Define position offsets for world randomization (on Table A)
        self.object_pos_offsets = np.array([
            [-0.05, -0.05, 0.0],
            [-0.05, 0.0, 0.0],
            [-0.05, 0.05, 0.0],
            [0.0, -0.05, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.05, -0.05, 0.0],
            [0.05, 0.0, 0.0],
            [0.05, 0.05, 0.0],
        ])  # [m]

    def _get_reward(self):
        """Progressive reward for the handover task.

        Stages (following MujocoAlohaHandoverEnv pattern):
            0.2  Robot A gripper near object
            0.5  Object grasped (lifted off table)
            0.7  Object in handover zone with both grippers nearby
            1.0  Object placed on target (Table B)
        """
        # Key positions
        object_pos = self.data.body("object").xpos.copy()
        target_pos = self.data.body("target").xpos.copy()
        handover_pos = self.data.body("handover_zone").xpos.copy()

        # End-effector positions (link06 is the wrist body)
        gripper_a_pos = self.data.body("robot_a/link06").xpos.copy()
        gripper_b_pos = self.data.body("robot_b/link06").xpos.copy()

        # Gripper states
        gripper_a_qpos = self.data.joint("robot_a/jointGripper").qpos[0]
        gripper_b_qpos = self.data.joint("robot_b/jointGripper").qpos[0]

        # Distances
        dist_a_to_obj = np.linalg.norm(object_pos - gripper_a_pos)
        dist_b_to_obj = np.linalg.norm(object_pos - gripper_b_pos)
        dist_obj_to_target = np.linalg.norm(object_pos - target_pos)
        dist_obj_to_handover = np.linalg.norm(object_pos[:2] - handover_pos[:2])  # XY only

        # Thresholds
        grasp_threshold = 0.1       # 10cm — gripper near object
        handover_threshold = 0.15   # 15cm — object near handover zone (XY)
        place_threshold = 0.1       # 10cm — object at target
        gripper_closed = -0.5       # Gripper joint value when closed enough

        reward = 0.0

        # Stage 4: Object placed at target on Table B → full success
        if dist_obj_to_target < place_threshold:
            reward = 1.0
        # Stage 3: Object in handover zone (Table C) and Robot B is approaching it
        elif (dist_obj_to_handover < handover_threshold
              and dist_b_to_obj < grasp_threshold):
            reward = 0.7
        # Stage 2: Object lifted from Table A and gripper A closed
        elif (object_pos[2] > 0.38  # Table surface is at ~0.35
              and gripper_a_qpos < gripper_closed):
            reward = 0.5
        # Stage 1: Gripper A is approaching the object
        elif dist_a_to_obj < grasp_threshold:
            reward = 0.2

        return reward

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world by randomizing object position on Table A."""
        if world_idx is None:
            world_idx = cumulative_idx % len(self.object_pos_offsets)

        delta_pos = self.object_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            delta_pos = delta_pos + np.random.uniform(
                low=-1.0 * self.world_random_scale,
                high=self.world_random_scale,
                size=3,
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
