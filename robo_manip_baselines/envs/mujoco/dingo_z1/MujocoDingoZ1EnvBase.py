from os import path

import mujoco
import numpy as np
from gymnasium.spaces import Box, Dict
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer

from robo_manip_baselines.common import (
    ArmConfig,
    DataKey,
    MobileOmniConfig,
    get_se3_from_pose,
)
from robo_manip_baselines.teleop import (
    KeyboardInputDevice,
    SpacemouseInputDevice,
    SpacemouseMobileInputDevice,
)

from ..MujocoEnvBase import MujocoEnvBase


class MujocoDingoZ1EnvBase(MujocoEnvBase):
    default_camera_config = {
        "azimuth": -135.0,
        "elevation": -45.0,
        "distance": 2.5,
        "lookat": [0.5, 0.0, 0.3],
    }
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            "mobile_vel": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        }
    )


    def setup_robot(self, init_qpos):
        self.init_qpos[: len(init_qpos)] = init_qpos
        self.init_qvel[:] = 0.0

        mujoco.mj_kinematics(self.model, self.data)

        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../../assets/common/robots/z1/z1.urdf"
                ),
                arm_root_pose=self.get_body_pose("base_link"),
                exclude_joint_names=["jointGripper"],
                ik_eef_joint_id=6,
                arm_joint_idxes=np.arange(6),
                gripper_joint_idxes=np.array([6]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[9:15],
                init_gripper_joint_pos=self.init_qpos[[15]],
                get_root_pose_func=lambda env: get_se3_from_pose(
                    env.get_body_pose("base_link")
                ),
            ),
            MobileOmniConfig(),
        ]

        # Mobile base joint names (from freejoint qpos: x, y, z, qw, qx, qy, qz)
        # We'll extract x, y, and theta from the freejoint
        self.base_freejoint_name = "base_freejoint"

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        default_kwargs = self.get_input_device_kwargs(input_device_name)

        if input_device_name == "spacemouse":
            return [
                SpacemouseInputDevice(
                    motion_manager.body_manager_list[0],
                    **{**default_kwargs.get(0, {}), **overwrite_kwargs.get(0, {})},
                ),
                SpacemouseMobileInputDevice(
                    motion_manager.body_manager_list[1],
                    **{**default_kwargs.get(1, {}), **overwrite_kwargs.get(1, {})},
                ),
            ]
        elif input_device_name == "keyboard":
            return [
                KeyboardInputDevice(
                    motion_manager.body_manager_list[0],
                    **{**default_kwargs.get(0, {}), **overwrite_kwargs.get(0, {})},
                ),
            ]
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid input device key: {input_device_name}"
            )

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return {0: {"gripper_scale": 0.05}, 1: {}}
        elif input_device_name == "keyboard":
            # Slower movement for precise grasping, gentle gripper
            return {0: {"pos_scale": 5e-3, "gripper_scale": 0.05}}
        else:
            return {}

    @property
    def command_keys_for_step(self):
        return [DataKey.COMMAND_MOBILE_OMNI_VEL, DataKey.COMMAND_JOINT_POS]

    @property
    def measured_keys_to_save(self):
        return [
            DataKey.MEASURED_JOINT_POS,
            DataKey.MEASURED_JOINT_VEL,
            DataKey.MEASURED_GRIPPER_JOINT_POS,
            DataKey.MEASURED_EEF_POSE,
            DataKey.MEASURED_EEF_WRENCH,
            DataKey.MEASURED_MOBILE_OMNI_VEL,
        ]

    @property
    def command_keys_to_save(self):
        return [
            DataKey.COMMAND_JOINT_POS,
            DataKey.COMMAND_GRIPPER_JOINT_POS,
            DataKey.COMMAND_EEF_POSE,
            DataKey.COMMAND_MOBILE_OMNI_VEL,
        ]

    def step(self, action):
        # action from Teleop: [vx, vy, vθ, joint1, joint2, joint3, joint4, joint5, joint6, gripper]
        mobile_vel = action[0:3].copy()
        joint_pos = action[3:10]
        
        mobile_vel_world = self.convert_mobile_vel_frame(mobile_vel, world_to_local=False)
        wheel_separation = 0.236
        wheel_radius = 0.049
        
        vx = mobile_vel_world[0]
        vtheta = mobile_vel_world[2]
        
        left_wheel_vel = (vx - vtheta * wheel_separation / 2) / wheel_radius
        right_wheel_vel = (vx + vtheta * wheel_separation / 2) / wheel_radius
        
        mujoco_action = np.concatenate([
            [left_wheel_vel, right_wheel_vel],
            joint_pos,
        ])
        
        self.do_simulation(mujoco_action, self.frame_skip)
        
        obs = self._get_obs()
        reward = self._get_reward()
        # Fast path: skip offscreen camera rendering to keep big window responsive
        info = self._get_info_fast()
            
        if self.render_mode == "human":
            self.render()

        return obs, reward, False, False, info

    def _get_info_fast(self):
        """Return info without rendering offscreen cameras. Keeps big window smooth."""
        info = {}
        if len(self.camera_names) == 0:
            return info
        info["rgb_images"] = {}
        info["depth_images"] = {}
        return info

    def get_images(self):
        """Render all offscreen cameras on demand. Called by Teleop when recording/drawing."""
        return self._get_info()

    def _get_obs(self):
        # Z1 arm joint names
        arm_joint_name_list = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        gripper_joint_name = "jointGripper"

        # Get arm joint positions and velocities
        arm_joint_pos = np.array(
            [self.data.joint(joint_name).qpos[0] for joint_name in arm_joint_name_list]
        )
        arm_joint_vel = np.array(
            [self.data.joint(joint_name).qvel[0] for joint_name in arm_joint_name_list]
        )
        gripper_joint_pos = np.array([self.data.joint(gripper_joint_name).qpos[0]])
        gripper_joint_vel = np.zeros(1)

        # TODO: Add force/torque sensor when available
        # For now, return zeros
        force = np.zeros(3)
        torque = np.zeros(3)

        # Get mobile base velocity from freejoint
        # Freejoint has 6 velocity components: [vx, vy, vz, wx, wy, wz]
        base_freejoint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self.base_freejoint_name
        )
        base_qvel_addr = self.model.jnt_dofadr[base_freejoint_id]
        
        # Extract planar velocity: vx, vy, wz (rotation around z-axis)
        mobile_vel = np.array(
            [
                self.data.qvel[base_qvel_addr],      # vx
                self.data.qvel[base_qvel_addr + 1],  # vy
                self.data.qvel[base_qvel_addr + 5],  # wz (theta_dot)
            ]
        )
        # Convert to local frame
        mobile_vel = self.convert_mobile_vel_frame(mobile_vel, world_to_local=True)

        return {
            "joint_pos": np.concatenate(
                (arm_joint_pos, gripper_joint_pos), dtype=np.float64
            ),
            "joint_vel": np.concatenate(
                (arm_joint_vel, gripper_joint_vel), dtype=np.float64
            ),
            "wrench": np.concatenate((force, torque), dtype=np.float64),
            "mobile_vel": mobile_vel.astype(np.float64),
        }

    def get_mobile_vel_from_obs(self, obs):
        """Get velocity of omni-directional mobile base from observation."""
        return obs["mobile_vel"]

    def convert_mobile_vel_frame(self, vel_in, world_to_local):
        """Convert velocity between world frame and robot local frame."""
        # Get base orientation from freejoint qpos
        base_freejoint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self.base_freejoint_name
        )
        base_qpos_addr = self.model.jnt_qposadr[base_freejoint_id]
        
        # Freejoint qpos: [x, y, z, qw, qx, qy, qz]
        # Extract quaternion and convert to yaw angle
        quat = self.data.qpos[base_qpos_addr + 3 : base_qpos_addr + 7]
        
        # Convert quaternion to rotation matrix, then extract yaw (theta)
        # For a ground robot, we only care about rotation around z-axis
        # qw, qx, qy, qz -> theta = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
        qw, qx, qy, qz = quat
        theta = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        
        if not world_to_local:
            theta *= -1

        rot_mat = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )

        vel_in_xy = vel_in[0:2]
        vel_out_xy = rot_mat @ vel_in_xy

        return np.concatenate([vel_out_xy, vel_in[[2]]])
