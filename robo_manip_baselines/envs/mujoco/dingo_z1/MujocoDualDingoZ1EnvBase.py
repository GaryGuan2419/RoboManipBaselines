"""
MujocoDualDingoZ1EnvBase: Base environment for dual Dingo+Z1 robot tasks.

Two Dingo+Z1 mobile manipulators in a shared scene.
Action space: 20-dim (10 per robot: [vx, vy, vθ, j1, j2, j3, j4, j5, j6, gripper])
Observation space: joint_pos (14), joint_vel (14), wrench (12), mobile_vel (6)

[LOCK_BASE] Set LOCK_BASE = True to zero out mobile base velocities,
            reducing effective action space to 14-dim.
"""

from os import path

import mujoco
import numpy as np
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import (
    ArmConfig,
    DataKey,
    MobileOmniConfig,
    get_se3_from_pose,
)
from robo_manip_baselines.teleop import (
    SpacemouseInputDevice,
    SpacemouseMobileInputDevice,
)
from robo_manip_baselines.teleop.GlfwKeyboardInputDevice import GlfwKeyboardInputDevice

from ..MujocoEnvBase import MujocoEnvBase


class MujocoDualDingoZ1EnvBase(MujocoEnvBase):
    """Base environment class for dual Dingo+Z1 mobile manipulators.

    Action space: 20-dim (10 per robot)
    [0:3]   Robot A mobile velocity (vx, vy, vθ)
    [3:10]  Robot A arm joints + gripper (7)
    [10:13] Robot B mobile velocity (vx, vy, vθ)
    [13:20] Robot B arm joints + gripper (7)
    """

    # ------------------------------------------------------------------ #
    #  [LOCK_BASE] Set to True to disable mobile base movement.          #
    #  When True, the 3-dim base velocity in each robot's action is      #
    #  forced to zero, effectively reducing action space from 20 to 14.  #
    # ------------------------------------------------------------------ #
    LOCK_BASE = True

    default_camera_config = {
        "azimuth": 90.0,
        "elevation": -35.0,
        "distance": 3.5,
        "lookat": [0.0, 0.0, 0.3],
    }

    observation_space = Dict(
        {
            # 7 joints per robot × 2 = 14
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64),
            # 6 wrench per robot × 2 = 12
            "wrench": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64),
            # 3 mobile vel per robot × 2 = 6
            "mobile_vel": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        }
    )

    action_space = Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float64)

    # Robot name prefixes (must match XML body/joint naming)
    ROBOT_PREFIXES = ["robot_a", "robot_b"]

    def __init__(self, xml_file, init_qpos, **kwargs):
        super().__init__(xml_file, init_qpos, **kwargs)
        # Override action_space set by MujocoEnv to match our 20-dim logic
        self.action_space = Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float64)

    # Arm joint names within each robot (unprefixed)
    ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    GRIPPER_JOINT_NAME = "jointGripper"

    def setup_robot(self, init_qpos):
        """Set up dual robot configurations.

        init_qpos layout for EACH robot (16 values):
            [x, y, z, qw, qx, qy, qz,     # freejoint (7)
             left_wheel, right_wheel,        # wheel joints (2)
             j1, j2, j3, j4, j5, j6,       # arm joints (6)
             gripper]                        # gripper (1)
        Total: 16 × 2 = 32 values (+ object freejoint 7 = 39 total qpos)
        """
        self.init_qpos[: len(init_qpos)] = init_qpos
        self.init_qvel[:] = 0.0

        mujoco.mj_kinematics(self.model, self.data)

        # Build config list: [arm_a, mobile_a, arm_b, mobile_b]
        # arm_joint_idxes must be RELATIVE to the observation space (14-dim),
        # NOT absolute MuJoCo qpos addresses.
        # Observation layout: [robot_a joints(7) | robot_b joints(7)]
        #   robot_a: arm=[0..5], gripper=[6]
        #   robot_b: arm=[7..12], gripper=[13]
        self.body_config_list = []
        for i, prefix in enumerate(self.ROBOT_PREFIXES):
            robot_qpos_offset = i * 16
            obs_offset = i * 7  # Each robot occupies 7 slots in obs

            self.body_config_list.append(
                ArmConfig(
                    arm_urdf_path=path.join(
                        path.dirname(__file__),
                        "../../assets/common/robots/z1/z1.urdf",
                    ),
                    arm_root_pose=self.get_body_pose(f"{prefix}/base_link"),
                    exclude_joint_names=["jointGripper"],
                    ik_eef_joint_id=6,
                    # Relative indices within the 14-dim observation/command space
                    arm_joint_idxes=np.arange(6) + obs_offset,
                    gripper_joint_idxes=np.array([6 + obs_offset]),
                    gripper_joint_idxes_in_gripper_joint_pos=np.array([i]),
                    eef_idx=i,
                    init_arm_joint_pos=init_qpos[robot_qpos_offset + 9 : robot_qpos_offset + 15],
                    init_gripper_joint_pos=init_qpos[[robot_qpos_offset + 15]],
                    get_root_pose_func=lambda env, p=prefix: get_se3_from_pose(
                        env.get_body_pose(f"{p}/base_link")
                    ),
                )
            )
            self.body_config_list.append(MobileOmniConfig())

        # Store freejoint names for mobile base control
        self.base_freejoint_names = [
            f"{prefix}/base_freejoint" for prefix in self.ROBOT_PREFIXES
        ]

    def _get_joint_qpos_idx(self, joint_name):
        """Get the qpos index for a named joint in the MuJoCo model."""
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        return self.model.jnt_qposadr[joint_id]

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        """Set up input devices for both robots.

        Returns a list of input devices: [arm_a, mobile_a, arm_b, mobile_b].
        """
        default_kwargs = self.get_input_device_kwargs(input_device_name)

        if input_device_name == "spacemouse":
            devices = []
            for robot_idx in range(2):
                arm_idx = robot_idx * 2
                mobile_idx = robot_idx * 2 + 1
                devices.append(
                    SpacemouseInputDevice(
                        motion_manager.body_manager_list[arm_idx],
                        **{
                            **default_kwargs.get(arm_idx, {}),
                            **overwrite_kwargs.get(arm_idx, {}),
                        },
                    )
                )
                devices.append(
                    SpacemouseMobileInputDevice(
                        motion_manager.body_manager_list[mobile_idx],
                        **{
                            **default_kwargs.get(mobile_idx, {}),
                            **overwrite_kwargs.get(mobile_idx, {}),
                        },
                    )
                )
            return devices
        elif input_device_name == "keyboard":
            # Use GLFW-based device (pynput doesn't work on WSL2)
            # Pass BOTH arm managers so user can toggle between robots with keys 1/2
            arm_managers = [
                motion_manager.body_manager_list[0],   # robot_a arm
                motion_manager.body_manager_list[2],   # robot_b arm
            ]
            device = GlfwKeyboardInputDevice(
                arm_managers,
                **{
                    **default_kwargs.get(0, {}),
                    **overwrite_kwargs.get(0, {}),
                },
            )
            # Store reference so we can attach to viewer later
            self._glfw_keyboard_device = device
            self._glfw_keyboard_attached = False
            return [device]
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid input device key: {input_device_name}"
            )

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return {
                0: {"gripper_scale": 0.05},
                1: {},
                2: {"gripper_scale": 0.05},
                3: {},
            }
        elif input_device_name == "keyboard":
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
        # Capture initial base positions on first call (for LOCK_BASE reset)
        if not hasattr(self, '_init_base_qpos'):
            self._init_base_qpos = {}
            for prefix in self.ROBOT_PREFIXES:
                fj_name = f"{prefix}/base_freejoint"
                fj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, fj_name)
                addr = self.model.jnt_qposadr[fj_id]
                self._init_base_qpos[prefix] = self.data.qpos[addr:addr+7].copy()
        """Step both robots simultaneously.

        Accepts two formats:
        1. Teleop format (17-dim): [mobile_vel(3), joint_pos_a(7), joint_pos_b(7)]
           - mobile_vel only applies to robot_a (keyboard controls robot_a)
           - robot_b stays still during teleop
        2. Policy format (20-dim): [mobile_a(3), arm_a(7), mobile_b(3), arm_b(7)]
        """
        action = np.array(action, dtype=np.float64)

        if len(action) <= 17:
            # Teleop format: [mobile_vel(3), joint_pos(14)]
            mobile_vel_a = action[0:3].copy()
            joint_pos_a = action[3:10]
            joint_pos_b = action[10:17] if len(action) >= 17 else np.zeros(7)

            # [LOCK_BASE]
            if self.LOCK_BASE:
                mobile_vel_a[:] = 0.0

            mujoco_ctrl_a = self._robot_action_to_ctrl(
                np.concatenate([mobile_vel_a, joint_pos_a]), "robot_a"
            )
            mujoco_ctrl_b = self._robot_action_to_ctrl(
                np.concatenate([np.zeros(3), joint_pos_b]), "robot_b"
            )
        else:
            # Policy format: [mobile_a(3), arm_a(7), mobile_b(3), arm_b(7)]
            action_a = action[0:10].copy()
            action_b = action[10:20].copy()

            if self.LOCK_BASE:
                action_a[0:3] = 0.0
                action_b[0:3] = 0.0

            mujoco_ctrl_a = self._robot_action_to_ctrl(action_a, "robot_a")
            mujoco_ctrl_b = self._robot_action_to_ctrl(action_b, "robot_b")

        # Combine: actuator order in XML is robot_a(9) then robot_b(9) = 18
        mujoco_action = np.concatenate([mujoco_ctrl_a, mujoco_ctrl_b])

        # [MOCAP GRASP] Dynamically attach object to gripper using kinematic override
        self._update_mocap_grasp("robot_a")
        self._update_mocap_grasp("robot_b")

        self.do_simulation(mujoco_action, self.frame_skip)

        # [LOCK_BASE] Physically reset base positions to prevent drift from arm reaction forces
        if self.LOCK_BASE and hasattr(self, '_init_base_qpos'):
            for prefix, init_qpos in self._init_base_qpos.items():
                fj_name = f"{prefix}/base_freejoint"
                fj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, fj_name)
                addr = self.model.jnt_qposadr[fj_id]
                self.data.qpos[addr:addr+7] = init_qpos
                # Also zero base velocities
                dof_addr = self.model.jnt_dofadr[fj_id]
                self.data.qvel[dof_addr:dof_addr+6] = 0.0

        obs = self._get_obs()
        reward = self._get_reward()
        info = self._get_info_fast()

        if self.render_mode == "human":
            self.render()
            # Auto-attach GLFW keyboard device to viewer on first render
            if (hasattr(self, '_glfw_keyboard_device') 
                    and not getattr(self, '_glfw_keyboard_attached', True)):
                viewer = getattr(self.mujoco_renderer, 'viewer', None)
                if viewer is not None:
                    self._glfw_keyboard_device.attach_to_viewer(viewer, env=self)
                    self._glfw_keyboard_attached = True

        return obs, reward, False, False, info

    def _robot_action_to_ctrl(self, robot_action, prefix):
        """Convert a single robot's 10-dim action to MuJoCo actuator commands.

        Args:
            robot_action: [vx, vy, vθ, j1, j2, j3, j4, j5, j6, gripper]
            prefix: 'robot_a' or 'robot_b'

        Returns:
            9-dim array: [left_wheel_vel, right_wheel_vel, j1-j6, gripper]
        """
        mobile_vel = robot_action[0:3].copy()
        joint_pos = robot_action[3:10]

        # Convert local mobile velocity to world frame
        mobile_vel_world = self._convert_mobile_vel_frame(
            mobile_vel, prefix, world_to_local=False
        )

        # Differential drive kinematics (same as single robot)
        wheel_separation = 0.236
        wheel_radius = 0.049

        vx = mobile_vel_world[0]
        vtheta = mobile_vel_world[2]

        left_wheel_vel = (vx - vtheta * wheel_separation / 2) / wheel_radius
        right_wheel_vel = (vx + vtheta * wheel_separation / 2) / wheel_radius

        return np.concatenate([
            [left_wheel_vel, right_wheel_vel],
            joint_pos,
        ])

    def _get_info_fast(self):
        """Return info without camera rendering (fast path for every step)."""
        return {"rgb_images": {}, "depth_images": {}}

    @property
    def camera_names(self):
        """Only expose overhead camera for the small window (low perf cost)."""
        return ["overhead"]

    def get_images(self):
        """Render only the overhead camera at low resolution for the small window."""
        info = {"rgb_images": {}, "depth_images": {}}
        if "overhead" in self.cameras:
            cam = self.cameras["overhead"]
            cam["viewer"].make_context_current()
            info["rgb_images"]["overhead"] = cam["viewer"].render(
                render_mode="rgb_array", camera_id=cam["id"]
            )
            info["depth_images"]["overhead"] = cam["viewer"].render(
                render_mode="depth_array", camera_id=cam["id"]
            )
        return info

    def reset_object(self):
        """Reset only the object (cylinder) to its initial position, keeping robot states."""
        import mujoco as mj
        obj_joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "object_freejoint")
        addr = self.model.jnt_qposadr[obj_joint_id]
        dof_addr = self.model.jnt_dofadr[obj_joint_id]
        # Reset position + quaternion (7 values)
        self.data.qpos[addr:addr+7] = self.init_qpos[addr:addr+7]
        # Reset velocity (6 values)
        self.data.qvel[dof_addr:dof_addr+6] = 0.0
        mj.mj_forward(self.model, self.data)

    def print_joint_state(self):
        """Print current joint angles for both robots (copy-paste ready)."""
        print(f"\n{'='*60}")
        for prefix in self.ROBOT_PREFIXES:
            joints = [self.data.joint(f"{prefix}/{jn}").qpos[0] for jn in self.ARM_JOINT_NAMES]
            gripper = self.data.joint(f"{prefix}/{self.GRIPPER_JOINT_NAME}").qpos[0]
            eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}/link06")
            gm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}/gripperMover")
            eef_pos = self.data.xpos[eef_id]
            gm_pos = self.data.xpos[gm_id]
            print(f"  {prefix}:")
            print(f"    joints:  [{', '.join(f'{v:.4f}' for v in joints)}]")
            print(f"    gripper: {gripper:.4f}")
            print(f"    link06:  [{eef_pos[0]:.4f}, {eef_pos[1]:.4f}, {eef_pos[2]:.4f}]")
            print(f"    gripper: [{gm_pos[0]:.4f}, {gm_pos[1]:.4f}, {gm_pos[2]:.4f}]")
        obj_pos = self.data.body("object").xpos
        print(f"  object:  [{obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f}]")
        print(f"{'='*60}")

    def _get_obs(self):
        """Get concatenated observations from both robots."""
        obs_a = self._get_obs_single_robot("robot_a")
        obs_b = self._get_obs_single_robot("robot_b")
        return {
            key: np.concatenate([obs_a[key], obs_b[key]])
            for key in obs_a.keys()
        }

    def _get_obs_single_robot(self, prefix):
        """Get observations for a single robot.

        Follows the same logic as MujocoDingoZ1EnvBase._get_obs().
        """
        # Arm joint positions and velocities
        arm_joint_pos = np.array([
            self.data.joint(f"{prefix}/{jn}").qpos[0]
            for jn in self.ARM_JOINT_NAMES
        ])
        arm_joint_vel = np.array([
            self.data.joint(f"{prefix}/{jn}").qvel[0]
            for jn in self.ARM_JOINT_NAMES
        ])

        # Gripper
        gripper_joint_pos = np.array([
            self.data.joint(f"{prefix}/{self.GRIPPER_JOINT_NAME}").qpos[0]
        ])
        gripper_joint_vel = np.zeros(1)

        # Wrench (placeholder zeros for now, same as single robot)
        force = np.zeros(3)
        torque = np.zeros(3)

        # Mobile base velocity from freejoint
        freejoint_name = f"{prefix}/base_freejoint"
        base_freejoint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, freejoint_name
        )
        base_qvel_addr = self.model.jnt_dofadr[base_freejoint_id]

        mobile_vel = np.array([
            self.data.qvel[base_qvel_addr],      # vx
            self.data.qvel[base_qvel_addr + 1],  # vy
            self.data.qvel[base_qvel_addr + 5],  # wz (theta_dot)
        ])
        # Convert to local frame
        mobile_vel = self._convert_mobile_vel_frame(
            mobile_vel, prefix, world_to_local=True
        )

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
        """Get velocity of omni-directional mobile bases from observation."""
        return obs["mobile_vel"]

    def _convert_mobile_vel_frame(self, vel_in, prefix, world_to_local):
        """Convert velocity between world frame and robot local frame.

        Same logic as MujocoDingoZ1EnvBase.convert_mobile_vel_frame(),
        but parameterized by robot prefix.
        """
        freejoint_name = f"{prefix}/base_freejoint"
        base_freejoint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, freejoint_name
        )
        base_qpos_addr = self.model.jnt_qposadr[base_freejoint_id]

        # Extract quaternion → yaw
        quat = self.data.qpos[base_qpos_addr + 3 : base_qpos_addr + 7]
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

    # Alias for compatibility with single-robot code paths
    def convert_mobile_vel_frame(self, vel_in, world_to_local):
        """Compatibility shim: converts for robot_a by default."""
        return self._convert_mobile_vel_frame(vel_in, "robot_a", world_to_local)

    def _update_mocap_grasp(self, prefix):
        """Kinematically force the object to follow the gripper when closed."""
        gripper_qpos = self.data.joint(f"{prefix}/{self.GRIPPER_JOINT_NAME}").qpos[0]
        # User confirmed current logic is inverted: 0.0 is closed, -1.0 is open.
        # So it should be > -0.4 for closed (holding).
        is_closed = gripper_qpos > -0.7
        
        state_var_name = f"_is_holding_{prefix}"
        offset_var_name = f"_hold_offset_{prefix}"
        if not hasattr(self, state_var_name):
            setattr(self, state_var_name, False)
            setattr(self, offset_var_name, None)
            
        is_holding = getattr(self, state_var_name)
        
        try:
            # gripperMover is the moving pad base. link06 is the fixed wrist.
            # Using link06 gives a completely stable reference frame that doesn't shift when gripping.
            wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}/link06")
            obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
            obj_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
            # We need geom id to zero out collisions
            obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object") # wait, object geom doesn't have a name in XML
        except:
            return
            
        # Let's find the geom dynamically: it's the first geom attached to the object body
        obj_geom_id = self.model.body_geomadr[obj_body_id]
            
        obj_qpos_addr = self.model.jnt_qposadr[obj_joint_id]
        obj_dof_addr = self.model.jnt_dofadr[obj_joint_id]
        
        # 释放延时计时器：防止刚松手时夹爪还没完全张开导致的穿模弹飞
        timer_var_name = f"_ghost_timer_{prefix}"
        if not hasattr(self, timer_var_name):
            setattr(self, timer_var_name, 0)
        
        # 夹爪中心点相对于 link06 原点的局部偏移量
        grasp_center_local = np.array([0.16, 0.0, 0.015])
        
        if is_closed:
            wrist_pos = self.data.xpos[wrist_id]
            mat = self.data.xmat[wrist_id].reshape(3, 3)
            grasp_center_global = wrist_pos + mat @ grasp_center_local
            
            # [可视化] 在画面中画出一个亮红色小球代表吸附中心
            self._add_visual_marker(grasp_center_global, prefix)

            obj_pos = self.data.qpos[obj_qpos_addr:obj_qpos_addr+3]
            dist = np.linalg.norm(grasp_center_global - obj_pos)
            
            # 只有当物体距离真正的夹爪中心点小于 6 厘米时，才开始吸附
            if not is_holding and dist < 0.06:
                setattr(self, state_var_name, True)
                is_holding = True
                setattr(self, timer_var_name, 0)  # 中断任何正在倒数的释放计时
                
                # 吸附时，强行让物体来到夹爪正中心
                setattr(self, offset_var_name, grasp_center_local)
                
                # 穿模保护：转化为 Layer 2，无视 Layer 1（夹爪），但被 Layer 3（桌面）托住
                self.model.geom_contype[obj_geom_id] = 2
                self.model.geom_conaffinity[obj_geom_id] = 2
                
            if is_holding:
                local_offset = getattr(self, offset_var_name)
                mat = self.data.xmat[wrist_id].reshape(3, 3)
                # 强制物体跟随夹爪位置
                target_pos = wrist_pos + mat @ local_offset
                self.data.qpos[obj_qpos_addr:obj_qpos_addr+3] = target_pos
                
                # [强制竖直] 无论抓取时是什么角度，吸住后瞬间修正为绝对竖直
                # 四元数 [1, 0, 0, 0] 代表无旋转，保证圆柱体轴线与世界坐标系 Z 轴重合
                self.data.qpos[obj_qpos_addr+3:obj_qpos_addr+7] = [1.0, 0.0, 0.0, 0.0]
                
                # 清除速度，防止惯性抖动
                self.data.qvel[obj_dof_addr:obj_dof_addr+6] = 0.0
                
                mujoco.mj_kinematics(self.model, self.data)
                mujoco.mj_comPos(self.model, self.data)
        else:
            if is_holding:
                # 刚松开的瞬间
                setattr(self, state_var_name, False)
                # 不立刻恢复碰撞！设定至少 3 秒的时间让它张开并安全撤离（保险起见设为 1500 step）
                setattr(self, timer_var_name, 1500)
                
        # Handle collision layer restore timer
        current_timer = getattr(self, timer_var_name)
        if current_timer > 0:
            setattr(self, timer_var_name, current_timer - 1)
            # 倒计时结束时，恢复为默认 Layer 1
            if current_timer - 1 == 0:
                self.model.geom_contype[obj_geom_id] = 1
                self.model.geom_conaffinity[obj_geom_id] = 1

    def _add_visual_marker(self, pos, prefix):
        """在仿真界面绘制一个临时的标记点."""
        if self.render_mode != "human":
            return
        
        # 尝试通过 Gymnasium 的 mujoco_renderer 访问 viewer
        viewer = getattr(self.mujoco_renderer, 'viewer', None)
        if viewer is not None:
            marker_color = [1, 0, 0, 0.8] if "robot_a" in prefix else [0, 1, 0, 0.8]
            viewer.add_marker(
                pos=pos,
                size=[0.015, 0.015, 0.015],
                rgba=marker_color,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label=f"{prefix} Target"
            )
