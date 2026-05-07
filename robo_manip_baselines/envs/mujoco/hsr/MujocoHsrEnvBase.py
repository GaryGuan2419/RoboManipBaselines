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

from ..MujocoEnvBase import MujocoEnvBase


class MujocoHsrEnvBase(MujocoEnvBase):
    # reset 内额外 mj_step 的次数。默认 0：不积分，机臂在数值上=你写的 init，不会出现「先掉一截」。
    # 若需让 free 物体(如瓶)在开局略 settle，可设 >0，并配合「机臂位姿每步回写 init」
    # （见 reset_kinematic_clamp_robot）。
    reset_settle_physics_steps = 0
    # True：在 reset 的每个子步后把「机器人+手指 mimic」(bottle2 之前) 的 qpos/qvel 强写回 init，机臂
    # 不随积分漂移；仅 bottle2 等外物体在子步内会动。False：普通物理积分(旧式 settle，机臂会漂到平衡)。
    reset_kinematic_clamp_robot = True
    # 缓存：bottle2_freejoint 的 qpos / 起始 qvel 下标（在 setup 后、首次 reset 时填充）
    _bottle2_qpos0 = -1
    _bottle2_dof0 = -1
    # 每次 reset 后、前 N 个 env.step：把 actuation 中「位置致动器」段 ctrl[3:9] 强设为 init，与
    # init_qpos[3:9] 完全一致，消掉首帧/遥操作 sync 与真值 1e-5 级误差 → PD 首段几乎零跟踪误差，避免
    # 「窗口已开、第一刀仿真里胳膊再沉一下再稳住」。底座仍是速度段 action[0:3]，不受影响。
    # 设 0 可关闭。若初值本身在重力下非静力平衡，仍会随时间微漂（物理正常）。
    arm_ctrl_lock_steps_after_reset = 2

    default_camera_config = {
        "azimuth": -120.0,
        "elevation": -25.0,
        "distance": 1.2,
        "lookat": [0.55, 0.08, 0.1],
    }
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            "mobile_vel": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        }
    )

    def setup_robot(self, init_qpos):
        self.init_qpos[: len(init_qpos)] = init_qpos
        self.init_qvel[:] = 0.0
        # hand_l/r_proximal 在 XML 里以 equality 与 hand_motor 1:1 绑死；Gym 首帧 data.qpos 的 9,10
        # 常未被 init_qpos[0:8] 覆盖，若电机为开而两 mimic 为 0，会违反约束。reset 的 mj_forward/首步
        # 上才会“弹”到一致，看起来就像夹爪先合再开。这里显式与电机对齐。
        if self.init_qpos.shape[0] >= 11:
            h8 = float(self.init_qpos[8])
            self.init_qpos[9] = h8
            self.init_qpos[10] = h8

        mujoco.mj_kinematics(self.model, self.data)

        try:
            b_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "bottle2_freejoint"
            )
            self._bottle2_qpos0 = int(self.model.jnt_qposadr[b_id])
            self._bottle2_dof0 = int(self.model.jnt_dofadr[b_id])
        except (ValueError, Exception):
            self._bottle2_qpos0 = int(self.model.nq)
            self._bottle2_dof0 = int(self.model.nv)

        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../../assets/common/robots/hsr/hsr.urdf"
                ),
                arm_root_pose=self.get_body_pose("base_link"),
                ik_eef_joint_id=5,
                arm_joint_idxes=np.arange(5),
                gripper_joint_idxes=np.array([5]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[3:8],
                init_gripper_joint_pos=self.init_qpos[[8]],
                get_root_pose_func=lambda env: get_se3_from_pose(
                    env.get_body_pose("base_link")
                ),
            ),
            MobileOmniConfig(),
        ]

        self.mobile_joint_name_list = [
            "mobile_x_joint",
            "mobile_y_joint",
            "mobile_theta_joint",
        ]

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
            from robo_manip_baselines.teleop.GlfwKeyboardHsrInputDevice import GlfwKeyboardHsrInputDevice
            return [
                GlfwKeyboardHsrInputDevice(
                    motion_manager.body_manager_list[0],
                    motion_manager.body_manager_list[1],
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
            return {
                0: {
                    # 5 mm per update is easier for precise final approach than 20 mm.
                    "pos_scale": 5e-3,
                    "gripper_scale": 0.05,
                    "mobile_x_scale": 0.25,
                    "mobile_y_scale": 0.25,
                }
            }
        else:
            return {}

    @property
    def command_keys_for_step(self):
        return [DataKey.COMMAND_MOBILE_OMNI_VEL, DataKey.COMMAND_JOINT_POS]

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._arm_ctrl_lock_remaining = int(
            getattr(self, "arm_ctrl_lock_steps_after_reset", 0) or 0
        )
        return obs, info

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        nu = len(self.data.ctrl)
        self.data.ctrl[:] = self.init_qpos[:nu]
        bq = getattr(self, "_bottle2_qpos0", -1)
        if bq < 0:
            bq, bv = int(self.model.nq), int(self.model.nv)
        else:
            bv = int(self._bottle2_dof0)
        n = int(getattr(self, "reset_settle_physics_steps", 0) or 0)
        do_clamp = bool(getattr(self, "reset_kinematic_clamp_robot", True))
        ctrl_hold = self.data.ctrl.copy()
        for _ in range(n):
            self.data.ctrl[:] = ctrl_hold
            mujoco.mj_step(self.model, self.data, nstep=1)
            if do_clamp:
                self.data.qpos[0:bq] = self.init_qpos[0:bq]
                self.data.qvel[0:bv] = 0.0
                self.data.ctrl[:] = ctrl_hold
            mujoco.mj_forward(self.model, self.data)
        if n == 0 and do_clamp:
            # 再同步一次，确保 init 与 keyframe/ctrl 完全一致（无子步时唯一一次 forward 已在 set_state 里做过）
            self.data.qpos[0:bq] = self.init_qpos[0:bq]
            self.data.qvel[0:bv] = 0.0
            self.data.ctrl[:] = ctrl_hold
            mujoco.mj_forward(self.model, self.data)
        mujoco.mj_rnePostConstraint(self.model, self.data)
        return self._get_obs()

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
        # Performance: MujocoEnvBase.step() calls `_get_info()` which renders RGB+Depth
        # for *all* offscreen cameras every simulation step. This makes keyboard teleop
        # feel "discrete" (low control update rate) and causes severe stutter.
        #
        # We skip camera rendering here and only render images on-demand via `get_images()`
        # (TeleopBase already re-fetches images right before recording/display).
        action_copy = np.asarray(action, dtype=np.float64).copy()
        r = int(getattr(self, "_arm_ctrl_lock_remaining", 0) or 0)
        if r > 0 and action_copy.shape[0] >= 9:
            action_copy[3:9] = self.init_qpos[3:9].copy()
            self._arm_ctrl_lock_remaining = r - 1
        action_copy[0:3] = self.convert_mobile_vel_frame(
            action_copy[0:3], world_to_local=False
        )

        self.do_simulation(action_copy, self.frame_skip)

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = False
        info = {}  # skip expensive camera rendering

        if self.render_mode == "human":
            # Keep the interactive viewer responsive (rendering the viewport only).
            if self._first_render:
                self._first_render = False
                self.mujoco_renderer.viewer._hide_menu = True
            self.render()

        # truncation=False as the time limit is handled by TimeLimit wrapper.
        return obs, reward, terminated, False, info

    def get_images(self):
        """Render all offscreen cameras on demand. Called during policy evaluation."""
        return self._get_info()

    def print_joint_state(self):
        """Print robot base pose, joint values, and key body poses."""
        arm_joint_names = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        base_joint_names = ["mobile_x_joint", "mobile_y_joint", "mobile_theta_joint"]
        gripper_joint_names = [
            "hand_motor_joint",
            "hand_l_proximal_joint",
            "hand_r_proximal_joint",
        ]

        base = [self.data.joint(jn).qpos[0] for jn in base_joint_names]
        arm = [self.data.joint(jn).qpos[0] for jn in arm_joint_names]
        grip = [self.data.joint(jn).qpos[0] for jn in gripper_joint_names]

        palm_pose = self.get_body_pose("hand_palm_link")
        left_tip_pose = self.get_body_pose("hand_l_finger_tip_frame")
        right_tip_pose = self.get_body_pose("hand_r_finger_tip_frame")

        print(f"\n{'=' * 72}")
        print("[HSR] Current robot state")
        print(
            f"  base (x, y, theta): [{base[0]: .4f}, {base[1]: .4f}, {base[2]: .4f}]"
        )
        print(
            "  arm joints [lift, flex, roll, wrist_flex, wrist_roll]: "
            f"[{', '.join(f'{v:.4f}' for v in arm)}]"
        )
        print(
            "  gripper joints [motor, left_proximal, right_proximal]: "
            f"[{', '.join(f'{v:.4f}' for v in grip)}]"
        )
        print(
            "  hand_palm pose [x, y, z, qw, qx, qy, qz]: "
            f"[{', '.join(f'{v:.4f}' for v in palm_pose)}]"
        )
        print(
            "  left_tip xyz: "
            f"[{left_tip_pose[0]:.4f}, {left_tip_pose[1]:.4f}, {left_tip_pose[2]:.4f}]"
        )
        print(
            "  right_tip xyz: "
            f"[{right_tip_pose[0]:.4f}, {right_tip_pose[1]:.4f}, {right_tip_pose[2]:.4f}]"
        )

        for obj_name in ("bottle1", "bottle2"):
            try:
                obj_pose = self.get_body_pose(obj_name)
                print(
                    f"  {obj_name} pose [x, y, z, qw, qx, qy, qz]: "
                    f"[{', '.join(f'{v:.4f}' for v in obj_pose)}]"
                )
            except KeyError:
                pass

        # Copy-paste helper for single-HSR init_qpos (without external objects).
        init_qpos_single_hsr = base + arm + [grip[0]]
        print(
            "  init_qpos (single HSR, 9 dims = base3 + arm5 + hand_motor):\n"
            f"    np.array([{', '.join(f'{v:.4f}' for v in init_qpos_single_hsr)}])"
        )
        print(f"{'=' * 72}")

    def _get_obs(self):
        arm_joint_name_list = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        gripper_joint_name = "hand_motor_joint"

        arm_joint_pos = np.array(
            [self.data.joint(joint_name).qpos[0] for joint_name in arm_joint_name_list]
        )
        arm_joint_vel = np.array(
            [self.data.joint(joint_name).qvel[0] for joint_name in arm_joint_name_list]
        )
        gripper_joint_pos = np.array([self.data.joint(gripper_joint_name).qpos[0]])
        gripper_joint_vel = np.zeros(1)
        force = self.data.sensor("force_sensor").data.flat.copy()
        torque = self.data.sensor("torque_sensor").data.flat.copy()

        mobile_vel = np.array(
            [
                self.data.joint(joint_name).qvel[0]
                for joint_name in self.mobile_joint_name_list
            ]
        )
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
        theta = self.data.joint(self.mobile_joint_name_list[-1]).qpos[0]
        if not world_to_local:
            theta *= -1

        rot_mat = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )

        vel_in_xy = vel_in[0:2]
        vel_out_xy = rot_mat @ vel_in_xy

        return np.concatenate([vel_out_xy, vel_in[[2]]])
