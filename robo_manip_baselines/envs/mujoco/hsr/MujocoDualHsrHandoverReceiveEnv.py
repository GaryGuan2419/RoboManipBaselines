"""Dual HSR handover-receive scene for policy rollout (Robot B only; A held fixed).

Training data uses short camera keys ``head`` / ``hand`` and split arm vs gripper state
keys; this env maps ``robot_b/*`` observations and builds 18-dim actions for
:class:`MujocoDualHsrDemoEnv`.

实验三 / Rollout 泛化（与 tidyup 瓶位 XY 扰动对称）：

- ``robot_a_xy_reset_perturb_half_extent_m`` / ``robot_b_xy_reset_perturb_half_extent_m``：
  每次 ``reset_model`` 在 **A 或 B 底盘** 世界系 XY 上叠加 U[-h,h]（每轴独立），名义约
  ``2h×2h`` cm；**不改 yaw / Z**。
- **定点 eval**：当 ``pick_eval_max_policy_action_steps`` 非空时，Rollout 在达到该 **policy step**
  （与 action 图横轴一致）后调用 ``compute_pick_eval_outcome()``。成功判据：**baton 两节 box 角点
  最低世界 z** ≥ ``handover_pick_eval_min_baton_lowest_z_m``（默认 **0.05 m**，离地约 5 cm），
  **不与 reset 基线比相对抬升**（交接后略低于开局仍可成功，只要未贴地）。
- **Rollout 协同松手（可选，默认开启）**：B 的 ``hand_motor`` qpos 连续低于阈值达到
  ``handover_rollout_b_grip_closed_hold_steps`` 个 **仿真步** 后 → A 夹爪改为张开指令并保持
  ``handover_rollout_release_steps`` 步 → A 以本地 ``vx<0`` 后退
  ``handover_rollout_retreat_steps`` 步（经 :class:`MujocoDualHsrDemoEnv` 旋到世界系，A 面向 +X
  时约沿 **-X**）。关闭：``gym.make(..., handover_rollout_coordinated_release=False)``。
"""

from __future__ import annotations

import os
import sys
from itertools import product
from os import path

import mujoco
import numpy as np

from robo_manip_baselines.common import ArmConfig, DataKey, MobileOmniConfig, get_se3_from_pose
from robo_manip_baselines.envs.mujoco.hsr.MujocoDualHsrDemoEnv import MujocoDualHsrDemoEnv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_BIN = os.path.join(_REPO_ROOT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from handover_config import (  # noqa: E402
    ARM_JOINT_NAMES,
    HANDOVER_BASE_B,
    HANDOVER_GRIPPER_A,
    HANDOVER_GRIPPER_B_OPEN,
    HANDOVER_POSE_A,
    build_handover_qpos,
)


class MujocoDualHsrHandoverReceiveEnv(MujocoDualHsrDemoEnv):
    """Same XML as sampling; adds ``body_config_list`` for Robot B and rollout hooks."""

    # Training stores arm (5) and gripper (1) in separate action keys; policy vector is 3+5+1.
    # RolloutBase merges arm+grip for MotionManager's COMMAND_JOINT_POS (expects 6-dim indexing).
    _rollout_split_arm_gripper_action = True

    def __init__(self, **kwargs):
        # Pop before MujocoEnv / Gymnasium see unknown kwargs.
        ra_xy = kwargs.pop("robot_a_xy_reset_perturb_half_extent_m", None)
        if ra_xy is None:
            self.robot_a_xy_reset_perturb_half_extent_m = 0.0
        else:
            self.robot_a_xy_reset_perturb_half_extent_m = max(0.0, float(ra_xy))

        rb_xy = kwargs.pop("robot_b_xy_reset_perturb_half_extent_m", None)
        if rb_xy is None:
            self.robot_b_xy_reset_perturb_half_extent_m = 0.0
        else:
            self.robot_b_xy_reset_perturb_half_extent_m = max(0.0, float(rb_xy))

        kwargs.pop("pick_eval_max_env_steps", None)
        pa_steps = kwargs.pop("pick_eval_max_policy_action_steps", None)
        self.pick_eval_max_policy_action_steps = (
            None if pa_steps is None else max(1, int(pa_steps))
        )
        # Baton length / ratio kept for API parity with tidyup; handover success uses palm XY + grip.
        baton_len = kwargs.pop("pick_eval_baton_length_m", 0.14)
        self.pick_eval_baton_length_m = float(baton_len)
        ratio = kwargs.pop("pick_eval_success_bottom_clearance_ratio", 0.5)
        self.pick_eval_success_bottom_clearance_ratio = float(ratio)
        self.pick_eval_success_min_z = (
            self.pick_eval_baton_length_m * self.pick_eval_success_bottom_clearance_ratio
        )

        self.handover_eval_max_palm_xy_m = float(
            kwargs.pop("handover_eval_max_palm_xy_m", 0.16) or 0.16
        )
        self.handover_eval_grip_qpos_max = float(
            kwargs.pop("handover_eval_grip_qpos_max", 0.52) or 0.52
        )
        # 第 N policy step 结算：baton 几何最低点世界 z ≥ 此值（米），相对 z=0 地面；默认 0.05=5cm。
        self.handover_pick_eval_min_baton_lowest_z_m = float(
            kwargs.pop("handover_pick_eval_min_baton_lowest_z_m", 0.05) or 0.05
        )
        kwargs.pop("handover_pick_eval_min_rise_m", None)  # 旧参数名，已改为绝对高度阈值

        # Rollout：B 夹紧持续若干步 → A 松开夹爪 → A 本地 -x 平移后退（世界系约 -X，与 A 面向 +X 一致）
        self.handover_rollout_coordinated_release = bool(
            kwargs.pop("handover_rollout_coordinated_release", True)
        )
        self.handover_rollout_b_grip_closed_qpos_max = float(
            kwargs.pop("handover_rollout_b_grip_closed_qpos_max", 0.52) or 0.52
        )
        self.handover_rollout_b_grip_closed_hold_steps = max(
            1,
            int(kwargs.pop("handover_rollout_b_grip_closed_hold_steps", 20) or 20),
        )
        self.handover_rollout_release_steps = max(
            1, int(kwargs.pop("handover_rollout_release_steps", 24) or 24)
        )
        self.handover_rollout_retreat_steps = max(
            1, int(kwargs.pop("handover_rollout_retreat_steps", 56) or 56)
        )
        self.handover_rollout_a_retreat_local_vel_x = float(
            kwargs.pop("handover_rollout_a_retreat_local_vel_x", -0.22) or -0.22
        )
        og = kwargs.pop("handover_rollout_a_open_grip_cmd", None)
        self.handover_rollout_a_open_grip_cmd = (
            float(og) if og is not None else float(HANDOVER_GRIPPER_B_OPEN)
        )

        super().__init__(**kwargs)

        self._handover_rollout_phase = 0
        self._handover_rollout_b_closed_ctr = 0
        self._handover_rollout_phase_ctr = 0

    def setup_robot(self, init_qpos):
        super().setup_robot(init_qpos)

        mujoco.mj_kinematics(self.model, self.data)

        self.mobile_joint_name_list = [
            "robot_b/mobile_x_joint",
            "robot_b/mobile_y_joint",
            "robot_b/mobile_theta_joint",
        ]

        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../../assets/common/robots/hsr/hsr.urdf"
                ),
                arm_root_pose=self.get_body_pose("robot_b/base_link"),
                ik_eef_joint_id=5,
                arm_joint_idxes=np.arange(5),
                gripper_joint_idxes=np.array([5]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[14:19].copy(),
                init_gripper_joint_pos=self.init_qpos[[19]].copy(),
                get_root_pose_func=lambda env: get_se3_from_pose(
                    env.get_body_pose("robot_b/base_link")
                ),
            ),
            MobileOmniConfig(),
        ]

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

    def _baton_lowest_world_z(self):
        """世界系 +Z 向上时，接力棒两节 box 几何的最低角点 z（米）。"""
        lowest = np.inf
        for gname in ("baton_lower", "baton_upper"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid < 0:
                continue
            pos = self.data.geom_xpos[gid]
            mat = self.data.geom_xmat[gid].reshape(3, 3)
            hs = self.model.geom_size[gid]
            for sx, sy, sz in product((-1.0, 1.0), repeat=3):
                corner = pos + mat @ (hs * np.array([sx, sy, sz], dtype=np.float64))
                lowest = min(lowest, float(corner[2]))
        return None if lowest == np.inf else lowest

    def compute_pick_eval_outcome(self, policy_action_step_count=None):
        """Rollout 在 ``pick_eval_max_policy_action_steps`` 达成时调用（与 tidyup 同名接口）。

        成功：baton 两节 box 角点 **最低世界 z** ≥ ``handover_pick_eval_min_baton_lowest_z_m``
        （默认 0.05 m，离地约 5 cm）。不与 reset 时高度比较。
        """
        try:
            baton_xy = self.data.body("bottle2").xpos[:2].copy()
            palm_xy = self.data.body("robot_b/hand_palm_link").xpos[:2].copy()
            dist_xy = float(np.linalg.norm(baton_xy - palm_xy))
        except Exception:
            dist_xy = float("inf")
        try:
            grip = float(self.data.joint("robot_b/hand_motor_joint").qpos[0])
        except Exception:
            grip = 1.0

        z_low = self._baton_lowest_world_z()
        z_th = float(getattr(self, "handover_pick_eval_min_baton_lowest_z_m", 0.05) or 0.05)
        z0 = getattr(self, "_handover_pick_eval_baseline_z_low", None)
        ok = z_low is not None and z_low >= z_th - 1e-6
        reward = 1.0 if ok else 0.0

        info = {
            "rollout_pick_eval_done": True,
            "pick_eval/trigger": "policy_action_list_length",
            "pick_eval/policy_action_steps": int(policy_action_step_count or -1),
            "handover_eval/palm_xy_dist_m": dist_xy,
            "handover_eval/max_palm_xy_m": float(self.handover_eval_max_palm_xy_m),
            "handover_eval/grip_qpos": grip,
            "handover_eval/grip_qpos_max": float(self.handover_eval_grip_qpos_max),
            "pick_eval/baton_lowest_z": None if z_low is None else float(z_low),
            "pick_eval/baton_baseline_lowest_z": None if z0 is None else float(z0),
            "pick_eval/min_baton_lowest_z_m": float(z_th),
            "pick_eval/success_min_z": float(self.pick_eval_success_min_z),
            "pick_eval/success": bool(ok),
        }
        nstr = "?" if policy_action_step_count is None else str(int(policy_action_step_count))
        z0s = "None" if z0 is None else f"{z0:.4f}"
        zls = "None" if z_low is None else f"{z_low:.4f}"
        print(
            f"[{self.__class__.__name__}] handover_receive_eval policy_step={nstr}: "
            f"baton_lowest_z={zls} m (need ≥{z_th:.4f} floor clearance), reset_baseline_z={z0s} m [debug] "
            f"palm_dist={dist_xy:.4f} m, B_grip={grip:.4f} "
            f"→ {'SUCCESS' if ok else 'FAIL'}",
            flush=True,
        )
        return reward, info

    def reset_model(self):
        q = build_handover_qpos(HANDOVER_BASE_B)
        h = float(getattr(self, "robot_a_xy_reset_perturb_half_extent_m", 0.0) or 0.0)
        if h > 0.0:
            dx = float(np.random.uniform(-h, h))
            dy = float(np.random.uniform(-h, h))
            q[0] += dx
            q[1] += dy
            print(
                f"[{self.__class__.__name__}] robot_a_xy_reset_perturb: "
                f"Δx={dx * 1e3:.2f} mm, Δy={dy * 1e3:.2f} mm "
                f"(half_extent={h * 1e3:.1f} mm)",
                flush=True,
            )
        hb = float(getattr(self, "robot_b_xy_reset_perturb_half_extent_m", 0.0) or 0.0)
        if hb > 0.0:
            dbx = float(np.random.uniform(-hb, hb))
            dby = float(np.random.uniform(-hb, hb))
            q[11] += dbx
            q[12] += dby
            print(
                f"[{self.__class__.__name__}] robot_b_xy_reset_perturb: "
                f"Δx={dbx * 1e3:.2f} mm, Δy={dby * 1e3:.2f} mm "
                f"(half_extent={hb * 1e3:.1f} mm)",
                flush=True,
            )
        self.init_qpos[:29] = q
        self.set_state(self.init_qpos, self.init_qvel)
        self.data.ctrl[:] = self.init_qpos[: len(self.data.ctrl)]
        self._handover_rollout_phase = 0
        self._handover_rollout_b_closed_ctr = 0
        self._handover_rollout_phase_ctr = 0
        # 仅作日志对照，不参与成败判定
        mujoco.mj_forward(self.model, self.data)
        self._handover_pick_eval_baseline_z_low = self._baton_lowest_world_z()
        return self._get_obs()

    def get_joint_pos_from_obs(self, obs):
        return obs["robot_b/joint_pos"]

    def get_joint_vel_from_obs(self, obs):
        return np.array(
            [
                self.data.joint(f"robot_b/{j}").qvel[0]
                for j in ARM_JOINT_NAMES
            ],
            dtype=np.float64,
        )

    def get_mobile_vel_from_obs(self, obs):
        return obs["robot_b/mobile_vel"]

    def get_eef_wrench_from_obs(self, obs):
        del obs
        try:
            force = self.data.sensor("robot_b/force_sensor").data.flat.copy()
            torque = self.data.sensor("robot_b/torque_sensor").data.flat.copy()
            return np.concatenate([force, torque])
        except Exception:
            return np.zeros(6, dtype=np.float64)

    def _read_robot_b_hand_motor_qpos(self) -> float:
        try:
            return float(self.data.joint("robot_b/hand_motor_joint").qpos[0])
        except Exception:
            return 1.0

    def get_rollout_hold_action_robot_a(self):
        """9-dim action slice for robot A: arm 固定；可选 B 夹紧后 A 松开再后退。"""
        a = np.zeros(9, dtype=np.float64)
        a[3:8] = HANDOVER_POSE_A

        if not getattr(self, "handover_rollout_coordinated_release", False):
            a[0:3] = 0.0
            a[8] = HANDOVER_GRIPPER_A
            return a

        b_grip = self._read_robot_b_hand_motor_qpos()
        thr = float(self.handover_rollout_b_grip_closed_qpos_max)
        ph = int(self._handover_rollout_phase)

        # ---- 0: 等 B 夹紧持续 hold_steps（连续 env 步）----
        if ph == 0:
            if b_grip <= thr + 1e-6:
                self._handover_rollout_b_closed_ctr += 1
            else:
                self._handover_rollout_b_closed_ctr = 0
            if self._handover_rollout_b_closed_ctr >= int(
                self.handover_rollout_b_grip_closed_hold_steps
            ):
                self._handover_rollout_phase = 1
                self._handover_rollout_phase_ctr = 0
                print(
                    f"[{self.__class__.__name__}] handover_rollout: B grip held closed "
                    f"(qpos≤{thr:.3f}) → release A",
                    flush=True,
                )
            a[0:3] = 0.0
            a[8] = HANDOVER_GRIPPER_A
            return a

        # ---- 1: A 张开若干步（让棒主要由 B 承担）----
        if ph == 1:
            a[0:3] = 0.0
            a[8] = float(self.handover_rollout_a_open_grip_cmd)
            self._handover_rollout_phase_ctr += 1
            if self._handover_rollout_phase_ctr >= int(self.handover_rollout_release_steps):
                self._handover_rollout_phase = 2
                self._handover_rollout_phase_ctr = 0
                print(
                    f"[{self.__class__.__name__}] handover_rollout: A retreat "
                    f"(local vx={self.handover_rollout_a_retreat_local_vel_x:.3f} m/s)",
                    flush=True,
                )
            return a

        # ---- 2: A 后退：本地 x 速度 <0 → 经 dual env 旋到世界系，面向 +X 时约沿 -X 平移 ----
        if ph == 2:
            a[0] = float(self.handover_rollout_a_retreat_local_vel_x)
            a[1] = 0.0
            a[2] = 0.0
            a[8] = float(self.handover_rollout_a_open_grip_cmd)
            self._handover_rollout_phase_ctr += 1
            if self._handover_rollout_phase_ctr >= int(self.handover_rollout_retreat_steps):
                self._handover_rollout_phase = 3
                self._handover_rollout_phase_ctr = 0
                print(
                    f"[{self.__class__.__name__}] handover_rollout: A stop retreat",
                    flush=True,
                )
            return a

        # ---- 3: 保持停住 + 张开（臂姿不变）----
        a[0:3] = 0.0
        a[8] = float(self.handover_rollout_a_open_grip_cmd)
        return a

    def wrap_rollout_action(self, motion_manager):
        """Build 18-dim env action: hold A, policy commands B (9-dim)."""
        b = np.concatenate(
            [
                motion_manager.get_command_data(DataKey.COMMAND_MOBILE_OMNI_VEL),
                motion_manager.get_command_data(DataKey.COMMAND_JOINT_POS),
            ]
        )
        a = self.get_rollout_hold_action_robot_a()
        return np.concatenate([a, b])

    def get_camera_fovy(self, camera_name):
        if camera_name in ("head", "hand"):
            camera_name = f"robot_b_{camera_name}"
        return super().get_camera_fovy(camera_name)

    def get_images(self):
        info = self._get_info()
        for short in ("head", "hand"):
            long_key = f"robot_b_{short}"
            if long_key in info.get("rgb_images", {}):
                info["rgb_images"][short] = info["rgb_images"][long_key]
            if long_key in info.get("depth_images", {}):
                info["depth_images"][short] = info["depth_images"][long_key]
        return info
