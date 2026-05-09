"""Single HSR B-side place environment."""

from itertools import product
from os import path

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer

from .MujocoHsrEnvBase import MujocoHsrEnvBase
from .hsr_b_side_place_config import (
    BATON_HOLD_STEPS,
    GRIP_LOCK_RELEASE_QPOS,
    HANDOVER_AREA_XY_DEFAULT,
    PLACE_BATON_NOMINAL_LENGTH_M,
    PLACE_TARGET_PAD_HALF_XY_M,
    TARGET_AREA_XY_DEFAULT,
    build_initial_qpos_18,
)

_HSR_ACTION_GRIP_IDX = 8
_INTENSE_CLOSE_CMD = -0.5


class MujocoHsrBSidePlaceEnv(MujocoHsrEnvBase):
    """Single-HSR B-side place sampling.

    Startup sequence
    ----------------
    1. Baton is kinematically held at its init pose for *at least*
       ``BATON_HOLD_STEPS`` sim steps (written back every step).
    2. During that phase the gripper ctrl is forced to ``-0.5``
       (same intense close used by TidyupPlace), which is far more
       aggressive than the default ``-0.15`` and produces much higher
       clamping force via the position actuator (force ∝ kp·(ctrl-qpos)).
    3. The baton is released only when **both** the step counter has
       elapsed *and* gripper qpos ≤ ``GRIP_LOCK_RELEASE_QPOS``.
    4. After release, ``action[8]`` is **never overridden** — the user
       controls the gripper entirely via Z/X keys.
    """

    default_camera_config = {
        "azimuth": -120.0,
        "elevation": -25.0,
        "distance": 3.0,
        "lookat": [1.0, 0.0, 0.15],
    }

    def __init__(self, **kwargs):
        xml = path.join(
            path.dirname(__file__),
            "../../assets/mujoco/envs/hsr/env_hsr_b_side_place.xml",
        )
        self._full_init_qpos = build_initial_qpos_18()

        self.place_base_xy_reset_perturb_half_extent_m = float(
            kwargs.pop("place_base_xy_reset_perturb_half_extent_m", 0.0) or 0.0
        )
        self.place_rollout_grip_lock_policy_steps = int(
            kwargs.pop("place_rollout_grip_lock_policy_steps", 0) or 0
        )
        self.place_rollout_grip_lock_cmd = float(
            kwargs.pop("place_rollout_grip_lock_cmd", -0.5)
        )
        self.place_rollout_grip_lock_verbose = bool(
            kwargs.pop("place_rollout_grip_lock_verbose", False)
        )
        kwargs.pop("pick_eval_max_env_steps", None)
        pa_steps = kwargs.pop("pick_eval_max_policy_action_steps", None)
        self.pick_eval_max_policy_action_steps = (
            None if pa_steps is None else int(pa_steps)
        )
        kwargs.pop("place_eval_max_xy_dist_m", None)
        kwargs.pop("place_eval_min_baton_lowest_z_m", None)
        _half = kwargs.pop("place_eval_target_pad_half_xy_m", None)
        if _half is None:
            self.place_eval_target_pad_half_xy = PLACE_TARGET_PAD_HALF_XY_M.copy()
        else:
            self.place_eval_target_pad_half_xy = np.asarray(
                _half, dtype=np.float64
            ).reshape(2)
        self.place_eval_baton_nominal_length_m = float(
            kwargs.pop(
                "place_eval_baton_nominal_length_m", PLACE_BATON_NOMINAL_LENGTH_M
            )
            or PLACE_BATON_NOMINAL_LENGTH_M
        )

        MujocoHsrEnvBase.__init__(
            self,
            xml,
            self._full_init_qpos[:9],
            **kwargs,
        )

        self._baton_hold_remaining = 0
        self._baton_hold_qpos = np.zeros(7)
        self._wait_user_grip_takeover = True
        self._last_user_grip_cmd = None

        self._hand_motor_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hand_motor_joint"
        )
        assert self._hand_motor_act_id == _HSR_ACTION_GRIP_IDX

        j_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "bottle2_freejoint"
        )
        self._baton_q_addr = self.model.jnt_qposadr[j_id]
        self._baton_v_addr = self.model.jnt_dofadr[j_id]
        self._gripper_q_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_motor_joint"
            )
        ]

    # ── cameras ──────────────────────────────────────────────────────────

    def setup_camera(self):
        self.cameras = {}
        for camera_id in range(self.model.ncam):
            camera_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id
            )
            self.cameras[camera_name.replace("/", "_")] = {
                "name": camera_name,
                "id": camera_id,
                "viewer": OffScreenViewer(
                    self.model, self.data, width=640, height=480
                ),
            }
        self.mujoco_renderer._viewers["dummy"] = None
        self._first_render = True

    # ── input device ─────────────────────────────────────────────────────

    def get_input_device_kwargs(self, input_device_name):
        kwargs = super().get_input_device_kwargs(input_device_name)
        if input_device_name == "keyboard":
            kwargs = dict(kwargs)
            k0 = dict(kwargs.get(0, {}))
            k0["gripper_scale"] = -abs(k0.get("gripper_scale", 0.05))
            kwargs[0] = k0
        return kwargs

    # ── helpers ───────────────────────────────────────────────────────────

    def _set_mocap_xy(self, body_name, xy):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mid = self.model.body_mocapid[bid]
        if mid < 0:
            return
        self.data.mocap_pos[mid][0] = xy[0]
        self.data.mocap_pos[mid][1] = xy[1]
        self.data.mocap_pos[mid][2] = 0.0

    def _restore_baton_hold(self):
        a = self._baton_q_addr
        v = self._baton_v_addr
        self.data.qpos[a : a + 7] = self._baton_hold_qpos
        self.data.qvel[v : v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _baton_highest_world_z(self):
        """两节 box 几何角点的最大世界 z（米）。"""
        highest = -np.inf
        for gname in ("baton_lower", "baton_upper"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid < 0:
                continue
            pos = self.data.geom_xpos[gid]
            mat = self.data.geom_xmat[gid].reshape(3, 3)
            hs = self.model.geom_size[gid]
            for sx, sy, sz in product((-1.0, 1.0), repeat=3):
                corner = pos + mat @ (hs * np.array([sx, sy, sz], dtype=np.float64))
                highest = max(highest, float(corner[2]))
        return None if highest == -np.inf else highest

    def compute_pick_eval_outcome(self, policy_action_step_count=None):
        """Rollout 在 ``pick_eval_max_policy_action_steps`` 达成时调用。

        成功需同时满足：
        - **bottle2 中心 XY** 落在 ``target_area`` 轴对齐矩形内（半宽 ``place_eval_target_pad_half_xy``）；
        - **接力棒几何最高点 z** ≤ ``target_area`` 参考 z + ``place_eval_baton_nominal_length_m``
          （默认 0.14 m，与棒长一致，排除「举在空中」的假成功）。
        """
        baton_xy = None
        tgt_xy = None
        tgt_z = None
        z_hi = self._baton_highest_world_z()
        try:
            baton_xy = self.data.body("bottle2").xpos[:2].copy()
            tgt_xy = self.data.body("target_area").xpos[:2].copy()
            tgt_z = float(self.data.body("target_area").xpos[2])
        except Exception:
            pass
        half = np.asarray(
            getattr(self, "place_eval_target_pad_half_xy", PLACE_TARGET_PAD_HALF_XY_M),
            dtype=np.float64,
        ).reshape(2)
        L = float(
            getattr(
                self,
                "place_eval_baton_nominal_length_m",
                PLACE_BATON_NOMINAL_LENGTH_M,
            )
            or PLACE_BATON_NOMINAL_LENGTH_M
        )
        ok_xy = False
        if baton_xy is not None and tgt_xy is not None:
            delta = np.abs(baton_xy - tgt_xy)
            ok_xy = bool(np.all(delta <= half + 1e-6))
        z_ceiling = None if tgt_z is None else float(tgt_z + L)
        ok_z = (
            z_hi is not None
            and z_ceiling is not None
            and z_hi <= z_ceiling + 1e-6
        )
        ok = bool(ok_xy and ok_z)
        reward = 1.0 if ok else 0.0
        info = {
            "rollout_pick_eval_done": True,
            "pick_eval/trigger": "policy_action_list_length",
            "pick_eval/policy_action_steps": int(policy_action_step_count or -1),
            "place_eval/baton_center_xy": None
            if baton_xy is None
            else baton_xy.astype(float).tolist(),
            "place_eval/target_center_xy": None
            if tgt_xy is None
            else tgt_xy.astype(float).tolist(),
            "place_eval/pad_half_xy_m": half.astype(float).tolist(),
            "place_eval/baton_highest_z_m": None if z_hi is None else float(z_hi),
            "place_eval/z_ceiling_m": z_ceiling,
            "place_eval/baton_nominal_length_m": float(L),
            "pick_eval/success": bool(ok),
        }
        nstr = "?" if policy_action_step_count is None else str(int(policy_action_step_count))
        bx = "None" if baton_xy is None else np.array2string(baton_xy, precision=4)
        tx = "None" if tgt_xy is None else np.array2string(tgt_xy, precision=4)
        hs = np.array2string(half, precision=4)
        zhs = "None" if z_hi is None else f"{z_hi:.4f}"
        zcs = "None" if z_ceiling is None else f"{z_ceiling:.4f}"
        print(
            f"[{self.__class__.__name__}] place_eval policy_step={nstr}: "
            f"baton_xy={bx} m, target_xy={tx} m, pad|dx|≤{half[0]:.4f} |dy|≤{half[1]:.4f} (half={hs}); "
            f"baton_top_z={zhs} m (need ≤{zcs}=target_z+{L:.3f} m) "
            f"→ {'SUCCESS' if ok else 'FAIL'}",
            flush=True,
        )
        return reward, info

    # ── reset / step ─────────────────────────────────────────────────────

    def reset(self, **kwargs):
        self.init_qpos[:] = build_initial_qpos_18().copy()
        self.init_qvel[:] = 0.0
        h = float(getattr(self, "place_base_xy_reset_perturb_half_extent_m", 0.0) or 0.0)
        if h > 0.0:
            dx = float(np.random.uniform(-h, h))
            dy = float(np.random.uniform(-h, h))
            self.init_qpos[0] += dx
            self.init_qpos[1] += dy
            print(
                f"[{self.__class__.__name__}] place_base_xy_reset_perturb: "
                f"Δx={dx * 1e3:.2f} mm, Δy={dy * 1e3:.2f} mm "
                f"(half_extent={h * 1e3:.1f} mm)",
                flush=True,
            )

        obs, info = super().reset(**kwargs)

        self._set_mocap_xy("handover_area", HANDOVER_AREA_XY_DEFAULT)
        self._set_mocap_xy("target_area", TARGET_AREA_XY_DEFAULT)
        mujoco.mj_forward(self.model, self.data)

        a = self._baton_q_addr
        self._baton_hold_qpos = self.data.qpos[a : a + 7].copy()
        self._baton_hold_remaining = int(BATON_HOLD_STEPS)
        self._wait_user_grip_takeover = True
        self._last_user_grip_cmd = None

        self.data.ctrl[:] = self.init_qpos[: len(self.data.ctrl)]
        self.data.ctrl[_HSR_ACTION_GRIP_IDX] = _INTENSE_CLOSE_CMD
        return self._get_obs(), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).copy()
        user_grip_cmd = float(action[_HSR_ACTION_GRIP_IDX])

        if self._baton_hold_remaining > 0:
            action[_HSR_ACTION_GRIP_IDX] = _INTENSE_CLOSE_CMD
        elif self._wait_user_grip_takeover:
            # Keep startup clamp until user actually touches Z/X once.
            # This avoids abrupt release from -0.5 -> 0.0 at the hold boundary.
            if self._last_user_grip_cmd is None:
                self._last_user_grip_cmd = user_grip_cmd
                action[_HSR_ACTION_GRIP_IDX] = _INTENSE_CLOSE_CMD
            elif abs(user_grip_cmd - self._last_user_grip_cmd) <= 1e-6:
                action[_HSR_ACTION_GRIP_IDX] = _INTENSE_CLOSE_CMD
            else:
                self._wait_user_grip_takeover = False

        obs, reward, terminated, truncated, info = super().step(action)

        if self._baton_hold_remaining > 0:
            self._restore_baton_hold()
            if self._baton_hold_remaining > 1:
                self._baton_hold_remaining -= 1
            else:
                if self.data.qpos[self._gripper_q_addr] <= GRIP_LOCK_RELEASE_QPOS:
                    self._baton_hold_remaining = 0

        return obs, reward, terminated, truncated, info

    def modify_world(self, world_idx=None, cumulative_idx=None):
        pass

    def after_rollout_reset(self):
        """Rollout only: run a few hold steps so grip/baton contacts resolve (collect script does this implicitly).

        Without this, the first render can show gripper/baton interpenetration before mj_step settles.
        """
        mujoco.mj_forward(self.model, self.data)
        arm_names = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        action = np.zeros(9, dtype=np.float64)
        for i, jn in enumerate(arm_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            action[3 + i] = float(self.data.qpos[self.model.jnt_qposadr[jid]])
        action[8] = _INTENSE_CLOSE_CMD
        n_warm = min(24, max(8, int(BATON_HOLD_STEPS) - 1))
        for _ in range(n_warm):
            self.step(action)
        return self._get_obs(), self._get_info()
