from itertools import product
from os import path

import mujoco
import numpy as np

from .MujocoHsrEnvBase import MujocoHsrEnvBase

# =============================================================================
#  HSR Tidyup（侧向 pick 等）场景说明 —— 与 rollout / 轻度泛化 eval 相关
# =============================================================================
# - 瓶位由 XML body + freejoint 的 init_qpos 决定；`modify_world` 只改 freejoint
#   的平移前三维，不改四元数 → 瓶「姿态」与模板一致，仅水平面内平移。
# - `bottle_xy_reset_perturb_half_extent_m`：每次 reset 选 world 后，在 **世界系 XY**
#   上再加独立均匀噪声 U[-h, h]，总平移范围是边长为 **2h 的正方形**（例如 h=0.025
#   → 约 5cm×5cm）。Z 与姿态不动。
# - 默认 **0.0**（与历史严格 nominal 一致）；Rollout 走 `OperationMujocoHsrTidyup` 时会传入
#   h=0.025。脚本里需要扰动时可 `gym.make(..., bottle_xy_reset_perturb_half_extent_m=0.025)`。
# - **Pick 定点 eval**：由 Rollout 在 **action 图横轴「policy step」**（即
#   `len(policy_action_list)`，每成功 `infer_policy` 追加一行）达到
#   `pick_eval_max_policy_action_steps` 时调用 `compute_pick_eval_outcome()` 结算：
#   棒子两节 box 角点世界 z 最低值 ≥ `ratio×baton_length` → reward=1，并写入
#   info[`rollout_pick_eval_done`]=True。关闭：不传 `pick_eval_max_policy_action_steps` 或 None。
# =============================================================================

# env_hsr_tidyup.xml：baton_lower + baton_upper 沿 body +Z 总高约 0.14 m。
BATON_NOMINAL_LENGTH_M = 0.14


class MujocoHsrTidyupEnv(MujocoHsrEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        # 必须从 kwargs 弹出：否则会原样传给 Gymnasium MuJoCo，报未知参数。
        bottle_xy_h = kwargs.pop("bottle_xy_reset_perturb_half_extent_m", None)
        if bottle_xy_h is None:
            # 默认关闭扰动；Rollout 侧在 OperationMujocoHsrTidyup 里显式传 0.025（约 5cm×5cm）。
            self.bottle_xy_reset_perturb_half_extent_m = 0.0
        else:
            self.bottle_xy_reset_perturb_half_extent_m = max(0.0, float(bottle_xy_h))

        kwargs.pop("pick_eval_max_env_steps", None)  # 旧参数名，忽略
        pa_steps = kwargs.pop("pick_eval_max_policy_action_steps", None)
        self.pick_eval_max_policy_action_steps = (
            None if pa_steps is None else max(1, int(pa_steps))
        )
        baton_len = kwargs.pop("pick_eval_baton_length_m", BATON_NOMINAL_LENGTH_M)
        self.pick_eval_baton_length_m = float(baton_len)
        ratio = kwargs.pop("pick_eval_success_bottom_clearance_ratio", 0.5)
        self.pick_eval_success_bottom_clearance_ratio = float(ratio)
        self.pick_eval_success_min_z = (
            self.pick_eval_baton_length_m * self.pick_eval_success_bottom_clearance_ratio
        )

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
        """在 Rollout 侧达到与 action 图横轴一致的 policy step 后调用，返回 (reward, info)。"""
        z_low = self._baton_lowest_world_z()
        floor_z = 0.0
        thr = floor_z + self.pick_eval_success_min_z
        ok = z_low is not None and z_low >= thr - 1e-4
        reward = 1.0 if ok else 0.0
        info = {
            "rollout_pick_eval_done": True,
            "pick_eval/trigger": "policy_action_list_length",
            "pick_eval/policy_action_steps": int(policy_action_step_count or -1),
            "pick_eval/baton_lowest_z": None if z_low is None else float(z_low),
            "pick_eval/success_min_z": float(thr),
            "pick_eval/success": bool(ok),
        }
        z_str = "None" if z_low is None else f"{z_low:.4f}"
        nstr = "?" if policy_action_step_count is None else str(int(policy_action_step_count))
        print(
            f"[{self.__class__.__name__}] pick_eval policy_step={nstr} (action 图横轴点数): "
            f"baton_lowest_z={z_str} m, threshold={thr:.4f} m "
            f"(ratio×length={self.pick_eval_success_bottom_clearance_ratio}×{self.pick_eval_baton_length_m}) "
            f"→ {'SUCCESS' if ok else 'FAIL'}",
            flush=True,
        )
        return reward, info

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

        # ------------------------------------------------------------------
        # 瓶位偏移：必须用 .copy()，否则下面 += 会 **写坏** `bottle_pos_offsets` 表里
        # 的常驻网格（再叠加 world_random_scale 时尤其明显）。
        # ------------------------------------------------------------------
        delta_pos = np.asarray(self.bottle_pos_offsets[world_idx], dtype=np.float64).copy()
        if self.world_random_scale is not None:
            delta_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )

        # ------------------------------------------------------------------
        # 【醒目】轻度位置泛化：世界系 XY 平面随机平移，边长约 2*h 的正方形；不改 Z、不改姿态。
        # 关闭：构造 env 时传 bottle_xy_reset_perturb_half_extent_m=0.0
        # ------------------------------------------------------------------
        h = float(getattr(self, "bottle_xy_reset_perturb_half_extent_m", 0.0) or 0.0)
        if h > 0.0:
            dx = float(np.random.uniform(-h, h))
            dy = float(np.random.uniform(-h, h))
            delta_pos[0] += dx
            delta_pos[1] += dy
            print(
                f"[{self.__class__.__name__}] bottle_xy_reset_perturb: "
                f"Δx={dx * 1e3:.2f} mm, Δy={dy * 1e3:.2f} mm "
                f"(half_extent={h * 1e3:.1f} mm → 约 {2 * h * 100:.1f}×{2 * h * 100:.1f} cm 范围)",
                flush=True,
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
