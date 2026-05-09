from itertools import product
from os import path

import mujoco
import numpy as np

from .MujocoHsrEnvBase import MujocoHsrEnvBase


class MujocoHsrGroundBatonGraspEnv(MujocoHsrEnvBase):
    """
    单机 HSR：接力棒 **平躺在地**（长轴沿 +X），适合训练/采样 **从上方向下** 闭合夹取；
    与之对照，`MujocoHsrTidyupEnv` 中棒为 **竖直** 站立（长轴 +Z），便于侧向/立棒抓取类演示。

    Rollout 实验三：``bottle_xy_reset_perturb_half_extent_m`` 在 ``modify_world`` 中对 **世界系 XY**
    施加 U[-h,h]（与 side_pick 一致）；``pick_eval_max_policy_action_steps`` 达成时
    ``compute_pick_eval_outcome``：棒两节 box **最低角点 z** ≥ ``pick_eval_min_lowest_z_above_floor_m``
    （默认 0.03 m，离地 3 cm）。
    """

    def __init__(self, **kwargs):
        bottle_xy_h = kwargs.pop("bottle_xy_reset_perturb_half_extent_m", None)
        if bottle_xy_h is None:
            self.bottle_xy_reset_perturb_half_extent_m = 0.0
        else:
            self.bottle_xy_reset_perturb_half_extent_m = max(0.0, float(bottle_xy_h))

        kwargs.pop("pick_eval_max_env_steps", None)
        pa_steps = kwargs.pop("pick_eval_max_policy_action_steps", None)
        self.pick_eval_max_policy_action_steps = (
            None if pa_steps is None else max(1, int(pa_steps))
        )
        self.pick_eval_min_lowest_z_above_floor_m = float(
            kwargs.pop("pick_eval_min_lowest_z_above_floor_m", 0.03) or 0.03
        )
        kwargs.pop("pick_eval_baton_length_m", None)
        kwargs.pop("pick_eval_success_bottom_clearance_ratio", None)

        MujocoHsrEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/hsr/env_hsr_ground_baton.xml",
            ),
            # >>> GROUND_BATON_INIT_RECORD (grep 此串可找回) <<<<<<<<<<<<<<<<<<<<<<
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.2600,
                    -2.0,
                    0.0039,
                    -1.05,
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

        self.bottle_pos_offsets = np.zeros((1, 3))
        self.target_task = None

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
        """Rollout 在 ``pick_eval_max_policy_action_steps`` 达成时调用。"""
        z_low = self._baton_lowest_world_z()
        floor_z = 0.0
        thr = floor_z + float(
            getattr(self, "pick_eval_min_lowest_z_above_floor_m", 0.03) or 0.03
        )
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
            f"[{self.__class__.__name__}] pick_eval policy_step={nstr}: "
            f"baton_lowest_z={z_str} m, threshold={thr:.4f} m (floor clearance ≥3 cm) "
            f"→ {'SUCCESS' if ok else 'FAIL'}",
            flush=True,
        )
        return reward, info

    def _get_reward(self):
        reward = 0.0
        container_half_extents = np.array([0.1, 0.15, 0.08])

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

        delta_pos = np.asarray(self.bottle_pos_offsets[world_idx], dtype=np.float64).copy()
        if self.world_random_scale is not None:
            delta_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )

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
            else:
                original_bottle_pos = self.original_bottle2_pos
            self.init_qpos[bottle_qpos_addr : bottle_qpos_addr + 3] = (
                original_bottle_pos + delta_pos
            )

        return world_idx
