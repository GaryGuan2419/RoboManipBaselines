#!/usr/bin/env python3
"""
Replay the manually captured ultimate-line dual-HSR baton relay keyframes.

This is a lightweight scripted expert trajectory:
  1. A reaches pick pose.
  2. A closes on the baton.
  3. A carries the baton to the handover area.
  4. B reaches the handover area.
  5. B closes on the baton.
  6. A opens and the stabilizer transfers the baton to B.
  7. A backs away from the handover area by 20 cm.
  8. B rotates in place to the place-heading.
  9. B translates to the place area.
  10. B opens.

The first two pick keyframes are shifted by the current initial baton XY relative
to the nominal captured baton XY. This keeps the captured A-to-baton relation
usable if the object start is later nudged slightly.

Step counts below are defaults for scripted convergence; shrink further with
``--stage_step_scale`` / ``--hold_steps``, or raise if arms lag on your machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy

import mujoco
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from robo_manip_baselines.common import DataKey  # noqa: E402
from robo_manip_baselines.envs.mujoco.hsr.MujocoDualHsrUltimateLineEnv import (  # noqa: E402
    MujocoDualHsrUltimateLineEnv,
)


ROBOT_NAMES = ("robot_a", "robot_b")
DEFAULT_SAMPLE_CAMERA_NAMES = (
    "robot_a_head",
    "robot_a_head_left",
    "robot_a_head_right",
    "robot_a_hand",
    "robot_b_head",
    "robot_b_head_left",
    "robot_b_head_right",
    "robot_b_hand",
    "ultimate_line_god",
)
ARM_JOINT_NAMES = (
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
)
GRIPPER_JOINT_NAME = "hand_motor_joint"
BASE_JOINT_NAMES = ("mobile_x_joint", "mobile_y_joint", "mobile_theta_joint")
ARM_JOINT_LIMITS = np.array(
    [
        [0.0, 0.69],
        [-2.62, 0.0],
        [-2.09, 3.84],
        [-1.92, 1.22],
        [-1.92, 3.67],
    ],
    dtype=np.float64,
)

NOMINAL_INITIAL_BATON_XY = np.array([-2.0, 0.0], dtype=np.float64)


def _kf(label, a_base, a_arm, a_grip, b_base, b_arm, b_grip, baton=None):
    return {
        "label": label,
        "robot_a": {
            "base_xy_yaw": np.array(a_base, dtype=np.float64),
            "arm_joints": np.array(a_arm, dtype=np.float64),
            "grip": float(a_grip),
        },
        "robot_b": {
            "base_xy_yaw": np.array(b_base, dtype=np.float64),
            "arm_joints": np.array(b_arm, dtype=np.float64),
            "grip": float(b_grip),
        },
        "baton": None if baton is None else np.array(baton, dtype=np.float64),
    }


KEYFRAMES = [
    _kf(
        "1_pick_settle",
        [-2.550465, -0.076799, 0.000001],
        [0.065777, -2.533079, 0.006920, 0.961149, 0.003848],
        0.800009,
        [2.500000, 0.000000, 3.141600],
        [0.215777, -2.533069, 0.006931, 1.004360, 0.003845],
        0.800042,
        [-2.000000, 0.000000, -0.000128, 1.000000, 0.000000, 0.000000, 0.000000],
    ),
    _kf(
        "2_pick_closed",
        [-2.550693, -0.083455, -0.003577],
        [0.065822, -2.533040, 0.022945, 0.961258, -0.012092],
        0.184749,
        [2.500000, 0.000000, 3.141600],
        [0.215777, -2.533069, 0.006931, 1.004360, 0.003845],
        0.800042,
        [-2.000010, -0.011907, -0.000122, 0.999965, -0.002774, 0.000061, -0.007847],
    ),
    _kf(
        "3_a_at_handover",
        [-0.547965, -0.076724, -0.008045],
        [0.239070, -2.539222, 0.007302, 0.957454, 0.003552],
        0.184690,
        [2.500000, 0.000000, 3.141600],
        [0.215777, -2.533069, 0.006931, 1.004360, 0.003845],
        0.800042,
        [0.000138, -0.004041, 0.169808, 0.999981, 0.000607, 0.004951, -0.003668],
    ),
    _kf(
        "4_b_at_handover",
        [-0.547864, -0.078724, -0.009147],
        [0.239070, -2.539222, 0.007302, 0.957454, 0.003552],
        0.184690,
        [0.548802, 0.081555, 3.141600],
        [0.160777, -2.533069, 0.006931, 1.004360, 0.003845],
        0.800042,
        [0.000319, -0.006646, 0.169808, 0.999979, 0.000610, 0.004951, -0.004219],
    ),
    _kf(
        "5_b_closed",
        [-0.548160, -0.074681, -0.006788],
        [0.240173, -2.535169, 0.003932, 0.959945, -0.001511],
        0.184705,
        [0.549103, 0.077223, 3.143883],
        [0.159668, -2.536962, 0.004791, 1.002088, -0.001412],
        0.183136,
        [0.001807, -0.001561, 0.173102, 0.999993, -0.002906, 0.001705, -0.001642],
    ),
    _kf(
        "6_a_open_b_holds",
        [-0.548162, -0.074677, -0.006766],
        [0.240777, -2.533077, 0.006917, 0.961152, 0.003843],
        0.900000,
        [0.549103, 0.077189, 3.143883],
        [0.159069, -2.539196, 0.007146, 1.000690, 0.003947],
        0.183104,
        [0.002764, -0.001112, 0.172959, 0.999988, -0.003968, -0.000133, -0.003016],
    ),
    _kf(
        "7_a_retreat_20cm",
        [-0.748162, -0.074677, -0.006766],
        [0.240777, -2.533077, 0.006917, 0.961152, 0.003843],
        0.899999,
        [0.549103, 0.077189, 3.143883],
        [0.159069, -2.539196, 0.007146, 1.000690, 0.003947],
        0.183104,
        None,
    ),
    _kf(
        "8_b_rotate_for_place",
        [-0.748162, -0.074677, -0.006766],
        [0.240777, -2.533077, 0.006917, 0.961152, 0.003843],
        0.899999,
        [0.549103, 0.077189, 6.267142],
        [0.019068, -2.539185, 0.007143, 1.000690, 0.003944],
        0.183104,
        None,
    ),
    _kf(
        "9_b_translate_to_place",
        [-0.748162, -0.074677, -0.006766],
        [0.240777, -2.533077, 0.006917, 0.961152, 0.003843],
        0.899999,
        [1.444198, -0.087778, 6.267142],
        [0.019068, -2.539185, 0.007143, 1.000690, 0.003944],
        0.183104,
        [1.991884, -0.019506, 0.032963, 0.012182, 0.000091, -0.003967, 0.999918],
    ),
    _kf(
        "10_b_open",
        [-0.748162, -0.074677, -0.006766],
        [0.240777, -2.533077, 0.006917, 0.961152, 0.003843],
        0.899999,
        [1.444198, -0.087778, 6.267142],
        [0.019068, -2.539185, 0.007143, 1.000690, 0.003944],
        0.900000,
        [1.991884, -0.019506, 0.032963, 0.012182, 0.000091, -0.003967, 0.999918],
    ),
]


DEFAULT_STAGE_STEPS = {
    "1_pick_settle": 220,
    "2_pick_closed": 90,
    "3_a_at_handover": 420,
    "4_b_at_handover": 30,
    "5_b_closed": 90,
    "6_a_open_b_holds": 120,
    "7_a_retreat_20cm": 90,
    "8_b_rotate_for_place": 220,
    "9_b_translate_to_place": 450,
    "10_b_open": 80,
}


def _joint_qpos(env, prefix: str, joint_name: str) -> float:
    return float(env.data.joint(f"{prefix}/{joint_name}").qpos[0])


def _joint_qvel(env, prefix: str, joint_name: str) -> float:
    return float(env.data.joint(f"{prefix}/{joint_name}").qvel[0])


def _joint_qpos_addr(env, full_joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, full_joint_name)
    return int(env.model.jnt_qposadr[joint_id])


def _set_joint_qpos(env, full_joint_name: str, value: float) -> None:
    env.data.qpos[_joint_qpos_addr(env, full_joint_name)] = float(value)


def _robot_base(env, robot_index: int) -> np.ndarray:
    prefix = ROBOT_NAMES[robot_index]
    return np.array([_joint_qpos(env, prefix, name) for name in BASE_JOINT_NAMES], dtype=np.float64)


def _robot_arm(env, robot_index: int) -> np.ndarray:
    prefix = ROBOT_NAMES[robot_index]
    return np.array([_joint_qpos(env, prefix, name) for name in ARM_JOINT_NAMES], dtype=np.float64)


def _robot_grip(env, robot_index: int) -> float:
    return _joint_qpos(env, ROBOT_NAMES[robot_index], GRIPPER_JOINT_NAME)


def _robot_arm_grip(env, robot_index: int) -> np.ndarray:
    return np.concatenate(
        [
            _robot_arm(env, robot_index),
            np.array([_robot_grip(env, robot_index)], dtype=np.float64),
        ]
    )


def _robot_arm_grip_vel(env, robot_index: int) -> np.ndarray:
    prefix = ROBOT_NAMES[robot_index]
    return np.array(
        [_joint_qvel(env, prefix, name) for name in ARM_JOINT_NAMES]
        + [_joint_qvel(env, prefix, GRIPPER_JOINT_NAME)],
        dtype=np.float64,
    )


def _all_robot_arm_grip(env) -> np.ndarray:
    return np.concatenate([_robot_arm_grip(env, robot_index) for robot_index in range(2)])


def _all_robot_arm_grip_vel(env) -> np.ndarray:
    return np.concatenate([_robot_arm_grip_vel(env, robot_index) for robot_index in range(2)])


def _all_robot_grip(env) -> np.ndarray:
    return np.array([_robot_grip(env, robot_index) for robot_index in range(2)], dtype=np.float64)


def _all_robot_mobile_vel(env) -> np.ndarray:
    obs = env._get_obs()
    return np.concatenate([obs[f"{prefix}/mobile_vel"] for prefix in ROBOT_NAMES]).astype(np.float64)


def _command_joint_pos_from_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(18)
    return np.concatenate([action[3:9], action[12:18]])


def _command_gripper_from_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(18)
    return np.array([action[8], action[17]], dtype=np.float64)


def _command_mobile_from_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(18)
    return np.concatenate([action[0:3], action[9:12]])


def _baton_qpos(env) -> np.ndarray:
    joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "bottle2_freejoint")
    qpos_addr = int(env.model.jnt_qposadr[joint_id])
    return np.asarray(env.data.qpos[qpos_addr : qpos_addr + 7], dtype=np.float64).copy()


def _set_baton_qpos(env, qpos: np.ndarray) -> None:
    qpos = np.asarray(qpos, dtype=np.float64).reshape(7)
    joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "bottle2_freejoint")
    qpos_addr = int(env.model.jnt_qposadr[joint_id])
    env.data.qpos[qpos_addr : qpos_addr + 7] = qpos


def _yaw_quat(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _sample_uniform_pair(rng: np.random.Generator, half_extent: float) -> np.ndarray:
    half_extent = float(half_extent)
    if half_extent <= 0.0:
        return np.zeros(2, dtype=np.float64)
    return rng.uniform(-half_extent, half_extent, size=2).astype(np.float64)


def _apply_initial_randomization(
    env,
    rng: np.random.Generator,
    *,
    enabled: bool,
    robot_xy_half_range: float,
    robot_yaw_half_range: float,
    baton_xy_half_range: float,
    baton_yaw_half_range: float,
) -> dict:
    noise = {
        "enabled": bool(enabled),
        "robot_a_base_delta": [0.0, 0.0, 0.0],
        "robot_b_base_delta": [0.0, 0.0, 0.0],
        "baton_xy_delta": [0.0, 0.0],
        "baton_yaw_delta": 0.0,
    }
    if not enabled:
        return noise

    for robot_index, prefix in enumerate(ROBOT_NAMES):
        dxy = _sample_uniform_pair(rng, robot_xy_half_range)
        dyaw = (
            float(rng.uniform(-robot_yaw_half_range, robot_yaw_half_range))
            if robot_yaw_half_range > 0.0
            else 0.0
        )
        _set_joint_qpos(env, f"{prefix}/mobile_x_joint", _joint_qpos(env, prefix, "mobile_x_joint") + dxy[0])
        _set_joint_qpos(env, f"{prefix}/mobile_y_joint", _joint_qpos(env, prefix, "mobile_y_joint") + dxy[1])
        _set_joint_qpos(
            env,
            f"{prefix}/mobile_theta_joint",
            _joint_qpos(env, prefix, "mobile_theta_joint") + dyaw,
        )
        noise[f"robot_{chr(ord('a') + robot_index)}_base_delta"] = [
            float(dxy[0]),
            float(dxy[1]),
            float(dyaw),
        ]

    baton = _baton_qpos(env)
    baton_xy_delta = _sample_uniform_pair(rng, baton_xy_half_range)
    baton_yaw_delta = (
        float(rng.uniform(-baton_yaw_half_range, baton_yaw_half_range))
        if baton_yaw_half_range > 0.0
        else 0.0
    )
    baton[:2] += baton_xy_delta
    baton[3:7] = _yaw_quat(baton_yaw_delta)
    _set_baton_qpos(env, baton)
    noise["baton_xy_delta"] = [float(baton_xy_delta[0]), float(baton_xy_delta[1])]
    noise["baton_yaw_delta"] = float(baton_yaw_delta)

    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    return noise


def _linearize_depth(env, raw_depth: np.ndarray) -> np.ndarray:
    extent = env.model.stat.extent
    near = env.model.vis.map.znear * extent
    far = env.model.vis.map.zfar * extent
    return near / (1.0 - raw_depth * (1.0 - near / far))


def _render_selected_cameras(env, camera_names: tuple[str, ...]):
    rgb_images = {}
    depth_images = {}
    for camera_name in camera_names:
        camera = env.cameras[camera_name]
        camera["viewer"].make_context_current()
        rgb_images[camera_name] = camera["viewer"].render(
            render_mode="rgb_array",
            camera_id=camera["id"],
        )
        raw_depth = camera["viewer"].render(
            render_mode="depth_array",
            camera_id=camera["id"],
        )
        depth_images[camera_name] = _linearize_depth(env, raw_depth).astype(np.float32)
    return rgb_images, depth_images


def _resize_rgb_nearest(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    y_idx = np.linspace(0, image.shape[0] - 1, int(height)).astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, int(width)).astype(np.int64)
    return image[y_idx][:, x_idx]


def _make_rgb_contact_sheet(
    rgb_images: dict[str, np.ndarray],
    camera_names: tuple[str, ...],
    *,
    tile_width: int = 320,
    tile_height: int = 240,
    columns: int = 3,
) -> np.ndarray:
    label_height = 28
    rows = int(np.ceil(len(camera_names) / float(columns)))
    sheet = np.zeros(
        (rows * (tile_height + label_height), columns * tile_width, 3),
        dtype=np.uint8,
    )

    try:
        import cv2  # noqa: WPS433
    except Exception:
        cv2 = None

    for index, camera_name in enumerate(camera_names):
        row = index // columns
        col = index % columns
        y0 = row * (tile_height + label_height)
        x0 = col * tile_width
        tile = _resize_rgb_nearest(rgb_images[camera_name], tile_width, tile_height)
        sheet[y0 + label_height : y0 + label_height + tile_height, x0 : x0 + tile_width] = tile
        if cv2 is not None:
            cv2.putText(
                sheet,
                camera_name,
                (x0 + 8, y0 + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return sheet


def _save_rgb_png(rgb: np.ndarray, out_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    frame = np.asarray(rgb, dtype=np.uint8)
    try:
        import cv2  # noqa: WPS433

        cv2.imwrite(out_path, frame[:, :, ::-1])
        return
    except Exception:
        pass
    try:
        import imageio.v2 as imageio  # noqa: WPS433

        imageio.imwrite(out_path, frame)
        return
    except Exception as exc:
        raise RuntimeError("Cannot save preview PNG. Install opencv-python or imageio.") from exc


class ReplayImageSampler:
    def __init__(
        self,
        env,
        out_dir: str,
        camera_names: tuple[str, ...],
        every: int,
        preview_every: int,
        show_preview_window: bool,
    ):
        self.out_dir = os.path.abspath(out_dir)
        self.preview_dir = os.path.join(self.out_dir, "previews")
        self.camera_names = tuple(camera_names)
        self.every = max(1, int(every))
        self.preview_every = max(0, int(preview_every))
        self.show_preview_window = bool(show_preview_window)
        self._preview_window_failed = False
        self.step_index = 0
        self.episode_step_index = 0
        self.frame_index = 0
        self.episode_index = 0

        missing = [name for name in self.camera_names if name not in env.cameras]
        if missing:
            available = ", ".join(env.camera_names)
            raise ValueError(
                "Sample camera(s) not found: "
                f"{missing}. Available camera keys: {available}"
            )

        os.makedirs(self.out_dir, exist_ok=True)
        if self.preview_every > 0:
            os.makedirs(self.preview_dir, exist_ok=True)
        self.episode_meta_path = os.path.join(self.out_dir, "episode_metadata.jsonl")
        with open(self.episode_meta_path, "w", encoding="utf-8"):
            pass
        with open(os.path.join(self.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "camera_names": self.camera_names,
                    "sample_every": self.every,
                    "preview_every_sample": self.preview_every,
                    "show_preview_window": self.show_preview_window,
                    "format": "frame_XXXXXX.npz with qpos/qvel/action plus rgb__* and depth__* arrays",
                    "preview_format": "previews/preview_XXXXXX.png RGB contact sheet",
                },
                f,
                indent=2,
            )

    def start_episode(self, episode_index: int, noise: dict) -> None:
        self.episode_index = int(episode_index)
        self.episode_step_index = 0
        with open(self.episode_meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"episode_index": self.episode_index, "noise": noise}) + "\n")

    def maybe_sample(self, env, *, stage_label: str, stage_step: int, action: np.ndarray):
        if self.episode_step_index % self.every != 0:
            self.step_index += 1
            self.episode_step_index += 1
            return

        rgb_images, depth_images = _render_selected_cameras(env, self.camera_names)
        payload = {
            "camera_names": np.asarray(self.camera_names),
            "episode_index": np.asarray(self.episode_index, dtype=np.int64),
            "global_step": np.asarray(self.step_index, dtype=np.int64),
            "episode_step": np.asarray(self.episode_step_index, dtype=np.int64),
            "frame_index": np.asarray(self.frame_index, dtype=np.int64),
            "stage_label": np.asarray(stage_label),
            "stage_step": np.asarray(stage_step, dtype=np.int64),
            "action": np.asarray(action, dtype=np.float32),
            "qpos": np.asarray(env.data.qpos, dtype=np.float32).copy(),
            "qvel": np.asarray(env.data.qvel, dtype=np.float32).copy(),
            "baton_qpos": _baton_qpos(env).astype(np.float32),
        }
        for camera_name in self.camera_names:
            payload[f"rgb__{camera_name}"] = rgb_images[camera_name]
            payload[f"depth__{camera_name}"] = depth_images[camera_name]

        out_path = os.path.join(self.out_dir, f"frame_{self.frame_index:06d}.npz")
        np.savez_compressed(out_path, **payload)
        if self.preview_every > 0 and self.frame_index % self.preview_every == 0:
            sheet = _make_rgb_contact_sheet(rgb_images, self.camera_names)
            preview_path = os.path.join(self.preview_dir, f"preview_{self.frame_index:06d}.png")
            _save_rgb_png(sheet, preview_path)
            self._maybe_show_preview(sheet)
        self.frame_index += 1
        self.step_index += 1
        self.episode_step_index += 1

    def _maybe_show_preview(self, sheet: np.ndarray) -> None:
        if not self.show_preview_window or self._preview_window_failed:
            return
        try:
            import cv2  # noqa: WPS433

            cv2.imshow("ultimate_line_9cam_preview", sheet[:, :, ::-1])
            cv2.waitKey(1)
        except Exception as exc:
            print(f"[Replay] Camera preview window disabled: {exc}")
            self._preview_window_failed = True


class RmbReplaySampler(ReplayImageSampler):
    def __init__(
        self,
        env,
        out_dir: str,
        camera_names: tuple[str, ...],
        every: int,
        preview_every: int,
        show_preview_window: bool,
        demo_name: str,
        task_desc: str,
    ):
        self.demo_name = str(demo_name)
        self.task_desc = str(task_desc)
        self._episode_data = None
        self._episode_noise = None
        super().__init__(
            env,
            out_dir=out_dir,
            camera_names=camera_names,
            every=every,
            preview_every=preview_every,
            show_preview_window=show_preview_window,
        )
        with open(os.path.join(self.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "camera_names": self.camera_names,
                    "sample_every": self.every,
                    "preview_every_sample": self.preview_every,
                    "show_preview_window": self.show_preview_window,
                    "format": "RmbData-Compact episode directories",
                    "episode_pattern": f"{self.demo_name}_world0_XXX.rmb",
                    "preview_format": "previews/preview_XXXXXX.png RGB contact sheet",
                },
                f,
                indent=2,
            )

    def start_episode(self, episode_index: int, noise: dict) -> None:
        super().start_episode(episode_index, noise)
        self._episode_noise = dict(noise)
        self._episode_data = {
            DataKey.TIME: [],
            DataKey.REWARD: [],
            DataKey.MEASURED_JOINT_POS: [],
            DataKey.MEASURED_JOINT_VEL: [],
            DataKey.MEASURED_GRIPPER_JOINT_POS: [],
            DataKey.MEASURED_MOBILE_OMNI_VEL: [],
            DataKey.COMMAND_JOINT_POS: [],
            DataKey.COMMAND_GRIPPER_JOINT_POS: [],
            DataKey.COMMAND_MOBILE_OMNI_VEL: [],
            "global_step": [],
            "episode_step": [],
            "stage_step": [],
            "baton_qpos": [],
        }
        for camera_name in self.camera_names:
            self._episode_data[DataKey.get_rgb_image_key(camera_name)] = []
            self._episode_data[DataKey.get_depth_image_key(camera_name)] = []

    def maybe_sample(self, env, *, stage_label: str, stage_step: int, action: np.ndarray):
        if self.episode_step_index % self.every != 0:
            self.step_index += 1
            self.episode_step_index += 1
            return

        if self._episode_data is None:
            raise RuntimeError("RMB sampler used before start_episode().")

        rgb_images, depth_images = _render_selected_cameras(env, self.camera_names)
        data = self._episode_data
        data[DataKey.TIME].append(float(env.data.time))
        data[DataKey.REWARD].append(0.0)
        data[DataKey.MEASURED_JOINT_POS].append(_all_robot_arm_grip(env).astype(np.float32))
        data[DataKey.MEASURED_JOINT_VEL].append(_all_robot_arm_grip_vel(env).astype(np.float32))
        data[DataKey.MEASURED_GRIPPER_JOINT_POS].append(_all_robot_grip(env).astype(np.float32))
        data[DataKey.MEASURED_MOBILE_OMNI_VEL].append(_all_robot_mobile_vel(env).astype(np.float32))
        data[DataKey.COMMAND_JOINT_POS].append(_command_joint_pos_from_action(action).astype(np.float32))
        data[DataKey.COMMAND_GRIPPER_JOINT_POS].append(_command_gripper_from_action(action).astype(np.float32))
        data[DataKey.COMMAND_MOBILE_OMNI_VEL].append(_command_mobile_from_action(action).astype(np.float32))
        data["global_step"].append(np.asarray(self.step_index, dtype=np.int64))
        data["episode_step"].append(np.asarray(self.episode_step_index, dtype=np.int64))
        data["stage_step"].append(np.asarray(stage_step, dtype=np.int64))
        data["baton_qpos"].append(_baton_qpos(env).astype(np.float32))
        for camera_name in self.camera_names:
            data[DataKey.get_rgb_image_key(camera_name)].append(rgb_images[camera_name])
            data[DataKey.get_depth_image_key(camera_name)].append(depth_images[camera_name])

        if self.preview_every > 0 and self.frame_index % self.preview_every == 0:
            sheet = _make_rgb_contact_sheet(rgb_images, self.camera_names)
            preview_path = os.path.join(self.preview_dir, f"preview_{self.frame_index:06d}.png")
            _save_rgb_png(sheet, preview_path)
            self._maybe_show_preview(sheet)

        self.frame_index += 1
        self.step_index += 1
        self.episode_step_index += 1

    def finish_episode(self, env) -> None:
        if self._episode_data is None:
            return

        filename = os.path.join(
            self.out_dir,
            f"{self.demo_name}_world0_{self.episode_index:03d}.rmb",
        )
        meta_data = {
            "demo_name": self.demo_name,
            "task_desc": self.task_desc,
            "env": env.__class__.__name__,
            "world_idx": 0,
            "episode_idx": self.episode_index,
            "camera_names": list(self.camera_names),
            "rgb_tactile_names": [],
            "pointcloud_camera_names": [],
            "sample_every": self.every,
            "initial_noise_json": json.dumps(self._episode_noise or {}, sort_keys=True),
        }
        for camera_name in self.camera_names:
            depth_key = DataKey.get_depth_image_key(camera_name)
            meta_data[depth_key + "_fovy"] = env.get_camera_fovy(camera_name)

        self._dump_episode_to_rmb_sequential(filename, self._episode_data, meta_data)
        print(f"[Replay] Saved RMB episode: {filename}")
        self._episode_data = None
        self._episode_noise = None

    def _dump_episode_to_rmb_sequential(self, filename: str, all_data_seq: dict, meta_data: dict) -> None:
        """Write compact RMB without DataManager's parallel video copies."""
        import shutil

        import h5py  # noqa: WPS433
        import videoio  # noqa: WPS433

        if os.path.isdir(filename):
            shutil.rmtree(filename)
        os.makedirs(filename, exist_ok=True)

        hdf5_filename = os.path.join(filename, "main.rmb.hdf5")
        with h5py.File(hdf5_filename, "w") as h5file:
            for key in list(all_data_seq.keys()):
                seq = all_data_seq[key]
                if DataKey.is_rgb_image_key(key):
                    video_filename = os.path.join(filename, f"{key}.rmb.mp4")
                    images = np.asarray(seq, dtype=np.uint8)
                    videoio.videosave(video_filename, images)
                elif DataKey.is_depth_image_key(key):
                    video_filename = os.path.join(filename, f"{key}.rmb.mp4")
                    images = (1e3 * np.asarray(seq, dtype=np.float32)).astype(np.uint16)
                    videoio.uint16save(video_filename, images)
                else:
                    h5file.create_dataset(key, data=np.asarray(seq))
                all_data_seq[key] = []

            for key, value in meta_data.items():
                if value is None:
                    value = ""
                h5file.attrs[key] = value
            h5file.attrs["format"] = "RmbData-Compact"


def _fmt(values, precision: int = 4) -> str:
    return np.array2string(np.asarray(values, dtype=np.float64), precision=precision, suppress_small=False)


def _current_arm_grip(env):
    return [
        {
            "arm": _robot_arm(env, robot_index),
            "grip": _robot_grip(env, robot_index),
        }
        for robot_index in range(2)
    ]


def _yaw_error(target: float, current: float) -> float:
    return float((target - current + np.pi) % (2.0 * np.pi) - np.pi)


def _base_command_to_target(current: np.ndarray, target: np.ndarray, kp_pos: float, kp_yaw: float) -> np.ndarray:
    err_xy = target[:2] - current[:2]
    yaw = current[2]
    v_world = kp_pos * err_xy
    v_local = np.array(
        [
            v_world[0] * np.cos(yaw) + v_world[1] * np.sin(yaw),
            -v_world[0] * np.sin(yaw) + v_world[1] * np.cos(yaw),
        ],
        dtype=np.float64,
    )
    v_local = np.clip(v_local, -0.45, 0.45)
    v_yaw = np.clip(kp_yaw * _yaw_error(float(target[2]), float(current[2])), -0.55, 0.55)
    return np.array([v_local[0], v_local[1], v_yaw], dtype=np.float64)


def _advance_arm_command_to_target(
    command: np.ndarray,
    actual: np.ndarray,
    previous_desired: np.ndarray,
    desired: np.ndarray,
    integral_gain: float,
    integral_step_clip: float,
) -> np.ndarray:
    """Continuous arm command with integral correction, matching the ultimate-line anti-sag idea."""
    feedforward = desired - previous_desired
    err = desired - actual
    correction = np.clip(
        float(integral_gain) * err,
        -float(integral_step_clip),
        float(integral_step_clip),
    )
    next_command = command + feedforward + correction
    return np.clip(next_command, ARM_JOINT_LIMITS[:, 0], ARM_JOINT_LIMITS[:, 1])


def _stage_targets_for_current_baton(env, align_pick_to_current_baton: bool):
    stages = deepcopy(KEYFRAMES)
    baton_xy = env.data.body("bottle2").xpos[:2].copy()
    pick_delta_xy = baton_xy - NOMINAL_INITIAL_BATON_XY
    if align_pick_to_current_baton:
        for idx in (0, 1):
            stages[idx]["robot_a"]["base_xy_yaw"][:2] += pick_delta_xy
    return stages, pick_delta_xy


def _print_nominal_relations(stages, pick_delta_xy):
    k1 = stages[0]
    baton_xy = (NOMINAL_INITIAL_BATON_XY + pick_delta_xy).astype(np.float64)
    a_base_xy = k1["robot_a"]["base_xy_yaw"][:2]
    rel = baton_xy - a_base_xy
    print("[Replay] Pick relation: baton_xy - A_base_xy =", np.array2string(rel, precision=4))
    print("[Replay] Applied initial baton XY delta =", np.array2string(pick_delta_xy, precision=4))
    print(
        "[Replay] Handover relation: "
        "B_base_xy - A_base_xy at stage 5 =",
        np.array2string(stages[4]["robot_b"]["base_xy_yaw"][:2] - stages[4]["robot_a"]["base_xy_yaw"][:2], precision=4),
    )


def _run_stage(
    env,
    stage: dict,
    steps: int,
    *,
    kp_pos: float,
    kp_yaw: float,
    arm_feedback_gain: float,
    arm_feedback_clip: float,
    hold_steps: int,
    print_every: int,
    seed_prev_action_18: np.ndarray | None = None,
    sampler: ReplayImageSampler | None = None,
):
    print(f"\n[Stage] {stage['label']} ({steps} steps)")
    start = _current_arm_grip(env)
    target_base = [stage[prefix]["base_xy_yaw"] for prefix in ROBOT_NAMES]
    target_arm = [stage[prefix]["arm_joints"] for prefix in ROBOT_NAMES]
    target_grip = [stage[prefix]["grip"] for prefix in ROBOT_NAMES]
    arm_cmd = []
    if seed_prev_action_18 is not None:
        seed = np.asarray(seed_prev_action_18, dtype=np.float64).reshape(18)
        for robot_index in range(2):
            offset = robot_index * 9
            arm_cmd.append(seed[offset + 3 : offset + 8].copy())
    else:
        arm_cmd = [start[robot_index]["arm"].copy() for robot_index in range(2)]
    previous_desired_arm = [start[robot_index]["arm"].copy() for robot_index in range(2)]
    last_action = np.zeros(18, dtype=np.float64)

    for t in range(max(1, steps)):
        alpha = min(1.0, float(t + 1) / float(max(1, steps)))
        # Smooth arm/gripper commands, while base uses feedback to reach the final pose.
        action = np.zeros(18, dtype=np.float64)
        for robot_index in range(2):
            offset = robot_index * 9
            base_now = _robot_base(env, robot_index)
            action[offset : offset + 3] = _base_command_to_target(
                base_now, target_base[robot_index], kp_pos, kp_yaw
            )
            desired_arm = (
                (1.0 - alpha) * start[robot_index]["arm"] + alpha * target_arm[robot_index]
            )
            arm_cmd[robot_index] = _advance_arm_command_to_target(
                arm_cmd[robot_index],
                _robot_arm(env, robot_index),
                previous_desired_arm[robot_index],
                desired_arm,
                arm_feedback_gain,
                arm_feedback_clip,
            )
            previous_desired_arm[robot_index] = desired_arm
            action[offset + 3 : offset + 8] = arm_cmd[robot_index]
            action[offset + 8] = (1.0 - alpha) * start[robot_index]["grip"] + alpha * target_grip[robot_index]

        last_action = action.copy()
        env.step(action)
        if sampler is not None:
            sampler.maybe_sample(env, stage_label=stage["label"], stage_step=t + 1, action=action)
        if print_every > 0 and (t + 1) % print_every == 0:
            _print_stage_error(env, stage, t + 1, steps)

    # Let the position controllers settle at the final command.
    for hold_idx in range(max(0, hold_steps)):
        action = np.zeros(18, dtype=np.float64)
        for robot_index in range(2):
            offset = robot_index * 9
            base_now = _robot_base(env, robot_index)
            action[offset : offset + 3] = _base_command_to_target(
                base_now, target_base[robot_index], kp_pos, kp_yaw
            )
            arm_cmd[robot_index] = _advance_arm_command_to_target(
                arm_cmd[robot_index],
                _robot_arm(env, robot_index),
                previous_desired_arm[robot_index],
                target_arm[robot_index],
                arm_feedback_gain,
                arm_feedback_clip,
            )
            previous_desired_arm[robot_index] = target_arm[robot_index]
            action[offset + 3 : offset + 8] = arm_cmd[robot_index]
            action[offset + 8] = target_grip[robot_index]
        last_action = action.copy()
        env.step(action)
        if sampler is not None:
            sampler.maybe_sample(
                env,
                stage_label=f"{stage['label']}_hold",
                stage_step=steps + hold_idx + 1,
                action=action,
            )

    _print_stage_error(env, stage, steps, steps)
    _print_stage_summary(env, stage)
    return last_action


def _print_stage_error(env, stage: dict, step: int, steps: int):
    parts = []
    for robot_index, name in enumerate(("A", "B")):
        current = _robot_base(env, robot_index)
        target = stage[ROBOT_NAMES[robot_index]]["base_xy_yaw"]
        pos_err = float(np.linalg.norm(target[:2] - current[:2]))
        yaw_err = abs(_yaw_error(float(target[2]), float(current[2])))
        parts.append(f"{name}:xy_err={pos_err:.3f},yaw_err={yaw_err:.3f}")
    baton = env.data.body("bottle2").xpos.copy()
    print(f"[Stage]   step {step}/{steps}  {'  '.join(parts)}  baton={np.array2string(baton, precision=3)}")


def _print_stage_summary(env, stage: dict):
    print(f"[Compare] {stage['label']} final actual vs captured target")
    for robot_index, robot_name in enumerate(("A", "B")):
        prefix = ROBOT_NAMES[robot_index]
        actual_base = _robot_base(env, robot_index)
        target_base = stage[prefix]["base_xy_yaw"]
        base_delta = actual_base - target_base
        base_delta[2] = _yaw_error(float(target_base[2]), float(actual_base[2])) * -1.0

        actual_arm = _robot_arm(env, robot_index)
        target_arm = stage[prefix]["arm_joints"]
        arm_delta = actual_arm - target_arm
        actual_grip = _robot_grip(env, robot_index)
        target_grip = float(stage[prefix]["grip"])

        print(f"[Compare]   Robot {robot_name} base target={_fmt(target_base)}")
        print(f"[Compare]   Robot {robot_name} base actual={_fmt(actual_base)} delta={_fmt(base_delta)}")
        print(f"[Compare]   Robot {robot_name} arm  target={_fmt(target_arm)}")
        print(
            f"[Compare]   Robot {robot_name} arm  actual={_fmt(actual_arm)} "
            f"delta={_fmt(arm_delta)} max_abs={float(np.max(np.abs(arm_delta))):.5f}"
        )
        print(
            f"[Compare]   Robot {robot_name} grip target={target_grip:.4f} "
            f"actual={actual_grip:.4f} delta={actual_grip - target_grip:+.5f}"
        )

    if stage.get("baton") is not None:
        actual_baton = _baton_qpos(env)
        target_baton = np.asarray(stage["baton"], dtype=np.float64)
        baton_delta = actual_baton - target_baton
        print(f"[Compare]   baton qpos target={_fmt(target_baton)}")
        print(
            f"[Compare]   baton qpos actual={_fmt(actual_baton)} "
            f"delta_xyz={_fmt(baton_delta[:3])}"
        )
    st = getattr(env, "_baton_grip_stabilizer", {})
    if st and st.get("enabled", False):
        print(f"[Compare]   stabilizer active_robot={st.get('active_robot')}")


def _configure_replay_stabilizer(env, args) -> None:
    if args.no_baton_grip_stabilizer:
        env.disable_baton_grip_stabilizer()
        return

    env.configure_baton_grip_stabilizer(
        enabled=True,
        robot_index=None,
        max_palm_dist=0.15,
        attach_grip_qpos=0.22,
        release_grip_qpos=0.3,
        verbose=bool(args.baton_grip_stabilizer_verbose),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay captured ultimate-line keyframes.")
    parser.add_argument("--no_render", action="store_true", help="Run without interactive viewer.")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of replay episodes to run.")
    parser.add_argument("--random_seed", type=int, default=0, help="Seed for initial-state randomization.")
    parser.add_argument("--settle_steps", type=int, default=50)
    parser.add_argument("--stage_step_scale", type=float, default=1.0)
    parser.add_argument(
        "--hold_steps",
        type=int,
        default=18,
        help="Post-interpolation settle steps per stage (reduces dead time for data).",
    )
    parser.add_argument("--kp_pos", type=float, default=2.0)
    parser.add_argument("--kp_yaw", type=float, default=2.0)
    parser.add_argument(
        "--arm_feedback_gain",
        type=float,
        default=0.07,
        help="Integral anti-sag gain, same idea as ultimate_line_demo_lib arm lock.",
    )
    parser.add_argument(
        "--arm_feedback_clip",
        type=float,
        default=0.004,
        help="Per-step per-joint max absolute correction for the arm anti-sag integrator.",
    )
    parser.add_argument("--print_every", type=int, default=0)
    parser.add_argument(
        "--list_cameras",
        action="store_true",
        help="Print available offscreen camera keys and exit.",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default=None,
        help="If set, save replay RGB+Depth samples in this directory.",
    )
    parser.add_argument(
        "--sample_format",
        choices=("rmb", "npz"),
        default="rmb",
        help="Save standard RMB episode directories by default; use npz only for frame-level debugging.",
    )
    parser.add_argument(
        "--demo_name",
        type=str,
        default="ultimate_line_9cam",
        help="Demo name prefix for RMB episode directories.",
    )
    parser.add_argument(
        "--task_desc",
        type=str,
        default="dual HSR ultimate-line baton relay",
        help="Task description stored in RMB metadata.",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=10,
        help="Save one RGB+Depth sample every N replay control steps when --sample_dir is set.",
    )
    parser.add_argument(
        "--sample_preview_every",
        type=int,
        default=1,
        help="Save one RGB contact-sheet PNG every N sampled frames; 0 disables previews.",
    )
    parser.add_argument(
        "--no_camera_preview_window",
        action="store_true",
        help="Do not show the live 9-camera RGB contact-sheet window while sampling.",
    )
    parser.add_argument(
        "--sample_cameras",
        nargs="+",
        default=list(DEFAULT_SAMPLE_CAMERA_NAMES),
        help="Camera keys to sample. Defaults to the fixed 8 robot cameras plus ultimate_line_god.",
    )
    parser.add_argument(
        "--no_align_pick_to_current_baton",
        action="store_true",
        help="Do not shift the first two A pick waypoints by the current initial baton XY.",
    )
    parser.add_argument(
        "--randomize_initial_state",
        action="store_true",
        help="Apply conservative initial noise to robot bases and baton pose before replay.",
    )
    parser.add_argument(
        "--robot_xy_noise",
        type=float,
        default=0.08,
        help="Uniform half-range in meters for each robot initial base x/y when randomized.",
    )
    parser.add_argument(
        "--robot_yaw_noise",
        type=float,
        default=0.08,
        help="Uniform half-range in radians for each robot initial yaw when randomized.",
    )
    parser.add_argument(
        "--baton_xy_noise",
        type=float,
        default=0.06,
        help="Uniform half-range in meters for baton initial x/y when randomized.",
    )
    parser.add_argument(
        "--baton_yaw_noise",
        type=float,
        default=0.35,
        help="Uniform half-range in radians for baton initial yaw when randomized.",
    )
    parser.add_argument("--no_baton_grip_stabilizer", action="store_true")
    parser.add_argument("--baton_grip_stabilizer_verbose", action="store_true")
    args = parser.parse_args()

    render_mode = "rgb_array" if args.no_render else "human"
    print("[Replay] Loading MujocoDualHsrUltimateLineEnv...")
    env = MujocoDualHsrUltimateLineEnv(render_mode=render_mode)
    rng = np.random.default_rng(int(args.random_seed))
    if args.list_cameras:
        print("[Replay] Available camera keys:")
        for camera_name in env.camera_names:
            print(f"  {camera_name}")
        env.close()
        return

    sampler = None
    if args.sample_dir:
        sampler_cls = RmbReplaySampler if args.sample_format == "rmb" else ReplayImageSampler
        sampler_kwargs = {}
        if args.sample_format == "rmb":
            sampler_kwargs = {
                "demo_name": str(args.demo_name),
                "task_desc": str(args.task_desc),
            }
        sampler = sampler_cls(
            env,
            out_dir=args.sample_dir,
            camera_names=tuple(args.sample_cameras),
            every=int(args.sample_every),
            preview_every=int(args.sample_preview_every),
            show_preview_window=not args.no_camera_preview_window,
            **sampler_kwargs,
        )
        print(
            "[Replay] Sampling RGB+Depth every "
            f"{sampler.every} step(s) to {sampler.out_dir}"
        )
        print(f"[Replay] Sample format: {args.sample_format}")
        print("[Replay] Sample cameras:", ", ".join(sampler.camera_names))
        if sampler.preview_every > 0:
            print(f"[Replay] RGB contact sheets: {sampler.preview_dir}")
        if sampler.show_preview_window:
            print("[Replay] Live camera preview window: ultimate_line_9cam_preview")
        if not args.no_render:
            print("[Replay] Note: sampling with the interactive viewer is slower; add --no_render for batch collection.")

    num_episodes = max(1, int(args.num_episodes))
    if num_episodes > 1 and not args.randomize_initial_state:
        print("[Replay] Note: --num_episodes > 1 without --randomize_initial_state repeats the same trajectory.")

    try:
        for episode_index in range(num_episodes):
            print(f"\n[Episode] {episode_index + 1}/{num_episodes}")
            env.reset()
            _configure_replay_stabilizer(env, args)
            if hasattr(env, "unlock_gripper"):
                env.unlock_gripper()

            noise = _apply_initial_randomization(
                env,
                rng,
                enabled=bool(args.randomize_initial_state),
                robot_xy_half_range=float(args.robot_xy_noise),
                robot_yaw_half_range=float(args.robot_yaw_noise),
                baton_xy_half_range=float(args.baton_xy_noise),
                baton_yaw_half_range=float(args.baton_yaw_noise),
            )
            if sampler is not None:
                sampler.start_episode(episode_index, noise)
            if noise["enabled"]:
                print("[Replay] Initial noise:", json.dumps(noise, sort_keys=True))

            for _ in range(max(0, args.settle_steps)):
                env.step(env.get_hold_action())
            mujoco.mj_forward(env.model, env.data)

            stages, pick_delta_xy = _stage_targets_for_current_baton(
                env,
                align_pick_to_current_baton=not args.no_align_pick_to_current_baton,
            )
            _print_nominal_relations(stages, pick_delta_xy)

            last_action_18 = None
            for stage in stages:
                steps = int(round(DEFAULT_STAGE_STEPS[stage["label"]] * float(args.stage_step_scale)))
                last_action_18 = _run_stage(
                    env,
                    stage,
                    max(1, steps),
                    kp_pos=float(args.kp_pos),
                    kp_yaw=float(args.kp_yaw),
                    arm_feedback_gain=float(args.arm_feedback_gain),
                    arm_feedback_clip=float(args.arm_feedback_clip),
                    hold_steps=int(args.hold_steps),
                    print_every=int(args.print_every),
                    seed_prev_action_18=last_action_18,
                    sampler=sampler,
                )
            if sampler is not None and hasattr(sampler, "finish_episode"):
                sampler.finish_episode(env)
        if sampler is not None:
            print(f"\n[Replay] Done. Saved {sampler.frame_index} sampled frame(s).")
        else:
            print("\n[Replay] Done.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
