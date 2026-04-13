"""Shared navigation + dual ManiFlow skill execution for ultimate-line HSR demo."""

from __future__ import annotations

import mujoco
import numpy as np

from handover_utils import build_dual_hold_targets_from_current, dual_hold_action_from_targets

# OpenCV preview window names (destroyed between policy phases).
_CV_PREVIEW_WINDOWS = []

_PICK_CAMERA_MAP = {
    "side_view": "pick_side_view",
    "overhead_view": "pick_overhead_view",
}
_TIGHT_GRIP_CMD = -0.30

# Outer-loop correction: cmd += gain * (q_ref - q_meas) each step to fight gravity sag
# while keeping q_ref = joint target at segment start (not pure measured tracking).
# Keep moderate — too large causes visible "lift"; handover A uses **no** integral (see below).
_NAV_ARM_LOCK_INTEGRAL_GAIN = 0.07
_SKILL_IDLE_ARM_INTEGRAL_GAIN = 0.06


def _refresh_locked_grip_from_qpos(env, locked, force_tight_grip_robots):
    ft = set(force_tight_grip_robots)
    for i, pfx in enumerate(["robot_a", "robot_b"]):
        gaddr = env.model.jnt_qposadr[
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{pfx}/hand_motor_joint")
        ]
        g = float(env.data.qpos[gaddr])
        if i in ft:
            locked[i]["grip"] = min(g - 0.18, _TIGHT_GRIP_CMD)
        elif g < 0.6:
            locked[i]["grip"] = g - 0.1
        else:
            locked[i]["grip"] = g


def _refresh_arm_lock_integral(env, locked, locked_ref_arm, per_robot_arm_addrs, i_gain: float):
    """Adjust commanded arm targets toward segment reference vs measured qpos."""
    max_step = 0.004  # rad / m per step — limits aggressive corrections
    for i in (0, 1):
        for j, addr in enumerate(per_robot_arm_addrs[i]):
            q = float(env.data.qpos[addr])
            e = float(locked_ref_arm[i, j]) - q
            delta = i_gain * e
            if delta > max_step:
                delta = max_step
            elif delta < -max_step:
                delta = -max_step
            locked[i]["arm"][j] = float(locked[i]["arm"][j]) + delta

# World-camera routing for logical keys side_view / overhead_view (see CAMERA_AND_POLICY_ALIGNMENT.txt).
# "pick"    -> pick_* (baton-centered, side_pick training in dual)
# "handover"-> dual side_view + overhead_view — **only used if** checkpoint meta lists them
# "place"   -> place_side_view + place_overhead_view (same relative offset from target as tidyup_place)
#
# Handover **demos** (collect_handover_demos.py) record only robot B ``head`` + ``hand``; the policy
# still uses whatever ``model_meta_info["image"]["camera_names"]`` says (often 2, sometimes 4 if trained
# with extra world cameras).


def _world_camera_physical_name(cam_key: str, cam_mode: str) -> str:
    if cam_key not in ("side_view", "overhead_view"):
        raise ValueError(cam_key)
    if cam_mode == "pick":
        return _PICK_CAMERA_MAP[cam_key]
    if cam_mode == "place":
        if cam_key == "side_view":
            return "place_side_view"
        return "place_overhead_view"
    if cam_mode == "handover":
        return cam_key
    raise ValueError(f"Unknown cam_mode={cam_mode!r}")


def robot_base_xy_yaw(env, robot_index: int):
    """World (x, y, yaw) for robot A (0) or B (1) base joints."""
    prefix = "robot_a" if robot_index == 0 else "robot_b"
    names = [
        f"{prefix}/mobile_x_joint",
        f"{prefix}/mobile_y_joint",
        f"{prefix}/mobile_theta_joint",
    ]
    return (
        float(env.data.joint(names[0]).qpos[0]),
        float(env.data.joint(names[1]).qpos[0]),
        float(env.data.joint(names[2]).qpos[0]),
    )


def navigate_to_dual(
    env,
    robot_index,
    target_xy,
    target_yaw=0.0,
    max_steps=1200,
    kp_pos=2.0,
    kp_yaw=2.0,
    force_tight_grip_robots=(),
    lock_arm_to_init=False,
    yaw_gate: float = 0.0,
    post_arrival_hold_steps: int = 8,
):
    robot_name = "A" if robot_index == 0 else "B"
    print(
        f"\n[Navigation] Robot {robot_name} -> "
        f"xy=({target_xy[0]:.3f},{target_xy[1]:.3f}) yaw={target_yaw:.3f}"
    )

    prefix = "robot_a" if robot_index == 0 else "robot_b"
    base_joints = [
        f"{prefix}/mobile_x_joint",
        f"{prefix}/mobile_y_joint",
        f"{prefix}/mobile_theta_joint",
    ]
    jnt_ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in base_joints]
    qpos_adrs = [env.model.jnt_qposadr[j] for j in jnt_ids]

    arm_joint_names = [
        "arm_lift_joint",
        "arm_flex_joint",
        "arm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ]

    force_tight_set = set(force_tight_grip_robots)
    locked = {}
    for i, pfx in enumerate(["robot_a", "robot_b"]):
        offset = i * 9
        arm_addrs = [
            env.model.jnt_qposadr[
                mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{pfx}/{j}")
            ]
            for j in arm_joint_names
        ]
        if lock_arm_to_init:
            locked_arm = [env.init_qpos[addr] for addr in arm_addrs]
        else:
            locked_arm = [env.data.qpos[addr] for addr in arm_addrs]

        locked_grip_addr = env.model.jnt_qposadr[
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{pfx}/hand_motor_joint")
        ]
        locked_grip = env.data.qpos[locked_grip_addr]

        if i in force_tight_set:
            locked_grip = min(locked_grip - 0.18, _TIGHT_GRIP_CMD)
        elif locked_grip < 0.6:
            locked_grip -= 0.1

        locked[i] = {"arm": locked_arm, "grip": locked_grip, "offset": offset}

    per_robot_arm_addrs = [
        [
            env.model.jnt_qposadr[
                mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{pfx}/{j}")
            ]
            for j in arm_joint_names
        ]
        for pfx in ["robot_a", "robot_b"]
    ]
    locked_ref_arm = np.zeros((2, 5), dtype=np.float64)
    for i in (0, 1):
        for j in range(5):
            locked_ref_arm[i, j] = float(locked[i]["arm"][j])

    for step in range(max_steps):
        current_x = env.data.qpos[qpos_adrs[0]]
        current_y = env.data.qpos[qpos_adrs[1]]
        current_yaw = env.data.qpos[qpos_adrs[2]]

        err_x = target_xy[0] - current_x
        err_y = target_xy[1] - current_y
        err_yaw = target_yaw - current_yaw
        err_yaw = (err_yaw + np.pi) % (2 * np.pi) - np.pi

        dist_err = np.sqrt(err_x**2 + err_y**2)
        yaw_err_abs = np.abs(err_yaw)

        # Optional: rotate toward target_yaw before translating (reduces "crab" motion).
        if yaw_gate > 0.0 and yaw_err_abs > yaw_gate:
            v_x_local = v_y_local = 0.0
            v_theta_local = kp_yaw * err_yaw
            v_theta_local = np.clip(v_theta_local, -0.8, 0.8)
            _refresh_arm_lock_integral(
                env,
                locked,
                locked_ref_arm,
                per_robot_arm_addrs,
                _NAV_ARM_LOCK_INTEGRAL_GAIN,
            )
            _refresh_locked_grip_from_qpos(env, locked, force_tight_grip_robots)
            action = np.zeros(18)
            idx_offset = robot_index * 9
            action[idx_offset : idx_offset + 3] = [0.0, 0.0, v_theta_local]
            for i in range(2):
                o = locked[i]["offset"]
                action[o + 3 : o + 8] = locked[i]["arm"]
                action[o + 8] = locked[i]["grip"]
            env.step(action)
            continue

        if dist_err < 0.005 and yaw_err_abs < 0.01:
            print(f"[Navigation] Robot {robot_name} arrived in {step} steps.")
            mujoco.mj_forward(env.model, env.data)
            # Re-snap arm+grip targets to **measured** qpos so PD does not fight stale nav-start
            # lock (reduces end-of-nav sag / jerk).
            snap = build_dual_hold_targets_from_current(env, force_tight_grip_robots)
            for _ in range(max(0, int(post_arrival_hold_steps))):
                env.step(dual_hold_action_from_targets(snap))
            return True

        v_x_global = kp_pos * err_x
        v_y_global = kp_pos * err_y
        v_x_local = v_x_global * np.cos(current_yaw) + v_y_global * np.sin(current_yaw)
        v_y_local = -v_x_global * np.sin(current_yaw) + v_y_global * np.cos(current_yaw)
        v_theta_local = kp_yaw * err_yaw

        v_local_xy = np.clip([v_x_local, v_y_local], -0.6, 0.6)
        v_theta_local = np.clip(v_theta_local, -0.8, 0.8)

        _refresh_arm_lock_integral(
            env,
            locked,
            locked_ref_arm,
            per_robot_arm_addrs,
            _NAV_ARM_LOCK_INTEGRAL_GAIN,
        )
        _refresh_locked_grip_from_qpos(env, locked, force_tight_grip_robots)
        action = np.zeros(18)
        idx_offset = robot_index * 9
        action[idx_offset : idx_offset + 3] = [v_local_xy[0], v_local_xy[1], v_theta_local]

        for i in range(2):
            o = locked[i]["offset"]
            action[o + 3 : o + 8] = locked[i]["arm"]
            action[o + 8] = locked[i]["grip"]

        env.step(action)

    print("[Navigation] WARNING: max steps reached.")
    return False


def navigate_to_xy_world_line_then_along_x(
    env,
    robot_index,
    end_xy_world,
    target_yaw=0.0,
    *,
    two_phase_threshold=0.04,
    yaw_gate=0.06,
    max_steps=1200,
    kp_pos=2.0,
    kp_yaw=2.0,
    force_tight_grip_robots=(),
):
    """
    Drive to end_xy_world with heading target_yaw, staying on world y = end_xy_world[1].

    If the base is off that y line, first move laterally (same x) to the line, then along +/-
    world X. With yaw ~0 this matches "face +X and move on one straight line" in the
    ultimate-line layout (goal/handover on y≈0).
    """
    robot_name = "A" if robot_index == 0 else "B"
    end_xy_world = [float(end_xy_world[0]), float(end_xy_world[1])]
    bx, by, _ = robot_base_xy_yaw(env, robot_index)
    ly = end_xy_world[1]
    ex = end_xy_world[0]
    if abs(by - ly) > two_phase_threshold:
        print(
            f"[Navigation] Robot {robot_name} line approach: lateral to y={ly:.4f} "
            f"(from by={by:.4f})"
        )
        navigate_to_dual(
            env,
            robot_index,
            [bx, ly],
            target_yaw=target_yaw,
            max_steps=max_steps,
            kp_pos=kp_pos,
            kp_yaw=kp_yaw,
            force_tight_grip_robots=force_tight_grip_robots,
            yaw_gate=yaw_gate,
        )
    print(
        f"[Navigation] Robot {robot_name} line approach: along x to "
        f"({ex:.4f},{ly:.4f}) yaw={target_yaw:.3f}"
    )
    navigate_to_dual(
        env,
        robot_index,
        [ex, ly],
        target_yaw=target_yaw,
        max_steps=max_steps,
        kp_pos=kp_pos,
        kp_yaw=kp_yaw,
        force_tight_grip_robots=force_tight_grip_robots,
        yaw_gate=yaw_gate,
    )


def _physical_rgb_source_for_policy_cam(cam_name: str, prefix: str, cam_mode: str) -> str:
    """MuJoCo / get_images dict key used to fill policy logical name ``cam_name``."""
    if cam_name in ("head", "hand"):
        return f"{prefix}_{cam_name}"
    if cam_name in ("side_view", "overhead_view"):
        return _world_camera_physical_name(cam_name, cam_mode)
    return cam_name


def _map_dual_images_to_single(
    env,
    prefix,
    cam_mode: str = "handover",
    *,
    policy_camera_names=None,
):
    """Map dual-env rgb keys to names expected by ManiFlow (``policy_camera_names`` from checkpoint).

    If ``policy_camera_names`` is None, fills the full legacy set head, hand, side_view, overhead_view.
    Otherwise only fills keys listed in the checkpoint meta (e.g. handover often only head + hand).
    """
    dual_images = env.get_images()
    mapped_info = {"rgb_images": {}, "depth_images": {}}
    if policy_camera_names:
        names = list(policy_camera_names)
    else:
        names = ["head", "hand", "side_view", "overhead_view"]

    for cam_name in names:
        src_key = _physical_rgb_source_for_policy_cam(cam_name, prefix, cam_mode)
        if "rgb_images" in dual_images and src_key in dual_images["rgb_images"]:
            mapped_info["rgb_images"][cam_name] = dual_images["rgb_images"][src_key]
        if "depth_images" in dual_images and src_key in dual_images["depth_images"]:
            mapped_info["depth_images"][cam_name] = dual_images["depth_images"][src_key]
    return mapped_info


def preview_cameras_close() -> None:
    """Close all ``--camera_preview`` OpenCV windows (call between policy phases or on exit)."""
    try:
        import cv2
    except ImportError:
        _CV_PREVIEW_WINDOWS.clear()
        return
    for w in list(_CV_PREVIEW_WINDOWS):
        try:
            cv2.destroyWindow(w)
        except Exception:
            pass
    _CV_PREVIEW_WINDOWS.clear()


def preview_cameras_show(
    env,
    mode: str | None,
    mapped_info: dict,
    phase_tag: str,
    *,
    max_edge: int = 360,
) -> None:
    """Show RGB previews during policy steps. ``mode`` is ``policy`` or ``all``."""
    if not mode or mode in ("none", ""):
        return
    try:
        import cv2
    except ImportError:
        return
    tag = phase_tag.replace("/", "_")
    if mode == "policy":
        rgb_map = dict(mapped_info.get("rgb_images", {}))
        title = "policy_input"
    elif mode == "all":
        rgb_map = dict(env.get_images().get("rgb_images", {}))
        title = "env_all"
    else:
        return
    for name in sorted(rgb_map.keys()):
        img = rgb_map[name]
        if img is None or getattr(img, "size", 0) == 0:
            continue
        h, w = img.shape[:2]
        scale = min(1.0, float(max_edge) / max(float(h), float(w)))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        win = f"ult_{title}_{tag}_{name}"
        cv2.imshow(win, bgr)
        if win not in _CV_PREVIEW_WINDOWS:
            _CV_PREVIEW_WINDOWS.append(win)
    cv2.waitKey(1)


def _print_checkpoint_camera_routing(planner, prefix: str, cam_mode: str) -> None:
    ck = getattr(planner, "camera_names", None) or []
    if not ck:
        print("[Cameras] checkpoint lists no image keys (e.g. pointcloud policy).")
        return
    parts = []
    for cn in ck:
        src = _physical_rgb_source_for_policy_cam(cn, prefix, cam_mode)
        parts.append(f"{cn}<-{src}")
    print(f"[Cameras] mode={cam_mode}  policy uses {len(ck)} stream(s): " + ", ".join(parts))


def execute_skill_dual(
    env,
    planner,
    skill_name,
    robot_index,
    max_steps=60,
    *,
    camera_preview=None,
):
    robot_name = "A" if robot_index == 0 else "B"
    print(f"\n[Skill] Robot {robot_name} '{skill_name}' ({max_steps} steps)")

    preview_cameras_close()
    obs = env._get_obs()
    prefix = "robot_a" if robot_index == 0 else "robot_b"
    if skill_name in ("pick", "side_pick"):
        cam_mode = "pick"
    elif skill_name in ("place", "side_place", "tidyup_place"):
        cam_mode = "place"
    else:
        cam_mode = "handover"
    _print_checkpoint_camera_routing(planner, prefix, cam_mode)
    planner.reset_buffers()

    arm_joint_names = [
        "arm_lift_joint",
        "arm_flex_joint",
        "arm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ]
    idx_offset = robot_index * 9
    grip_addr = env.model.jnt_qposadr[
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/hand_motor_joint")
    ]

    idle_idx = 1 - robot_index
    idle_prefix = "robot_b" if robot_index == 0 else "robot_a"
    idle_offset = idle_idx * 9
    idle_arm_addrs = [
        env.model.jnt_qposadr[
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{idle_prefix}/{j}")
        ]
        for j in arm_joint_names
    ]
    idle_grip_addr = env.model.jnt_qposadr[
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{idle_prefix}/hand_motor_joint")
    ]

    # Idle robot: reference pose at skill start + integral correction (reduces sag vs fixed cmd).
    idle_arm_ref = np.array([float(env.data.qpos[addr]) for addr in idle_arm_addrs], dtype=np.float64)
    idle_arm_cmd = idle_arm_ref.copy()

    ck_cams = getattr(planner, "camera_names", None)
    phase_tag = f"{robot_name}_{skill_name}"
    for _step in range(max_steps):
        info = _map_dual_images_to_single(
            env, prefix, cam_mode, policy_camera_names=ck_cams
        )
        preview_cameras_show(env, camera_preview, info, phase_tag)
        single_obs = {
            "joint_pos": obs[f"{prefix}/joint_pos"],
            "mobile_vel": obs[f"{prefix}/mobile_vel"],
        }
        single_action = planner.get_action(single_obs, info)

        q_idle = np.array([float(env.data.qpos[addr]) for addr in idle_arm_addrs], dtype=np.float64)
        d_idle = _SKILL_IDLE_ARM_INTEGRAL_GAIN * (idle_arm_ref - q_idle)
        d_idle = np.clip(d_idle, -0.004, 0.004)
        idle_arm_cmd = idle_arm_cmd + d_idle
        g_idle = float(env.data.qpos[idle_grip_addr])
        idle_g_cmd = g_idle - 0.1 if g_idle < 0.6 else g_idle

        full_action = np.zeros(18)
        full_action[idx_offset : idx_offset + 9] = single_action
        full_action[idle_offset + 3 : idle_offset + 8] = idle_arm_cmd
        full_action[idle_offset + 8] = idle_g_cmd

        obs, _r, _t, _tr, _i = env.step(full_action)

    preview_cameras_close()
    print(
        f"[Skill] Robot {robot_name} done. grip_qpos={float(env.data.qpos[grip_addr]):.4f}"
    )


def execute_handover_b_release_a_when_closed(
    env,
    planner,
    skill_name,
    max_steps=90,
    b_grip_takeover_qpos=0.22,
    release_dwell_steps=12,
    a_open_cmd=0.85,
    *,
    camera_preview=None,
    freeze_b_base_steps: int = 0,
):
    """Run handover policy on robot B; when B gripper qpos is closed enough, open A after dwell."""
    print(
        f"\n[Handover] B policy '{skill_name}' + release A when "
        f"g_B<={b_grip_takeover_qpos} for {release_dwell_steps} steps"
    )
    if freeze_b_base_steps > 0:
        print(
            f"[Handover] First {freeze_b_base_steps} steps: B mobile cmd forced to 0 "
            "(policy still updates arm/grip — reduces early base drift / premature grasp)."
        )
    preview_cameras_close()
    _print_checkpoint_camera_routing(planner, "robot_b", "handover")

    obs = env._get_obs()
    planner.reset_buffers()

    arm_joint_names = [
        "arm_lift_joint",
        "arm_flex_joint",
        "arm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ]
    a_addrs = [
        env.model.jnt_qposadr[
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_a/{j}")
        ]
        for j in arm_joint_names
    ]
    # Fixed joint targets for A (no outer-loop integral): B's motion + contact otherwise
    # falsely looks like "sag" and pushes A upward.
    a_arm_hold = np.array([float(env.data.qpos[a]) for a in a_addrs], dtype=np.float64)
    b_grip_addr = env.model.jnt_qposadr[
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_b/hand_motor_joint")
    ]

    dwell = 0
    released = False

    ck_cams = getattr(planner, "camera_names", None)
    phase_tag = f"B_{skill_name}"
    for _step in range(max_steps):
        info = _map_dual_images_to_single(
            env, "robot_b", "handover", policy_camera_names=ck_cams
        )
        preview_cameras_show(env, camera_preview, info, phase_tag)
        single_obs = {
            "joint_pos": obs["robot_b/joint_pos"],
            "mobile_vel": obs["robot_b/mobile_vel"],
        }
        single_action = planner.get_action(single_obs, info)

        full_action = np.zeros(18)
        full_action[9:18] = single_action
        if freeze_b_base_steps > 0 and _step < freeze_b_base_steps:
            full_action[9:12] = 0.0
        full_action[3:8] = a_arm_hold
        g_b = float(env.data.qpos[b_grip_addr])
        if not released:
            if g_b <= b_grip_takeover_qpos:
                dwell += 1
            else:
                dwell = 0
            if dwell >= release_dwell_steps:
                full_action[8] = a_open_cmd
                released = True
                print(f"[Handover] Released A gripper at step {_step} (g_B={g_b:.4f})")
            else:
                g_a_addr = env.model.jnt_qposadr[
                    mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_a/hand_motor_joint")
                ]
                g_a = float(env.data.qpos[g_a_addr])
                full_action[8] = min(g_a - 0.12, _TIGHT_GRIP_CMD)
        else:
            full_action[8] = a_open_cmd

        obs, _r, _t, _tr, _i = env.step(full_action)
        if released:
            print(
                f"[Handover] B policy stopped after A release (step {_step + 1}/{max_steps})."
            )
            break

    if not released:
        print("[Handover] WARNING: A not released (B gripper threshold not met).")
    preview_cameras_close()
