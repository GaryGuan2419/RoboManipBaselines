"""
Shared utilities for dual HSR handover: snap baton, PID arm alignment, joint read/print.
"""

import numpy as np
import mujoco

from handover_config import ARM_JOINT_NAMES, GRIPPER_JOINT_NAME

# Match bin/ultimate_line_demo_lib.navigate_to_dual(force_tight_grip): squeeze so grasp does not open.
_TIGHT_GRIP_CMD = -0.30
# Idle robot during align: outer-loop correction toward pose at align start (reduces gravity sag).
_ALIGN_OTHER_ARM_INTEGRAL_GAIN = 0.06
# Hold segments inherit only a small anti-gravity command offset from the previous action.
# This prevents a policy's last active motion command from being replayed as a hold command.
_HOLD_ARM_CMD_QPOS_MIN = np.array([0.020, 0.018, -0.008, 0.002, -0.006], dtype=np.float64)
_HOLD_ARM_CMD_QPOS_MAX = np.array([0.040, 0.045, 0.008, 0.014, 0.006], dtype=np.float64)


def _palm_zs(env) -> np.ndarray:
    out = []
    for prefix in ("robot_a", "robot_b"):
        palm_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{prefix}/hand_palm_link",
        )
        out.append(float(env.data.xpos[palm_id][2]))
    return np.asarray(out, dtype=np.float64)


class PalmZSegmentMonitor:
    """Track min/max palm height inside a phase without printing every step."""

    def __init__(self, env, label: str, enabled: bool = True):
        self.env = env
        self.label = label
        self.enabled = bool(enabled)
        self.n = 0
        self.start = None
        self.end = None
        self.min_z = None
        self.max_z = None
        self.min_step = np.zeros(2, dtype=np.int64)
        self.max_step = np.zeros(2, dtype=np.int64)
        if self.enabled:
            self.sample()

    def sample(self) -> None:
        if not self.enabled:
            return
        z = _palm_zs(self.env)
        if self.start is None:
            self.start = z.copy()
            self.min_z = z.copy()
            self.max_z = z.copy()
        else:
            lower = z < self.min_z
            higher = z > self.max_z
            self.min_z[lower] = z[lower]
            self.max_z[higher] = z[higher]
            self.min_step[lower] = self.n
            self.max_step[higher] = self.n
        self.end = z.copy()
        self.n += 1

    def finish(self) -> None:
        if not self.enabled or self.start is None:
            return
        print(f"[DiagSeg] {self.label}: samples={self.n}")
        for idx, name in enumerate(("A", "B")):
            drop_from_start = float(self.start[idx] - self.min_z[idx])
            drop_from_peak = float(self.max_z[idx] - self.min_z[idx])
            end_delta = float(self.end[idx] - self.start[idx])
            print(
                f"[DiagSeg]   {name} palm_z start={self.start[idx]:+.4f} "
                f"min={self.min_z[idx]:+.4f}@{int(self.min_step[idx])} "
                f"max={self.max_z[idx]:+.4f}@{int(self.max_step[idx])} "
                f"end={self.end[idx]:+.4f} "
                f"drop_start={drop_from_start:+.4f} "
                f"drop_peak={drop_from_peak:+.4f} "
                f"end-start={end_delta:+.4f}"
            )


def _seed_action_or_ctrl(env, seed_prev_action_18=None):
    if seed_prev_action_18 is not None:
        return np.asarray(seed_prev_action_18, dtype=np.float64).reshape(18)
    if getattr(env.model, "nu", 0) >= 18 and getattr(env.data, "ctrl", None) is not None:
        return np.asarray(env.data.ctrl[:18], dtype=np.float64).copy()
    return None


def _seed_arm_grip_command(
    env,
    robot_index: int,
    seed_prev_action_18,
    fallback_arm,
    fallback_grip,
):
    action = _seed_action_or_ctrl(env, seed_prev_action_18)
    if action is None:
        return np.asarray(fallback_arm, dtype=np.float64).copy(), float(fallback_grip)
    off = int(robot_index) * 9
    return (
        np.asarray(action[off + 3 : off + 8], dtype=np.float64).copy(),
        float(action[off + 8]),
    )


def get_arm_qpos_addrs(env, prefix):
    """Return list of qpos addresses for the 5 arm joints of the given robot."""
    return [
        env.model.jnt_qposadr[
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/{j}")
        ]
        for j in ARM_JOINT_NAMES
    ]


def get_gripper_qpos_addr(env, prefix):
    return env.model.jnt_qposadr[
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/{GRIPPER_JOINT_NAME}")
    ]


def read_arm_joints(env, robot_index):
    """Read current 5-dim arm joint positions for the given robot."""
    prefix = "robot_a" if robot_index == 0 else "robot_b"
    addrs = get_arm_qpos_addrs(env, prefix)
    return np.array([env.data.qpos[a] for a in addrs])


def read_gripper(env, robot_index):
    """Read current gripper position for the given robot."""
    prefix = "robot_a" if robot_index == 0 else "robot_b"
    addr = get_gripper_qpos_addr(env, prefix)
    return env.data.qpos[addr]


def snap_baton_to_robot_a(env):
    """Attach bottle2 near A's hand while keeping baton orientation upright.

    Position is anchored to A hand_palm frame, but orientation is forced to world-upright
    to avoid sudden horizontal flips during synthetic scene setup.
    """
    mujoco.mj_forward(env.model, env.data)
    palm_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "robot_a/hand_palm_link")
    palm_xpos = env.data.xpos[palm_id].copy()
    palm_xmat = env.data.xmat[palm_id].reshape(3, 3)
    offset_local = np.array([0.0, 0.0, 0.04])
    target_pos = palm_xpos + palm_xmat @ offset_local

    baton_jnt = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "bottle2_freejoint")
    q_addr = env.model.jnt_qposadr[baton_jnt]
    v_addr = env.model.jnt_dofadr[baton_jnt]

    env.data.qpos[q_addr:q_addr + 3] = target_pos
    # Keep baton upright (identity quaternion in world frame).
    env.data.qpos[q_addr + 3:q_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    env.data.qvel[v_addr:v_addr + 6] = 0.0


def align_arm_to_pose(
    env,
    robot_index,
    target_arm,
    target_gripper,
    steps=100,
    smooth_steps=80,
    snap_baton=False,
    hold_start_gripper=False,
    transition_diag=False,
    diag_label=None,
    seed_prev_action_18=None,
):
    """Smoothly interpolate one robot's arm to target joints. Other robot holds position.

    Args:
        snap_baton: if True, call snap_baton_to_robot_a every step (for --skip_pick mode).
        hold_start_gripper: if True, do not interpolate gripper toward ``target_gripper`` (which may
            be a squeeze command in a different convention than qpos). Instead keep a firm grasp
            using the same squeeze rule as navigate_to_dual(force_tight_grip), so the baton is not
            dropped during arm motion.
    """
    current_arm = read_arm_joints(env, robot_index)
    current_grip = read_gripper(env, robot_index)
    current_arm, current_grip = _seed_arm_grip_command(
        env,
        robot_index,
        seed_prev_action_18,
        current_arm,
        current_grip,
    )
    idx_offset = robot_index * 9
    label = diag_label or f"align robot {robot_index}"
    seg_mon = PalmZSegmentMonitor(env, label, transition_diag)

    # Lock the other robot: hold reference pose at align start + integral vs measured (reduces sag).
    other_idx = 1 - robot_index
    other_ref = read_arm_joints(env, other_idx).copy()
    other_grip = read_gripper(env, other_idx)
    other_cmd, other_grip_cmd = _seed_arm_grip_command(
        env,
        other_idx,
        seed_prev_action_18,
        other_ref,
        other_grip,
    )
    other_offset = other_idx * 9

    def _tight_grip_command(g_qpos: float) -> float:
        """Grasping robot: same squeeze as navigate_to_dual(force_tight_grip)."""
        g = float(g_qpos)
        return min(g - 0.18, _TIGHT_GRIP_CMD)

    action = np.zeros(18)
    for t in range(steps):
        alpha = min(1.0, t / max(smooth_steps, 1))
        interp_arm = (1.0 - alpha) * current_arm + alpha * np.asarray(target_arm)
        if hold_start_gripper:
            interp_grip = _tight_grip_command(read_gripper(env, robot_index))
        else:
            interp_grip = (1.0 - alpha) * current_grip + alpha * target_gripper

        qm_other = read_arm_joints(env, other_idx)
        delta = _ALIGN_OTHER_ARM_INTEGRAL_GAIN * (other_ref - qm_other)
        delta = np.clip(delta, -0.004, 0.004)
        other_cmd = other_cmd + delta

        action = np.zeros(18)
        # Active robot: interpolated arm + gripper
        action[idx_offset + 3:idx_offset + 8] = interp_arm
        action[idx_offset + 8] = interp_grip
        # Other robot: hold arm; live grip (open = pass-through, grasp = tight squeeze)
        action[other_offset + 3:other_offset + 8] = other_cmd
        og = read_gripper(env, other_idx)
        if og < 0.6:
            other_grip_cmd = _tight_grip_command(og)
        action[other_offset + 8] = other_grip_cmd

        env.step(action)
        seg_mon.sample()
        if snap_baton:
            snap_baton_to_robot_a(env)
            seg_mon.sample()
    seg_mon.finish()
    return action.copy()


def print_robot_state(env, robot_index):
    """Print arm joints, gripper, base pose, and hand_palm position for one robot."""
    prefix = "robot_a" if robot_index == 0 else "robot_b"
    label = "A" if robot_index == 0 else "B"
    arm = read_arm_joints(env, robot_index)
    grip = read_gripper(env, robot_index)

    base_joints = [f"{prefix}/mobile_x_joint", f"{prefix}/mobile_y_joint", f"{prefix}/mobile_theta_joint"]
    base = [env.data.joint(j).qpos[0] for j in base_joints]

    palm_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}/hand_palm_link")
    palm_xyz = env.data.xpos[palm_id].copy()

    print(f"\n{'='*60}")
    print(f"  Robot {label} ({prefix})")
    print(f"  base (x, y, yaw): [{base[0]:.4f}, {base[1]:.4f}, {base[2]:.4f}]")
    print(f"  arm joints [lift, flex, roll, wflex, wroll]:")
    print(f"    np.array([{', '.join(f'{v:.4f}' for v in arm)}])")
    print(f"  gripper: {grip:.4f}")
    print(f"  hand_palm xyz: [{palm_xyz[0]:.4f}, {palm_xyz[1]:.4f}, {palm_xyz[2]:.4f}]")
    print(f"{'='*60}")


def render_robot_cameras(env, robot_prefix, rgb_only=False):
    """Render head+hand cameras for the given robot.

    Args:
        rgb_only: if True, skip depth rendering (faster; use for display only).

    Returns a dict with "rgb_images" and optionally "depth_images", both
    keyed by short camera name (e.g. "head", "hand").  Depth images are in
    metres, matching the conversion used by MujocoEnvBase._get_info().
    """
    extent = env.model.stat.extent
    near = env.model.vis.map.znear * extent
    far  = env.model.vis.map.zfar  * extent

    rgb_images   = {}
    depth_images = {}
    for cam_suffix in ["head", "hand"]:
        cam_key = f"{robot_prefix}_{cam_suffix}"
        if cam_key not in env.cameras:
            continue
        cam = env.cameras[cam_key]
        cam["viewer"].make_context_current()
        rgb_images[cam_suffix] = cam["viewer"].render(
            render_mode="rgb_array", camera_id=cam["id"]
        )
        if not rgb_only:
            raw_depth = cam["viewer"].render(
                render_mode="depth_array", camera_id=cam["id"]
            )
            depth_images[cam_suffix] = near / (1.0 - raw_depth * (1.0 - near / far))

    result = {"rgb_images": rgb_images}
    if not rgb_only:
        result["depth_images"] = depth_images
    return result


def build_dual_hold_targets_from_current(env, force_tight_grip_robots=()):
    """Arm command = measured qpos; grip command matches ``navigate_to_dual`` squeeze rule."""
    ft = set(force_tight_grip_robots)
    out = {}
    for i in (0, 1):
        arm = read_arm_joints(env, i).copy()
        g = float(read_gripper(env, i))
        if i in ft:
            g_cmd = min(g - 0.18, _TIGHT_GRIP_CMD)
        elif g < 0.6:
            g_cmd = g - 0.1
        else:
            g_cmd = g
        out[i] = {"arm": arm, "grip": g_cmd}
    return out


def dual_hold_action_from_targets(locked_by_robot: dict) -> np.ndarray:
    """18-dim action: zero base velocity; arms + grip from ``locked_by_robot`` [0,1]."""
    action = np.zeros(18, dtype=np.float64)
    for i, off in ((0, 0), (1, 9)):
        action[off + 3 : off + 8] = locked_by_robot[i]["arm"]
        action[off + 8] = locked_by_robot[i]["grip"]
    return action


def _locked_dict_grip_refresh(env, locked, force_tight_grip_robots):
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


def hold_dual_pose_steps(
    env,
    n_steps: int,
    *,
    force_tight_grip_robots=(),
    render_fn=None,
    mj_forward_first=True,
    arm_integral_gain: float = 0.06,
    transition_diag=False,
    diag_label=None,
    seed_prev_action_18=None,
    zero_arm_qvel_first=False,
):
    """Hold both arms near the snapshot pose using integral correction (reduces long-hold sag).

    Returns:
        Last 18-dim env action applied (or the initial hold command if ``n_steps == 0``), for seeding
        the next policy / navigation segment.
    """
    if mj_forward_first:
        mujoco.mj_forward(env.model, env.data)
    seg_mon = PalmZSegmentMonitor(
        env,
        diag_label or f"hold_dual_pose_steps n={int(n_steps)}",
        transition_diag,
    )
    snap = build_dual_hold_targets_from_current(env, force_tight_grip_robots)
    locked = {}
    for i in (0, 1):
        arm_cmd, grip_cmd = _seed_arm_grip_command(
            env,
            i,
            seed_prev_action_18,
            snap[i]["arm"],
            snap[i]["grip"],
        )
        arm_cmd = np.asarray(arm_cmd, dtype=np.float64).copy()
        measured_arm = np.asarray(snap[i]["arm"], dtype=np.float64)
        # A hold segment should preserve small anti-gravity compensation, not continue a
        # policy's last active motion command.
        arm_cmd = measured_arm + np.clip(
            arm_cmd - measured_arm,
            _HOLD_ARM_CMD_QPOS_MIN,
            _HOLD_ARM_CMD_QPOS_MAX,
        )
        locked[i] = {
            "arm": [float(x) for x in arm_cmd],
            "grip": float(grip_cmd),
        }
    per_robot_arm_addrs = [
        [
            env.model.jnt_qposadr[
                mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{pfx}/{j}")
            ]
            for j in ARM_JOINT_NAMES
        ]
        for pfx in ["robot_a", "robot_b"]
    ]
    if zero_arm_qvel_first:
        for pfx in ["robot_a", "robot_b"]:
            for jname in ARM_JOINT_NAMES:
                jid = mujoco.mj_name2id(
                    env.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{pfx}/{jname}",
                )
                vaddr = int(env.model.jnt_dofadr[jid])
                env.data.qvel[vaddr] = 0.0
        mujoco.mj_forward(env.model, env.data)
    locked_ref = np.zeros((2, 5), dtype=np.float64)
    for i in (0, 1):
        for j, addr in enumerate(per_robot_arm_addrs[i]):
            locked_ref[i, j] = float(env.data.qpos[addr])
    _locked_dict_grip_refresh(env, locked, force_tight_grip_robots)
    last_act = dual_hold_action_from_targets(locked).copy()
    for _ in range(n_steps):
        for i in (0, 1):
            for j, addr in enumerate(per_robot_arm_addrs[i]):
                q = float(env.data.qpos[addr])
                e = float(locked_ref[i, j]) - q
                d = float(np.clip(arm_integral_gain * e, -0.004, 0.004))
                locked[i]["arm"][j] += d
        _locked_dict_grip_refresh(env, locked, force_tight_grip_robots)
        last_act = dual_hold_action_from_targets(locked).copy()
        env.step(last_act)
        seg_mon.sample()
        if render_fn is not None:
            render_fn()
    seg_mon.finish()
    return last_act
