"""
Handover pose constants for dual HSR baton relay.

Run bin/calibrate_handover_pose.py to determine these values, then paste here.
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# All values below are from the calibrated pre-handover capture.
# ══════════════════════════════════════════════════════════════════════════════

# ── Robot A (holding baton, facing +X) ────────────────────────────────────────
HANDOVER_BASE_A = np.array([0.55, -0.0800, 0.0000])   # x, y, yaw
HANDOVER_POSE_A = np.array([0.30, -2.4996, 0.0108, 1.08, 0.0048])
HANDOVER_GRIPPER_A = -0.15   # command (actual ≈0.19 blocked by baton; negative → squeeze)

# ── Robot B (ready to receive, facing -X) ─────────────────────────────────────
HANDOVER_BASE_B = np.array([1.65, 0.0768, 3.1416])    # x, y, yaw
# Joint order: [arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll]
# wrist_flex: 增大→手腕向下弯, 减小→手腕上扬 (0 ~ π, 当前≈0.99rad)
HANDOVER_POSE_B = np.array([0.1787, -2.5480, 0.0102, 1.08, 0.0052])
HANDOVER_GRIPPER_B_OPEN = 0.8

# ── Baton ─────────────────────────────────────────────────────────────────────
HANDOVER_BATON_POS = np.array([1.0020, -0.0020, 0.2023])

# Oracle navigation (bin/run_ultimate_hsr_line_demo.py, robo_manip_baselines/mllm/dual_hsr_handover_geometry.py):
#   robot_base_target_xy = handover_landmark_xy + ORACLE_OFFSET_*_HANDOVER_XY
# Calibrated as (robot_base - baton_xy) when baton sits on the handover landmark (same as demos).
ORACLE_OFFSET_A_HANDOVER_XY = (HANDOVER_BASE_A[:2] - HANDOVER_BATON_POS[:2]).astype(np.float64)
ORACLE_OFFSET_B_HANDOVER_XY = (HANDOVER_BASE_B[:2] - HANDOVER_BATON_POS[:2]).astype(np.float64)

# Runtime nav tweak (world XY, metres), added **after** ``landmark + ORACLE_OFFSET_*`` math.
# Does not move MJCF bodies; when landmarks move, targets still follow — this is an extra relative trim.
# Applied to: A_handover, B_handover_pick, and B place nav (including ultimate line reorient branch).
NAV_RUNTIME_XY_EXTRA = np.array([-0.05, 0.0], dtype=np.float64)

# ── B chassis at **place** only (does not change A handover or B handover oracle) ─────────────
# World: B base_xy = target_area_xy[:2] + B_PLACE_OFFSET_FROM_GOAL_XY
# ``target_area`` comes from the env (ultimate line: body target_area; b_side_place XML: pad ~ (1,1)).
# Reference for the single-env pad must match ``TARGET_AREA_XY_DEFAULT`` in
# ``hsr_b_side_place_config.py`` (default [1, 1]).
TARGET_AREA_XY_REFERENCE_FOR_B_PLACE = np.array([1.0, 1.0], dtype=np.float64)
# Set to a length-2 ndarray to tune B place without touching HANDOVER_BASE_A / handover offsets.
# None → keep legacy ``goal_to_base_offset_xy_for_place("b_side_place")`` (hybrid dx + dy=-0.08)
# and legacy spawn ``ROBOT_BASE`` in ``build_initial_qpos_18``.
B_PLACE_OFFSET_FROM_GOAL_XY = None  # e.g. np.array([-0.45, -0.08], dtype=np.float64)


def b_place_offset_from_goal_xy_legacy_hybrid() -> np.ndarray:
    """Legacy dual-line b_side_place nav: dx = HANDOVER_BASE_A.x - ref.x, dy = -0.08."""
    dx = float(HANDOVER_BASE_A[0] - TARGET_AREA_XY_REFERENCE_FOR_B_PLACE[0])
    return np.array([dx, -0.08], dtype=np.float64)


def get_b_place_offset_from_goal_xy() -> np.ndarray:
    """Offset added to goal xy for B at place; used when B_PLACE_OFFSET_FROM_GOAL_XY is set."""
    if B_PLACE_OFFSET_FROM_GOAL_XY is not None:
        return np.asarray(B_PLACE_OFFSET_FROM_GOAL_XY, dtype=np.float64).reshape(2)
    return b_place_offset_from_goal_xy_legacy_hybrid()

# Actual gripper qpos when A is holding the baton (from calibration; used in build_handover_qpos).
GRIP_A_ACTUAL = 0.19


def build_handover_qpos(b_base):
    """Build the 29-dim qpos for the calibrated handover scene (no flash)."""
    q = np.zeros(29)
    q[0:3] = HANDOVER_BASE_A
    q[3:8] = HANDOVER_POSE_A
    q[8] = GRIP_A_ACTUAL
    q[9] = GRIP_A_ACTUAL
    q[10] = GRIP_A_ACTUAL
    q[11:14] = b_base
    q[14:19] = HANDOVER_POSE_B
    q[19] = HANDOVER_GRIPPER_B_OPEN
    q[20] = HANDOVER_GRIPPER_B_OPEN
    q[21] = HANDOVER_GRIPPER_B_OPEN
    q[22:25] = HANDOVER_BATON_POS
    q[25:29] = [1.0, 0.0, 0.0, 0.0]
    return q

# ── Randomization for P2 sampling ─────────────────────────────────────────────
B_BASE_JITTER_XY = np.array([0.03, 0.04])
B_BASE_JITTER_YAW = 0.10

# ── Side-pick navigation offset (Robot A → bottle) ───────────────────────────
# Derived from single-robot MujocoHsrTidyupEnv training conditions:
#   base init:   (-0.176, 0.0)
#   bottle2 pos: ( 0.37,  0.08)   [env_hsr_tidyup.xml body pos, set by modify_world]
#   offset = bottle - base = (0.546, 0.08)
# In dual env: A navigates to  bottle_xy - this offset  so the relative geometry
# matches what the side_pick policy saw during training.
SIDE_PICK_BASE_TO_OBJECT_XY = np.array([0.546, 0.08])
SIDE_PICK_YAW_A = 0.0

# ── Joint names (shared across scripts) ──────────────────────────────────────
ARM_JOINT_NAMES = [
    "arm_lift_joint", "arm_flex_joint", "arm_roll_joint",
    "wrist_flex_joint", "wrist_roll_joint",
]
GRIPPER_JOINT_NAME = "hand_motor_joint"

# ── Interactive tuning step sizes ─────────────────────────────────────────────
JOINT_STEPS = np.array([
    0.005,   # arm_lift  (meters)
    0.03,    # arm_flex  (rad)
    0.03,    # arm_roll  (rad)
    0.03,    # wrist_flex (rad)
    0.03,    # wrist_roll (rad)
])
GRIPPER_STEP = 0.05
FINE_SCALE = 0.25
