"""
Single-HSR handover-A sampling config.

目标：把环境裁剪为“只保留 handover A 的机器人与接力棒关键参数”，用于稳定采样。
"""

import importlib.util
import os

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _load_handover_config():
    path = os.path.join(_REPO_ROOT, "bin", "handover_config.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("handover_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hc = _load_handover_config()

if _hc is not None:
    # ── Robot A（与 handover_config 逐字段一致，供对照 / 文档）──────────────────
    HANDOVER_BASE_A = _hc.HANDOVER_BASE_A.copy()
    HANDOVER_POSE_A = _hc.HANDOVER_POSE_A.copy()
    HANDOVER_GRIPPER_A = float(_hc.HANDOVER_GRIPPER_A)
    GRIP_A_ACTUAL = float(_hc.GRIP_A_ACTUAL)
    # ── Baton（与 handover_config 一致）──────────────────────────────────────
    HANDOVER_BATON_POS = _hc.HANDOVER_BATON_POS.copy()
    # ── Robot B（保留作为可选参考）──────────────────────────────────────────────
    HANDOVER_BASE_B = _hc.HANDOVER_BASE_B.copy()
    HANDOVER_POSE_B = _hc.HANDOVER_POSE_B.copy()
    HANDOVER_GRIPPER_B_OPEN = float(_hc.HANDOVER_GRIPPER_B_OPEN)
    B_BASE_JITTER_XY = _hc.B_BASE_JITTER_XY.copy()
    B_BASE_JITTER_YAW = float(_hc.B_BASE_JITTER_YAW)
else:
    # 与 bin/handover_config.py 同文件快照（无 handover_config 时使用）
    HANDOVER_BASE_A = np.array([0.4498, -0.0800, 0.0000])
    HANDOVER_POSE_A = np.array([0.2685, -2.4996, 0.0108, 0.9645, 0.0048])
    HANDOVER_GRIPPER_A = -0.15
    GRIP_A_ACTUAL = 0.19
    HANDOVER_BATON_POS = np.array([1.0020, -0.0020, 0.2523])
    HANDOVER_BASE_B = np.array([1.6640, 0.0768, 3.1416])
    HANDOVER_POSE_B = np.array([0.1787, -2.5480, 0.0102, 1.08, 0.0052])
    HANDOVER_GRIPPER_B_OPEN = 0.8
    B_BASE_JITTER_XY = np.array([0.03, 0.04])
    B_BASE_JITTER_YAW = 0.10

# ---- 单机采样：直接采用 handover A 机器人配置 ----
ROBOT_BASE = HANDOVER_BASE_A.copy()
ROBOT_POSE = HANDOVER_POSE_A.copy()

# 与 A 握棒一致：初始 qpos；snap 窗口 ctrl 与双机 A rollout 一致（HANDOVER_GRIPPER_A）
GRIP_HOLD = GRIP_A_ACTUAL
GRIP_HOLD_SNAP_CMD = HANDOVER_GRIPPER_A

# 棒的初始世界 xyz；None = 使用 handover 的 HANDOVER_BATON_POS
BATON_INIT_POS_OVERRIDE = HANDOVER_BATON_POS + np.array([-0.02, 0.0, 0.05])
# 棒的初始姿态 wxyz；None = 世界竖直 [1,0,0,0]
BATON_INIT_QUAT_OVERRIDE = None

# reset 后若干 env step 内将棒**固定**在初始 qpos（每步末写回）；之后纯物理
BATON_HOLD_STEPS = 45

# MuJoCo HSR 夹爪：position 执行器，ctrl = 关节位置目标；力 ≈ kp*(ctrl−qpos)。
#
# GRIP_HOLD_MODE:
#   "none"        — 默认：BATON_HOLD + GRIP_SQUEEZE 内强制闭合；挤压阶段若 grip_cmd≥GRIP_TELEOP_OPEN_THRESHOLD
#       则放行（可 X 张开）。窗口结束后不覆盖。
#   "squeeze_cmd" — 若再设 GRIP_ALWAYS_SQUEEZE=True，则每步恒写 GRIP_HOLD_SNAP_CMD（键盘夹爪无效）。
#
# collect_b_side_place_demos 里 COMMAND_GRIPPER 录的是键盘 grip_cmd；窗口内 env 与键盘可能不一致。
GRIP_HOLD_MODE = "none"
GRIP_ALWAYS_SQUEEZE = False

# 棒**运动学固定结束**后，再强制闭合指令若干步，让摩擦稳定（双机 get_hold_action 用 qpos−offset 也是为
# 避免「ctrl≈qpos → 无力矩 → 物体滑落」）。默认 0 时一松棒只靠遥操作；若 grip_cmd 仍接近 qpos 会掉棒。
# 竖直 place 用 lock_bottle+snap 代替；侧向棒无 snap，需这段 + 下方 OPEN_THRESHOLD 配合。
GRIP_SQUEEZE_STEPS = 0

# 在 GRIP_SQUEEZE 窗口内：若遥操作 grip_cmd ≥ 此值，视为「要张开」（类比 TidyupPlace 里 gripper_cmd>0.7 取消自动捏紧），
# 不再覆盖 action[8]，以便 X 生效。
GRIP_TELEOP_OPEN_THRESHOLD = 0.10

# 夹爪“锁存保持”：
# 一旦检测到明显闭合命令（<= CLOSE_THRESHOLD），后续即使不再按键也维持轻压夹持；
# 直到检测到明确张开命令（>= OPEN_THRESHOLD）才解除锁存。
GRIP_LATCH_ENABLE = True
GRIP_LATCH_CLOSE_THRESHOLD = 0.25
GRIP_LATCH_OPEN_THRESHOLD = 0.30
# 锁存保持时的轻压偏置：target = qpos - offset（类似双机 get_hold_action 的思路）。
GRIP_LATCH_TRACK_OFFSET = 0.3

# 参照 MujocoHsrTidyupPlaceEnv 的抓取流程：在预固定后，短暂 snap 到掌心并强制闭合。
# B-side 默认关闭，避免「过一段时间物体位姿突然跳变」。
GRASP_SNAP_ENABLE = False
GRASP_SNAP_OFFSET_LOCAL = np.array([0.0, 0.0, 0.1])
GRASP_CLOSE_DONE_QPOS = 0.12

# 开局强制闭合：从 reset 开始持续发送闭合命令固定若干步（不看 qpos 阈值），
# 避免“闭合不到阈值导致一直覆盖 X 打不开”。None 表示与 BATON_HOLD_STEPS 同步结束。
GRIP_FORCE_CLOSE_FROM_RESET = True
GRIP_FORCE_CLOSE_STEPS = None

# 解除长方体固定的夹爪闭合阈值：到达该 qpos（越小越闭合）才允许释放固定。
GRIP_LOCK_RELEASE_QPOS = 0.22


def grip_force_close_steps():
    if GRIP_FORCE_CLOSE_STEPS is None:
        return int(BATON_HOLD_STEPS)
    return int(GRIP_FORCE_CLOSE_STEPS)


def grip_squeeze_steps():
    if GRIP_SQUEEZE_STEPS is None:
        return int(BATON_HOLD_STEPS)
    return int(GRIP_SQUEEZE_STEPS)

# World XY mocap markers（可视化用，默认不随机）
HANDOVER_AREA_XY_DEFAULT = np.array([1.0, 0.0])
TARGET_AREA_XY_DEFAULT = np.array([1.0, 1.0])

# 与 env_hsr_b_side_place.xml 中 target_area 绿垫 geom ``size="0.05 0.05 0.001"`` 一致（半宽，世界 XY）
PLACE_TARGET_PAD_HALF_XY_M = np.array([0.05, 0.05], dtype=np.float64)

# 两节 box 接力棒沿竖直方向总尺度约 0.14 m（用于「不得举过高」判据：最高点 ≤ 垫面 z + 此长度）
PLACE_BATON_NOMINAL_LENGTH_M = 0.14

TARGET_XY_JITTER = 0.0
HANDOVER_XY_JITTER = 0.0

# 额外底座偏移（默认关闭，避免 reset 视觉突变）
PLACE_BASE_OFFSET = np.array([0.0, 0.0, 0.0])

def _baton_world_xyz():
    if BATON_INIT_POS_OVERRIDE is not None:
        return np.asarray(BATON_INIT_POS_OVERRIDE, dtype=np.float64).reshape(3)
    return HANDOVER_BATON_POS.copy()


def _baton_quat_wxyz():
    if BATON_INIT_QUAT_OVERRIDE is not None:
        return np.asarray(BATON_INIT_QUAT_OVERRIDE, dtype=np.float64).reshape(4)
    return np.array([1.0, 0.0, 0.0, 0.0])


def build_initial_qpos_18():
    """Full model qpos (18): base3, arm5, gripper1, mimic2, bottle free7."""
    q = np.zeros(18)
    g = GRIP_A_ACTUAL
    # Optional: B place chassis only — match dual-line nav when B_PLACE_OFFSET_FROM_GOAL_XY is set.
    if _hc is not None and getattr(_hc, "B_PLACE_OFFSET_FROM_GOAL_XY", None) is not None:
        off = np.asarray(_hc.B_PLACE_OFFSET_FROM_GOAL_XY, dtype=np.float64).reshape(2)
        q[0] = float(TARGET_AREA_XY_DEFAULT[0] + off[0])
        q[1] = float(TARGET_AREA_XY_DEFAULT[1] + off[1])
        q[2] = float(_hc.HANDOVER_BASE_A[2])
        q[0:3] += PLACE_BASE_OFFSET
    else:
        q[0:3] = ROBOT_BASE + PLACE_BASE_OFFSET
    q[3:8] = ROBOT_POSE
    q[8] = g
    q[9:11] = g
    q[11:14] = _baton_world_xyz()
    q[14:18] = _baton_quat_wxyz()
    return q
