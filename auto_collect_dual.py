#!/usr/bin/env python
"""
auto_collect_dual.py: Automated Data Collection for Dual Dingo-Z1 Handover Task

Layout:
  Table A (-1.1) -- Robot A (-0.55) -- Table C (0) -- Robot B (0.55) -- Table B (1.1)

Action format (20-dim):
  [0:3]   Robot A mobile vel (vx, vy, vθ)  — keep zero to lock base
  [3:9]   Robot A arm joints j1~j6         — absolute position in rad
  [9]     Robot A gripper                   — open=0.0, close=-1.5
  [10:13] Robot B mobile vel               — keep zero
  [13:19] Robot B arm joints j1~j6         — absolute position in rad
  [19]    Robot B gripper                   — open=0.0, close=-1.5

Usage:
    conda run -n robomanip python -u auto_collect_dual.py              # visible
    conda run -n robomanip python -u auto_collect_dual.py --headless   # batch
"""

import os
import sys
import argparse
import numpy as np
import mujoco
import gymnasium as gym
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robo_manip_baselines.envs  # registers gymnasium environments
from robo_manip_baselines.common.manager.DataManager import DataManager, DataKey
from waypoints_config import (
    JA_PRE_GRASP, JA_GRASP, JA_CLOSE, JA_PLACE, JA_RETREAT,
    JB_PRE_GRASP, JB_GRASP, JB_CLOSE, JB_PLACE, JB_RETREAT,
    GA_PRE_GRASP, GA_GRASP, GA_CLOSE, GA_PLACE_HOLD, GA_PLACE_OPEN, GA_RETREAT,
    GB_PRE_GRASP, GB_GRASP, GB_CLOSE, GB_PLACE_HOLD, GB_PLACE_OPEN, GB_RETREAT,
    GRIPPER_OPEN, GRIPPER_CLOSE,
)


# ─── IK Utils ─────────────────────────────────────────────────────────────────

def pos_ik_step(model, data, target_pos, eef_body_id, arm_dof_indices, gain=0.6):
    """One Jacobian-IK step (position only).

    Returns delta_q (in rad) for the 6 arm joints.
    gain controls how aggressively we step towards the target (0 < gain < 1).
    """
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, None, eef_body_id)

    # Extract 3×6 Jacobian for the arm joints only
    J = jacp[:, arm_dof_indices]

    err = target_pos - data.xpos[eef_body_id]
    # Hard clip: max 2cm per step to stay smooth
    err = np.clip(err, -0.02, 0.02)

    # Damped least squares (λ=0.1 prevents singularity blowup)
    lam = 0.1
    dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), err * gain)
    return dq

def pos_ori_ik_step(model, data, target_pos, target_quat, eef_body_id, arm_dof_indices, gain=0.6):
    """One Jacobian-IK step (position + orientation).
    Returns delta_q (in rad) for the 6 arm joints.
    """
    curr_pos = data.xpos[eef_body_id]
    curr_quat = np.zeros(4)
    mujoco.mju_mat2Quat(curr_quat, data.xmat[eef_body_id])
    
    err_pos = target_pos - curr_pos
    step_err_pos = np.clip(err_pos * 2.0, -0.02, 0.02)
    
    q_diff = np.zeros(4)
    curr_quat_inv = np.zeros(4)
    mujoco.mju_negQuat(curr_quat_inv, curr_quat)
    mujoco.mju_mulQuat(q_diff, target_quat, curr_quat_inv)
    err_ori = np.zeros(3)
    mujoco.mju_quat2Vel(err_ori, q_diff, 1.0)
    step_err_ori = np.clip(err_ori * 0.1, -0.05, 0.05)
    
    err = np.concatenate([step_err_pos * gain, step_err_ori * gain])
    
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, eef_body_id)
    
    J_pos = jacp[:, arm_dof_indices]
    J_ori = jacr[:, arm_dof_indices]
    J = np.vstack([J_pos, J_ori])
    
    lam = 0.1
    dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)
    return dq


def clamp_joints(q, model, arm_actuator_indices):
    """Clamp joint targets to actuator ctrl range."""
    for i, ai in enumerate(arm_actuator_indices):
        lo, hi = model.actuator_ctrlrange[ai]
        q[i] = np.clip(q[i], lo, hi)
    return q


def solve_global_ik(model, data, target_pos, target_quat, eef_id, qpos_idx, act_idx, q_init):
    """
    Finds a kinematically feasible joint configuration using global optimization (SLSQP).
    Minimizes positional error + orientation error + distance from HOME.
    """
    saved_qpos = data.qpos.copy()
    
    # Actuator limits for bounds
    bounds = []
    for ai in act_idx:
        lo, hi = model.actuator_ctrlrange[ai]
        bounds.append((lo, hi))
        
    def cost_fn(q):
        data.qpos[qpos_idx] = q
        mujoco.mj_kinematics(model, data)
        
        # Position error
        err_pos = np.linalg.norm(data.xpos[eef_id] - target_pos)
        
        # Orientation error
        curr_quat = np.zeros(4)
        mujoco.mju_mat2Quat(curr_quat, data.xmat[eef_id])
        q_diff = np.zeros(4)
        curr_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(curr_quat_inv, curr_quat)
        mujoco.mju_mulQuat(q_diff, target_quat, curr_quat_inv)
        err_ori = np.linalg.norm(q_diff[1:]) * 2.0  # approximate angular err
        
        # Home state penalty to avoid unnatural poses (especially elbows hitting the table)
        HOME_JOINTS = np.array([0.0, 0.785, -0.261, -0.523, 0.0, 0.0])
        err_home = np.linalg.norm(q - HOME_JOINTS)
        
        return err_pos * 100.0 + err_ori * 10.0 + err_home * 0.1

# ─── Main ─────────────────────────────────────────────────────────────────────

import random

def auto_collect_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    render_mode = "rgb_array" if args.headless else "human"
    print(f"[auto_collect] Creating environment (render_mode={render_mode})...")
    env = gym.make('robo_manip_baselines/MujocoDualDingoZ1HandoverEnv-v0',
                   render_mode=render_mode)

    model = env.unwrapped.model
    data  = env.unwrapped.data

    # --- Setup DataManager for Dual Handover ---
    # Patch the environment's camera_names property so we record everything we want
    env.unwrapped.__class__.camera_names = property(lambda self: ["overhead", "front", "robot_a_wrist", "robot_b_wrist"])
    
    # We also need to patch get_images to render all cameras instead of just 'overhead'
    def patched_get_images():
        info = {"rgb_images": {}, "depth_images": {}}
        for cam_name in ["overhead", "front", "robot_a_wrist", "robot_b_wrist"]:
            if cam_name in env.unwrapped.cameras:
                cam = env.unwrapped.cameras[cam_name]
                cam["viewer"].make_context_current()
                info["rgb_images"][cam_name] = cam["viewer"].render(
                    render_mode="rgb_array", camera_id=cam["id"]
                )
                
                # Fetch depth
                depth_image = cam["viewer"].render(
                    render_mode="depth_array", camera_id=cam["id"]
                )
                extent = model.stat.extent
                near = model.vis.map.znear * extent
                far = model.vis.map.zfar * extent
                depth_image = near / (1 - depth_image * (1 - near / far))
                info["depth_images"][cam_name] = depth_image
        return info
    env.unwrapped.get_images = patched_get_images

    print("Setting up DataManager...")
    data_manager = DataManager(env, demo_name="DualHandover_DualDingoZ1", task_desc="Dual Dingo-Z1 Handover")
    data_manager.setup_camera_info()

    # ── Identify joint indices ──────────────────────────────────────────────
    def robot_indices(prefix):
        """Return dof_indices, qpos_indices, eef_body_id, arm_actuator_indices."""
        joint_names = [f"{prefix}/joint{i}" for i in range(1, 7)]
        dof_idx  = [model.jnt_dofadr [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joint_names]
        qpos_idx = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joint_names]
        eef_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}/link06")
        # Actuator indices for the 6 arm joints (to get ctrlrange)
        act_idx  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}/motor{i}") for i in range(1, 7)]
        return dof_idx, qpos_idx, eef_id, act_idx

    dof_a, qpos_a, eef_a, act_a = robot_indices("robot_a")
    dof_b, qpos_b, eef_b, act_b = robot_indices("robot_b")

    # Indices and body IDs for randomization
    obj_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    obj_qpos_addr = model.jnt_qposadr[obj_jnt_id]
    hz_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "handover_zone")
    tgt_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")

    # Store default (original) positions to randomize relative to them
    default_obj_qpos_xy = env.unwrapped.init_qpos[obj_qpos_addr : obj_qpos_addr+2].copy()
    default_hz_pos_xy = model.body_pos[hz_body_id][:2].copy()
    default_tgt_pos_xy = model.body_pos[tgt_body_id][:2].copy()

    # Home pose (resting position)
    HOME_JOINTS = np.array([0.0, 0.785, -0.261, -0.523, 0.0, 0.0])

    table_z   = 0.36   # table surface height (pos=0.35 + half_geom=0.01)
    hover_z   = table_z + 0.18
    grasp_z   = table_z + 0.05

    os.makedirs("auto_demos", exist_ok=True)
    saved, ep = 0, 0

    # Orientations will be physically extracted per-episode to avoid Euler frame mismatches.

    table_z   = 0.36   # table surface height
    hover_z   = table_z + 0.10
    grasp_z   = table_z + 0.05

    os.makedirs("auto_demos", exist_ok=True)
    saved, ep = 0, 0

    while saved < args.episodes:
        ep += 1
        
        # 1. Randomize positions for each episode
        r_scale = 0.05 # +/- 5cm
        obj_noise = np.random.uniform(-r_scale, r_scale, size=2)
        hz_noise  = np.random.uniform(-r_scale, r_scale, size=2)
        tgt_noise = np.random.uniform(-r_scale, r_scale, size=2)
        
        env.unwrapped.init_qpos[obj_qpos_addr : obj_qpos_addr+2] = default_obj_qpos_xy + obj_noise
        model.body_pos[hz_body_id][:2] = default_hz_pos_xy + hz_noise
        model.body_pos[tgt_body_id][:2] = default_tgt_pos_xy + tgt_noise

        obs, _ = env.reset()
        data_manager.reset()
        mujoco.mj_forward(model, data)

        obj_pos = data.body("object").xpos.copy()
        tgt_pos = data.body("target").xpos.copy()
        hz_pos  = data.body("handover_zone").xpos.copy()

        print(f"\n--- Attempt {ep} (Saved {saved}/{args.episodes}) ---")
        print(f"  RANDOM: obj={obj_pos.round(3)} handover={hz_pos.round(3)} target={tgt_pos.round(3)}")
        cmd_a = data.qpos[qpos_a].copy()
        cmd_b = data.qpos[qpos_b].copy()

        # Generate Kinematically Feasible Waypoints
        # For grasping, the wrist is 0.16m away from the object.
        # If object is at -X, wrist is at +0.16, pointing -X
        
        # 从标定点推算抓取位的笛卡尔坐标
        # A robot in Table A: link06 在物体 +X 方向
        # 注意: link06 到夹爪中心约 0.10m，夹爪中心需在物体 6cm 内才触发吸附
        # 所以 link06 目标 = obj_x + 0.135，夹爪中心 ≈ obj_x + 0.035 (留值 2-3cm 边距)
        a_grasp_pos  = np.array([obj_pos[0] + 0.145, obj_pos[1], 0.4122])

        # B robot in Handover Zone: 同理
        b_grasp_pos  = np.array([hz_pos[0]  + 0.145, hz_pos[1],  0.415])

        # 放置位置 (Cartesian, 由标定点确定坐标)
        a_place_pos  = np.array([-0.1602, hz_pos[1],  0.4115])   # A 放入中转区
        b_place_pos  = np.array([ 0.9447, tgt_pos[1], 0.4115])   # B 放入 Table B

        # 在关节空间预备点后，读取当前绯姿作为 Cartesian IK 的目标姿态
        # (让 IK 不改姿态只正位置，避免手腔舟动)
        def get_eef_quat(eef_id):
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, data.xmat[eef_id])
            return q

        # 预先读取当前姿态，关节空间移动后重新读取 (IK 的姿态硬苸)
        qa_pick  = get_eef_quat(eef_a)
        qb_pick  = get_eef_quat(eef_b)
        qa_place = get_eef_quat(eef_a)
        qb_place = get_eef_quat(eef_b)

        # ── 预先离线 IK 适配: 根据随机位置调整预备点关节角 ──────────────────────
        # 思路: 标定时 JA_PRE_GRASP 的 link06 在某一固定位置。
        # 随机化后, 我们把 "预备点的 link06 目标" 也平移相同的 XY 偏移,
        # 然后在后台跑几步 Jacobian IK 算出新的关节角 JA_PRE_GRASP_ADJ,
        # 主循环的关节空间插值就能从这个更好的起点出发, 只剩纯 X 轴靠近.
        # 原 A 预备点 link06: [-0.8349, -0.0358, 0.4163]
        # 原 B 预备点 link06: [0.2900, 0.0243, 0.4211]
        CALIB_PRE_A_POS = np.array([-0.8349, -0.0358, 0.4163])
        CALIB_PRE_B_POS = np.array([ 0.2900,  0.0243, 0.4211])

        def offline_ik_adapt(calib_pre_pos, calib_joints, eef_id, dof_idx, qpos_idx, act_idx, obj_pos_now, obj_pos_default):
            """离线计算适配后的预备点关节角."""
            delta_xy = np.array([obj_pos_now[0] - obj_pos_default[0],
                                  obj_pos_now[1] - obj_pos_default[1],
                                  0.0])
            target_pre = calib_pre_pos + delta_xy

            # 保存仿真状态
            saved_qpos = data.qpos.copy()
            saved_qvel = data.qvel.copy()
            saved_ctrl = data.ctrl.copy()

            # 临时置入标定关节角, 迭代 IK
            q_adj = np.array(calib_joints)
            data.qpos[qpos_idx] = q_adj
            for _ in range(40):
                mujoco.mj_forward(model, data)
                err = np.linalg.norm(data.xpos[eef_id] - target_pre)
                if err < 0.005:
                    break
                dq = pos_ik_step(model, data, target_pre, eef_id, dof_idx)
                q_adj = q_adj + dq * 0.8
                q_adj = clamp_joints(q_adj, model, act_idx)
                data.qpos[qpos_idx] = q_adj

            # 恢复仿真状态
            data.qpos[:] = saved_qpos
            data.qvel[:] = saved_qvel
            data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(model, data)
            return q_adj


        OBJ_DEFAULT_POS = np.array([-1.1, 0.0, 0.4])   # XML 中 object 的默认位置
        HZ_DEFAULT_POS  = np.array([ 0.0, 0.0, 0.365])  # handover_zone 默认位置

        JA_PRE_GRASP_ADJ = offline_ik_adapt(
            CALIB_PRE_A_POS, JA_PRE_GRASP, eef_a, dof_a, qpos_a, act_a,
            obj_pos, OBJ_DEFAULT_POS)
        JB_PRE_GRASP_ADJ = offline_ik_adapt(
            CALIB_PRE_B_POS, JB_PRE_GRASP, eef_b, dof_b, qpos_b, act_b,
            hz_pos, HZ_DEFAULT_POS)


        # ────────────────────────────────────────────────────────────────────────
        waypoints = [
            # ── Phase 1: Robot A 抓取 Table A ────────────────────────
            # 1. 适配后的预备点 (关节空间, 已提前适配随机偏移)
            {"n": "A pre-grasp",    "type": "joint", "ja": JA_PRE_GRASP_ADJ, "ga": GRIPPER_OPEN, "tol": 0.10},

            # 2. 纯位置IK精细进入 - 实时跟踪物体实际位置
            {"n": "A grasp reach",  "pa": a_grasp_pos, "ga": GRIPPER_OPEN, "tol": 0.03, "track_obj": True},
            # 3. 夹爪闭合 (触发 6cm 吸附，持续 60步等吸附确认) - 持续跟踪
            {"n": "A close grip",   "pa": a_grasp_pos, "ga": GA_CLOSE, "tol": 0.04, "hold": 60, "track_obj": True},
            # 4. 抬起 (关节空间回预备点)
            {"n": "A lift",         "type": "joint", "ja": JA_PRE_GRASP, "ga": GA_CLOSE, "tol": 0.10},
            {"n": "CHECK_lifted",   "type": "check", "chk": "obj_lifted"},

            # -- Phase 2: Robot A 放置到中转区 --
            {"n": "A to place",   "type": "joint", "ja": JA_PLACE, "ga": GA_CLOSE, "tol": 0.10},
            {"n": "A release",    "type": "joint", "ja": JA_PLACE, "ga": GRIPPER_OPEN, "tol": 0.10, "hold": 30},
            {"n": "A retreat",    "type": "joint", "ja": JA_RETREAT, "ga": GRIPPER_OPEN, "tol": 0.10},

            # -- Phase 3: Robot B 抓取中转区 --
            {"n": "B pre-grasp",    "type": "joint", "jb": JB_PRE_GRASP_ADJ, "gb": GRIPPER_OPEN, "tol": 0.10},
            # 9. B 纯位置IK精细进入 - 实时跟踪物体实际位置
            {"n": "B grasp reach",  "pb": b_grasp_pos, "gb": GRIPPER_OPEN, "tol": 0.03, "track_obj": True},
            # 10. B 夹爪闭合 (持续 60步等吸附) - 同样实时跟踪
            {"n": "B close grip",   "pb": b_grasp_pos, "gb": GB_CLOSE, "tol": 0.04, "hold": 60, "track_obj": True},

            # 11. B 抬起
            {"n": "B lift",         "type": "joint", "jb": JB_PRE_GRASP_ADJ, "gb": GB_CLOSE, "tol": 0.10},
            {"n": "CHECK_lifted_b", "type": "check", "chk": "obj_lifted_b"},
            # ── Phase 4: Robot B 放置到 Table B ─────────────────────
            # 12. B 关节空间跷到放置位置
            {"n": "B to place",   "type": "joint", "jb": JB_PLACE, "gb": GB_CLOSE, "tol": 0.10},
            # 13. B 松开夹爪
            {"n": "B release",    "type": "joint", "jb": JB_PLACE, "gb": GRIPPER_OPEN, "tol": 0.10, "hold": 30},
            # 14. B 退回
            {"n": "B retreat",    "type": "joint", "jb": JB_RETREAT, "gb": GRIPPER_OPEN, "tol": 0.10},

            {"n": "CHECK_placed", "type": "check", "chk": "obj_placed"},
        ]

        # Increased max_steps to allow for potentially longer paths if randomized far
        max_steps = 700
        wp_i, hold_cnt, step_cnt = 0, 0, 0
        da, db = 0.0, 0.0
        failed = False

        print(f"\n--- Attempt {ep} (Saved {saved}/{args.episodes}) ---")
        print(f"  obj={obj_pos.round(3)}  handover={hz_pos.round(3)}  target={tgt_pos.round(3)}")

        while wp_i < len(waypoints) and step_cnt < max_steps:
            wp = waypoints[wp_i]

            # ── Validation checks ──
            if wp.get("type") == "check":
                mujoco.mj_forward(model, data)
                obj_z = data.body("object").xpos[2]
                if wp["chk"] in ("obj_lifted", "obj_lifted_b"):
                    if obj_z < obj_pos[2] + 0.01:   # 1cm 即可确认吸附成功
                        print(f"  ✗ {wp['n']}: 物体未抬起 (z={obj_z:.3f}, need>{obj_pos[2]+0.01:.3f})")
                        failed = True; break
                elif wp["chk"] == "obj_placed":
                    d = np.linalg.norm(data.body("object").xpos[:2] - tgt_pos[:2])
                    if d > 0.12:
                        print(f"  ✗ {wp['n']}: 未到目标 (d={d:.3f}m)")
                        failed = True; break
                print(f"  ✓ {wp['n']}")
                wp_i += 1
                continue

            # ── IK step ──
            if wp.get("type") == "joint":
                # Pure joint space interpolation to avoid any Cartesian deadlock
                ja_target = wp.get("ja", cmd_a)
                jb_target = wp.get("jb", cmd_b)
                
                # Step size per iteration
                step_size = 0.05
                
                cmd_a = cmd_a + np.clip(ja_target - cmd_a, -step_size, step_size)
                cmd_a = clamp_joints(cmd_a, model, act_a)
                
                cmd_b = cmd_b + np.clip(jb_target - cmd_b, -step_size, step_size)
                cmd_b = clamp_joints(cmd_b, model, act_b)
                
            else:
                # Normal Jacobian IK phase for approaching limits
                pa_target = np.array(wp.get("pa")) if wp.get("pa") is not None else None
                qa_target = wp.get("qa")
                pb_target = np.array(wp.get("pb")) if wp.get("pb") is not None else None
                qb_target = wp.get("qb")

                # 实时追踪物体位置 (A/B 的抓取阶段均适用)
                if wp.get("track_obj"):
                    mujoco.mj_forward(model, data)
                    actual_obj = data.body("object").xpos
                    if pa_target is not None:   # A robot grasping
                        pa_target = np.array([actual_obj[0] + 0.145,
                                              actual_obj[1],
                                              0.4122])
                    if pb_target is not None:   # B robot grasping
                        pb_target = np.array([actual_obj[0] + 0.145,
                                              actual_obj[1],
                                              0.415])

                # 关键: hold期间不运行IK，就地冻结关节，避免振荡撞飞物体
                if hold_cnt > 0:
                    # 已在允差内，正在计数，锁住关节即可
                    cmd_a = np.array(data.qpos[qpos_a])
                    cmd_b = np.array(data.qpos[qpos_b])
                else:
                    if pa_target is not None:
                        if qa_target is not None:
                            # 位置+姿态 IK
                            dq_a = pos_ori_ik_step(model, data, pa_target, qa_target, eef_a, dof_a)
                        else:
                            # 纯位置 IK (无姿态约束, 避免翘起来)
                            dq_a = pos_ik_step(model, data, pa_target, eef_a, dof_a)
                        cmd_a = cmd_a + dq_a * 0.8
                        actual_a = data.qpos[qpos_a]
                        cmd_a = np.clip(cmd_a, actual_a - 0.4, actual_a + 0.4)
                        cmd_a = clamp_joints(cmd_a, model, act_a)
                    # else: pa 未指定, A 机器人保持不动

                    if pb_target is not None:
                        if qb_target is not None:
                            dq_b = pos_ori_ik_step(model, data, pb_target, qb_target, eef_b, dof_b)
                        else:
                            dq_b = pos_ik_step(model, data, pb_target, eef_b, dof_b)
                        cmd_b = cmd_b + dq_b * 0.8
                        actual_b = data.qpos[qpos_b]
                        cmd_b = np.clip(cmd_b, actual_b - 0.4, actual_b + 0.4)
                        cmd_b = clamp_joints(cmd_b, model, act_b)
                    # else: B 机器人保持不动

            # ── 组装 20-dim action ──
            action = np.zeros(20)
            action[3:9]   = cmd_a            # Robot A: arm j1-j6 (rad)
            action[9]     = wp.get("ga", GRIPPER_OPEN)          # Robot A: gripper
            action[13:19] = cmd_b            # Robot B: arm j1-j6 (rad)
            action[19]    = wp.get("gb", GRIPPER_OPEN)          # Robot B: gripper

            obs, reward, terminated, _, info = env.step(action)

            # ── Data Collection (every 5 steps = 10Hz if sim_skip=8, env steps at 50Hz) ──
            if step_cnt % 5 == 0:
                if hasattr(env.unwrapped, "get_images"):
                     info.update(env.unwrapped.get_images())
                
                data_manager.append_single_data(DataKey.TIME, data.time)
                data_manager.append_single_data(DataKey.REWARD, reward)
                data_manager.append_single_data(DataKey.MEASURED_JOINT_POS, obs["joint_pos"])
                data_manager.append_single_data(DataKey.MEASURED_JOINT_VEL, obs["joint_vel"])
                data_manager.append_single_data(DataKey.MEASURED_MOBILE_OMNI_VEL, obs["mobile_vel"])
                if "wrench" in obs:
                     data_manager.append_single_data(DataKey.MEASURED_EEF_WRENCH, obs["wrench"])
                
                # Command tracking for behavior cloning
                # Actually, our observation space for joint_pos is 14 dim. 
                # action[3:9] is arm_a, action[9] is gripper_a
                # action[13:19] is arm_b, action[19] is gripper_b
                # Let's save the precise command layout matching obs layout:
                cmd_joint_pos = np.zeros(14)
                cmd_joint_pos[0:6] = action[3:9]
                cmd_joint_pos[6] = action[9]
                cmd_joint_pos[7:13] = action[13:19]
                cmd_joint_pos[13] = action[19]
                data_manager.append_single_data(DataKey.COMMAND_JOINT_POS, cmd_joint_pos)

                cmd_mobile_vel = np.zeros(6)
                cmd_mobile_vel[0:3] = action[0:3]
                cmd_mobile_vel[3:6] = action[10:13]
                data_manager.append_single_data(DataKey.COMMAND_MOBILE_OMNI_VEL, cmd_mobile_vel)

                if "rgb_images" in info:
                    for cam_name, img in info["rgb_images"].items():
                        data_manager.append_single_data(DataKey.get_rgb_image_key(cam_name), img)
                    
                    # Live GUI rendering for non-headless mode
                    if not args.headless:
                        # Stack images into a grid: 
                        # [[overhead, front], 
                        #  [robot_a,  robot_b]]
                        # Images are RGB, cv2 expects BGR
                        img_oh = cv2.cvtColor(info["rgb_images"].get("overhead", np.zeros((120, 160, 3), dtype=np.uint8)), cv2.COLOR_RGB2BGR)
                        img_fr = cv2.cvtColor(info["rgb_images"].get("front", np.zeros((120, 160, 3), dtype=np.uint8)), cv2.COLOR_RGB2BGR)
                        img_ra = cv2.cvtColor(info["rgb_images"].get("robot_a_wrist", np.zeros((120, 160, 3), dtype=np.uint8)), cv2.COLOR_RGB2BGR)
                        img_rb = cv2.cvtColor(info["rgb_images"].get("robot_b_wrist", np.zeros((120, 160, 3), dtype=np.uint8)), cv2.COLOR_RGB2BGR)
                        
                        top_row = np.hstack([img_oh, img_fr])
                        bot_row = np.hstack([img_ra, img_rb])
                        grid = np.vstack([top_row, bot_row])
                        
                        # Scale up for better visibility
                        grid = cv2.resize(grid, (0,0), fx=2.0, fy=2.0)
                        cv2.imshow("Dual Dingo-Z1 Auto Collection", grid)
                        cv2.waitKey(1)

                if "depth_images" in info:
                    for cam_name, img in info["depth_images"].items():
                        data_manager.append_single_data(DataKey.get_depth_image_key(cam_name), img)

            # ── Waypoint 推进 ──
            if wp.get("type") == "joint":
                # For joint space targets, use joint distance as threshold metric
                ja_target = wp.get("ja", cmd_a)
                jb_target = wp.get("jb", cmd_b)
                da = np.linalg.norm(data.qpos[qpos_a] - ja_target) if wp.get("ja") is not None else 0.0
                db = np.linalg.norm(data.qpos[qpos_b] - jb_target) if wp.get("jb") is not None else 0.0
            else:
                mujoco.mj_forward(model, data)
                da = np.linalg.norm(data.xpos[eef_a] - pa_target) if pa_target is not None else 0.0
                db = np.linalg.norm(data.xpos[eef_b] - pb_target) if pb_target is not None else 0.0
            
            dist = max(da, db)

            if dist < wp["tol"]:
                if "hold" in wp:
                    hold_cnt += 1
                    if hold_cnt >= wp["hold"]:
                        print(f"  → {wp['n']} done (step {step_cnt})")
                        wp_i += 1; hold_cnt = 0
                else:
                    print(f"  → {wp['n']} done (step {step_cnt})")
                    wp_i += 1

            # ── 高频调试：抓取阶段每步打印位置 ──
            wp_name = wp.get("n", "")
            if wp_name in ("A grasp reach", "A close grip"):
                mujoco.mj_forward(model, data)
                eef_pos   = data.xpos[eef_a]                    # link06 位置
                obj_now   = data.body("object").xpos             # 物体位置
                # 夹爪中心 ≈ link06 沿 -X 延伸 0.10m
                grip_dir  = data.xmat[eef_a].reshape(3, 3)[:, 0]  # link06 local X轴
                grip_cen  = eef_pos - grip_dir * 0.10
                dist_grip = np.linalg.norm(grip_cen - obj_now)
                print(f"    [step {step_cnt:3d}] link06={eef_pos.round(3)} "
                      f"gripper_center={grip_cen.round(3)} "
                      f"obj={obj_now.round(3)} dist={dist_grip:.3f}m")
            elif step_cnt % 50 == 0:
                print(f"  DEBUG: {wp['n']} | da={da:.3f} db={db:.3f} | hold={hold_cnt}")
            step_cnt += 1


        # ── 结果 ──
        if failed or wp_i < len(waypoints):
            print(f"  ✗ Discarded (wp={wp_i}/{len(waypoints)}, step={step_cnt})")
        else:
            saved += 1
            print(f"  ✅ Demo {saved:03d} done ({step_cnt} steps)")
            data_manager.save_data(f"auto_demos/dual_handover_{saved:03d}.rmb")

    env.close()
    print(f"\n{'='*50}")
    print(f"完成！收集 {saved} 个 demos。")
    print(f"{'='*50}")


if __name__ == "__main__":
    auto_collect_main()
