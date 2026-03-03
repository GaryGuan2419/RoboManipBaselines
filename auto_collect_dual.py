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
import scipy.optimize
import mujoco
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robo_manip_baselines.envs  # registers gymnasium environments


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


# ─── Main ─────────────────────────────────────────────────────────────────────

def auto_collect_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    render_mode = None if args.headless else "human"
    print(f"[auto_collect] Creating environment (render_mode={render_mode})...")
    env = gym.make('robo_manip_baselines/MujocoDualDingoZ1HandoverEnv-v0',
                   render_mode=render_mode)

    model = env.unwrapped.model
    data  = env.unwrapped.data

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

    # Home pose (resting position)
    HOME_JOINTS = np.array([0.0, 0.785, -0.261, -0.523, 0.0, 0.0])
    GRIPPER_OPEN  =  0.0
    GRIPPER_CLOSE = -1.5   # range: [-1.51844, 0]

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
        obs, _ = env.reset()
        mujoco.mj_forward(model, data)

        obj_pos = data.body("object").xpos.copy()
        tgt_pos = data.body("target").xpos.copy()
        hz_pos  = data.body("handover_zone").xpos.copy()

        cmd_a = data.qpos[qpos_a].copy()
        cmd_b = data.qpos[qpos_b].copy()

        # Generate Kinematically Feasible Waypoints
        # For grasping, the wrist is 0.16m away from the object.
        # If object is at -X, wrist is at +0.16, pointing -X
        
        # ── Setup dynamic kinematics solver for pre-grasping joint configurations ──
        # Robot A Left (Points to -X / Table A)
        saved_qpos = data.qpos.copy()
        
        ja_pick_dyn = np.array([1.57, 1.0, -0.5, 0.0, -0.5, 0.0])
        data.qpos[qpos_a] = ja_pick_dyn
        mujoco.mj_kinematics(model, data)
        qa_pick = np.zeros(4)
        mujoco.mju_mat2Quat(qa_pick, data.xmat[eef_a])
        
        # Robot A Right (Points to +X / Table C)
        ja_place_dyn = np.array([-1.57, 1.0, -0.5, 0.0, -0.5, 0.0])
        data.qpos[qpos_a] = ja_place_dyn
        mujoco.mj_kinematics(model, data)
        qa_place = np.zeros(4)
        mujoco.mju_mat2Quat(qa_place, data.xmat[eef_a])

        # Robot B Left (Points to -X / Table C)
        jb_pick_dyn = np.array([1.57, 1.0, -0.5, 0.0, -0.5, 0.0])
        data.qpos[qpos_b] = jb_pick_dyn
        mujoco.mj_kinematics(model, data)
        qb_pick = np.zeros(4)
        mujoco.mju_mat2Quat(qb_pick, data.xmat[eef_b])

        # Robot B Right (Points to +X / Table B)
        jb_place_dyn = np.array([-1.57, 1.0, -0.5, 0.0, -0.5, 0.0])
        data.qpos[qpos_b] = jb_place_dyn
        mujoco.mj_kinematics(model, data)
        qb_place = np.zeros(4)
        mujoco.mju_mat2Quat(qb_place, data.xmat[eef_b])
        data.qpos[:] = saved_qpos
        mujoco.mj_kinematics(model, data)

        # ── Phase 1: Robot A picks from Table A (which is at -X) ──
        ax_grasp_a = obj_pos[0] + 0.16
        ax_hover_a = ax_grasp_a + 0.10

        # ── Phase 2: Robot A places on Table C (which is at +X) ──
        ax_place_c = hz_pos[0] - 0.16
        ax_hover_c = ax_place_c - 0.10
        
        # ── Phase 3: Robot B picks from Table C (which is at -X) ──
        bx_grasp_c = hz_pos[0] + 0.16
        bx_hover_c = bx_grasp_c + 0.10
        
        # ── Phase 4: Robot B places on Table B (which is at +X) ──
        bx_place_b = tgt_pos[0] - 0.16
        bx_hover_b = bx_place_b - 0.10

        waypoints = [
            # ── Phase 1: Robot A picks from Table A ────────────────
            {"n": "A joint pick",  "type": "joint", "ja": ja_pick_dyn, "ga": GRIPPER_OPEN, "tol": 0.05},
            {"n": "A hover obj",   "pa": [ax_hover_a, obj_pos[1], hover_z], "qa": qa_pick, "ga": GRIPPER_OPEN, "tol": 0.04},
            {"n": "A reach fwd",   "pa": [ax_grasp_a, obj_pos[1], grasp_z], "qa": qa_pick, "ga": GRIPPER_OPEN, "tol": 0.02},
            {"n": "A grasp",       "pa": [ax_grasp_a, obj_pos[1], grasp_z], "qa": qa_pick, "ga": GRIPPER_CLOSE, "tol": 0.04, "hold": 30},
            {"n": "A lift",        "pa": [ax_grasp_a, obj_pos[1], hover_z], "qa": qa_pick, "ga": GRIPPER_CLOSE, "tol": 0.04},
            {"n": "CHECK_lifted",  "type": "check", "chk": "obj_lifted"},

            # ── Phase 2: Robot A places on Table C ──────────────────
            {"n": "A joint place", "type": "joint", "ja": ja_place_dyn, "ga": GRIPPER_CLOSE, "tol": 0.05},
            {"n": "A to C hover",  "pa": [ax_hover_c, hz_pos[1], hover_z], "qa": qa_place, "ga": GRIPPER_CLOSE, "tol": 0.04},
            {"n": "A to C place",  "pa": [ax_place_c, hz_pos[1], grasp_z], "qa": qa_place, "ga": GRIPPER_CLOSE, "tol": 0.025},
            {"n": "A release",     "pa": [ax_place_c, hz_pos[1], grasp_z], "qa": qa_place, "ga": GRIPPER_OPEN, "tol": 0.04, "hold": 180},
            {"n": "A retreat",     "pa": [ax_hover_c, hz_pos[1], hover_z], "qa": qa_place, "ga": GRIPPER_OPEN, "tol": 0.03},
            {"n": "A Home",        "type": "joint", "ja": HOME_JOINTS, "ga": GRIPPER_OPEN, "tol": 0.05},

            # ── Phase 3: Robot B picks from Table C ──────────────────
            {"n": "B joint pick",  "type": "joint", "jb": jb_pick_dyn, "gb": GRIPPER_OPEN, "tol": 0.05},
            {"n": "B hover C",     "pb": [bx_hover_c, hz_pos[1], hover_z], "qb": qb_pick, "gb": GRIPPER_OPEN, "tol": 0.04},
            {"n": "B reach C",     "pb": [bx_grasp_c, hz_pos[1], grasp_z], "qb": qb_pick, "gb": GRIPPER_OPEN, "tol": 0.02},
            {"n": "B grasp",       "pb": [bx_grasp_c, hz_pos[1], grasp_z], "qb": qb_pick, "gb": GRIPPER_CLOSE, "tol": 0.04, "hold": 30},
            {"n": "B lift C",      "pb": [bx_grasp_c, hz_pos[1], hover_z], "qb": qb_pick, "gb": GRIPPER_CLOSE, "tol": 0.04},
            {"n": "CHECK_lifted_b","type": "check", "chk": "obj_lifted_b"},

            # ── Phase 4: Robot B places on Table B ───────────────────
            {"n": "B joint place", "type": "joint", "jb": jb_place_dyn, "gb": GRIPPER_CLOSE, "tol": 0.05},
            {"n": "B to B hover",  "pb": [bx_hover_b, tgt_pos[1], hover_z], "qb": qb_place, "gb": GRIPPER_CLOSE, "tol": 0.04},
            {"n": "B to B place",  "pb": [bx_place_b, tgt_pos[1], grasp_z], "qb": qb_place, "gb": GRIPPER_CLOSE, "tol": 0.025},
            {"n": "B release",     "pb": [bx_place_b, tgt_pos[1], grasp_z], "qb": qb_place, "gb": GRIPPER_OPEN, "tol": 0.04, "hold": 180},
            {"n": "B retreat",     "pb": [bx_hover_b, tgt_pos[1], hover_z], "qb": qb_place, "gb": GRIPPER_OPEN, "tol": 0.03},
            {"n": "B Home",        "type": "joint", "jb": HOME_JOINTS, "gb": GRIPPER_OPEN, "tol": 0.05},

            
            {"n": "CHECK_placed",  "type": "check", "chk": "obj_placed"},
        ]

        max_steps = 300
        wp_i, hold_cnt, step_cnt = 0, 0, 0
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
                    if obj_z < obj_pos[2] + 0.02:
                        print(f"  ✗ {wp['n']}: 物体未抬起 (z={obj_z:.3f})")
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

                if pa_target is not None and qa_target is not None:
                    dq_a = pos_ori_ik_step(model, data, pa_target, qa_target, eef_a, dof_a)
                    cmd_a = cmd_a + dq_a * 1.5
                    actual_a = data.qpos[qpos_a]
                    cmd_a = np.clip(cmd_a, actual_a - 0.4, actual_a + 0.4)
                    cmd_a = clamp_joints(cmd_a, model, act_a)
                else:
                    cmd_a = cmd_a + np.clip(HOME_JOINTS - cmd_a, -0.05, 0.05)

                if pb_target is not None and qb_target is not None:
                    dq_b = pos_ori_ik_step(model, data, pb_target, qb_target, eef_b, dof_b)
                    cmd_b = cmd_b + dq_b * 1.5
                    actual_b = data.qpos[qpos_b]
                    cmd_b = np.clip(cmd_b, actual_b - 0.4, actual_b + 0.4)
                    cmd_b = clamp_joints(cmd_b, model, act_b)
                else:
                    cmd_b = cmd_b + np.clip(HOME_JOINTS - cmd_b, -0.05, 0.05)

            # ── 组装 20-dim action ──
            action = np.zeros(20)
            action[3:9]   = cmd_a            # Robot A: arm j1-j6 (rad)
            action[9]     = wp.get("ga", GRIPPER_OPEN)          # Robot A: gripper
            action[13:19] = cmd_b            # Robot B: arm j1-j6 (rad)
            action[19]    = wp.get("gb", GRIPPER_OPEN)          # Robot B: gripper

            obs, _, *_ = env.step(action)

            # ── 进度日志 ──
            if step_cnt % 50 == 0:
                print(f"  DEBUG: {wp['n']} | da={da:.3f} | hold={hold_cnt}")
                mujoco.mj_forward(model, data)
                da = np.linalg.norm(data.xpos[eef_a] - (pa_target if pa_target is not None else data.xpos[eef_a])) if pa_target is not None else 0
                db = np.linalg.norm(data.xpos[eef_b] - (pb_target if pb_target is not None else data.xpos[eef_b])) if pb_target is not None else 0
                print(f"  step {step_cnt:4d}: [{wp['n']}] da={da:.3f} db={db:.3f}")

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

            if step_cnt % 50 == 0:
                print(f"  DEBUG: {wp['n']} | da={da:.3f} db={db:.3f} | hold={hold_cnt}")
            step_cnt += 1

        # ── 结果 ──
        if failed or wp_i < len(waypoints):
            print(f"  ✗ Discarded (wp={wp_i}/{len(waypoints)}, step={step_cnt})")
        else:
            saved += 1
            print(f"  ✅ Demo {saved:03d} done ({step_cnt} steps)")

    env.close()
    print(f"\n{'='*50}")
    print(f"完成！收集 {saved} 个 demos。")
    print(f"{'='*50}")


if __name__ == "__main__":
    auto_collect_main()
