#!/usr/bin/env python
"""
Dingo-Z1 Gripper & EEF Diagnostic Tool (standalone, no gym.make needed).
Directly loads the MuJoCo XML model to inspect gripper behavior and EEF offsets.
"""

import os
import numpy as np
import mujoco

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(
    SCRIPT_DIR,
    "robo_manip_baselines/envs/assets/mujoco/envs/dingo_z1/env_dingo_z1_grasp.xml",
)


def main():
    print("=" * 70)
    print("  Dingo-Z1 Gripper & EEF Diagnostic Tool")
    print("=" * 70)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # ── SECTION 1: Joint Inventory ──
    print(f"\n[1/5] ALL JOINTS ({model.njnt} total):")
    print(f"  {'Name':<25} {'Type':<8} {'Range':<30} {'qpos_adr':<10} {'dof_adr'}")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"unnamed_{i}"
        jtype = ['free', 'ball', 'slide', 'hinge'][model.jnt_type[i]]
        lo, hi = model.jnt_range[i]
        print(f"  {name:<25} {jtype:<8} [{lo:+.5f}, {hi:+.5f}]    {model.jnt_qposadr[i]:<10} {model.jnt_dofadr[i]}")

    # ── SECTION 2: Actuator Inventory ──
    print(f"\n[2/5] ALL ACTUATORS ({model.nu} total):")
    print(f"  {'Name':<20} {'Joint':<20} {'ctrlrange':<30} {'gain':<10} {'biasprm[:3]'}")
    for i in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
        jnt_id = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or "?"
        lo, hi = model.actuator_ctrlrange[i]
        gain = model.actuator_gainprm[i, 0]
        bias = model.actuator_biasprm[i, :3]
        print(f"  {aname:<20} {jname:<20} [{lo:+.4f}, {hi:+.4f}]   g={gain:<8.0f} b=[{bias[0]:.0f},{bias[1]:.0f},{bias[2]:.0f}]")

    # ── SECTION 3: EEF Body Chain Offsets ──
    print("\n[3/5] EEF BODY CHAIN POSITIONS (at init pose):")

    # Set initial arm pose matching the env: [j1, j2, j3, j4, j5, j6] = [0, 0.785, -0.261, -0.523, 0, 0]
    arm_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    init_arm_qpos = [0.0, 0.785, -0.261, -0.523, 0.0, 0.0]
    for jname, qval in zip(arm_joints, init_arm_qpos):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data.qpos[model.jnt_qposadr[jid]] = qval
    mujoco.mj_forward(model, data)

    for bname in ["link05", "link06", "gripperMover", "object", "target"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            print(f"  {bname:<20} NOT FOUND")
            continue
        pos = data.xpos[bid]
        mat = data.xmat[bid].reshape(3, 3)
        print(f"  {bname:<20} pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")
        print(f"  {'':<20} x-axis=[{mat[0,0]:+.3f}, {mat[1,0]:+.3f}, {mat[2,0]:+.3f}]  "
              f"z-axis=[{mat[0,2]:+.3f}, {mat[1,2]:+.3f}, {mat[2,2]:+.3f}]")

    l06_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link06")
    gm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripperMover")
    l06_pos = data.xpos[l06_id].copy()
    gm_pos = data.xpos[gm_id].copy()
    offset_vec = gm_pos - l06_pos
    print(f"\n  >>> link06 → gripperMover offset: [{offset_vec[0]:+.4f}, {offset_vec[1]:+.4f}, {offset_vec[2]:+.4f}]  dist={np.linalg.norm(offset_vec):.4f}m")

    # Stator pad center in link06 local frame
    l06_mat = data.xmat[l06_id].reshape(3, 3)
    stator_tip_local = np.array([0.186, 0.0, -0.0125])
    stator_tip_world = l06_pos + l06_mat @ stator_tip_local
    tip_off = stator_tip_world - l06_pos
    print(f"  >>> link06 → stator pad tip (world): [{tip_off[0]:+.4f}, {tip_off[1]:+.4f}, {tip_off[2]:+.4f}]  dist={np.linalg.norm(tip_off):.4f}m")

    # ── SECTION 4: Gripper Closing Response ──
    print("\n[4/5] GRIPPER CLOSING TEST (50 sim steps per command):")
    test_cmds = [0.0, -0.3, -0.5, -0.8, -1.0, -1.2, -1.5, -1.51844]

    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motorGripper")
    gripper_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jointGripper")
    gripper_qpos_adr = model.jnt_qposadr[gripper_jnt_id]

    # Also find arm actuators to hold position
    arm_act_names = ["motor1", "motor2", "motor3", "motor4", "motor5", "motor6"]
    arm_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in arm_act_names]

    for cmd in test_cmds:
        mujoco.mj_resetData(model, data)
        # Set arm pose
        for jname, qval in zip(arm_joints, init_arm_qpos):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            data.qpos[model.jnt_qposadr[jid]] = qval
        mujoco.mj_forward(model, data)

        # Run simulation with gripper command
        for step in range(50):
            # Hold arm joints at initial pose
            for aid, qval in zip(arm_act_ids, init_arm_qpos):
                data.ctrl[aid] = qval
            data.ctrl[gripper_act_id] = cmd
            mujoco.mj_step(model, data)

        gripper_qpos = data.qpos[gripper_qpos_adr]
        gm_pos_now = data.xpos[gm_id].copy()
        print(f"  cmd={cmd:+.6f} → jointGripper qpos={gripper_qpos:+.6f}  "
              f"gripperMover=[{gm_pos_now[0]:+.4f},{gm_pos_now[1]:+.4f},{gm_pos_now[2]:+.4f}]")

    # ── SECTION 5: Orientation at Seed Pose ──
    print("\n[5/5] EEF ORIENTATION AT SEED POSE [0, 1.0, -0.5, 0, -0.5, 0]:")
    mujoco.mj_resetData(model, data)
    seed_pose = [0.0, 1.0, -0.5, 0.0, -0.5, 0.0]
    for jname, qval in zip(arm_joints, seed_pose):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data.qpos[model.jnt_qposadr[jid]] = qval
    mujoco.mj_forward(model, data)

    l06_pos_s = data.xpos[l06_id].copy()
    l06_mat_s = data.xmat[l06_id].reshape(3, 3).copy()
    quat_s = np.zeros(4)
    mujoco.mju_mat2Quat(quat_s, data.xmat[l06_id])

    print(f"  link06 pos:  [{l06_pos_s[0]:+.4f}, {l06_pos_s[1]:+.4f}, {l06_pos_s[2]:+.4f}]")
    print(f"  link06 quat: [{quat_s[0]:+.4f}, {quat_s[1]:+.4f}, {quat_s[2]:+.4f}, {quat_s[3]:+.4f}]")
    print(f"  x-axis (gripper forward): [{l06_mat_s[0,0]:+.3f}, {l06_mat_s[1,0]:+.3f}, {l06_mat_s[2,0]:+.3f}]")
    print(f"  y-axis:                    [{l06_mat_s[0,1]:+.3f}, {l06_mat_s[1,1]:+.3f}, {l06_mat_s[2,1]:+.3f}]")
    print(f"  z-axis:                    [{l06_mat_s[0,2]:+.3f}, {l06_mat_s[1,2]:+.3f}, {l06_mat_s[2,2]:+.3f}]")
    print(f"\n  Dot(x_axis, [0,0,-1]) = {np.dot(l06_mat_s[:,0], [0,0,-1]):.3f}  (1.0 = perfectly downward)")

    # Object position
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_pos = data.xpos[obj_id]
    print(f"  Object at:  [{obj_pos[0]:+.4f}, {obj_pos[1]:+.4f}, {obj_pos[2]:+.4f}]")
    print(f"  link06 → Object dist: {np.linalg.norm(obj_pos - l06_pos_s):.4f}m")

    # gripperMover position at seed pose
    gm_pos_s = data.xpos[gm_id].copy()
    print(f"  gripperMover at seed: [{gm_pos_s[0]:+.4f}, {gm_pos_s[1]:+.4f}, {gm_pos_s[2]:+.4f}]")
    print(f"  gripperMover → Object dist: {np.linalg.norm(obj_pos - gm_pos_s):.4f}m")

    print("\n" + "=" * 70)
    print("  Diagnostic Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
