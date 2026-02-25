#!/usr/bin/env python
"""
Robust Pure Python Damped Least Squares IK Auto Collector for MujocoDingoZ1GraspEnv
Generates high-quality demonstrations using MuJoCo's built-in Jacobian inverse kinematics.
Features strict error-clipping to prevent integral windup, and explicit success validation.
"""

import os
import sys
import numpy as np
import mujoco
from tqdm import tqdm
import gymnasium as gym

# Setup imports correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robo_manip_baselines.common.manager.DataManager import DataManager, DataKey

def compute_ik_step(model, data, target_pos, target_quat, eef_id, arm_dof_indices, damping=0.01):
    """
    Computes a small, safe joint-space step towards the target pose using DLS IK.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, eef_id)
    
    J_p = jacp[:, arm_dof_indices]
    J_r = jacr[:, arm_dof_indices]
    J = np.vstack([J_p, J_r])
    
    current_pos = data.xpos[eef_id]
    current_mat = data.xmat[eef_id]
    current_quat = np.zeros(4)
    mujoco.mju_mat2Quat(current_quat, current_mat)
    
    err_pos = target_pos - current_pos
    err_rot = np.zeros(3)
    mujoco.mju_subQuat(err_rot, target_quat, current_quat)
    
    # --- STRICT ERROR CLIPPING TO PREVENT INSTABILITY (Integral Windup) ---
    # We never want the IK to compute a massive jump if the target is far away.
    # We treat the error as a bounded "velocity" pulling the end-effector.
    step_err_pos = np.clip(err_pos * 2.0, -0.01, 0.01)   # Max 1cm Cartesian movement per step
    step_err_rot = np.clip(err_rot * 0.5, -0.05, 0.05)   # Max rotation movement per step
    
    # Weight orientation vs position
    err = np.hstack([step_err_pos, step_err_rot])
    
    # DLS formula: dq = J.T * (J * J.T + lambda * I)^-1 * err
    I = np.eye(6)
    delta_q = J.T @ np.linalg.inv(J @ J.T + damping**2 * I) @ err
    
    return delta_q

def auto_collect_main():
    num_episodes = 50
    max_steps_per_episode = 400
    env_id = 'robo_manip_baselines/MujocoDingoZ1GraspEnv-v0'
    
    print(f"Creating environment {env_id} with rgb_array render mode...")
    env = gym.make(env_id, render_mode="rgb_array")
    
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    arm_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    arm_dof_indices = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in arm_joints]
    arm_qpos_indices = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in arm_joints]
    
    eef_name = "link06"
    eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, eef_name)
    
    print("Setting up DataManager...")
    data_manager = DataManager(env, demo_name="AutoDingoZ1Grasp", task_desc="Robust Auto Collection with Strict Validation")
    data_manager.setup_camera_info()
    
    results = {"success": [], "reward": [], "duration": []}
    
    # Clean output dir
    os.makedirs("auto_demos", exist_ok=True)
    
    ep = 0
    saved_episodes = 0
    
    while saved_episodes < num_episodes:
        ep += 1
        obs, info = env.reset()
        data_manager.reset()
        
        # Read exact starting positions of objects
        obj_pos_init = data.body("object").xpos.copy()
        tgt_pos_init = data.body("target").xpos.copy()
        
        # We define a strict downward target orientation
        # (Assuming initial env reset is safe to extract a downwardish orientation)
        saved_qpos = data.qpos.copy()
        data.qpos[arm_qpos_indices] = [0.0, 1.0, -0.5, 0.0, -0.5, 0.0]
        mujoco.mj_kinematics(model, data)
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, data.xmat[eef_id])
        data.qpos[:] = saved_qpos
        mujoco.mj_kinematics(model, data)
        
        # Waypoint Sequence (Includes explicit validation 'checks')
        waypoints = [
            {"name": "Hover Obj", "pos": obj_pos_init + np.array([0, 0, 0.20]), "gripper": 1.0, "tol": 0.03},
            {"name": "Descend",   "pos": obj_pos_init + np.array([0, 0, 0.00]), "gripper": 1.0, "tol": 0.10},
            {"name": "Grasp Wait","pos": obj_pos_init + np.array([0, 0, 0.00]), "gripper": -0.8, "tol": 0.10, "wait": 15},
            {"name": "Lift",      "pos": obj_pos_init + np.array([0, 0, 0.25]), "gripper": -0.8, "tol": 0.03},
            {"name": "VALIDATE_GRASP", "type": "check", "check_type": "lifted"}, # Explicit Check
            {"name": "Hover Tgt", "pos": tgt_pos_init + np.array([0, 0, 0.25]), "gripper": -0.8, "tol": 0.03},
            {"name": "Lower Tgt", "pos": tgt_pos_init + np.array([0, 0, 0.00]), "gripper": -0.8, "tol": 0.10},
            {"name": "Drop Wait", "pos": tgt_pos_init + np.array([0, 0, 0.00]), "gripper": 1.0, "tol": 0.10, "wait": 10},
            {"name": "Lift Away", "pos": tgt_pos_init + np.array([0, 0, 0.20]), "gripper": 1.0, "tol": 0.03},
            {"name": "VALIDATE_PLACE", "type": "check", "check_type": "placed"}, # Explicit Check
        ]
        
        wp_idx = 0
        wait_counter = 0
        duration = 0
        episode_failed = False
        
        current_cmd_q = data.qpos[arm_qpos_indices].copy()
        
        pbar = tqdm(total=len(waypoints), desc=f"Attempt {ep} (Saved {saved_episodes}/{num_episodes})")
        
        while wp_idx < len(waypoints) and duration < max_steps_per_episode:
            wp = waypoints[wp_idx]
            
            # --- Strict Validation Checks ---
            if wp.get("type") == "check":
                if wp["check_type"] == "lifted":
                    # Check if object is physically lifted off ground
                    curr_obj_z = data.body("object").xpos[2]
                    if curr_obj_z < obj_pos_init[2] + 0.02:  # At least 2cm up
                        print(f"\n[Validation Failed] Object slipped or not lifted! (Z={curr_obj_z:.3f})")
                        episode_failed = True
                        break
                elif wp["check_type"] == "placed":
                    # Check if object is close to target horizontally
                    curr_obj_pos = data.body("object").xpos.copy()
                    dist_to_tgt = np.linalg.norm(curr_obj_pos[:2] - tgt_pos_init[:2])
                    if dist_to_tgt > 0.05: # Must be within 5cm radius
                        print(f"\n[Validation Failed] Object missed the target! (Dist={dist_to_tgt:.3f}m)")
                        episode_failed = True
                        break
                wp_idx += 1
                pbar.update(1)
                continue

            # --- Jacobian IK calculation ---
            # Compute limited delta step based on PHYSICAL state
            delta_q = compute_ik_step(model, data, wp["pos"], target_quat, eef_id, arm_dof_indices, damping=0.08)
            
            # Pure integration to overcome gravity (Allows PD to fight steady-state error)
            current_cmd_q += delta_q * 1.5
            
            # Anti-windup: limit the command to be at most 0.4 rad ahead of the physical arm
            # This allows strong PD response without violent crashing (like 10 rad errors)
            actual_q = data.qpos[arm_qpos_indices]
            current_cmd_q = np.clip(current_cmd_q, actual_q - 0.4, actual_q + 0.4)
            
            # Formulate the 10D action [vx, vy, vtheta, q1, q2, q3, q4, q5, q6, gripper]
            action = np.zeros(10)
            action[3:9] = current_cmd_q
            action[9] = wp["gripper"]
            
            # Step the environment
            obs, reward, terminated, _, info = env.step(action)
                
            # Render images & Log data at ~12.5Hz (skip frames to save time/space)
            if duration % 5 == 0:
                if hasattr(env.unwrapped, "get_images"):
                    info.update(env.unwrapped.get_images())
                    
                data_manager.append_single_data(DataKey.TIME, data.time)
                data_manager.append_single_data(DataKey.REWARD, reward)
                data_manager.append_single_data(DataKey.MEASURED_JOINT_POS, obs["joint_pos"])
                data_manager.append_single_data(DataKey.MEASURED_JOINT_VEL, obs["joint_vel"])
                data_manager.append_single_data(DataKey.MEASURED_MOBILE_OMNI_VEL, obs["mobile_vel"])
                if "wrench" in obs:
                     data_manager.append_single_data(DataKey.MEASURED_EEF_WRENCH, obs["wrench"])
                
                current_quat = np.zeros(4)
                mujoco.mju_mat2Quat(current_quat, data.xmat[eef_id])
                measured_eef_pose = np.concatenate([data.xpos[eef_id], current_quat])
                data_manager.append_single_data(DataKey.MEASURED_EEF_POSE, measured_eef_pose)
                
                if "rgb_images" in info:
                    for cam_name, img in info["rgb_images"].items():
                        data_manager.append_single_data(DataKey.get_rgb_image_key(cam_name), img)
                if "depth_images" in info:
                    for cam_name, img in info["depth_images"].items():
                        data_manager.append_single_data(DataKey.get_depth_image_key(cam_name), img)
                
                # Raw action vector not saved as 'ACTION', but captured via COMMAND_ keys below
                
                if hasattr(env.unwrapped, "command_keys_to_save"):
                    command_keys = env.unwrapped.command_keys_to_save
                    for key in command_keys:
                        if key == DataKey.COMMAND_JOINT_POS:
                            data_manager.append_single_data(key, current_cmd_q.copy())
                        elif key == DataKey.COMMAND_GRIPPER_JOINT_POS:
                            data_manager.append_single_data(key, np.array([wp["gripper"]]))
                        elif key == DataKey.COMMAND_EEF_POSE:
                            eef_pose = np.concatenate([data.xpos[eef_id], target_quat]) 
                            data_manager.append_single_data(key, eef_pose)
                        elif key == DataKey.COMMAND_MOBILE_OMNI_VEL:
                            data_manager.append_single_data(key, np.zeros(3))
            
            # Waypoint Transition Logic
            dist = np.linalg.norm(data.xpos[eef_id] - wp["pos"])
            if "wait" in wp:
                if dist < wp["tol"]:
                    wait_counter += 1
                    if wait_counter >= wp["wait"]:
                        wp_idx += 1
                        wait_counter = 0
                        pbar.update(1)
            else:
                if dist < wp["tol"]:
                    wp_idx += 1
                    pbar.update(1)
            
            duration += 1
            
        pbar.close()
        
        # Determine strict success
        if episode_failed or wp_idx < len(waypoints):
            print(f"Discarding attempt {ep} due to failure or timeout ({duration} steps).")
        else:
            print(f"✅ Success! Saved demo {saved_episodes:03d} ({duration} steps).")
            # Only save successful episodes
            data_manager.save_data(f"auto_demos/auto_grasp_{saved_episodes:03d}.hdf5")
            saved_episodes += 1
            results["success"].append(True)
            results["duration"].append(duration)
            
        print("-" * 50)
        
    env.close()
    
    print(f"\n==========================================")
    print(f"Data Collection Completed!")
    print(f"Total Flawless Demos Saved: {saved_episodes}")
    print(f"Average Duration: {np.mean(results['duration']):.1f} steps")
    print(f"==========================================")

if __name__ == "__main__":
    auto_collect_main()
