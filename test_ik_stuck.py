import mujoco
from robo_manip_baselines.envs.mujoco.dingo_z1.MujocoDualDingoZ1HandoverEnv import MujocoDualDingoZ1HandoverEnv
import numpy as np
import scipy.optimize
import time

def solve_ik_global(model, data, eef_id, dof_indices, target_pos, target_quat, home_q):
    qpos_orig = data.qpos.copy()
    
    def objective(q):
        data.qpos[dof_indices] = q
        mujoco.mj_kinematics(model, data)
        pos = data.xpos[eef_id]
        pos_err = np.sum((pos - target_pos)**2)
        curr_quat = np.zeros(4)
        mujoco.mju_mat2Quat(curr_quat, data.xmat[eef_id])
        dot = np.dot(curr_quat, target_quat)
        ori_err = 1.0 - dot**2
        home_penalty = 0.01 * np.sum((q - home_q)**2)
        return pos_err * 100 + ori_err * 10 + home_penalty

    bounds = [(-2.6, 2.6), (0.0, 2.96), (-2.8, 0.0), (-1.5, 1.5), (-1.3, 1.3), (-2.7, 2.7)]
    t0 = time.time()
    res = scipy.optimize.minimize(objective, x0=home_q, method='SLSQP', bounds=bounds, options={'maxiter': 100})
    t1 = time.time()
    
    data.qpos[:] = qpos_orig
    mujoco.mj_kinematics(model, data)
    print(f"SLSQP took {t1-t0:.2f}s, Success: {res.success}, Iter: {res.nit}")
    return res.x

def test_dynamic_ik():
    env = MujocoDualDingoZ1HandoverEnv(render_mode=None)
    model = env.unwrapped.model
    data = env.unwrapped.data
    env.reset()
    mujoco.mj_forward(model, data)

    dof_a = [model.jnt_dofadr [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'robot_a/joint{i}')] for i in range(1, 7)]
    qpos_a = [model.jnt_qposadr [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'robot_a/joint{i}')] for i in range(1, 7)]
    eef_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot_a/link06')
    HOME_JOINTS = np.array([0.0, 0.785, -0.261, -0.523, 0.0, 0.0])

    obj_pos = data.body("object").xpos.copy()
    ax_hover = obj_pos[0] + 0.16
    hover_z = 0.36 + 0.18
    pa_target = np.array([ax_hover, obj_pos[1], hover_z])

    qa_pick = np.zeros(4)
    saved_qpos = data.qpos.copy()
    data.qpos[qpos_a] = [1.57, 1.0, -0.5, 0.0, -0.5, 0.0]
    mujoco.mj_kinematics(model, data)
    mujoco.mju_mat2Quat(qa_pick, data.xmat[eef_a])
    data.qpos[:] = saved_qpos
    mujoco.mj_kinematics(model, data)
    
    q_opt = solve_ik_global(model, data, eef_a, dof_a, pa_target, qa_pick, HOME_JOINTS)
    print(f"Optimized Joints: {q_opt.round(3)}")

if __name__ == '__main__':
    test_dynamic_ik()
