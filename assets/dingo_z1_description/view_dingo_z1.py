#!/usr/bin/env python3
"""
Dingo-Z1 Mobile Manipulator Viewer
移动双臂机器人可视化
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # 切换到正确的目录
    import os
    os.chdir('/home/bguan/RoboManipBaselines/assets/dingo_z1_description')
    
    # 加载模型
    print("加载 Dingo-Z1 移动双臂机器人...")
    model = mujoco.MjModel.from_xml_path('dingo_z1_complete.xml')
    data = mujoco.MjData(model)
    
    print("\n=== 模型信息 ===")
    print(f"总自由度: {model.nv} DOF")
    print(f"  - 移动底盘: 2 DOF (左右轮)")
    print(f"  - 机械臂: 6 DOF")
    print(f"  - 夹爪: 1 DOF")
    
    print("\n=== 启动3D可视化 ===")
    print("观看机器人执行复杂动作：")
    print("  1. 移动底盘前进/转向")
    print("  2. 机械臂伸展/收缩")
    print("  3. 夹爪开合")
    print("\n控制提示:")
    print("  - 鼠标拖动旋转视角")
    print("  - 滚轮缩放")
    print("  - Ctrl+C 退出")
    print()
    
    # 设置初始位置到 home
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 创建viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 运行仿真
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            t = time.time() - start_time
            
            # === 底盘运动 (差速驱动) ===
            # 模拟前进 + 缓慢转向
            forward_speed = 1.5 * np.sin(0.2 * t)
            turn_speed = 0.8 * np.sin(0.15 * t)
            
            data.ctrl[0] = forward_speed + turn_speed  # 左轮
            data.ctrl[1] = forward_speed - turn_speed  # 右轮
            
            # === 机械臂运动 (协调动作) ===
            # Joint 1: 底座旋转
            data.ctrl[2] = 1.0 * np.sin(0.3 * t)
            
            # Joint 2: 肩关节 (上下摆动)
            data.ctrl[3] = 0.785 + 0.5 * np.sin(0.25 * t)
            
            # Joint 3: 肘关节
            data.ctrl[4] = -0.5 - 0.3 * np.sin(0.28 * t)
            
            # Joint 4: 腕关节1
            data.ctrl[5] = 0.4 * np.sin(0.35 * t)
            
            # Joint 5: 腕关节2 (旋转)
            data.ctrl[6] = 0.6 * np.sin(0.4 * t)
            
            # Joint 6: 腕关节3
            data.ctrl[7] = 0.5 * np.sin(0.45 * t)
            
            # === 夹爪 (开合) ===
            data.ctrl[8] = -0.8 * (0.5 + 0.5 * np.sin(0.5 * t))
            
            # 仿真步进
            mujoco.mj_step(model, data)
            
            # 同步viewer
            viewer.sync()
            
            # 控制帧率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ 已退出可视化")
