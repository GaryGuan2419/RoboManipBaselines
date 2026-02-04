#!/usr/bin/env python3
"""
Dingo MuJoCo Model Viewer
可视化检查 Dingo 移动底盘模型
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # 加载模型
    print("加载 Dingo 模型...")
    model = mujoco.MjModel.from_xml_path('dingo_complete.xml')
    data = mujoco.MjData(model)
    
    print("\n=== 模型信息 ===")
    print(f"关节数: {model.njnt}")
    print(f"执行器数: {model.nu}")
    print(f"几何体数: {model.ngeom}")
    
    print("\n关节列表:")
    for i in range(model.njnt):
        joint = model.joint(i)
        print(f"  {i+1}. {joint.name} (type: hinge)")
    
    print("\n执行器列表:")
    for i in range(model.nu):
        print(f"  {i+1}. {model.actuator(i).name}")
    
    print("\n=== 启动可视化 ===")
    print("控制说明:")
    print("  - 鼠标拖动: 旋转/平移视角")
    print("  - 轮子会模拟差速驱动运动")
    print("\n按 Ctrl+C 或关闭窗口退出\n")
    
    # 创建viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 运行仿真
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            
            # 模拟差速驱动：前进 + 左转
            t = time.time() - start_time
            
            # 前进速度 + 转弯
            forward_vel = 2.0 * np.sin(0.5 * t)
            turn_vel = 1.0 * np.sin(0.3 * t)
            
            # 差速驱动公式
            data.ctrl[0] = forward_vel + turn_vel  # 左轮
            data.ctrl[1] = forward_vel - turn_vel  # 右轮
            
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
        print("\n\n已退出")
