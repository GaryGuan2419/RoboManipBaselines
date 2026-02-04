#!/usr/bin/env python3
"""
Z1 MuJoCo Model Viewer
可视化检查 Z1 模型，测试关节运动
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # 加载模型
    print("加载 Z1 模型...")
    model = mujoco.MjModel.from_xml_path('z1_complete.xml')
    data = mujoco.MjData(model)
    
    print("\n=== 模型信息 ===")
    print(f"关节数: {model.njnt}")
    print(f"执行器数: {model.nu}")
    
    print("\n关节列表:")
    for i in range(model.njnt):
        joint = model.joint(i)
        jnt_range = model.jnt_range[i]
        print(f"  {i+1}. {joint.name}: [{jnt_range[0]:.3f}, {jnt_range[1]:.3f}]")
    
    # 设置到 home 位置
    mujoco.mj_resetDataKeyframe(model, data, 0)  # 使用 keyframe "home"
    
    print("\n=== 启动可视化 ===")
    print("控制说明:")
    print("  - 鼠标左键拖动: 旋转视角")
    print("  - 鼠标右键拖动: 平移视角")
    print("  - 滚轮: 缩放")
    print("  - 双击: 选择body")
    print("  - Ctrl+P: 暂停/继续")
    print("  - Backspace: 重置")
    print("\n关节会自动循环运动以展示模型...")
    print("按 Ctrl+C 或关闭窗口退出\n")
    
    # 创建viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 运行仿真
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            
            # 简单的正弦运动测试所有关节
            t = time.time() - start_time
            for i in range(model.nu):
                # 获取关节范围
                jnt_range = model.jnt_range[i]
                mid = (jnt_range[0] + jnt_range[1]) / 2
                amp = (jnt_range[1] - jnt_range[0]) / 4
                
                # 不同频率的正弦波
                freq = 0.3 + i * 0.1
                data.ctrl[i] = mid + amp * np.sin(2 * np.pi * freq * t)
            
            # 仿真步进
            mujoco.mj_step(model, data)
            
            # 同步viewer
            viewer.sync()
            
            # 控制帧率 (60 FPS)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已退出")
