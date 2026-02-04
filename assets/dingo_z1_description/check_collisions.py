#!/usr/bin/env python3
"""
检查模型碰撞和穿透问题
"""
import mujoco
import os

os.chdir('/home/bguan/RoboManipBaselines/assets/dingo_z1_description')

print("=== 检查 Dingo-Z1 模型 ===")

try:
    model = mujoco.MjModel.from_xml_path('dingo_z1_complete.xml')
    data = mujoco.MjData(model)
    
    print("✅ 模型加载成功")
    
    # 重置到 home 位置
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 前向运动学
    mujoco.mj_forward(model, data)
    
    # 检查碰撞
    print(f"\n碰撞检查:")
    print(f"  接触对数: {data.ncon}")
    
    if data.ncon > 0:
        print(f"\n⚠️  检测到 {data.ncon} 个接触/碰撞:")
        for i in range(min(data.ncon, 10)):  # 最多显示10个
            contact = data.contact[i]
            geom1 = model.geom(contact.geom1).name
            geom2 = model.geom(contact.geom2).name
            dist = contact.dist
            print(f"    {i+1}. {geom1} <-> {geom2}, 距离: {dist:.6f}m")
            
            if dist < -0.001:  # 穿透超过1mm
                print(f"       ❌ 严重穿透！")
    else:
        print("  ✅ 无碰撞")
    
    # 检查各body位置
    print(f"\n关键部件Z坐标:")
    print(f"  base_link: {data.xpos[model.body('base_link').id][2]:.4f}m")
    print(f"  chassis_link: {data.xpos[model.body('chassis_link').id][2]:.4f}m")
    print(f"  z1_mount: {data.xpos[model.body('z1_mount').id][2]:.4f}m")
    print(f"  link00: {data.xpos[model.body('link00').id][2]:.4f}m")
    print(f"  left_wheel: {data.xpos[model.body('left_wheel_link').id][2]:.4f}m")
    
    # 计算轮子底部
    wheel_center_z = data.xpos[model.body('left_wheel_link').id][2]
    wheel_radius = 0.049
    wheel_bottom = wheel_center_z - wheel_radius
    print(f"\n轮子底部Z: {wheel_bottom:.4f}m")
    
    if wheel_bottom > 0.001:
        print(f"❌ 轮子离地 {wheel_bottom:.4f}m")
    elif wheel_bottom < -0.001:
        print(f"❌ 轮子穿地 {abs(wheel_bottom):.4f}m")
    else:
        print("✅ 轮子正确接触地面")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
