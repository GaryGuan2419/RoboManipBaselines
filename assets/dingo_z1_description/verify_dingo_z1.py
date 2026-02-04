#!/usr/bin/env python3
"""
Dingo-Z1 Mobile Manipulator Verification
验证组合模型
"""
import mujoco
import os

os.chdir('/home/bguan/RoboManipBaselines/assets/dingo_z1_description')

print("=== 验证 Dingo-Z1 MuJoCo 模型 ===")
try:
    model = mujoco.MjModel.from_xml_path('dingo_z1_complete.xml')
    
    print("✅ 组合模型加载成功！")
    print(f"\n模型信息:")
    print(f"  关节数: {model.njnt} (应该是9: 2轮 + 6臂 + 1夹爪)")
    print(f"  自由度: {model.nv}")  
    print(f"  执行器: {model.nu} (应该是9)")
    print(f"  几何体: {model.ngeom}")
    print(f"  body数: {model.nbody}")
    
    # 列出所有关节
    print(f"\n关节列表:")
    print("  Dingo (移动底盘):")
    for i in range(2):
        print(f"    {i+1}. {model.joint(i).name}")
    
    print("  Z1 (机械臂):")
    for i in range(2, model.njnt):
        print(f"    {i+1}. {model.joint(i).name}")
    
    # 列出执行器
    print(f"\n执行器列表:")
    print("  Dingo:")
    for i in range(2):
        print(f"    {i+1}. {model.actuator(i).name}")
    
    print("  Z1:")
    for i in range(2, model.nu):
        print(f"    {i+1}. {model.actuator(i).name}")
    
    print("\n✅ Dingo-Z1 移动双臂机器人模型创建成功！")
    print(f"文件位置: {os.path.abspath('dingo_z1_complete.xml')}")
    
    print("\n📊 系统总览:")
    print(f"  - 移动底盘: Dingo 差速驱动 (2 DOF)")
    print(f"  - 机械臂: Z1 6自由度")
    print(f"  - 末端执行器: 单指夹爪 (1 DOF)")
    print(f"  - 总自由度: {model.nv} DOF")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
