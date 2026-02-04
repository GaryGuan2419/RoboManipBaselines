#!/usr/bin/env python3
import mujoco
import os

os.chdir('/home/bguan/RoboManipBaselines/assets/dingo_description')

print("=== 验证 Dingo MuJoCo 模型 ===")
try:
    model = mujoco.MjModel.from_xml_path('dingo_complete.xml')
    
    print("✅ 模型加载成功！")
    print(f"\n模型信息:")
    print(f"  关节数: {model.njnt}")
    print(f"  自由度: {model.nv}")  
    print(f"  执行器: {model.nu}")
    print(f"  几何体: {model.ngeom}")
    print(f"  body数: {model.nbody}")
    
    # 列出关节名
    print(f"\n关节列表:")
    for i in range(model.njnt):
        print(f"  {i+1}. {model.joint(i).name}")
    
    # 列出执行器
    print(f"\n执行器列表:")
    for i in range(model.nu):
        print(f"  {i+1}. {model.actuator(i).name}")
    
    print("\n✅ Dingo 差速驱动底盘模型创建成功！")
    print(f"文件位置: {os.path.abspath('dingo_complete.xml')}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
