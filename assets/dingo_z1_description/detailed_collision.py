#!/usr/bin/env python3
"""
详细碰撞诊断 - 找出是哪个几何体穿透地面
"""
import mujoco
import os

os.chdir('/home/bguan/RoboManipBaselines/assets/dingo_z1_description')

print("=== 详细碰撞诊断 ===")

model = mujoco.MjModel.from_xml_path('dingo_z1_complete.xml')
data = mujoco.MjData(model)

# 重置到 home 位置
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print(f"\n检测到 {data.ncon} 个碰撞:\n")

for i in range(data.ncon):
    contact = data.contact[i]
    geom1_id = contact.geom1
    geom2_id = contact.geom2
    
    # 获取几何体名称
    geom1_name = model.geom(geom1_id).name if geom1_id >= 0 else "未命名"
    geom2_name = model.geom(geom2_id).name if geom2_id >= 0 else "未命名"
    
    # 获取几何体所属body
    geom1_body = model.geom(geom1_id).bodyid if geom1_id >= 0 else -1
    geom2_body = model.geom(geom2_id).bodyid if geom2_id >= 0 else -1
    
    body1_name = model.body(geom1_body).name if geom1_body >= 0 else "world"
    body2_name = model.body(geom2_body).name if geom2_body >= 0 else "world"
    
    dist = contact.dist
    
    print(f"碰撞 {i+1}:")
    print(f"  Geom1: {geom1_name} (body: {body1_name})")
    print(f"  Geom2: {geom2_name} (body: {body2_name})")
    print(f"  穿透距离: {dist*1000:.2f}mm")
    print(f"  位置: {contact.pos}")
    
    if dist < -0.001:
        print(f"  ❌ 严重穿透！\n")
    else:
        print(f"  ✅ 正常接触\n")

# 检查后万向轮位置
print("\n=== 后万向轮检查 ===")
caster_z = data.xpos[model.body('rear_caster').id][2]
caster_radius = 0.01  # 球半径
caster_bottom = caster_z - caster_radius

print(f"后万向轮中心Z: {caster_z:.6f}m")
print(f"后万向轮底部Z: {caster_bottom:.6f}m")

if caster_bottom < -0.001:
    print(f"❌ 后万向轮穿地 {abs(caster_bottom)*1000:.2f}mm！")
elif caster_bottom > 0.001:
    print(f"⚠️  后万向轮离地 {caster_bottom*1000:.2f}mm")
else:
    print("✅ 后万向轮接触正常")
