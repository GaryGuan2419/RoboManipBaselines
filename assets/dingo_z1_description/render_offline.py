#!/usr/bin/env python3
"""
离线渲染 Dingo-Z1 模型
生成图片而不是打开窗口
"""
import mujoco
import numpy as np
import os
from PIL import Image

os.chdir('/home/bguan/RoboManipBaselines/assets/dingo_z1_description')

print("=== 离线渲染 Dingo-Z1 ===")

# 加载模型
model = mujoco.MjModel.from_xml_path('dingo_z1_complete.xml')
data = mujoco.MjData(model)

# 重置到 home 位置
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print("✅ 模型加载成功")

# 创建离线渲染器（使用较小分辨率以适应默认framebuffer）
width, height = 640, 480
renderer = mujoco.Renderer(model, height=height, width=width)

output_dir = "renders"
os.makedirs(output_dir, exist_ok=True)

print(f"\n渲染图片到 {output_dir}/ ...")

# 简单渲染一个默认视角
renderer.update_scene(data)
pixels = renderer.render()

# 保存为图片
img = Image.fromarray(pixels)
filepath = f"{output_dir}/dingo_z1_default.png"
img.save(filepath)
print(f"  ✅ {filepath}")

print(f"\n✅ 完成！图片保存在:")
print(f"   {os.path.abspath(output_dir)}/")
print(f"\n你可以用图片查看器打开这些文件查看机器人模型！")
