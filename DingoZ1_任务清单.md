# Dingo+Z1 语言驱动移动交接任务 - 完整任务清单

## 📋 Phase 0: 环境准备与基础验证

### 0.1 获取机器人模型
- [ ] 获取 Dingo 底座的 URDF/MJCF 模型
- [ ] 获取宇树 Z1 机械臂的 URDF/MJCF 模型
- [ ] 确认真实机器人的具体规格
  - [ ] Dingo 尺寸、轮距、最大速度
  - [ ] Z1 关节限位、夹爪规格

### 0.2 MuJoCo 建模
- [ ] 创建 `dingo_base.xml`（移动底座）
- [ ] 创建 `z1_arm.xml`（机械臂）
- [ ] 组合为 `dingo_z1.xml`（完整机器人）
- [ ] 添加摄像头、传感器定义

### 0.3 基础验证（仿真）
- [ ] 在 MuJoCo 中加载模型，检查可视化
- [ ] 用键盘控制 Dingo 移动（前后左右）
- [ ] 用 SpaceMouse/键盘控制 Z1 关节
- [ ] 验证夹爪开合
- [ ] 检查碰撞检测是否正常

**验证标准：** 模型能正常加载，手动控制无异常

---

## 📋 Phase 1: 创建环境类

### 1.1 单机抓取环境
- [ ] 创建 `MujocoDingoZ1GraspEnv.py`
  - [ ] 继承 `MujocoEnvBase`
  - [ ] 定义观测空间（摄像头 + 关节角度）
  - [ ] 定义动作空间（10维：vx, vy, vθ, 6关节, 夹爪）
  - [ ] 实现 `_get_reward()` 方法（抓取成功判定）
  - [ ] 实现 `modify_world()` 方法（随机物体位置）

### 1.2 双机交接环境
- [ ] 创建 `MujocoDualDingoZ1HandoverEnv.py`
  - [ ] 定义两个机器人的观测空间
  - [ ] 定义两个机器人的动作空间（20维）
  - [ ] 实现交接成功判定逻辑

### 1.3 注册环境
- [ ] 在 `robo_manip_baselines/envs` 中注册新环境
- [ ] 测试环境初始化和重置

**验证标准：** `python -c "import robo_manip_baselines.envs.mujoco.dingo_z1"` 无报错

---

## 📋 Phase 2: 单机技能训练（仿真）

### 2.1 数据收集 - 移动抓取
- [ ] 运行 Teleop 工具
  ```bash
  python ./bin/Teleop.py MujocoDingoZ1Grasp --input_device spacemouse
  ```
- [ ] 收集 30-50 个成功演示
  - [ ] 覆盖不同物体位置
  - [ ] 覆盖不同初始姿态
- [ ] 检查数据质量（播放数据，确认无错误）

### 2.2 训练 ManiFlow
- [ ] 运行训练脚本
  ```bash
  python ./bin/Train.py ManiFlowPolicy image \
    --dataset_dir ./dataset/DingoZ1Grasp_DatasetXX \
    --checkpoint_dir ./checkpoint/ManiFlowPolicy/DingoZ1_Grasp
  ```
- [ ] 监控训练损失（应稳定下降）
- [ ] 保存训练日志和曲线

### 2.3 仿真验证
- [ ] 运行推理测试
  ```bash
  python ./bin/Rollout.py ManiFlowPolicy MujocoDingoZ1Grasp \
    --checkpoint ./checkpoint/ManiFlowPolicy/DingoZ1_Grasp/policy_last.ckpt
  ```
- [ ] 测试 20+ 次，记录成功率
- [ ] 分析失败案例（物体掉落？导航失败？抓取失败？）

**成功标准：** 仿真成功率 ≥ 70%

---

## 📋 Phase 3: MLLM 框架搭建（并行开发）

### 3.1 设计技能库接口
- [ ] 创建 `skill_library.py`
- [ ] 定义基础技能接口
  - [ ] `grasp(object_name, object_location)`
  - [ ] `navigate_to(location)`
  - [ ] `place(location)`
  - [ ] `wait()`

### 3.2 实现 GPT-4 规划器
- [ ] 创建 `language_planner.py`
- [ ] 设计 System Prompt
- [ ] 实现场景描述生成（从图像提取信息）
- [ ] 实现任务规划函数
  ```python
  plan = planner.plan(instruction, scene_image)
  # 返回: [{"skill": "grasp", "params": {...}}, ...]
  ```

### 3.3 单机端到端测试
- [ ] 整合 GPT-4 + 单机 ManiFlow
- [ ] 测试简单指令："去桌子拿红色杯子"
- [ ] 验证规划 → 执行链路

**成功标准：** 能正确解析语言指令并生成技能序列

---

## 📋 Phase 4: 双机协作训练（仿真）

### 4.1 双机环境验证
- [ ] 加载双机环境
- [ ] 手动控制两个机器人（确认同步正常）
- [ ] 设计交接区域和姿态

### 4.2 数据收集 - 交接任务
- [ ] 运行双机 Teleop
  ```bash
  python ./bin/Teleop.py MujocoDualDingoZ1Handover
  ```
- [ ] 收集 40-60 个成功演示
  - [ ] Robot1: 抓取 → 移动到交接区 → 等待
  - [ ] Robot2: 移动到交接区 → 接收 → 移动到目标 → 放置

### 4.3 训练双机 ManiFlow
- [ ] 训练 Robot1 策略（抓取+移动到交接区）
- [ ] 训练 Robot2 策略（接收+移动+放置）
- [ ] 或训练联合策略（20维动作空间）

### 4.4 仿真验证
- [ ] 测试双机协作成功率
- [ ] 测试交接的稳定性和成功率

**成功标准：** 仿真双机交接成功率 ≥ 60%

---

## 📋 Phase 5: MLLM 双机集成

### 5.1 扩展技能库
- [ ] 添加双机技能
  - [ ] `handover(from_robot, to_robot, object)`
  - [ ] `receive(robot_id, from_location)`

### 5.2 双机端到端测试
- [ ] 测试复杂指令："Robot1去左边拿杯子交给Robot2，然后Robot2放到右边桌子上"
- [ ] 验证规划合理性
- [ ] 验证执行流畅性

**成功标准：** 能完成完整的双机语言驱动任务

---

## 📋 Phase 6: 真实机器人部署

### 6.1 真实环境准备
- [ ] 确认实验室场地布置
- [ ] 标定摄像头位置
- [ ] 设置安全区域和急停装置
- [ ] 测试 Dingo+Z1 真机基础控制

### 6.2 Sim-to-Real 数据收集
- [ ] 在真实环境收集少量数据（10-15个演示）
  - [ ] 单机抓取任务
  - [ ] 双机交接任务
- [ ] 对比真实数据和仿真数据的差异

### 6.3 模型微调
- [ ] 从仿真模型开始微调
  ```bash
  python ./bin/Train.py ManiFlowPolicy image \
    --dataset_dir ./dataset/RealDingoZ1_DatasetXX \
    --load_checkpoint ./checkpoint/.../policy_last.ckpt
  ```

### 6.4 真实环境验证
- [ ] 单机抓取测试（成功率目标 ≥ 50%）
- [ ] 双机交接测试（成功率目标 ≥ 40%）
- [ ] 完整语言驱动任务演示

**成功标准：** 真实环境能完成基本的语言驱动任务

---

## 📋 Phase 7: 系统优化与论文

### 7.1 性能优化
- [ ] 分析失败案例
- [ ] 调整超参数
- [ ] 收集更多边界case数据

### 7.2 实验与评估
- [ ] 设计评估指标（成功率、时间、鲁棒性）
- [ ] 对比实验（有无语言、单机vs双机）
- [ ] 记录定量结果

### 7.3 演示准备
- [ ] 录制演示视频
- [ ] 准备PPT/海报
- [ ] 撰写毕业论文

---

## 🎯 关键里程碑检查点

- [ ] Milestone 1: MuJoCo 模型可视化和手动控制 ✅
- [ ] Milestone 2: 单机仿真成功率 ≥ 70% ✅
- [ ] Milestone 3: MLLM 能正确规划简单任务 ✅
- [ ] Milestone 4: 双机仿真成功率 ≥ 60% ✅
- [ ] Milestone 5: 真实环境成功演示 ✅

---

## ⚠️ 风险点与应对

| 风险 | 应对措施 |
|------|---------|
| MuJoCo建模耗时 | 参考HSR/ALOHA现有模型，快速迭代 |
| 仿真成功率低 | 增加演示数量，调整任务难度 |
| Sim-to-Real gap大 | 在仿真中增加随机化，收集更多真实数据 |
| 双机同步困难 | 先做单机验证，双机作为可选 |
| 硬件故障 | 备份仿真结果，论文可以纯仿真 |

---

## 📌 注意事项

1. **每个阶段都要验证通过再进入下一阶段**
2. **及时记录数据、日志、视频**（论文需要）
3. **代码及时备份和版本管理**
4. **真机测试务必注意安全**（设置限位、急停）
5. **定期和导师汇报进度**
