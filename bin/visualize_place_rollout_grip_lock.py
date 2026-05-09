#!/usr/bin/env python3
"""
可渲染演示：B-side place + 开局底盘 XY 随机偏移 + 前 N 个 policy step 夹爪锁定。

你会看到：
1. **MuJoCo 窗口**：机器人底盘每次 reset 在设定半宽内随机平移（yaw 不变）；前 N 个 policy
   推理步内手指保持强力闭合（策略输出的夹爪维被覆盖为 ``place_rollout_grip_lock_cmd``，默认 -0.5）。
2. **OpenCV 策略图窗口**（未加 ``--no_plot``）：action 曲线最后一维为夹爪；锁定阶段应贴近水平线，
   解锁后才会跟随网络输出变化。
3. **终端**：加 ``--place_rollout_grip_lock_verbose`` 时每个 policy step 打一行
   ``policy_output_was`` vs ``command_grip``。

成败在结算步（默认 ``policy_step==75``，与 action 图横轴一致）由 env 判定：
**棒心 XY ∈ 绿垫矩形** 且 **棒几何最高点 z ≤ 垫面 z + 棒名义长度（默认 0.14 m）**。

用法（仓库根、有图形界面 / WSLg）::

  python bin/visualize_place_rollout_grip_lock.py \\
    --checkpoint robo_manip_baselines/checkpoint/ManiFlowPolicy/hsr_side_place/policy_best.ckpt

按 ``n`` 可提前结束当前局（与 Rollout 一致）；或等 ``--max_duration`` 自动退出。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "robo_manip_baselines/checkpoint/ManiFlowPolicy/hsr_side_place/policy_best.ckpt"
        ),
    )
    ap.add_argument(
        "--half_extent_m",
        type=float,
        default=0.025,
        help="底盘 XY 扰动半宽 [m]；名义约 (2h)×(2h) cm，默认 2.5 cm → 5×5 cm",
    )
    ap.add_argument(
        "--grip_lock_policy_steps",
        type=int,
        default=25,
        help="前多少个 policy step 强制夹爪指令（与柱状图实验一致）",
    )
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument(
        "--pick_eval_max_policy_action_steps",
        type=int,
        default=75,
        help="与 action 图横轴一致；设很大可延后结算便于纯观察",
    )
    ap.add_argument("--max_duration", type=float, default=120.0)
    ap.add_argument(
        "--no_plot",
        action="store_true",
        help="只开 MuJoCo、不开 OpenCV action 图（仍建议开终端 verbose）",
    )
    args = ap.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "robo_manip_baselines.bin.Rollout",
        "ManiFlowPolicy",
        "MujocoHsrBSidePlace",
        "--checkpoint",
        str(args.checkpoint),
        "--world_idx",
        "0",
        "--world_idx_repeat_count",
        "1",
        "--skip",
        str(args.skip),
        "--seed",
        str(args.seed),
        "--place_base_xy_reset_perturb_half_extent_m",
        str(args.half_extent_m),
        "--place_rollout_grip_lock_policy_steps",
        str(args.grip_lock_policy_steps),
        "--place_rollout_grip_lock_verbose",
        "--pick_eval_max_policy_action_steps",
        str(args.pick_eval_max_policy_action_steps),
        "--auto_exit",
        "--max_duration",
        str(args.max_duration),
    ]
    if args.no_plot:
        cmd.append("--no_plot")
    # 故意不加 --no_render：弹出仿真窗口

    print(
        "[visualize_place_rollout_grip_lock] 说明：\n"
        "  - 夹爪「锁住」= Rollout 把策略 9 维动作最后一维 **改写** 为闭合指令（默认 -0.5），\n"
        "    网络仍按原逻辑推理，只是执行层不采纳其夹爪输出。\n"
        "  - 成功判据：棒心 XY 在绿垫内，且棒最高点 z ≤ 垫 z + 0.14 m；\n"
        "    默认在第 75 个 policy step 结算（与 action 图横轴一致）。\n",
        flush=True,
    )
    print("[visualize_place_rollout_grip_lock] RUN:", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=str(repo)))


if __name__ == "__main__":
    main()
