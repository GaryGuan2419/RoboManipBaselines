#!/usr/bin/env python3
"""
实验三（B-side place）：底盘世界系 XY 扰动档下 side_place 策略成功率柱状图。

扰动档：``bin/exp3_xy_perturb_bands.py``（1×1 … 15×15 cm 名义，含 12、15）。

Rollout：前 ``--grip_lock_policy_steps``（默认 25）个 **policy step** 强制夹爪闭合指令；
``--pick_eval_policy_steps``（默认 **75**，与 action 图横轴 policy step 一致）步结算 ``compute_pick_eval_outcome``：
**bottle2 中心 XY ∈ 绿垫矩形** 且 **棒几何最高点 z ≤ 垫面 z + 棒名义长度**（默认 0.14 m）。

用法（仓库根目录）::

  python bin/plot_place_robot_base_xy_perturb_success.py \\
    --checkpoint robo_manip_baselines/checkpoint/ManiFlowPolicy/hsr_side_place/policy_best.ckpt \\
    --episodes 150
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

_bin_dir = Path(__file__).resolve().parent
if str(_bin_dir) not in sys.path:
    sys.path.insert(0, str(_bin_dir))
from exp3_xy_perturb_bands import PERTURB_CM_SPECS  # noqa: E402

DEFAULT_EXP3_RUN_DIR = Path("runs/exp3_hsr_side_place")
DEFAULT_EXP3_OUT_PNG = DEFAULT_EXP3_RUN_DIR / "place_base_xy_perturb_bars.png"


def wilson_95_interval(k: int, n: int) -> tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 1.0
    z = 1.96
    ph = k / n
    denom = 1.0 + z**2 / n
    center = (ph + z**2 / (2.0 * n)) / denom
    rad = z * np.sqrt(max(0.0, ph * (1.0 - ph) / n + z**2 / (4.0 * n**2))) / denom
    lo = max(0.0, center - rad)
    hi = min(1.0, center + rad)
    return ph, lo, hi


def run_rollout_batch(
    repo_root: Path,
    yaml_dir: Path,
    checkpoint: Path,
    half_extent_m: float,
    episodes: int,
    skip: int,
    seed: int,
    max_duration: float,
    grip_lock_policy_steps: int,
    pick_eval_policy_steps: int,
) -> tuple[int, int, Path]:
    yaml_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False, dir=str(yaml_dir)
    )
    tmp.close()
    out_yaml = Path(tmp.name)

    cmd = [
        sys.executable,
        "-m",
        "robo_manip_baselines.bin.Rollout",
        "ManiFlowPolicy",
        "MujocoHsrBSidePlace",
        "--checkpoint",
        str(checkpoint),
        "--world_idx",
        "0",
        "--world_idx_repeat_count",
        str(episodes),
        "--skip",
        str(skip),
        "--no_render",
        "--no_plot",
        "--auto_exit",
        "--max_duration",
        str(max_duration),
        "--seed",
        str(seed),
        "--place_base_xy_reset_perturb_half_extent_m",
        str(half_extent_m),
        "--place_rollout_grip_lock_policy_steps",
        str(grip_lock_policy_steps),
        "--pick_eval_max_policy_action_steps",
        str(pick_eval_policy_steps),
        "--result_filename",
        str(out_yaml),
    ]
    print("[plot_place_robot_base_xy_perturb_success] RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(repo_root))
    if r.returncode != 0:
        raise RuntimeError(
            f"Rollout failed with code {r.returncode} (place_base_xy h={half_extent_m})"
        )

    with open(out_yaml, "r") as f:
        data = yaml.safe_load(f)
    succ = data.get("success", [])
    n = len(succ)
    k = int(sum(1 for x in succ if bool(x)))
    return k, n, out_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "robo_manip_baselines/checkpoint/ManiFlowPolicy/hsr_side_place/policy_best.ckpt"
        ),
    )
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EXP3_OUT_PNG,
    )
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_EXP3_RUN_DIR)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--max_duration", type=float, default=120.0)
    ap.add_argument(
        "--grip_lock_policy_steps",
        type=int,
        default=25,
        help="前 N 个 policy step 锁夹爪（Rollout --place_rollout_grip_lock_policy_steps）",
    )
    ap.add_argument(
        "--pick_eval_policy_steps",
        type=int,
        default=75,
        help="pick_eval / action 图横轴 policy step==该值时结算（Rollout --pick_eval_max_policy_action_steps）",
    )
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--keep_intermediate_yaml", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    labels = [lbl for _, lbl in PERTURB_CM_SPECS]
    rates: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    raw_rows: list[tuple[str, int, int, float, float, float]] = []

    yaml_paths: list[Path] = []
    try:
        for h_m, lbl in PERTURB_CM_SPECS:
            k, n, ypath = run_rollout_batch(
                repo_root,
                args.run_dir,
                args.checkpoint,
                h_m,
                args.episodes,
                args.skip,
                args.seed,
                args.max_duration,
                args.grip_lock_policy_steps,
                args.pick_eval_policy_steps,
            )
            yaml_paths.append(ypath)
            ph, lo, hi = wilson_95_interval(k, n)
            rates.append(ph)
            lows.append(lo)
            highs.append(hi)
            raw_rows.append((lbl, k, n, ph, lo, hi))
    finally:
        if not args.keep_intermediate_yaml:
            for p in yaml_paths:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, rates, yerr=[np.array(rates) - np.array(lows), np.array(highs) - np.array(rates)], capsize=4, color="steelblue", ecolor="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"B-side place (hsr_side_place): base XY perturb vs success "
        f"(n={args.episodes}/band, grip lock first {args.grip_lock_policy_steps} policy steps)"
    )
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"[plot_place_robot_base_xy_perturb_success] Wrote {args.out}", flush=True)

    csv_path = args.csv
    if csv_path is None:
        csv_path = args.out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band_label", "success_count", "total", "rate", "wilson_lo", "wilson_hi"])
        w.writerows(raw_rows)
    print(f"[plot_place_robot_base_xy_perturb_success] Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
