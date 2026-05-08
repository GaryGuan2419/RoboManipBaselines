#!/usr/bin/env python3
"""
实验三主图：不同 XY 瓶位扰动带（1/2/3/5 cm 见方）下 hsr_side_pick 成功率柱状图。

依赖：本仓库可 import；已安装 matplotlib、PyYAML；能跑 Rollout（GPU/CUDA 等与本机一致）。

用法（在仓库根目录）:
  python bin/plot_pick_xy_perturb_success.py \\
    --checkpoint robo_manip_baselines/checkpoint/ManiFlowPolicy/hsr_side_pick/policy_best.ckpt \\
    --episodes 150 \\
    --out runs/exp3_pick_xy_bars.png

每档扰动会子进程跑一次 Rollout（--no_render --no_plot --auto_exit），再汇总 Wilson 近似 95% 误差条。
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


# 半宽 h [m]：每轴 U[-h,h]，名义约 (2h)×(2h) cm
PERTURB_CM_SPECS = [
    (0.005, "1×1 cm"),
    (0.01, "2×2 cm"),
    (0.015, "3×3 cm"),
    (0.025, "5×5 cm"),
]


def wilson_95_interval(k: int, n: int) -> tuple[float, float, float]:
    """返回 (p_hat, lo, hi)，Wilson score 区间端点。"""
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
    checkpoint: Path,
    half_extent_m: float,
    episodes: int,
    skip: int,
    seed: int,
    max_duration: float,
) -> tuple[int, int, Path]:
    """子进程跑 Rollout；返回 (成功数, 总局数, yaml 路径)。"""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False, dir=repo_root / "runs"
    )
    tmp.close()
    out_yaml = Path(tmp.name)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "robo_manip_baselines.bin.Rollout",
        "ManiFlowPolicy",
        "MujocoHsrTidyup",
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
        "--bottle_xy_reset_perturb_half_extent_m",
        str(half_extent_m),
        "--result_filename",
        str(out_yaml),
    ]
    print("[plot_pick_xy_perturb_success] RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(repo_root))
    if r.returncode != 0:
        raise RuntimeError(f"Rollout failed with code {r.returncode} (h={half_extent_m})")

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
            "robo_manip_baselines/checkpoint/ManiFlowPolicy/hsr_side_pick/policy_best.ckpt"
        ),
        help="policy_best.ckpt 路径（相对 cwd 或绝对）",
    )
    ap.add_argument(
        "--episodes",
        type=int,
        default=150,
        help="每个扰动档跑多少局（150 够用；论文可 200）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("runs/exp3_pick_xy_perturb_bars.png"),
        help="输出主图 PNG",
    )
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--seed", type=int, default=-1, help="Rollout --seed（-1 表随机）")
    ap.add_argument(
        "--max_duration",
        type=float,
        default=120.0,
        help="每局 RolloutPhase 仿真时间兜底 [s]",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="可选：写出汇总 CSV（默认与 PNG 同 stem）",
    )
    ap.add_argument(
        "--keep_intermediate_yaml",
        action="store_true",
        help="保留每档 Rollout 的临时 yaml；默认跑完删除",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    (repo_root / "runs").mkdir(parents=True, exist_ok=True)

    ckpt = args.checkpoint
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt.resolve()}")

    labels: list[str] = []
    p_list: list[float] = []
    yerr_lo: list[float] = []
    yerr_hi: list[float] = []
    rows: list[dict] = []

    for h_m, lab in PERTURB_CM_SPECS:
        k, n, ypath = run_rollout_batch(
            repo_root,
            ckpt,
            h_m,
            args.episodes,
            args.skip,
            args.seed,
            args.max_duration,
        )
        ph, lo, hi = wilson_95_interval(k, n)
        labels.append(lab)
        p_list.append(ph)
        yerr_lo.append(ph - lo)
        yerr_hi.append(hi - ph)
        rows.append(
            {
                "label": lab,
                "half_extent_m": h_m,
                "successes": k,
                "n": n,
                "p_hat": ph,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "yaml": str(ypath),
            }
        )
        print(
            f"[plot_pick_xy_perturb_success] {lab}: {k}/{n} = {100*ph:.1f}% "
            f"(95% CI {100*lo:.1f}%–{100*hi:.1f}%)",
            flush=True,
        )
        if not args.keep_intermediate_yaml:
            try:
                ypath.unlink(missing_ok=True)
            except TypeError:
                if ypath.exists():
                    ypath.unlink()

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    x = np.arange(len(labels))
    ax.bar(
        x,
        p_list,
        yerr=[yerr_lo, yerr_hi],
        capsize=6,
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("ManiFlow hsr_side_pick (XY bottle perturbation)")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"[plot_pick_xy_perturb_success] Wrote figure: {args.out.resolve()}", flush=True)

    csv_path = args.csv
    if csv_path is None:
        csv_path = args.out.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[plot_pick_xy_perturb_success] Wrote table: {csv_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
