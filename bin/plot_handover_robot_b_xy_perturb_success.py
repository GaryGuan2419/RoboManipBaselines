#!/usr/bin/env python3
"""
实验三（handover receive）：Robot B 底盘世界系 XY 扰动带下，B 策略成功率柱状图。

扰动档与 pick / handover A 汇演一致：``bin/exp3_xy_perturb_bands.py``（1×1 … 10×10 cm 名义）。

成功判据：第 70 个 policy step 后 baton 最低点世界 z ≥ 0.05 m（离地约 5 cm，见 env）。

用法::

  python bin/plot_handover_robot_b_xy_perturb_success.py \\
    --checkpoint .../policy_best.ckpt \\
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

DEFAULT_EXP3_RUN_DIR = Path("runs/exp3_hsr_handover_b_xy")
DEFAULT_EXP3_OUT_PNG = DEFAULT_EXP3_RUN_DIR / "handover_robot_b_xy_perturb_bars.png"


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
        "ManiFlowPolicyHsrDualHandoverReceive",
        "MujocoDualHsrHandoverReceive",
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
        "--robot_b_xy_reset_perturb_half_extent_m",
        str(half_extent_m),
        "--result_filename",
        str(out_yaml),
    ]
    print("[plot_handover_robot_b_xy_perturb_success] RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(repo_root))
    if r.returncode != 0:
        raise RuntimeError(
            f"Rollout failed with code {r.returncode} (robot_b_xy h={half_extent_m})"
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
            "robo_manip_baselines/checkpoint/ManiFlowPolicy/hsr_handover_receive/policy_best.ckpt"
        ),
        help="policy_best.ckpt",
    )
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--out", type=Path, default=DEFAULT_EXP3_OUT_PNG)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_EXP3_RUN_DIR)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--max_duration", type=float, default=240.0)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--keep_intermediate_yaml", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    run_dir = (repo_root / args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

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
            run_dir,
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
                "robot_b_xy_half_extent_m": h_m,
                "successes": k,
                "n": n,
                "p_hat": ph,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "yaml": str(ypath),
            }
        )
        print(
            f"[plot_handover_robot_b_xy_perturb_success] {lab}: {k}/{n} = {100*ph:.1f}% "
            f"(95% CI {100*lo:.1f}%–{100*hi:.1f}%)",
            flush=True,
        )
        if not args.keep_intermediate_yaml:
            try:
                ypath.unlink(missing_ok=True)
            except TypeError:
                if ypath.exists():
                    ypath.unlink()

    n_bars = max(1, len(labels))
    fig_w = min(14.0, 5.5 + 1.1 * n_bars)
    fig, ax = plt.subplots(figsize=(fig_w, 4.2), dpi=150)
    x = np.arange(len(labels))
    ax.bar(
        x,
        p_list,
        yerr=[yerr_lo, yerr_hi],
        capsize=6,
        color="#8172B3",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("ManiFlow dual-HSR handover receive (Robot B base XY perturbation)")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"[plot_handover_robot_b_xy_perturb_success] Wrote figure: {args.out.resolve()}")

    csv_path = args.csv if args.csv is not None else args.out.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[plot_handover_robot_b_xy_perturb_success] Wrote table: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
