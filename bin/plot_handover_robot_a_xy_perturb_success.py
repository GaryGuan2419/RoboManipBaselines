#!/usr/bin/env python3
"""
实验三（handover receive）：Robot A 底盘世界系 XY 扰动带下，B 策略成功率柱状图。

与 ``bin/plot_pick_xy_perturb_success.py`` 对称：扰动档见 ``bin/exp3_xy_perturb_bands.py``。
子进程跑 ``Rollout``（ManiFlowPolicyHsrDualHandoverReceive + MujocoDualHsrHandoverReceive）。

成功判据与 ``OperationMujocoDualHsrHandoverReceive`` / env 一致：第 70 个 policy step 后
``compute_pick_eval_outcome``，baton 几何最低点世界 **z ≥ 0.05 m**（离地约 5 cm，不比 reset 基线）。

用法（仓库根目录）::

  CUDA_VISIBLE_DEVICES=0 nohup env PYTHONUNBUFFERED=1 \\
    python bin/plot_handover_robot_a_xy_perturb_success.py \\
    --checkpoint robo_manip_baselines/checkpoint/ManiFlowPolicy/hsr_handover_receive/policy_best.ckpt \\
    --episodes 150 \\
    > runs/exp3_hsr_handover_receive/nohup.log 2>&1 &

默认输出目录 ``runs/exp3_hsr_handover_receive/``（图、CSV、临时 yaml）；``--run-dir`` /
``--out`` 可覆盖。总局数 = len(PERTURB_CM_SPECS) × --episodes。
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

DEFAULT_EXP3_RUN_DIR = Path("runs/exp3_hsr_handover_receive")
DEFAULT_EXP3_OUT_PNG = DEFAULT_EXP3_RUN_DIR / "handover_robot_a_xy_perturb_bars.png"


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
        "--robot_a_xy_reset_perturb_half_extent_m",
        str(half_extent_m),
        "--result_filename",
        str(out_yaml),
    ]
    print("[plot_handover_robot_a_xy_perturb_success] RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(repo_root))
    if r.returncode != 0:
        raise RuntimeError(
            f"Rollout failed with code {r.returncode} (robot_a_xy h={half_extent_m})"
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
        help="policy_best.ckpt（按你训练输出路径改）",
    )
    ap.add_argument(
        "--episodes",
        type=int,
        default=150,
        help="每个扰动档跑多少局",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EXP3_OUT_PNG,
        help="输出主图 PNG",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_EXP3_RUN_DIR,
        help="Rollout 临时 yaml 目录",
    )
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--seed", type=int, default=-1, help="Rollout --seed（-1 表随机）")
    ap.add_argument(
        "--max_duration",
        type=float,
        default=240.0,
        help="每局 RolloutPhase 仿真时间兜底 [s]（交接比 pick 长）",
    )
    ap.add_argument("--csv", type=Path, default=None, help="汇总 CSV（默认与 PNG 同 stem）")
    ap.add_argument(
        "--keep_intermediate_yaml",
        action="store_true",
        help="保留每档 Rollout 的临时 yaml；默认跑完删除",
    )
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
                "robot_a_xy_half_extent_m": h_m,
                "successes": k,
                "n": n,
                "p_hat": ph,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "yaml": str(ypath),
            }
        )
        print(
            f"[plot_handover_robot_a_xy_perturb_success] {lab}: {k}/{n} = {100*ph:.1f}% "
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
        color="#55A868",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("ManiFlow dual-HSR handover receive (Robot A base XY perturbation)")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(
        f"[plot_handover_robot_a_xy_perturb_success] Wrote figure: {args.out.resolve()}",
        flush=True,
    )

    csv_path = args.csv
    if csv_path is None:
        csv_path = args.out.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(
        f"[plot_handover_robot_a_xy_perturb_success] Wrote table: {csv_path.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
