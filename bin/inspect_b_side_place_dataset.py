#!/usr/bin/env python3
"""
Inspect b_side_place .rmb episodes: first-frame pose, gripper, and step-to-step jumps.

Usage (from repo root):
  python bin/inspect_b_side_place_dataset.py \\
      robo_manip_baselines/dataset/hsr_b_side_place_20260407_182410

Recursively finds episode folders containing ``main.rmb.hdf5`` (or a single ``*.rmb`` path).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from robo_manip_baselines.common.data.DataKey import DataKey
from robo_manip_baselines.common.data.RmbData import RmbData


def _find_episode_paths(root: str) -> list[str]:
    out: list[str] = []
    if os.path.isfile(root) and root.endswith(".rmb"):
        return [root]
    if os.path.isdir(root):
        for dirpath, _dirnames, filenames in os.walk(root):
            if "main.rmb.hdf5" in filenames:
                out.append(dirpath)
    return sorted(out)


def _summarize_episode(ep_path: str) -> Optional[dict]:
    with RmbData(ep_path) as rmb:
        t = rmb[DataKey.TIME][:]
        jp = rmb[DataKey.MEASURED_JOINT_POS][:]
        gp = rmb[DataKey.MEASURED_GRIPPER_JOINT_POS][:]
        n = min(len(t), len(jp), len(gp))
        if n == 0:
            print(f"  [skip] empty: {ep_path}")
            return None
        jp = jp[:n]
        gp = gp.reshape(-1)[:n]

        cjp = cgp = None
        if DataKey.COMMAND_JOINT_POS in rmb:
            cjp = rmb[DataKey.COMMAND_JOINT_POS][:n]
        if DataKey.COMMAND_GRIPPER_JOINT_POS in rmb:
            cgp = rmb[DataKey.COMMAND_GRIPPER_JOINT_POS][:n].reshape(-1)

        span = float(t[n - 1] - t[0]) if n >= 2 else 0.0
        print(f"  frames={n}  time span≈{span:.3f}s")
        print(f"  measured_joint_pos[0] = {np.array2string(jp[0], precision=4)}")
        print(f"  measured_gripper_qpos[0] = {float(gp[0]):.4f}  (smaller≈more closed)")
        if cjp is not None and len(cjp):
            print(f"  command_joint_pos[0]   = {np.array2string(cjp[0], precision=4)}")
        if cgp is not None and len(cgp):
            print(f"  command_gripper[0]     = {float(cgp[0]):.4f}")

        if n >= 2:
            djp = np.linalg.norm(jp[1] - jp[0])
            dgp = abs(float(gp[1] - gp[0]))
            print(f"  |Δarm| step0→1: {djp:.5f} rad-equivalent L2  |Δgrip|: {dgp:.5f}")

        # Early-window max jump (possible bad trim / teleop hit N mid-motion)
        w = min(30, n)
        max_arm_jump = 0.0
        max_g_jump = 0.0
        for i in range(1, w):
            max_arm_jump = max(max_arm_jump, float(np.linalg.norm(jp[i] - jp[i - 1])))
            max_g_jump = max(max_g_jump, abs(float(gp[i] - gp[i - 1])))
        print(f"  max |Δarm| in first {w} steps: {max_arm_jump:.5f}  max |Δgrip|: {max_g_jump:.5f}")

        # Grip statistics at episode starts (across files, caller prints batch)
        return {
            "n": n,
            "g0": float(gp[0]),
            "dj1": abs(float(gp[1] - gp[0])) if n >= 2 else 0.0,
            "max_jump_arm": max_arm_jump,
            "max_jump_g": max_g_jump,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", help="Folder with episode subdirs, or one .rmb episode path")
    args = ap.parse_args()

    paths = _find_episode_paths(args.dataset_dir)
    if not paths:
        print(
            f"No episodes found under {args.dataset_dir!r}\n"
            "Expected: .../episode_dir/main.rmb.hdf5 or a path ending in .rmb"
        )
        sys.exit(1)

    print(f"Found {len(paths)} episode(s).\n")
    g0_list = []
    for p in paths:
        print(os.path.basename(p.rstrip("/")))
        stats = _summarize_episode(p)
        if stats:
            g0_list.append(stats["g0"])
        print()

    if g0_list:
        g0_list = np.array(g0_list)
        print(
            "Across episodes: gripper qpos[0]  mean={:.4f}  std={:.4f}  "
            "min={:.4f}  max={:.4f}".format(
                float(g0_list.mean()),
                float(g0_list.std()),
                float(g0_list.min()),
                float(g0_list.max()),
            )
        )
        print(
            "Interpretation: values ~0.1–0.25 often mean fairly closed; "
            "~0.6+ is more open. Compare to your deploy align end-state."
        )


if __name__ == "__main__":
    main()
