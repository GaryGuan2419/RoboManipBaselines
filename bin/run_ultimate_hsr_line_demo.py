#!/usr/bin/env python3
"""
Ultimate scripted demo: pick (A) -> handover -> place (B) on one world-x line (-2, 0, +2).

Usage (from repo root):
  python bin/run_ultimate_hsr_line_demo.py --skip_policies   # navigation + alignment only
  python bin/run_ultimate_hsr_line_demo.py --config bin/configs/ultimate_hsr_line_demo.yaml

Requires checkpoints + model_meta_info.pkl next to each ckpt unless --skip_policies.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import mujoco
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BIN = os.path.join(_REPO_ROOT, "bin")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import handover_config as hc  # noqa: E402
from handover_utils import align_arm_to_pose, hold_dual_pose_steps  # noqa: E402

from robo_manip_baselines.envs.mujoco.hsr.MujocoDualHsrUltimateLineEnv import (  # noqa: E402
    MujocoDualHsrUltimateLineEnv,
)
from robo_manip_baselines.envs.mujoco.hsr.MujocoHsrTidyupEnv import MujocoHsrTidyupEnv  # noqa: E402
from robo_manip_baselines.mllm.dual_hsr_handover_geometry import (  # noqa: E402
    compute_navigation_waypoints,
    compute_navigation_waypoints_from_landmark_xy,
    goal_to_base_offset_xy_for_place,
)
from robo_manip_baselines.mllm.maniflow_executor import (  # noqa: E402
    ManiFlowExecutor,
    ManiFlowExecutorHsrDualHandoverB,
)
from robo_manip_baselines.mllm.overhead_ray_ground import (  # noqa: E402
    estimate_dual_demo_landmarks_world_xy_overhead_ray_plane,
)

from ultimate_line_demo_lib import (  # noqa: E402
    execute_handover_b_release_a_when_closed,
    execute_skill_dual,
    navigate_to_dual,
    navigate_to_xy_world_line_then_along_x,
    preview_cameras_close,
    robot_base_xy_yaw,
)


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("Install PyYAML to use --config") from e
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _release_policy_memory() -> None:
    """Free weights between sequential ManiFlow loads (avoids Linux OOM killer on limited RAM/VRAM)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--nav_source",
        choices=["oracle", "ray_plane"],
        default=None,
        help="Overrides yaml; oracle is recommended for stable demos.",
    )
    parser.add_argument("--no_nav_fallback", action="store_true")
    parser.add_argument(
        "--skip_policies",
        action="store_true",
        help="Only run navigation + arm alignment (no ManiFlow checkpoints).",
    )
    parser.add_argument("--render", action="store_true", help="Force human render window")
    parser.add_argument("--no_render", action="store_true", help="No viewer (rgb_array)")
    parser.add_argument(
        "--camera_preview",
        choices=["none", "policy", "all"],
        default=None,
        help="OpenCV windows during ManiFlow: policy=inputs to network; all=all RGB in env.",
    )
    args = parser.parse_args()

    cfg = {}
    if args.config:
        cfg = _load_yaml(args.config)

    nav_source = args.nav_source or cfg.get("nav_source", "oracle")
    no_fb = args.no_nav_fallback or bool(cfg.get("no_nav_fallback", False))
    kp_pos = float(cfg.get("kp_pos", 2.0))
    kp_yaw = float(cfg.get("kp_yaw", 2.0))
    max_steps_nav = int(cfg.get("max_steps_nav", 1600))
    warmup = int(cfg.get("warmup_hold_steps", 50))
    pick_steps = int(cfg.get("pick_skill_steps", 60))
    hand_steps = int(cfg.get("handover_skill_steps", 90))
    place_steps = int(cfg.get("place_skill_steps", 30))
    b_q = float(cfg.get("b_grip_takeover_qpos", 0.22))
    dwell = int(cfg.get("release_dwell_steps", 12))
    a_open = float(cfg.get("a_open_cmd", 0.85))
    handover_settle_steps = int(cfg.get("handover_settle_steps", 40))
    handover_freeze_b_base_steps = int(cfg.get("handover_freeze_b_base_steps", 0))
    place_pre_policy_hold_steps = int(cfg.get("place_pre_policy_hold_steps", 20))
    place_pre_hold_arm_integral_gain = float(
        cfg.get("place_pre_hold_arm_integral_gain", 0.04)
    )

    cam_prev = args.camera_preview if args.camera_preview is not None else cfg.get(
        "camera_preview", "none"
    )
    if cam_prev in (None, "none", ""):
        camera_preview = None
    else:
        camera_preview = cam_prev
        import atexit

        atexit.register(preview_cameras_close)

    # Extra world-frame xy added to B_handover_pick after geometry (fine-tune lateral approach).
    handover_b_xy_extra = np.asarray(
        cfg.get("handover_b_xy_extra", [0.0, 0.0]), dtype=np.float64
    ).reshape(2)
    # After handover: B turns to face +X, then drives to place with that heading.
    b_reorient_to_plus_x = bool(cfg.get("b_reorient_to_plus_x", True))
    b_place_yaw_plus_x = float(cfg.get("b_place_yaw_plus_x", 0.0))
    # Which place dataset layout your hsr_side_place ckpt matches (see HSR_MANIFLOW_SAMPLING_BIAS/REFERENCE.txt §3).
    b_place_sampling_env = cfg.get("b_place_sampling_env", "b_side_place")
    if b_place_sampling_env == "custom":
        if "b_place_offset_facing_plus_x" not in cfg:
            raise ValueError(
                "b_place_sampling_env==custom requires b_place_offset_facing_plus_x in yaml"
            )
        b_place_offset_facing_plus_x = np.asarray(
            cfg["b_place_offset_facing_plus_x"], dtype=np.float64
        ).reshape(2)
    else:
        b_place_offset_facing_plus_x = goal_to_base_offset_xy_for_place(
            b_place_sampling_env
        )
    b_place_two_phase = bool(cfg.get("b_place_two_phase", True))
    b_place_nav_yaw_gate = float(cfg.get("b_place_nav_yaw_gate", 0.06))
    # Override world y of the approach line; null = target_area y + offset[1] (default: goal line).
    b_place_world_line_y = cfg.get("b_place_world_line_y", None)

    ck_pick = os.path.join(
        _REPO_ROOT,
        cfg.get(
            "checkpoint_pick",
            "robo_manip_baselines/checkpoint/ManiFlowPolicy/hsr_side_pick/policy_best.ckpt",
        ),
    )
    ck_hand = os.path.join(
        _REPO_ROOT,
        cfg.get(
            "checkpoint_handover",
            "robo_manip_baselines/checkpoint/ManiFlowPolicy/hsr_side_handover/policy_best.ckpt",
        ),
    )
    ck_place = os.path.join(
        _REPO_ROOT,
        cfg.get(
            "checkpoint_place",
            "robo_manip_baselines/checkpoint/ManiFlowPolicy/hsr_side_place/policy_best.ckpt",
        ),
    )

    render_mode = "human"
    if args.no_render:
        render_mode = "rgb_array"
    if args.render:
        render_mode = "human"

    print("=" * 60)
    print(" Ultimate HSR line demo (pick -> handover -> place)")
    print(f" nav_source={nav_source}  skip_policies={args.skip_policies}")
    print(
        f" b_place_sampling_env={b_place_sampling_env}  "
        f"place_offset_xy={b_place_offset_facing_plus_x.tolist()}"
    )
    print(f" checkpoint_pick={os.path.basename(os.path.dirname(ck_pick))}")
    print(f" checkpoint_handover={os.path.basename(os.path.dirname(ck_hand))}")
    print(f" checkpoint_place={os.path.basename(os.path.dirname(ck_place))}")
    if not args.skip_policies:
        print(
            f" handover_config oracle offsets: "
            f"A_handover=handover_lm+{hc.ORACLE_OFFSET_A_HANDOVER_XY.tolist()} "
            f"B_handover=handover_lm+{hc.ORACLE_OFFSET_B_HANDOVER_XY.tolist()}"
        )
        print(
            f" handover_settle_steps={handover_settle_steps}  "
            f"handover_freeze_b_base_steps={handover_freeze_b_base_steps}"
        )
    if camera_preview:
        print(
            f" camera_preview={camera_preview}  "
            f"(policy=ManiFlow inputs, all=every env rgb_images key)"
        )
    print("=" * 60)

    env = MujocoDualHsrUltimateLineEnv(render_mode=render_mode)
    obs, _info = env.reset()

    for _ in range(warmup):
        env.step(env.get_hold_action())
        if render_mode == "human":
            env.render()

    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

    if nav_source == "oracle":
        waypoints = compute_navigation_waypoints(env.unwrapped)
        print("[Nav] oracle waypoints:", {k: waypoints[k] for k in waypoints})
    else:
        info_img = env.get_images()
        rgb = info_img["rgb_images"]["overhead_view"]
        lm, dbg = estimate_dual_demo_landmarks_world_xy_overhead_ray_plane(
            env,
            camera_key="overhead_view",
            plane_z=0.0,
            rgb=rgb,
            fallback_oracle=not no_fb,
        )
        e = env.unwrapped
        for k in ("bottle", "handover", "goal"):
            body = {"bottle": "bottle2", "handover": "handover_area", "goal": "target_area"}[k]
            o = e.data.body(body).xpos[:2]
            err = float(np.linalg.norm(lm[k] - o))
            print(f"[ray_plane] {k}: lm={lm[k]} source={dbg[k].get('source')} oracle_err={err:.4f}m")
        waypoints = compute_navigation_waypoints_from_landmark_xy(
            lm["bottle"], lm["handover"], lm["goal"]
        )
        print("[Nav] ray_plane waypoints:", {k: waypoints[k] for k in waypoints})

    w = waypoints
    # Fine-tune B handover stance (optional yaml)
    bh = np.asarray(w["B_handover_pick"]["target_xy"], dtype=np.float64) + handover_b_xy_extra
    w["B_handover_pick"] = {
        "target_xy": bh.tolist(),
        "target_yaw": w["B_handover_pick"]["target_yaw"],
    }

    # --- Phase: A to pick ---
    navigate_to_dual(
        env.unwrapped,
        0,
        w["A_pick"]["target_xy"],
        target_yaw=w["A_pick"]["target_yaw"],
        max_steps=max_steps_nav,
        kp_pos=kp_pos,
        kp_yaw=kp_yaw,
    )

    dummy_tidyup = None
    if not args.skip_policies:
        for ck, name in [(ck_pick, "pick"), (ck_hand, "handover"), (ck_place, "place")]:
            if not os.path.isfile(ck):
                raise FileNotFoundError(
                    f"Missing checkpoint {name}: {ck}\n"
                    "Use --skip_policies to test navigation only, or fix paths in yaml."
                )
            meta = os.path.join(os.path.dirname(ck), "model_meta_info.pkl")
            if not os.path.isfile(meta):
                raise FileNotFoundError(f"Missing meta next to {name} ckpt: {meta}")

        # Load one policy at a time: three ManiFlow checkpoints at once often exceeds RAM/VRAM (WSL2 -> Killed).
        # One shared single-robot dummy for pick + handover + place MotionManager (RGB comes from main dual env).
        dummy_tidyup = MujocoHsrTidyupEnv(render_mode="rgb_array")
        pick_ex = ManiFlowExecutor(ck_pick, dummy_tidyup)
        execute_skill_dual(
            env.unwrapped,
            pick_ex,
            "side_pick",
            0,
            max_steps=pick_steps,
            camera_preview=camera_preview,
        )
        env.unwrapped.unlock_gripper()
        del pick_ex
        _release_policy_memory()
    else:
        print("[Skip] pick policy")

    # --- A to handover ---
    navigate_to_dual(
        env.unwrapped,
        0,
        w["A_handover"]["target_xy"],
        target_yaw=w["A_handover"]["target_yaw"],
        max_steps=max_steps_nav,
        kp_pos=kp_pos,
        kp_yaw=kp_yaw,
        force_tight_grip_robots=(0,),
    )

    # --- B to handover ---
    navigate_to_dual(
        env.unwrapped,
        1,
        w["B_handover_pick"]["target_xy"],
        target_yaw=w["B_handover_pick"]["target_yaw"],
        max_steps=max_steps_nav,
        kp_pos=kp_pos,
        kp_yaw=kp_yaw,
    )

    print("[Align] Interpolating arms to calibrated handover poses...")
    align_arm_to_pose(
        env.unwrapped,
        0,
        hc.HANDOVER_POSE_A,
        float(hc.HANDOVER_GRIPPER_A),
        steps=140,
        smooth_steps=120,
        snap_baton=False,
        hold_start_gripper=True,
    )
    align_arm_to_pose(
        env.unwrapped,
        1,
        hc.HANDOVER_POSE_B,
        float(hc.HANDOVER_GRIPPER_B_OPEN),
        steps=140,
        smooth_steps=120,
        snap_baton=False,
    )

    if handover_settle_steps > 0:
        print(
            f"[Handover] Pre-policy settle: {handover_settle_steps} hold steps "
            "(measured arm pose; A tight grip; bases still before ManiFlow)."
        )
        hold_dual_pose_steps(
            env.unwrapped,
            handover_settle_steps,
            force_tight_grip_robots=(0,),
            render_fn=(lambda: env.render()) if render_mode == "human" else None,
        )
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

    if not args.skip_policies:
        hand_ex = ManiFlowExecutorHsrDualHandoverB(ck_hand, dummy_tidyup)
        execute_handover_b_release_a_when_closed(
            env.unwrapped,
            hand_ex,
            "side_handover",
            max_steps=hand_steps,
            b_grip_takeover_qpos=b_q,
            release_dwell_steps=dwell,
            a_open_cmd=a_open,
            camera_preview=camera_preview,
            freeze_b_base_steps=handover_freeze_b_base_steps,
        )
        env.unwrapped.unlock_gripper()
        del hand_ex
        _release_policy_memory()
        # Do NOT close dummy_tidyup here: gymnasium/MuJoCo teardown runs glfw/OS GL cleanup that
        # breaks the *main* env's viewer + offscreen GL (WSL: window vanishes, place -> segfault).
        # Tear down dummy_tidyup once at process exit after place (see below).
    else:
        print("[Skip] handover policy")

    # --- B to place (optional: 180° to face +X, then approach goal from -X side) ---
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
    if b_reorient_to_plus_x:
        bx, by, _ = robot_base_xy_yaw(env.unwrapped, 1)
        print(f"[Nav] B reorient: stand at ({bx:.3f},{by:.3f}) -> yaw={b_place_yaw_plus_x:.3f} (+X)")
        navigate_to_dual(
            env.unwrapped,
            1,
            [bx, by],
            target_yaw=b_place_yaw_plus_x,
            max_steps=max_steps_nav,
            kp_pos=kp_pos,
            kp_yaw=kp_yaw,
            force_tight_grip_robots=(1,),
            yaw_gate=b_place_nav_yaw_gate,
        )
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        g_xy = env.unwrapped.data.body("target_area").xpos[:2].copy()
        off = b_place_offset_facing_plus_x
        if b_place_world_line_y is not None:
            line_y = float(b_place_world_line_y)
        else:
            line_y = float(g_xy[1] + off[1])
        place_xy = [float(g_xy[0] + off[0]), line_y]
        nav_ex = np.asarray(
            getattr(hc, "NAV_RUNTIME_XY_EXTRA", [0.0, 0.0]), dtype=np.float64
        ).reshape(2)
        place_xy[0] = float(place_xy[0] + nav_ex[0])
        place_xy[1] = float(place_xy[1] + nav_ex[1])
        print(
            f"[Nav] B -> place (facing +world_X) goal_xy={g_xy} -> line_y={line_y:.4f} "
            f"target_xy={place_xy}"
        )
        if b_place_two_phase:
            navigate_to_xy_world_line_then_along_x(
                env.unwrapped,
                1,
                place_xy,
                target_yaw=b_place_yaw_plus_x,
                two_phase_threshold=0.04,
                yaw_gate=b_place_nav_yaw_gate,
                max_steps=max_steps_nav,
                kp_pos=kp_pos,
                kp_yaw=kp_yaw,
                force_tight_grip_robots=(1,),
            )
        else:
            navigate_to_dual(
                env.unwrapped,
                1,
                place_xy,
                target_yaw=b_place_yaw_plus_x,
                max_steps=max_steps_nav,
                kp_pos=kp_pos,
                kp_yaw=kp_yaw,
                force_tight_grip_robots=(1,),
                yaw_gate=b_place_nav_yaw_gate,
            )
    else:
        navigate_to_dual(
            env.unwrapped,
            1,
            w["B_place"]["target_xy"],
            target_yaw=w["B_place"]["target_yaw"],
            max_steps=max_steps_nav,
            kp_pos=kp_pos,
            kp_yaw=kp_yaw,
            force_tight_grip_robots=(1,),
        )

    if not args.skip_policies:
        # Arm init vs training: b_side_place demos use hsr_b_side_place_config.build_initial_qpos_18()
        # → arm = HANDOVER_POSE_A, grip ≈ GRIP_A_ACTUAL (same as handover_config). Align B before policy.
        if b_place_sampling_env in ("b_side_place", "b_side_place_full", "custom"):
            print(
                "[Align] B arm → side_place dataset init "
                "(bin/handover_config HANDOVER_POSE_A; see hsr_b_side_place_config.ROBOT_POSE)"
            )
            align_arm_to_pose(
                env.unwrapped,
                1,
                hc.HANDOVER_POSE_A,
                float(hc.HANDOVER_GRIPPER_A),
                steps=140,
                smooth_steps=120,
                snap_baton=False,
                hold_start_gripper=True,
            )
        elif b_place_sampling_env == "tidyup_place":
            tidyup_arm = np.array([0.25, -2.0, 0.0, -1.0, 0.0], dtype=np.float64)
            print(
                "[Align] B arm → env_hsr_tidyup_place.xml keyframe arm (tidyup_place ckpt); grip open"
            )
            align_arm_to_pose(
                env.unwrapped,
                1,
                tidyup_arm,
                0.8,
                steps=140,
                smooth_steps=120,
                snap_baton=False,
                hold_start_gripper=False,
            )
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

        if place_pre_policy_hold_steps > 0:
            print(
                f"[Place] Pre-policy hold: {place_pre_policy_hold_steps} steps "
                f"(B tight grip; settle contact before ManiFlow)."
            )
            hold_dual_pose_steps(
                env.unwrapped,
                place_pre_policy_hold_steps,
                force_tight_grip_robots=(1,),
                render_fn=(lambda: env.render()) if render_mode == "human" else None,
                arm_integral_gain=place_pre_hold_arm_integral_gain,
            )
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

        # Reuse dummy_tidyup — a third MuJoCo (old place_dummy) + place ckpt often OOM-kills here.
        # Policy images are filled from the dual env in execute_skill_dual; dummy is for MotionManager only.
        gc.collect()
        _release_policy_memory()
        print("[Memory] place policy: reusing pick/handover dummy_tidyup (no extra MujocoHsrTidyupPlaceEnv).")
        place_ex = ManiFlowExecutor(ck_place, dummy_tidyup)
        execute_skill_dual(
            env.unwrapped,
            place_ex,
            "place",
            1,
            max_steps=place_steps,
            camera_preview=camera_preview,
        )
        env.unwrapped.unlock_gripper()
        del place_ex
        _release_policy_memory()
    else:
        print("[Skip] place policy")

    print("\n[System] Ultimate line demo sequence finished.")
    hold_dual_pose_steps(
        env.unwrapped,
        80,
        force_tight_grip_robots=(),
        render_fn=(lambda: env.render()) if render_mode == "human" else None,
    )

    preview_cameras_close()
    if dummy_tidyup is not None:
        try:
            dummy_tidyup.close()
        except Exception:
            pass
        dummy_tidyup = None

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
