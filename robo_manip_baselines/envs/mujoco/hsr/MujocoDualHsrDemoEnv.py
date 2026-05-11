from os import path
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
from gymnasium.spaces import Box, Dict
from robo_manip_baselines.envs.mujoco.MujocoEnvBase import MujocoEnvBase

class MujocoDualHsrDemoEnv(MujocoEnvBase):
    default_camera_config = {
        "azimuth": -120.0,
        "elevation": -25.0,
        "distance": 3.0,
        "lookat": [0.5, 0.0, 0.1],
    }
    
    def __init__(
        self,
        xml_filename="env_dual_hsr_demo.xml",
        init_qpos=None,
        **kwargs,
    ):
        # Action space: 18-dim (9 actuators per robot: 3 mobile vel + 5 arm pos + 1 gripper)
        self.action_space = Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float64)
        
        self.observation_space = Dict({
            "robot_a/joint_pos": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            "robot_a/mobile_vel": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "robot_b/joint_pos": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            "robot_b/mobile_vel": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        })
        
        # init_qpos layout per robot (11 values each):
        #   mobile_x, mobile_y, mobile_theta,        (3 base)
        #   arm_lift, arm_flex, arm_roll,              (3 arm)
        #   wrist_flex, wrist_roll,                    (2 wrist)
        #   hand_motor,                                (1 gripper)
        #   hand_l_proximal, hand_r_proximal           (2 mimics)
        # bottle2_freejoint: 7 (pos + quat)
        # Total: 11 + 11 + 7 = 29 dims
        
        # Single-robot keyframe reference: 0 0 0 | 0.25 -2.0 0 -1.0 0 | 0.8 | 0.8 0.8
        #                                  base    arm                  grip   mimics
        # User specified positions:
        #   Robot A start: (-1, 0), facing +X (yaw=0) — offset from bottle (-0.2,0) so arm does not occlude
        #   Robot B start: (1.08, -0.76), facing +Y (yaw=pi/2)
        # 与单机 tidyup 侧抓姿态同风格；arm_lift 抬高便于「下探→抓→抬」
        if init_qpos is None:
            init_qpos = np.array([
                # Robot A (11 values)
                -1.0, 0.0, 0.0,                       # base: x, y, theta
                0.2600, -2.4844, 0.0039, 1.0132, 0.0021,  # arm: lift, flex, roll, wrist_flex, wrist_roll
                0.8,                                    # gripper: hand_motor
                0.8, 0.8,                               # mimics: hand_l_proximal, hand_r_proximal
                # Robot B (11 values)
                2.0, 0.0, 3.1416,                      # base: x, y, theta (facing -X, toward A)
                0.2600, -2.4844, 0.0039, 1.0132, 0.0021,  # arm (same style as A)
                0.8,                                    # gripper: hand_motor
                0.8, 0.8,                               # mimics: hand_l_proximal, hand_r_proximal
                # Bottle2 freejoint (7 values): world xy at pick test pose
                -0.2, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0,  # pos + quaternion
            ])
        else:
            init_qpos = np.asarray(init_qpos, dtype=np.float64).copy()

        xml_path = path.join(
            path.dirname(__file__), "../../assets/mujoco/envs/hsr", xml_filename
        )
        
        super().__init__(xml_path, init_qpos, **kwargs)
        self.action_space = Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float64)
        self._baton_grip_stabilizer = {
            "enabled": False,
            "robot_index": None,
            "active_robot": None,
            "relative_pos": None,
            "relative_mat": None,
            "max_palm_dist": 0.16,
            "attach_grip_qpos": 0.45,
            "release_grip_qpos": 0.72,
            "verbose": False,
        }

    def configure_baton_grip_stabilizer(
        self,
        *,
        enabled=True,
        robot_index=None,
        max_palm_dist=0.16,
        attach_grip_qpos=0.45,
        release_grip_qpos=0.72,
        verbose=False,
    ):
        """Optionally hold ``bottle2`` rigidly relative to a closed HSR hand.

        This is a simulation stabilizer for the rectangular baton case where the
        finger pads can contact only an edge/line and the object slowly creeps
        under gravity despite high friction. It activates only after the baton is
        near a closed gripper and releases when that gripper opens.
        """
        self._baton_grip_stabilizer.update(
            {
                "enabled": bool(enabled),
                "robot_index": None if robot_index is None else int(robot_index),
                "active_robot": None,
                "relative_pos": None,
                "relative_mat": None,
                "max_palm_dist": float(max_palm_dist),
                "attach_grip_qpos": float(attach_grip_qpos),
                "release_grip_qpos": float(release_grip_qpos),
                "verbose": bool(verbose),
            }
        )

    def disable_baton_grip_stabilizer(self):
        self.configure_baton_grip_stabilizer(enabled=False)

    # Higher resolution only for the legacy overhead debug camera. The 9-camera
    # replay sampler keeps its policy cameras at one size to avoid partial
    # offscreen renders on some MuJoCo/GL backends.
    _OVERHEAD_OFFSCREEN_W = 1920
    _OVERHEAD_OFFSCREEN_H = 1080

    def setup_camera(self):
        self.cameras = {}
        for camera_id in range(self.model.ncam):
            camera = {}
            camera_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id
            )
            camera["name"] = camera_name
            camera["id"] = camera_id
            if camera_name == "overhead_view":
                ow, oh = self._OVERHEAD_OFFSCREEN_W, self._OVERHEAD_OFFSCREEN_H
            else:
                ow, oh = 640, 480
            camera["viewer"] = OffScreenViewer(self.model, self.data, width=ow, height=oh)
            self.cameras[camera_name.replace("/", "_")] = camera

        self.mujoco_renderer._viewers["dummy"] = None
        self._first_render = True

    def setup_robot(self, init_qpos):
        self.init_qpos[: len(init_qpos)] = init_qpos
        self.init_qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, self.data)

    def modify_world(self, world_idx=None, cumulative_idx=None):
        return 0

    # Joint names for arm position actuators (indices 3:8 per robot in action space)
    ARM_JOINT_NAMES = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]

    def get_hold_action(self):
        """Build a 'hold position' action matching single-robot navigate_to pattern.
        
        Arms: Use FIXED init positions (gravity compensation).
        Gripper: Lock ONCE with -0.1 squeeze offset when holding something.
                 (command=actual → zero error → zero torque → object drops!)
        """
        if not hasattr(self, '_hold_arm_cache'):
            self._hold_arm_cache = {}
            for i, prefix in enumerate(["robot_a", "robot_b"]):
                for j, jname in enumerate(self.ARM_JOINT_NAMES):
                    addr = self.model.jnt_qposadr[
                        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/{jname}")
                    ]
                    self._hold_arm_cache[(i, j)] = self.init_qpos[addr]
            self._locked_gripper = [None, None]
        
        action = np.zeros(18)
        for i, prefix in enumerate(["robot_a", "robot_b"]):
            offset = i * 9
            for j in range(5):
                action[offset + 3 + j] = self._hold_arm_cache[(i, j)]
            
            grip_addr = self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/hand_motor_joint")
            ]
            # Lock gripper if not yet locked, with squeeze offset
            if self._locked_gripper[i] is None:
                grip_pos = self.data.qpos[grip_addr]
                if grip_pos < 0.6:  # Holding something
                    grip_pos -= 0.1  # Extra squeeze torque!
                self._locked_gripper[i] = grip_pos
            action[offset + 8] = self._locked_gripper[i]
        return action
    
    def unlock_gripper(self):
        """Call after a skill (pick/place) to re-read and re-lock gripper state."""
        self._locked_gripper = [None, None]

    def _get_obs(self):
        obs = {}
        for prefix in ["robot_a", "robot_b"]:
            arm_joints = [f"{prefix}/arm_lift_joint", f"{prefix}/arm_flex_joint", f"{prefix}/arm_roll_joint", f"{prefix}/wrist_flex_joint", f"{prefix}/wrist_roll_joint", f"{prefix}/hand_motor_joint"]
            mobile_joints = [f"{prefix}/mobile_x_joint", f"{prefix}/mobile_y_joint", f"{prefix}/mobile_theta_joint"]
            
            joint_pos = np.array([self.data.joint(j).qpos[0] for j in arm_joints])
            
            # mobile vel to local map
            mobile_vel_world = np.array([self.data.joint(j).qvel[0] for j in mobile_joints])
            theta = self.data.joint(mobile_joints[2]).qpos[0]
            rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            local_xy = rot_mat @ mobile_vel_world[0:2]
            mobile_vel = np.array([local_xy[0], local_xy[1], mobile_vel_world[2]])
            
            obs[f"{prefix}/joint_pos"] = joint_pos
            obs[f"{prefix}/mobile_vel"] = mobile_vel
        return obs
        
    def step(self, action):
        """Override step to SKIP automatic camera rendering for massive speedup.
        
        MujocoEnvBase.step() calls _get_info() which renders RGB+depth from ALL
        cameras every single step. With 6 cameras that's 12 offscreen render passes
        PER STEP — the dominant bottleneck. We skip this and only render cameras
        on-demand via get_images() when the policy actually needs them.
        """
        # 1. Transform local vel commands to world frame
        action_copy = action.copy()
        for i, prefix in enumerate(["robot_a", "robot_b"]):
            mobile_joints = [f"{prefix}/mobile_x_joint", f"{prefix}/mobile_y_joint", f"{prefix}/mobile_theta_joint"]
            theta = self.data.joint(mobile_joints[2]).qpos[0]
            rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            
            idx = i * 9
            vel_in_local_xy = action_copy[idx:idx+2]
            vel_world_xy = rot_mat @ vel_in_local_xy
            action_copy[idx:idx+2] = vel_world_xy
        
        # 2. Physics step (without camera rendering)
        self.do_simulation(action_copy, self.frame_skip)
        self._apply_baton_grip_stabilizer()
        
        # 3. Lightweight obs + empty info (no camera rendering!)
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = False
        info = {}  # Skip _get_info() which does expensive camera rendering
        
        # 4. Visual window rendering (fast, only the viewport)
        if self.render_mode == "human":
            if self._first_render:
                self._first_render = False
                self.mujoco_renderer.viewer._hide_menu = True
            self.render()
        
        return obs, reward, terminated, False, info
        
    def get_images(self):
        """On-demand camera rendering. Call this ONLY when the policy needs images."""
        return self._get_info()

    def _apply_baton_grip_stabilizer(self):
        st = getattr(self, "_baton_grip_stabilizer", None)
        if not st or not st.get("enabled", False):
            return

        baton_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "bottle2_freejoint"
        )
        baton_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bottle2")
        if baton_jid < 0 or baton_bid < 0:
            return

        active = st.get("active_robot")
        if active is None:
            active = self._select_baton_grip_stabilizer_robot(st, baton_bid)
            if active is None:
                return
            self._attach_baton_grip_stabilizer(st, active, baton_bid)
        else:
            prefix = "robot_a" if active == 0 else "robot_b"
            if self._gripper_qpos(prefix) >= st["release_grip_qpos"]:
                if st["verbose"]:
                    print(f"[BatonGrip] released from robot {active}")
                st["active_robot"] = None
                st["relative_pos"] = None
                st["relative_mat"] = None
                return

        self._restore_baton_grip_stabilizer(st, int(st["active_robot"]), baton_jid)

    def _select_baton_grip_stabilizer_robot(self, st, baton_bid):
        candidates = []
        for idx, prefix in enumerate(("robot_a", "robot_b")):
            requested = st.get("robot_index")
            if requested is not None and idx != requested:
                continue
            gq = self._gripper_qpos(prefix)
            if gq > st["attach_grip_qpos"]:
                continue
            palm_bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}/hand_palm_link"
            )
            if palm_bid < 0:
                continue
            dist = float(np.linalg.norm(self.data.xpos[baton_bid] - self.data.xpos[palm_bid]))
            if dist <= st["max_palm_dist"]:
                candidates.append((dist, idx))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def _attach_baton_grip_stabilizer(self, st, robot_index, baton_bid):
        prefix = "robot_a" if robot_index == 0 else "robot_b"
        palm_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}/hand_palm_link"
        )
        palm_pos = self.data.xpos[palm_bid].copy()
        palm_mat = self.data.xmat[palm_bid].reshape(3, 3).copy()
        baton_pos = self.data.xpos[baton_bid].copy()
        baton_mat = self.data.xmat[baton_bid].reshape(3, 3).copy()
        st["active_robot"] = int(robot_index)
        st["relative_pos"] = palm_mat.T @ (baton_pos - palm_pos)
        st["relative_mat"] = palm_mat.T @ baton_mat
        if st["verbose"]:
            dist = float(np.linalg.norm(baton_pos - palm_pos))
            print(f"[BatonGrip] attached to robot {robot_index} (palm_dist={dist:.4f} m)")

    def _restore_baton_grip_stabilizer(self, st, robot_index, baton_jid):
        prefix = "robot_a" if robot_index == 0 else "robot_b"
        palm_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}/hand_palm_link"
        )
        palm_pos = self.data.xpos[palm_bid].copy()
        palm_mat = self.data.xmat[palm_bid].reshape(3, 3).copy()
        rel_pos = np.asarray(st["relative_pos"], dtype=np.float64)
        rel_mat = np.asarray(st["relative_mat"], dtype=np.float64).reshape(3, 3)
        target_pos = palm_pos + palm_mat @ rel_pos
        target_mat = palm_mat @ rel_mat
        target_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(target_quat, target_mat.reshape(-1))

        qadr = int(self.model.jnt_qposadr[baton_jid])
        vadr = int(self.model.jnt_dofadr[baton_jid])
        self.data.qpos[qadr : qadr + 3] = target_pos
        self.data.qpos[qadr + 3 : qadr + 7] = target_quat
        self.data.qvel[vadr : vadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _gripper_qpos(self, prefix):
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/hand_motor_joint"
        )
        return float(self.data.qpos[self.model.jnt_qposadr[joint_id]])

