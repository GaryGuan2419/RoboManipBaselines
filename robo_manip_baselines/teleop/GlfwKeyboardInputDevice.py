"""
GlfwKeyboardInputDevice: a keyboard input device that uses GLFW key callbacks
from the MuJoCo viewer window, instead of pynput (which doesn't work on WSL2).

Supports dual-robot mode: press 1/2 to switch which robot's arm_manager
receives commands. Visual feedback is printed when switching.
"""

import numpy as np
import pinocchio as pin
import glfw

from .InputDeviceBase import InputDeviceBase


class GlfwKeyboardInputDevice(InputDeviceBase):
    """Keyboard input device using GLFW key callbacks from MuJoCo viewer.
    
    Supports controlling multiple arm_managers with toggle keys (1, 2, ...).
    """

    def __init__(
        self,
        arm_manager,
        pos_scale=1e-2,
        rpy_scale=5e-2,
        gripper_scale=50.0,
    ):
        super().__init__()

        # Support single or multiple arm_managers
        if isinstance(arm_manager, (list, tuple)):
            self.arm_managers = list(arm_manager)
        else:
            self.arm_managers = [arm_manager]
        
        self.active_idx = 0  # Which robot is currently controlled
        self.pos_scale = pos_scale
        self.rpy_scale = rpy_scale
        self.gripper_scale = gripper_scale

        # Track key states
        self.state = {
            # position control keys (GLFW key constants)
            glfw.KEY_W: False,   # forward
            glfw.KEY_S: False,   # backward
            glfw.KEY_A: False,
            glfw.KEY_D: False,
            glfw.KEY_Q: False,
            glfw.KEY_E: False,
            # rotation control keys
            glfw.KEY_I: False,
            glfw.KEY_K: False,
            glfw.KEY_J: False,
            glfw.KEY_L: False,
            glfw.KEY_U: False,
            glfw.KEY_O: False,
            # gripper control keys
            glfw.KEY_Z: False,
            glfw.KEY_X: False,
        }

        # Robot switching keys: 1 = robot A, 2 = robot B
        self._switch_keys = {
            glfw.KEY_1: 0,
            glfw.KEY_2: 1,
        }

        self._viewer = None
        self._original_key_callback = None
        self._env = None  # set via attach_to_viewer for print_joint_state

    @property
    def arm_manager(self):
        """Current active arm manager."""
        return self.arm_managers[self.active_idx]

    def connect(self):
        if self.connected:
            return

        self.connected = True
        n = len(self.arm_managers)
        robot_names = ["Robot A", "Robot B", "Robot C", "Robot D"][:n]
        print(f"[{self.__class__.__name__}] Connected (GLFW mode).")
        print(f"[{self.__class__.__name__}] Controlling: {robot_names[self.active_idx]}")
        print(f"[{self.__class__.__name__}] Key Bindings (press in MuJoCo viewer window):")
        print("  - WASD : X/Y-axis forward/backward, left/right")
        print("  - QE   : Z-axis up/down")
        print("  - IJKL : Roll and Pitch rotation")
        print("  - UO   : Yaw rotation")
        print("  - Z/X  : Gripper close/open")
        print("  - P    : Print joint angles")
        print("  - B    : Reset object position (bring back)")
        if n > 1:
            keys_str = ", ".join(f"{k+1}={robot_names[k]}" for k in range(n))
            print(f"  - {keys_str} : Switch robot")

    def attach_to_viewer(self, viewer, env=None):
        """Attach GLFW key callback to an existing MuJoCo viewer window."""
        self._viewer = viewer
        self._env = env
        if hasattr(viewer, 'window') and viewer.window is not None:
            self._original_key_callback = glfw.set_key_callback(
                viewer.window, self._glfw_key_callback
            )
            print(f"[{self.__class__.__name__}] GLFW key callback attached to viewer.")
            print(f"[{self.__class__.__name__}] Press P in viewer to print joint angles.")

    def _glfw_key_callback(self, window, key, scancode, action, mods):
        """GLFW key callback: update key state dictionary."""
        # Print joint state on P press
        if key == glfw.KEY_P and action == glfw.PRESS:
            if self._env and hasattr(self._env, 'print_joint_state'):
                self._env.print_joint_state()
            else:
                print("[P] print_joint_state not available")

        # Reset object position on B press (bring back)
        if key == glfw.KEY_B and action == glfw.PRESS:
            if self._env and hasattr(self._env, 'reset_object'):
                self._env.reset_object()
                print("[B] 圆柱体已归位")
            else:
                print("[B] reset_object not available")

        # Handle robot switching on press only
        if key in self._switch_keys and action == glfw.PRESS:
            new_idx = self._switch_keys[key]
            if new_idx < len(self.arm_managers):
                self.active_idx = new_idx
                robot_names = ["Robot A", "Robot B", "Robot C", "Robot D"]
                print(f"[{self.__class__.__name__}] >>> Switched to {robot_names[new_idx]} <<<")

        # Handle control keys
        if key in self.state:
            if action == glfw.PRESS or action == glfw.REPEAT:
                self.state[key] = True
            elif action == glfw.RELEASE:
                self.state[key] = False

        # Intercept WASD keys to prevent conflicts with default MuJoCo viewer hotkeys (like S for shadows)
        if key in [glfw.KEY_W, glfw.KEY_A, glfw.KEY_S, glfw.KEY_D]:
            return

        # Chain to original callback if it existed
        if self._original_key_callback is not None:
            self._original_key_callback(window, key, scancode, action, mods)

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")
        # GLFW events are polled in the render loop; nothing extra to do here.

    def set_command_data(self):
        delta_pos = np.zeros(3)

        # X-axis (forward / backward)
        if self.state[glfw.KEY_W]:
            delta_pos[0] += self.pos_scale
        if self.state[glfw.KEY_S]:
            delta_pos[0] -= self.pos_scale

        # Y-axis
        if self.state[glfw.KEY_A]:
            delta_pos[1] += self.pos_scale
        if self.state[glfw.KEY_D]:
            delta_pos[1] -= self.pos_scale

        # Z-axis
        if self.state[glfw.KEY_Q]:
            delta_pos[2] += self.pos_scale
        if self.state[glfw.KEY_E]:
            delta_pos[2] -= self.pos_scale

        delta_rpy = np.zeros(3)

        # Roll
        if self.state[glfw.KEY_J]:
            delta_rpy[0] -= self.rpy_scale
        if self.state[glfw.KEY_L]:
            delta_rpy[0] += self.rpy_scale

        # Pitch
        if self.state[glfw.KEY_I]:
            delta_rpy[1] += self.rpy_scale
        if self.state[glfw.KEY_K]:
            delta_rpy[1] -= self.rpy_scale

        # Yaw
        if self.state[glfw.KEY_U]:
            delta_rpy[2] += self.rpy_scale * 2.0
        if self.state[glfw.KEY_O]:
            delta_rpy[2] -= self.rpy_scale * 2.0

        # Apply to the ACTIVE arm manager
        target_se3 = self.arm_manager.target_se3.copy()
        target_se3.translation += delta_pos

        # Multiply rotation
        new_rot = pin.rpy.rpyToMatrix(*delta_rpy) @ target_se3.rotation

        # Re-orthogonalize the rotation matrix
        quat = pin.Quaternion(new_rot)
        quat.normalize()
        target_se3.rotation = quat.matrix()

        self.arm_manager.set_command_eef_pose(target_se3)

        # Set gripper command
        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()

        if self.state[glfw.KEY_Z] and not self.state[glfw.KEY_X]:
            gripper_joint_pos += self.gripper_scale
        elif self.state[glfw.KEY_X] and not self.state[glfw.KEY_Z]:
            gripper_joint_pos -= self.gripper_scale

        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)

    def close(self):
        if self.connected:
            self.connected = False
            print(f"[{self.__class__.__name__}] Disconnected.")
