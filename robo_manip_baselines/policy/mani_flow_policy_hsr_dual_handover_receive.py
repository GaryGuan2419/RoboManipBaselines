"""Rollout ManiFlow policy on dual HSR handover-receive env (dataset-aligned state split).

Imported as ``robo_manip_baselines.policy.mani_flow_policy_hsr_dual_handover_receive`` (Rollout CLI).

Eval / 与训练对齐的默认环境参数见 ``OperationMujocoDualHsrHandoverReceive.setup_env``：
``pick_eval_max_policy_action_steps=70``、``handover_pick_eval_min_baton_lowest_z_m=0.05``
（baton 最低世界 z ≥ 5 cm 判成功，相对地面而非相对 reset 基线）；扰动
``robot_{a,b}_xy_reset_perturb_half_extent_m`` 由 Rollout CLI 或 gym.make 覆盖。
"""

import cv2
import numpy as np
import torch

from robo_manip_baselines.common import DataKey, normalize_data
from robo_manip_baselines.policy.mani_flow_policy.RolloutManiFlowPolicy import (
    RolloutManiFlowPolicy,
)


class RolloutManiFlowPolicyHsrDualHandoverReceive(RolloutManiFlowPolicy):
    """Use ``robot_b/joint_pos`` as 6-dim then split into arm(5)+gripper(1) like training."""

    def update_state_buf(self):
        obs = self.obs
        jp = obs["robot_b/joint_pos"]
        parts = []
        for key in self.state_keys:
            if key == DataKey.MEASURED_JOINT_POS:
                parts.append(jp[:5])
            elif key == DataKey.MEASURED_GRIPPER_JOINT_POS:
                parts.append(jp[5:6])
            else:
                parts.append(self.motion_manager.get_measured_data(key, obs))
        state = np.concatenate(parts)
        state = normalize_data(state, self.model_meta_info["state"])
        state = torch.tensor(state, dtype=torch.float32)

        if self.state_buf is None:
            self.state_buf = [
                state for _ in range(self.model_meta_info["data"]["n_obs_steps"])
            ]
        else:
            self.state_buf.pop(0)
            self.state_buf.append(state)

    def update_images_buf(self):
        images = []
        for camera_name in self.camera_names:
            long_key = f"robot_b_{camera_name}"
            if "rgb_images" in self.info and long_key in self.info["rgb_images"]:
                image = self.info["rgb_images"][long_key]
            elif "rgb_images" in self.info and camera_name in self.info["rgb_images"]:
                image = self.info["rgb_images"][camera_name]
            else:
                rendered_info = self.env.unwrapped.get_images()
                if camera_name in rendered_info["rgb_images"]:
                    image = rendered_info["rgb_images"][camera_name]
                else:
                    image = rendered_info["rgb_images"][long_key]

            image = cv2.resize(image, self.model_meta_info["data"]["image_size"])
            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image, dtype=torch.uint8)
            image = self.image_transforms(image)
            image = image * 2.0 - 1.0
            images.append(image)

        if self.images_buf is None:
            self.images_buf = [
                [image for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for image in images
            ]
        else:
            for single_images_buf, image in zip(self.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)
