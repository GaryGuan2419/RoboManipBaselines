import gymnasium as gym


class OperationMujocoHsrBSidePlace:
    """Default TeleopBase entry for B-side place collection.

    This keeps the project-standard teleop/data pipeline (same style as side_a_pick),
    while the B-side specifics remain in MujocoHsrBSidePlaceEnv config/XML.
    """

    def setup_env(self, render_mode="human", **kwargs):
        self.env = gym.make(
            "robo_manip_baselines/MujocoHsrBSidePlaceEnv-v0",
            render_mode=render_mode,
        )

    def get_pre_motion_phases(self):
        return []

    def set_additional_args(self, parser):
        # Teleop.py's --config is passed as kwargs to env setup, not as TeleopBase CLI args.
        # Set sane defaults here so this operation behaves like the custom keyboard collector.
        parser.set_defaults(
            input_device="keyboard",
            demo_name="hsr_b_side_place",
            task_desc="HSR B side place baton on pad",
            file_format="rmb",
            record_every_steps=2,
            image_every_steps=8,
        )
