class BodyManagerBase:
    """Manager for each body component (e.g., single arm, mobile base)."""

    def __init__(self, env, body_config):
        self.env = env
        self.body_config = body_config

    def sync_with_obs(self, obs):
        """Synchronize the internal target state with the actual observed state. Override if needed."""
        pass


class BodyConfigBase:
    """Configuration  for each body component (e.g., single arm, mobile base)."""

    pass
