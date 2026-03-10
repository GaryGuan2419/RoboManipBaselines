import dataclasses

import numpy as np

from ..data.DataKey import DataKey
from .BodyManagerBase import BodyConfigBase, BodyManagerBase


class MobileOmniManager(BodyManagerBase):
    """Manager for omni-directional mobile base."""

    SUPPORTED_DATA_KEYS = [
        DataKey.MEASURED_MOBILE_OMNI_VEL,
        DataKey.COMMAND_MOBILE_OMNI_VEL,
    ]

    def reset(self, init=False):
        self.target_vel = np.zeros(3)

    def set_command_data(self, key, command, is_skip=False):
        if key == DataKey.COMMAND_MOBILE_OMNI_VEL:
            self.set_command_vel(command[self.body_config.mobile_vel_idxes])
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid command data key: {key}"
            )

    def set_command_vel(self, vel):
        self.target_vel = vel

    def get_command_data(self, key):
        if key == DataKey.COMMAND_MOBILE_OMNI_VEL:
            return self.get_command_vel()
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid command data key: {key}"
            )

    def get_command_vel(self):
        return self.target_vel

    def draw_markers(self):
        pass


@dataclasses.dataclass
class MobileOmniConfig(BodyConfigBase):
    """Configuration for omni-directional mobile base."""

    BodyManagerClass = MobileOmniManager

    # Relative indices within the mobile velocity observation/command space
    mobile_vel_idxes: np.ndarray = dataclasses.field(
        default_factory=lambda: np.arange(3)
    )
