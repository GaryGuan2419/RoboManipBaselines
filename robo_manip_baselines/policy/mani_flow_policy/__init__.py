from __future__ import annotations

from .ManiFlowImageDataset import ManiFlowImageDataset
from .ManiFlowPointcloudDataset import ManiFlowPointcloudDataset
from .TrainManiFlowPolicy import TrainManiFlowPolicy

__all__ = [
    "ManiFlowImageDataset",
    "ManiFlowPointcloudDataset",
    "TrainManiFlowPolicy",
    "RolloutManiFlowPolicy",
]


def __getattr__(name: str):
    """Rollout 依赖 pytorch3d；仅 ``import ...mani_flow_policy`` 时不强制加载 Rollout。"""
    if name == "RolloutManiFlowPolicy":
        from .RolloutManiFlowPolicy import RolloutManiFlowPolicy as _RolloutManiFlowPolicy

        return _RolloutManiFlowPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
