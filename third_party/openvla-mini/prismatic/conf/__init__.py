from .models import ModelConfig, ModelRegistry

__all__ = [
    "DatasetConfig",
    "DatasetRegistry",
    "ModelConfig",
    "ModelRegistry",
    "VLAConfig",
    "VLARegistry",
]


def __getattr__(name: str):
    if name in {"DatasetConfig", "DatasetRegistry"}:
        from .datasets import DatasetConfig, DatasetRegistry

        return {"DatasetConfig": DatasetConfig, "DatasetRegistry": DatasetRegistry}[name]
    if name in {"VLAConfig", "VLARegistry"}:
        from .vla import VLAConfig, VLARegistry

        return {"VLAConfig": VLAConfig, "VLARegistry": VLARegistry}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
