__all__ = ["DummyDataset", "EpisodicRLDSDataset", "RLDSBatchTransform", "RLDSDataset"]


def __getattr__(name: str):
    if name in set(__all__):
        from .datasets import DummyDataset, EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset

        return {
            "DummyDataset": DummyDataset,
            "EpisodicRLDSDataset": EpisodicRLDSDataset,
            "RLDSBatchTransform": RLDSBatchTransform,
            "RLDSDataset": RLDSDataset,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
