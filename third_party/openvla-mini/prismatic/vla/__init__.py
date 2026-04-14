__all__ = ["get_vla_dataset_and_collator"]


def __getattr__(name: str):
    if name == "get_vla_dataset_and_collator":
        from .materialize import get_vla_dataset_and_collator

        return get_vla_dataset_and_collator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
