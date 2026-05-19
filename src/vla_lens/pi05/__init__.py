"""PI0.5-specific interpretability primitives."""

from __future__ import annotations

from vla_lens.pi05.interventions import InterventionSpec, InterventionType

__all__ = [
    "ActivationSelector",
    "InterventionSpec",
    "InterventionType",
    "parse_selector",
]


def __getattr__(name: str):
    if name in {"ActivationSelector", "parse_selector"}:
        from vla_lens.pi05.selectors import ActivationSelector, parse_selector

        return {"ActivationSelector": ActivationSelector, "parse_selector": parse_selector}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
