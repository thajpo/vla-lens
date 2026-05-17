"""Train a simple probe suite and save it as a VLA-lens artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vla_lens.probes import (
    dump_probe_spec,
    load_probe_spec,
    train_probe_artifact,
    train_probe_artifact_from_spec,
)
from vla_lens.selectors import ActivationQuery
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Trace dataset root or one .vlatrace bundle")
    parser.add_argument("--spec", default=None, help="YAML probe spec path. Use '-' for stdin.")
    parser.add_argument(
        "--print-spec",
        action="store_true",
        help="Print a complete YAML spec from the provided flags and exit.",
    )
    parser.add_argument("--name", default="Outcome probe")
    parser.add_argument("--target", default="outcome", help="Episode/sample column to predict")
    parser.add_argument(
        "--module",
        default=None,
        help="Activation module glob, e.g. 'backbone.layers.*.resid'",
    )
    parser.add_argument("--name-glob", default=None, help="Activation name glob")
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma/range layer list, e.g. '0,4,8' or '0-15'",
    )
    parser.add_argument("--tensor-type", default=None)
    parser.add_argument("--token-kind", default=None, help="Example: image_patch or action")
    parser.add_argument(
        "--timesteps",
        default="all",
        help="'all', one integer, comma list, or range",
    )
    parser.add_argument("--generation-step", default=None)
    parser.add_argument("--reduce-tokens", default="mean", choices=["mean", "flat", "none"])
    parser.add_argument(
        "--episode-filter",
        action="append",
        default=[],
        help="Episode filter key=value",
    )
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-value", default="train")
    parser.add_argument("--test-value", default="test")
    parser.add_argument(
        "--metadata-baseline",
        default="",
        help="Comma-separated metadata baseline columns",
    )
    parser.add_argument(
        "--sweep",
        default="layer",
        help="Row column to sweep, usually layer. Use 'none' to train once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TraceDataset.open(args.root)
    if args.spec:
        spec = load_probe_spec(args.spec)
        if args.print_spec:
            print(dump_probe_spec(spec))
            return
        saved = train_probe_artifact_from_spec(dataset, spec)
    else:
        selector = ActivationQuery(
            episodes=_parse_filters(args.episode_filter),
            name=args.name_glob,
            module=args.module,
            layers=_parse_int_list(args.layers),
            tensor_type=args.tensor_type,
            token_kind=args.token_kind,
            timesteps=_parse_timesteps(args.timesteps),
            generation_step=_parse_value(args.generation_step),
            reduce_tokens=args.reduce_tokens,
        )
        if args.print_spec:
            print(
                dump_probe_spec(
                    {
                        "name": args.name,
                        "target": {"kind": args.target},
                        "features": selector.to_dict(),
                        "split": {
                            "kind": "random_episode",
                            "column": args.split_column,
                            "train_value": args.train_value,
                            "test_value": args.test_value,
                        },
                        "baseline": [
                            item.strip()
                            for item in args.metadata_baseline.split(",")
                            if item.strip()
                        ],
                        "sweep": args.sweep,
                    }
                )
            )
            return
        saved = train_probe_artifact(
            dataset,
            name=args.name,
            selector=selector,
            target=args.target,
            split_column=args.split_column,
            train_value=args.train_value,
            test_value=args.test_value,
            metadata_baseline_columns=[
                item.strip() for item in args.metadata_baseline.split(",") if item.strip()
            ],
            sweep=args.sweep,
        )
    metrics = saved.artifact.metrics
    print(f"artifact_id={saved.artifact.artifact_id}")
    print(f"artifact_type={saved.artifact.artifact_type}")
    print(f"results={len(saved.results)}")
    print(f"best_score={metrics.get('best_score')}")
    print(f"best_delta={metrics.get('best_delta')}")
    print(f"path={saved.artifact.path}")


def _parse_filters(items: list[str]) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Episode filters must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        filters[key.strip()] = _parse_value(value.strip())
    return filters


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return out


def _parse_timesteps(value: str) -> Any:
    value = value.strip()
    if value == "all":
        return "all"
    parsed = _parse_int_list(value)
    if parsed is None:
        return "all"
    if len(parsed) == 1:
        return parsed[0]
    return parsed


def _parse_value(value: str | None) -> Any:
    if value is None or value == "":
        return None
    if value.lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
