"""Run a staged VLA-lens probe campaign from YAML specs."""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from vla_lens.artifacts import LensArtifact, make_artifact_id
from vla_lens.probes import load_probe_spec, train_probe_artifact_from_spec
from vla_lens.traces import TraceDataset


@dataclass(frozen=True, slots=True)
class ProbeBatchRecord:
    spec_path: str
    name: str
    status: str
    artifact_id: str | None = None
    best_score: float | None = None
    best_baseline: float | None = None
    best_delta: float | None = None
    best_model: str | None = None
    best_eval_split: str | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Trace dataset root")
    parser.add_argument("--campaign", type=Path, default=None, help="YAML campaign file")
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="Probe YAML spec path. Can be supplied more than once.",
    )
    parser.add_argument(
        "--spec-glob",
        action="append",
        default=[],
        help="Glob for probe YAML specs, resolved from the current working directory.",
    )
    parser.add_argument("--name", default=None, help="Override campaign artifact/display name")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run", action="store_true", help="Train probes. Omit for dry-run.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed spec")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TraceDataset.open(args.root)
    campaign = _load_campaign(args.campaign)
    spec_paths = _campaign_specs(campaign, args.spec, args.spec_glob)
    if args.limit is not None:
        spec_paths = spec_paths[: args.limit]
    if not spec_paths:
        raise SystemExit("No probe specs were provided")

    campaign_name = args.name or str(campaign.get("name") or "VLA-lens probe campaign")
    records: list[ProbeBatchRecord] = []
    print(f"campaign={campaign_name}")
    print(f"dataset={dataset.root}")
    print(f"specs={len(spec_paths)}")
    print(f"mode={'run' if args.run else 'dry-run'}")

    for spec_path in spec_paths:
        spec = load_probe_spec(spec_path)
        name = str(spec.get("name") or spec_path.stem)
        if not args.run:
            target = spec.get("target", {})
            split = spec.get("split", {})
            probe = spec.get("probe", {})
            print(
                f"dry_run spec={spec_path} name={name!r} "
                f"target={target} split={split} models={probe.get('models', ['linear'])}"
            )
            records.append(ProbeBatchRecord(str(spec_path), name, "dry_run"))
            continue

        print(f"running spec={spec_path} name={name!r}")
        try:
            saved = train_probe_artifact_from_spec(dataset, spec)
        except Exception as exc:
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            print(f"failed spec={spec_path} error={error}")
            records.append(ProbeBatchRecord(str(spec_path), name, "failed", error=error))
            if args.fail_fast:
                break
            continue
        metrics = dict(saved.artifact.metrics or {})
        records.append(
            ProbeBatchRecord(
                spec_path=str(spec_path),
                name=name,
                status="completed",
                artifact_id=saved.artifact.artifact_id,
                best_score=_optional_float(metrics.get("best_score")),
                best_baseline=_optional_float(metrics.get("best_baseline")),
                best_delta=_optional_float(metrics.get("best_delta")),
                best_model=_optional_str(metrics.get("best_model")),
                best_eval_split=_optional_str(metrics.get("best_eval_split")),
            )
        )
        print(
            f"completed artifact_id={saved.artifact.artifact_id} "
            f"best_delta={metrics.get('best_delta')}"
        )

    if args.run:
        campaign_artifact = _save_campaign_artifact(dataset, campaign_name, campaign, records)
        print(f"campaign_artifact_id={campaign_artifact.artifact_id}")
        print(f"campaign_path={campaign_artifact.path}")


def _load_campaign(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError("Campaign YAML must be a mapping")
    base = dict(payload)
    base["_campaign_path"] = str(path)
    return base


def _campaign_specs(
    campaign: Mapping[str, Any],
    explicit_specs: Sequence[str],
    glob_patterns: Sequence[str],
) -> list[Path]:
    paths: list[Path] = []
    base_dir = Path(str(campaign.get("_campaign_path") or ".")).resolve().parent
    for item in campaign.get("specs") or []:
        paths.append(_resolve_path(base_dir, item))
    for pattern in campaign.get("spec_globs") or []:
        paths.extend(sorted(Path.cwd().glob(str(pattern))))
    paths.extend(Path(item) for item in explicit_specs)
    for pattern in glob_patterns:
        paths.extend(sorted(Path.cwd().glob(pattern)))
    return list(dict.fromkeys(path.resolve() for path in paths))


def _resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    direct = (Path.cwd() / path).resolve()
    if direct.exists():
        return direct
    return (base_dir / path).resolve()


def _save_campaign_artifact(
    dataset: TraceDataset,
    name: str,
    campaign: Mapping[str, Any],
    records: Sequence[ProbeBatchRecord],
) -> LensArtifact:
    artifact_id = make_artifact_id(name, "probe_campaign")
    relative_dir = Path("artifacts") / artifact_id
    outputs = {
        "summary": str(relative_dir / "campaign_summary.parquet"),
        "manifest": str(relative_dir / "run_manifest.json"),
    }
    frame = pd.DataFrame.from_records([asdict(record) for record in records])
    metrics = {
        "spec_count": int(len(records)),
        "completed_count": int((frame["status"] == "completed").sum()) if not frame.empty else 0,
        "failed_count": int((frame["status"] == "failed").sum()) if not frame.empty else 0,
    }
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="probe_campaign",
        name=name,
        group_id="probe_campaigns",
        scope="dataset",
        selector={},
        method={
            "workflow": "run_vla_lens_probe_batch",
            "campaign": _jsonable_mapping(campaign),
            "outputs": outputs,
        },
        metrics=metrics,
        display={"kind": "probe_campaign", "records": frame.to_dict("records")},
        tags=("probe", "campaign"),
    )
    saved = dataset.save_artifact(artifact)
    artifact_dir = dataset.root / relative_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(artifact_dir / "campaign_summary.parquet", index=False)
    (artifact_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "campaign": _jsonable_mapping(campaign),
                "records": [asdict(record) for record in records],
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return saved


def _jsonable_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): str(value) if isinstance(value, Path) else value
        for key, value in payload.items()
        if str(key) != "_campaign_path"
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


if __name__ == "__main__":
    main()
