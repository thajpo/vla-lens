"""Save a dataset summary report as a VLA-lens artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_lens.artifacts import LensArtifact
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="LeRobot v3 dataset root, trace dataset root, or one .vlatrace bundle",
    )
    parser.add_argument("--name", default="Dataset summary report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TraceDataset.open(args.root)
    episode_index = dataset.episode_index
    artifact = LensArtifact.create(
        artifact_type="dataset_report",
        name=args.name,
        group_id="dataset_reports",
        scope="dataset",
        selector={"episodes": "all"},
        method={"workflow": "save_vla_lens_dataset_report"},
        metrics={
            "episode_count": int(len(episode_index)),
            "activation_site_count": int(len(dataset.model_site_index)),
            "artifact_count_before_save": int(len(dataset.artifact_index)),
        },
        display={
            "kind": "dataset_report",
            "outcomes": _value_counts(episode_index, "outcome"),
            "tasks": _value_counts(episode_index, "task_id"),
            "models": _value_counts(episode_index, "model_id"),
            "activation_coverage": dataset.stats.activation_coverage().to_dict("records"),
        },
        tags=("report", "dataset"),
        source_trace_ids=tuple(sorted(str(value) for value in episode_index["trace_id"])),
    )
    saved = dataset.save_artifact(artifact)
    print(f"artifact_id={saved.artifact_id}")
    print(f"artifact_type={saved.artifact_type}")
    print(f"path={saved.path}")


def _value_counts(frame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].astype(str).value_counts(dropna=False).to_dict().items()
    }


if __name__ == "__main__":
    main()
