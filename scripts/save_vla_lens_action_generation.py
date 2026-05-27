"""Save an ActionGeneration artifact for a LeRobot-backed VLA Lens dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_lens.action_generation import save_action_generation_artifact
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="LeRobot v3 dataset root or top-level directory containing nested LeRobot v3 roots",
    )
    parser.add_argument("--name", default="Action generation summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = save_action_generation_artifact(TraceDataset.open(args.root), name=args.name)
    metrics = saved.artifact.metrics
    print(f"artifact_id={saved.artifact.artifact_id}")
    print(f"artifact_type={saved.artifact.artifact_type}")
    print(f"episodes={metrics.get('episode_count')}")
    print(f"policy_decisions={metrics.get('policy_decision_count')}")
    print(f"mean_initial_commitment={metrics.get('mean_initial_commitment')}")
    print(f"mean_executed_vs_predicted={metrics.get('mean_executed_vs_predicted')}")
    print(f"path={saved.artifact.path}")


if __name__ == "__main__":
    main()
