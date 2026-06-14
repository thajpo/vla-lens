"""Build one-row-per-policy-call labels from PI0.5 object-flow labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_lens.pi05.policy_call_labels import save_pi05_policy_call_labels_artifact
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Trace dataset root")
    parser.add_argument("--name", default="PI0.5 policy-call labels")
    parser.add_argument("--object-flow-artifact-id", default=None)
    parser.add_argument(
        "--no-rebuild-index",
        action="store_true",
        help="Skip refreshing the dashboard artifact index after saving labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TraceDataset.open(args.root)
    saved = save_pi05_policy_call_labels_artifact(
        dataset,
        name=args.name,
        object_flow_artifact_id=args.object_flow_artifact_id,
        rebuild_index=not args.no_rebuild_index,
    )
    outputs = dict(saved.artifact.method.get("outputs") or {})
    print(f"artifact_id={saved.artifact.artifact_id}")
    print(f"artifact_type={saved.artifact.artifact_type}")
    print(f"path={saved.artifact.path}")
    print(f"policy_call_labels={outputs.get('policy_call_labels', '')}")
    for key, value in saved.artifact.metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
