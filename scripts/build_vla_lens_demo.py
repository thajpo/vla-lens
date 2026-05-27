"""Build a synthetic LeRobot-backed VLA Lens dataset for the live dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_lens import create_synthetic_trace_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/vla_lens_demo"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=24)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = create_synthetic_trace_dataset(
        args.out,
        num_episodes=args.episodes,
        timesteps=args.timesteps,
        layers=args.layers,
        overwrite=args.overwrite,
    )
    print(f"dataset={dataset.root}")
    print(f"serve_dashboard=uv run python scripts/serve_vla_lens_dashboard.py {dataset.root}")


if __name__ == "__main__":
    main()
