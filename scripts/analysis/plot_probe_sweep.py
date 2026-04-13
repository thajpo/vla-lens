#!/usr/bin/env python3
"""
Fit probes on collected activations and plot accuracy heatmap (layer × token_position).

Usage:
    python scripts/analysis/plot_probe_sweep.py --run-id pilot_20ep

Reads:
    artifacts/activations/{run_id}/activations.parquet
    artifacts/activations/{run_id}/episodes.parquet
    artifacts/activations/{run_id}/tensors/*.pt

Writes:
    artifacts/figures/probe_sweep/{run_id}_heatmap.png
    artifacts/figures/probe_sweep/{run_id}_sweep.parquet
    artifacts/probes/{run_id}_probes.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from openvla_steering.interp.probes import fit_probe, probe_sweep, save_probe
from openvla_steering.utils.io import ensure_parent


def load_activations_for_probe(
    run_dir: Path,
    task_a: int,
    task_b: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load activation tensors for two tasks and return per-(layer, position) arrays.

    Returns:
        {f"{layer}|{pos}": (X, y)}
        X: (n_samples, hidden_dim), y: (n_samples,) int {0=task_a, 1=task_b}
    """
    act_df = pd.read_parquet(run_dir / "activations.parquet")
    ep_df = pd.read_parquet(run_dir / "episodes.parquet")

    # Filter to only the two probe tasks
    ep_df = ep_df[ep_df["task_id"].isin([task_a, task_b])].copy()
    label_map = {task_a: 0, task_b: 1}

    # One row per (episode_id, layer, token_position) in activations.parquet
    # Each row points to a tensor of shape (n_steps, hidden_dim)
    # We aggregate by taking the mean over steps per episode (simple baseline)
    # More sophisticated: take step at a fixed index (e.g. step 10)

    result: dict[str, tuple[list, list]] = {}

    for _, row in act_df.iterrows():
        ep_id = row["episode_id"]
        ep_rows = ep_df[ep_df["episode_id"] == ep_id]
        if ep_rows.empty:
            continue
        task_id = ep_rows.iloc[0]["task_id"]
        if task_id not in label_map:
            continue
        label = label_map[task_id]

        tensor_path = ROOT / row["tensor_path"]
        if not tensor_path.exists():
            print(f"[WARN] Missing tensor: {tensor_path}")
            continue

        tensor = torch.load(tensor_path, map_location="cpu").float().numpy()
        # tensor: (n_steps, hidden_dim)
        # Use mean across steps as the episode-level feature
        feat = tensor.mean(axis=0)

        key = f"{row['layer']}|{row['token_position']}"
        if key not in result:
            result[key] = ([], [])
        result[key][0].append(feat)
        result[key][1].append(label)

    return {
        key: (np.stack(feats), np.array(labels))
        for key, (feats, labels) in result.items()
        if len(feats) >= 4
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--task-a", type=int, default=71, help="Task ID for class 0 (e.g. red mug)")
    p.add_argument("--task-b", type=int, default=72, help="Task ID for class 1 (e.g. white mug)")
    p.add_argument("--output-dir", default="artifacts/figures/probe_sweep")
    p.add_argument("--probes-dir", default="artifacts/probes")
    p.add_argument("--C", type=float, default=1.0, help="Logistic regression regularization")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--permutation-test", action="store_true",
                   help="Run permutation test for significance (slow)")
    args = p.parse_args()

    run_dir = ROOT / "artifacts" / "activations" / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    print(f"Loading activations from {run_dir}")
    data = load_activations_for_probe(run_dir, args.task_a, args.task_b)

    if not data:
        raise SystemExit("No activation data found. Check run_id and task IDs.")

    print(f"Found {len(data)} (layer, position) pairs. Fitting probes...")

    # Build activations dict for probe_sweep
    X_dict = {key: Xy[0] for key, Xy in data.items()}
    # All keys share the same y (same episode set), use any
    first_key = next(iter(data))
    y = data[first_key][1]

    sweep_df = probe_sweep(X_dict, y, C=args.C, cv=args.cv)
    sweep_df["run_id"] = args.run_id
    sweep_df["task_a"] = args.task_a
    sweep_df["task_b"] = args.task_b

    # Parse layer and position from key
    sweep_df[["layer", "token_position"]] = sweep_df["module"].str.split("|", expand=True)

    # Save sweep results
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / f"{args.run_id}_sweep.parquet"
    sweep_df.to_parquet(sweep_path, index=False)
    print(f"\nSweep results saved to {sweep_path}")

    # Print top results
    print(f"\nTop 10 probes by CV accuracy:")
    print(sweep_df.head(10)[["layer", "token_position", "cv_accuracy", "cv_auc"]].to_string(index=False))

    best_row = sweep_df.iloc[0]
    print(f"\nBest: layer={best_row['layer']}  position={best_row['token_position']}")
    print(f"      CV accuracy={best_row['cv_accuracy']:.3f}  AUROC={best_row['cv_auc']:.3f}")

    if best_row["cv_accuracy"] < 0.60:
        print("\n[WARN] Peak accuracy < 0.60. Check data pipeline before scaling.")
        print("       Common causes: wrong layer names, label mismatch, too few episodes.")
    elif best_row["cv_accuracy"] >= 0.70:
        print("\n[OK] Peak accuracy >= 0.70. Safe to proceed with full 200-episode collection.")

    # Fit and save best probe for downstream use
    best_key = best_row["module"]
    X_best, y_best = data[best_key]
    best_probe = fit_probe(X_best, y_best, C=args.C, cv=args.cv)
    probe_dir = ROOT / args.probes_dir
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_dir / f"{args.run_id}_best_probe.pkl"
    save_probe({**best_probe, "layer": best_row["layer"], "token_position": best_row["token_position"]},
               probe_path)
    print(f"Best probe saved to {probe_path}")

    # Heatmap — requires matplotlib
    try:
        import matplotlib.pyplot as plt

        # Parse layer index for ordering
        def layer_sort_key(layer_name: str) -> int:
            try:
                return int(layer_name.split(".")[-1])
            except Exception:
                return 0

        positions = sweep_df["token_position"].unique().tolist()
        layers = sorted(sweep_df["layer"].unique().tolist(), key=layer_sort_key)

        matrix = np.full((len(layers), len(positions)), np.nan)
        for _, row in sweep_df.iterrows():
            li = layers.index(row["layer"])
            pi = positions.index(row["token_position"])
            matrix[li, pi] = row["cv_accuracy"]

        fig, ax = plt.subplots(figsize=(max(4, len(positions) * 2), max(6, len(layers) * 0.4)))
        im = ax.imshow(matrix, aspect="auto", vmin=0.4, vmax=1.0, cmap="RdYlGn")
        plt.colorbar(im, ax=ax, label="CV Accuracy")
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions, rotation=15)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([l.split(".")[-1] for l in layers])
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer Index")
        ax.set_title(f"Color Probe Accuracy: task {args.task_a} vs {args.task_b}\n({args.run_id})")

        # Annotate cells
        for i in range(len(layers)):
            for j in range(len(positions)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="black")

        fig_path = out_dir / f"{args.run_id}_heatmap.png"
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        print(f"Heatmap saved to {fig_path}")
        plt.close(fig)

    except ImportError:
        print("[WARN] matplotlib not available; skipping heatmap.")


if __name__ == "__main__":
    main()
