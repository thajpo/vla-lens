from __future__ import annotations

import pandas as pd
import pytest

from vla_lens import create_synthetic_trace_dataset
from vla_lens.dataset import DatasetIndexError, build_dataset_index, validate_dataset_index
from vla_lens.dataset.index import (
    EPISODE_INDEX,
    INDEX_MANIFEST,
    MODEL_SITE_INDEX,
    REQUIRED_EPISODE_COLUMNS,
)


def test_dataset_index_overwrite_rebuild_is_deterministic(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=3)
    before = pd.read_parquet(dataset.root / EPISODE_INDEX)

    result = build_dataset_index(dataset.root, overwrite=True)
    after = pd.read_parquet(dataset.root / EPISODE_INDEX)

    assert result.mode == "rebuild"
    assert set(REQUIRED_EPISODE_COLUMNS) <= set(after.columns)
    pd.testing.assert_frame_equal(before, after)
    manifest = validate_dataset_index(dataset.root)
    assert manifest["indexed_episode_count"] == 2


def test_dataset_index_append_mode_adds_only_missing_trace_rows(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=3)
    episodes = pd.read_parquet(dataset.root / EPISODE_INDEX)
    model_sites = pd.read_parquet(dataset.root / MODEL_SITE_INDEX)
    first_trace_id = str(episodes.iloc[0]["trace_id"])

    episodes.head(1).to_parquet(dataset.root / EPISODE_INDEX, index=False)
    model_sites.loc[model_sites["trace_id"].astype(str) == first_trace_id].to_parquet(
        dataset.root / MODEL_SITE_INDEX,
        index=False,
    )

    result = build_dataset_index(dataset.root)
    rebuilt = pd.read_parquet(dataset.root / EPISODE_INDEX)

    assert result.mode == "append"
    assert rebuilt["trace_id"].astype(str).tolist() == ["synthetic_000", "synthetic_001"]
    assert rebuilt["trace_id"].astype(str).is_unique


def test_dataset_index_validation_fails_on_stale_manifest_schema(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    manifest_path = dataset.root / INDEX_MANIFEST
    manifest = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(manifest.replace('"0.1.0"', '"stale"', 1), encoding="utf-8")

    with pytest.raises(DatasetIndexError, match="Dataset index schema mismatch"):
        validate_dataset_index(dataset.root)


def test_dataset_index_validation_fails_on_stale_source_fingerprint(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    manifest_path = next((dataset.root / "vla_lens" / "episodes").glob("*/manifest.json"))
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(DatasetIndexError, match="fingerprint is stale"):
        validate_dataset_index(dataset.root)
