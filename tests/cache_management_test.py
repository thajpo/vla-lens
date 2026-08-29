from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path

import pandas as pd
import pytest
import zarr

from vla_lens.artifacts import LensArtifact
from vla_lens.cache import CacheBuildMetadata, CacheManager
from vla_lens.cache_cli import main as cache_main
from vla_lens.campaigns import prepare_feature_campaign
from vla_lens.selectors import ActivationQuery
from vla_lens.synthetic import create_synthetic_trace_dataset
from vla_lens.traces import TraceDataset


def _concurrent_cache_worker(
    cache_root: str,
    markers: str,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    manager = CacheManager(cache_root)
    start.wait()

    def build(path: Path) -> CacheBuildMetadata:
        Path(markers, str(os.getpid())).write_text("built", encoding="utf-8")
        time.sleep(0.15)
        (path / "payload.bin").write_bytes(b"shared")
        return CacheBuildMetadata(shape=(1, 6), dtype="uint8", axes=("row", "byte"))

    _, manifest, built = manager.get_or_build(
        namespace="features",
        key="shared-key",
        recipe={"selector": "same"},
        source_fingerprint="sha256:source",
        builder=build,
        validator=lambda path: (path / "payload.bin").read_bytes() == b"shared",
    )
    results.put((built, manifest.content_fingerprint))


def _concurrent_artifact_worker(
    dataset_root: str,
    index: int,
    start: multiprocessing.synchronize.Event,
) -> None:
    start.wait()
    TraceDataset.open(dataset_root).save_artifact(
        LensArtifact(
            artifact_id=f"parallel-artifact-{index}",
            artifact_type="test",
            name=f"Parallel artifact {index}",
            scope="dataset",
        )
    )


def test_cache_builds_once_across_processes(tmp_path):
    cache_root = tmp_path / ".vla_cache"
    markers = tmp_path / "markers"
    markers.mkdir()
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_cache_worker,
            args=(str(cache_root), str(markers), start, results),
        )
        for _ in range(3)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    outcomes = [results.get(timeout=1) for _ in processes]
    assert sum(built for built, _ in outcomes) == 1
    assert len({fingerprint for _, fingerprint in outcomes}) == 1
    assert len(list(markers.iterdir())) == 1
    manifest = CacheManager(cache_root).manifest("features", "shared-key")
    assert manifest.complete
    assert manifest.shape == (1, 6)
    assert manifest.size_bytes == 6


def test_cache_recovers_incomplete_entry_and_preserves_pin(tmp_path):
    manager = CacheManager(tmp_path / ".vla_cache")

    def build(path: Path) -> CacheBuildMetadata:
        (path / "payload.bin").write_bytes(b"first")
        return CacheBuildMetadata()

    entry, _, _ = manager.get_or_build(
        namespace="features",
        key="recoverable",
        recipe={"version": 1},
        source_fingerprint="sha256:first",
        builder=build,
    )
    manager.set_pinned("features", "recoverable", pinned=True)
    (entry.parent / f".{entry.name}.tmp-interrupted").mkdir()

    def rebuild(path: Path) -> CacheBuildMetadata:
        (path / "payload.bin").write_bytes(b"second")
        return CacheBuildMetadata()

    rebuilt_path, manifest, built = manager.get_or_build(
        namespace="features",
        key="recoverable",
        recipe={"version": 1},
        source_fingerprint="sha256:second",
        builder=rebuild,
        validator=lambda path: (path / "payload.bin").exists(),
    )
    assert built
    assert manifest.pinned
    assert (rebuilt_path / "payload.bin").read_bytes() == b"second"
    assert not list(rebuilt_path.parent.glob(".recoverable.tmp-*"))


def test_prune_is_dry_run_by_default_and_never_removes_pins(tmp_path):
    manager = CacheManager(tmp_path / ".vla_cache")
    for key in ("keep", "drop"):
        def build(path: Path, value: str = key) -> CacheBuildMetadata:
            (path / "data").write_text(value, encoding="utf-8")
            return CacheBuildMetadata()

        manager.get_or_build(
            namespace="features",
            key=key,
            recipe={"key": key},
            source_fingerprint="sha256:source",
            builder=build,
        )
    manager.set_pinned("features", "keep", pinned=True)

    planned = manager.prune(max_bytes=0, min_free_bytes=0)
    assert not planned.apply
    assert planned.entries == ("features/drop",)
    assert manager.entry_path("features", "drop").exists()
    assert planned.blocked_by_pins

    applied = manager.prune(max_bytes=0, min_free_bytes=0, apply=True)
    assert applied.entries == ("features/drop",)
    assert not manager.entry_path("features", "drop").exists()
    assert manager.entry_path("features", "keep").exists()
    with pytest.raises(ValueError, match="named .vla_cache"):
        CacheManager(tmp_path / "not-cache").prune(max_bytes=0, min_free_bytes=0)


def test_prune_does_not_delete_entry_replaced_after_plan(tmp_path, monkeypatch):
    manager = CacheManager(tmp_path / ".vla_cache")

    def build(path: Path, payload: bytes) -> CacheBuildMetadata:
        (path / "payload.bin").write_bytes(payload)
        return CacheBuildMetadata()

    _, stale, _ = manager.get_or_build(
        namespace="features",
        key="replaced",
        recipe={"version": 1},
        source_fingerprint="sha256:stale",
        builder=lambda path: build(path, b"stale"),
    )
    entry, replacement, _ = manager.get_or_build(
        namespace="features",
        key="replaced",
        recipe={"version": 1},
        source_fingerprint="sha256:replacement",
        builder=lambda path: build(path, b"replacement"),
    )
    assert replacement.source_fingerprint != stale.source_fingerprint

    # Simulate a prune plan becoming stale before its apply phase. In
    # production this is another worker rebuilding the same cache key.
    monkeypatch.setattr(manager, "entries", lambda: [stale])
    result = manager.prune(max_bytes=0, min_free_bytes=0, apply=True)

    assert result.entries == ()
    assert not result.blocked_by_pins
    assert (entry / "payload.bin").read_bytes() == b"replacement"


def test_campaign_prepare_deduplicates_identical_selectors(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "dataset", num_episodes=2, timesteps=4)
    selector = {
        "module": "backbone.layers.*.resid",
        "layers": [0],
        "token_kind": "image_patch",
        "timesteps": [0],
        "reduce_tokens": "mean",
    }
    campaign = {
        "campaign_id": "object-research",
        "precomputes": [
            {"name": "identity", "selector": selector},
            {"name": "localization", "selector": selector},
        ],
    }

    first = prepare_feature_campaign(dataset, campaign)
    second = prepare_feature_campaign(dataset, campaign)

    assert first.requested_count == 2
    assert first.unique_count == 1
    assert first.features[0].names == ("identity", "localization")
    assert first.features[0].built
    assert not second.features[0].built


def test_feature_scientific_key_ignores_source_timestamps_but_freshness_does_not(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "dataset", num_episodes=1, timesteps=4)
    query = ActivationQuery(
        module="backbone.layers.*.resid",
        layers=[0],
        token_kind="image_patch",
        timesteps=[0],
        reduce_tokens="mean",
    )
    view = dataset.select_model_sites(query)
    key = view.cache_key()
    first = view.materialize(cache=True)
    site = view._matching_model_sites().iloc[0]
    array_path = Path(str(site["bundle_path"])) / str(site["relative_path"])
    metadata_path = array_path / ".zarray"
    metadata_stat = metadata_path.stat()
    os.utime(
        metadata_path,
        ns=(metadata_stat.st_atime_ns, metadata_stat.st_mtime_ns + 1_000_000),
    )

    second_view = TraceDataset.open(dataset.root).select_model_sites(query)
    second = second_view.materialize(cache=True)

    assert second_view.cache_key() == key
    assert first.cache_built
    assert second.cache_built
    assert len(list((dataset.cache_dir() / "features").glob(key))) == 1


def test_feature_selector_fails_before_caching_when_no_model_sites_match(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "dataset", num_episodes=1, timesteps=4)
    query = ActivationQuery(module="model.site.that.does.not.exist")

    with pytest.raises(ValueError, match="matched no model sites"):
        dataset.select_model_sites(query).materialize(cache=True)

    assert not (dataset.root / ".vla_cache" / "features").exists()


def test_feature_selector_rejects_an_existing_empty_cache(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "dataset", num_episodes=1, timesteps=4)
    query = ActivationQuery(
        module="backbone.layers.*.resid",
        layers=[0],
        token_kind="image_patch",
    )
    view = dataset.select_model_sites(query)
    sites = view._matching_model_sites()
    manager = CacheManager(dataset.cache_dir())
    key = view.cache_key()

    def build_empty(path: Path) -> CacheBuildMetadata:
        zarr.open_array(
            str(path / "X.zarr"),
            mode="w",
            shape=(0, 0),
            chunks=(1, 1),
            dtype="float32",
        )
        pd.DataFrame().to_parquet(path / "rows.parquet", index=False)
        return CacheBuildMetadata(shape=(0, 0), dtype="float32", row_count=0)

    manager.get_or_build(
        namespace="features",
        key=key,
        recipe=view._cache_recipe(sites),
        source_fingerprint=view._source_fingerprint(sites),
        builder=build_empty,
    )
    for bundle in dataset.bundles:
        bundle.__dict__["tokens"] = bundle.tokens.assign(token_kind="not_image_patch")

    with pytest.raises(ValueError, match="produced no feature rows"):
        view.materialize(cache=True)

    empty = zarr.open_array(str(manager.entry_path("features", key) / "X.zarr"), mode="r")
    assert empty.shape == (0, 0)


def test_cache_status_does_not_open_or_scan_dataset(tmp_path, monkeypatch, capsys):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    monkeypatch.setattr(
        "vla_lens.cache_cli.TraceDataset.open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dataset opened")),
    )

    assert cache_main(["--root", str(dataset_root), "--json", "status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"entries": [], "entry_count": 0, "pinned_count": 0, "size_bytes": 0}


def test_concurrent_dataset_artifact_saves_keep_every_index_row(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "dataset", num_episodes=1, timesteps=4)
    context = multiprocessing.get_context("fork")
    start = context.Event()
    processes = [
        context.Process(
            target=_concurrent_artifact_worker,
            args=(str(dataset.root), index, start),
        )
        for index in range(4)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    reopened = TraceDataset.open(dataset.root)
    dataset_rows = reopened.dataset_artifact_index
    assert {
        "parallel-artifact-0",
        "parallel-artifact-1",
        "parallel-artifact-2",
        "parallel-artifact-3",
    }.issubset(set(dataset_rows["artifact_id"].astype(str)))


def test_failed_artifact_index_write_keeps_previous_valid_file(tmp_path, monkeypatch):
    dataset = create_synthetic_trace_dataset(tmp_path / "dataset", num_episodes=1, timesteps=4)
    dataset.save_artifact(
        LensArtifact(
            artifact_id="first-artifact",
            artifact_type="test",
            name="First",
            scope="dataset",
        )
    )
    original_index = dataset._dataset_artifact_root() / "tables" / "artifact_index.parquet"
    original_bytes = original_index.read_bytes()

    def fail_write(path: Path, _frame: object) -> None:
        path.write_bytes(b"incomplete parquet")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr("vla_lens.traces.dataset._write_table", fail_write)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        dataset.save_artifact(
            LensArtifact(
                artifact_id="second-artifact",
                artifact_type="test",
                name="Second",
                scope="dataset",
            )
        )

    assert original_index.read_bytes() == original_bytes
    reopened = TraceDataset.open(dataset.root)
    assert set(reopened.dataset_artifact_index["artifact_id"].astype(str)) == {"first-artifact"}
