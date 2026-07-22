from __future__ import annotations

import os

import numpy as np

import vla_lens.selectors as selectors


class _TrackingOrthogonalIndex:
    def __init__(self, owner: "_TrackingArray") -> None:
        self.owner = owner

    def __getitem__(self, selection):
        sample_indices = np.asarray(selection[0])
        self.owner.batch_sizes.append(len(sample_indices))
        return self.owner.values[selection]


class _TrackingArray:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.batch_sizes: list[int] = []
        self.oindex = _TrackingOrthogonalIndex(self)


def test_vectorized_mean_reads_raw_tokens_in_bounded_batches(monkeypatch):
    values = np.arange(10 * 5 * 3, dtype=np.float32).reshape(10, 5, 3)
    array = _TrackingArray(values)
    monkeypatch.setattr(selectors, "VECTORIZED_READ_TARGET_BYTES", 120)

    vectors = selectors._vectorized_mean_samples(
        array=array,
        axes=["policy_call", "token", "channel"],
        samples=[("policy_call", index) for index in range(10)],
        token_indices=None,
        generation_step=None,
        reduction="mean",
    )

    assert array.batch_sizes == [2, 2, 2, 2, 2]
    np.testing.assert_allclose(np.stack(vectors), values.mean(axis=1))


def test_directory_signature_uses_only_root_zarr_metadata(tmp_path, monkeypatch):
    array_path = tmp_path / "activation.zarr"
    array_path.mkdir()
    (array_path / ".zarray").write_text('{"shape": [1, 2]}', encoding="utf-8")
    (array_path / "0.0").write_bytes(b"tensor chunk")

    def reject_recursive_walk(*args, **kwargs):
        raise AssertionError("cache checks must not walk tensor chunks")

    monkeypatch.setattr(type(array_path), "rglob", reject_recursive_walk)
    signature = selectors._path_signature(array_path)

    assert signature["kind"] == "dir"
    assert ".zarray_size" in signature
    assert "files" not in signature


def test_directory_signature_changes_when_zarr_is_rewritten(tmp_path):
    array_path = tmp_path / "activation.zarr"
    array_path.mkdir()
    metadata = array_path / ".zarray"
    metadata.write_text('{"shape": [1, 2]}', encoding="utf-8")
    first = selectors._path_signature(array_path)

    metadata.write_text('{"shape": [2, 2]}', encoding="utf-8")
    next_mtime = int(first[".zarray_mtime_ns"]) + 1
    os.utime(metadata, ns=(next_mtime, next_mtime))
    second = selectors._path_signature(array_path)

    assert second != first
