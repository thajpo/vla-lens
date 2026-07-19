from __future__ import annotations

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
