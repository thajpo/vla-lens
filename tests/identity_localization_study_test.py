from __future__ import annotations

import numpy as np
import pandas as pd

from vla_lens.probes.identity_localization_study import (
    _grouped_bootstrap,
    _patch_metrics,
    _summary_table,
    linear_token_contributions,
)
from vla_lens.probes.token_representations import ProjectionState


def test_linear_token_contributions_reconstruct_scores_exactly():
    tokens = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 1.0], [0.0, 3.0]],
        ]
    )
    projection = ProjectionState(
        input_center=np.array([0.5, 0.5, 0.5, 0.5]),
        input_scale=np.array([1.0, 2.0, 1.0, 2.0]),
        pca_center=np.array([0.1, -0.2, 0.3, 0.0]),
        components=np.array(
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, -0.5],
                [0.2, 0.1, 0.0, 0.7],
            ]
        ),
        explained_variance_ratio=np.ones(3),
    )
    coefficients = np.array([[2.0, -1.0], [0.5, 3.0]])
    intercepts = np.array([0.25, -0.75])

    contributions, scores, static = linear_token_contributions(
        tokens, projection, coefficients, intercepts
    )

    flat = tokens.reshape(len(tokens), -1)
    readout = projection.transform(flat)[:, :2]
    expected = readout @ coefficients.T + intercepts
    np.testing.assert_allclose(scores, expected, atol=1e-6)
    np.testing.assert_allclose(
        contributions.sum(axis=2) + intercepts[None, :], expected, atol=1e-6
    )
    assert contributions.shape == (2, 2, 2)
    assert static.shape == (2, 2)


def test_patch_metrics_reports_lift_over_exact_random_ranking():
    metrics = _patch_metrics(
        np.array([0.9, 0.8, 0.2, 0.1]),
        np.array([True, True, False, False]),
        np.array([False, False, True, False]),
    )

    assert metrics["average_precision"] == 1.0
    assert metrics["average_precision_minus_random"] > 0.0
    assert metrics["target_absolute_mass_fraction"] > 0.5
    assert metrics["wrong_object_average_precision"] < 1.0


def test_grouped_bootstrap_weights_tasks_instead_of_object_rows():
    values = np.array([1.0] * 9 + [-1.0])
    groups = np.array(["repeated"] * 9 + ["single"])

    summary = _grouped_bootstrap(
        values, groups, bootstrap_samples=100, seed=0
    )

    assert summary["group_count"] == 2
    assert np.isclose(summary["mean"], 0.0)


def test_grouped_bootstrap_drops_nonfinite_values():
    summary = _grouped_bootstrap(
        np.array([0.4, np.nan]),
        np.array(["kept", "dropped"]),
        bootstrap_samples=10,
        seed=0,
    )

    assert summary["group_count"] == 1
    assert np.isclose(summary["mean"], 0.4)


def test_localization_summary_separates_probe_positives_and_misses():
    metrics = pd.DataFrame(
        {
            "method": ["positive_contribution", "positive_contribution"],
            "probe_supported": [True, True],
            "probe_predicted_present": [True, False],
            "average_precision": [0.8, 0.3],
            "average_precision_minus_random": [0.6, 0.1],
            "static_average_precision": [0.5, 0.2],
            "target_minus_wrong_object": [0.2, -0.1],
            "trace_id": ["predicted", "missed"],
            "task_key": ["suite:1", "suite:2"],
            "instruction_key": ["one", "two"],
        }
    )

    summary = _summary_table(metrics, bootstrap_samples=10)

    assert set(summary["cohort"]) == {
        "all_visible",
        "probe_predicted_present",
        "probe_missed_present",
    }
    primary = summary.loc[
        (summary["cohort"] == "probe_predicted_present")
        & (summary["metric"] == "average_precision_minus_static")
        & (summary["unit"] == "episode")
    ].iloc[0]
    assert primary["group_count"] == 1
    assert np.isclose(primary["mean"], 0.3)


def test_unsupported_identity_never_enters_primary_cohort():
    metrics = pd.DataFrame(
        {
            "method": ["positive_contribution"],
            "probe_supported": [False],
            "probe_predicted_present": [False],
            "average_precision": [0.8],
            "average_precision_minus_random": [0.6],
            "static_average_precision": [0.5],
            "target_minus_wrong_object": [0.2],
            "trace_id": ["unsupported"],
            "task_key": ["suite:1"],
            "instruction_key": ["one"],
        }
    )

    summary = _summary_table(metrics, bootstrap_samples=10)

    assert "probe_predicted_present" not in set(summary["cohort"])
    assert "probe_unsupported" in set(summary["cohort"])
