from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("artifacts/pi05_analysis/target_binding_controls")
OUT = ROOT / "stage2_failure_flow_synthesis"
FLOW = ROOT / "flow_binding_refined" / "flow_binding_refined_calls.csv"
ROLLOUTS = ROOT / "rollouts.csv"
CALLS = ROOT / "calls.csv"
FAILURE = ROOT / "failure_case_selection"


def rel(value: object) -> str:
    if pd.isna(value) or str(value) == "":
        return "none"
    return str(value)


def top_join(values: pd.Series, n: int = 3) -> str:
    counts = values.dropna().astype(str).value_counts().head(n)
    return ";".join(f"{idx}:{count}" for idx, count in counts.items())


def score_close_to_zero_positive(
    median_margin: float, mean_margin: float, beat_rate: float
) -> float:
    # Highest near zero; large positive margins are less diagnostic.
    if pd.isna(median_margin):
        return -999.0
    sign_bonus = 1.0 if median_margin >= 0 else 0.25
    return (
        sign_bonus - abs(float(median_margin)) + 0.25 * float(mean_margin) + 0.1 * float(beat_rate)
    )


def relation_frame(rollouts: pd.DataFrame) -> pd.DataFrame:
    frame = rollouts.copy()
    frame["moved_target_relation"] = frame["first_moved_is_target"].map(
        {True: "target", False: "distractor"}
    )
    frame.loc[
        frame["first_moved_object"].isna() | (frame["first_moved_object"].astype(str) == ""),
        "moved_target_relation",
    ] = "none"
    frame["lifted_target_relation"] = frame["first_lifted_is_target"].map(
        {True: "target", False: "distractor"}
    )
    frame.loc[
        frame["first_lifted_object"].isna() | (frame["first_lifted_object"].astype(str) == ""),
        "lifted_target_relation",
    ] = "none"
    return frame


def aggregate_flow(flow: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    grouped = flow.groupby(keys, dropna=False)
    out = grouped.agg(
        rows=("rollout_id", "size"),
        rollouts=("rollout_id", "nunique"),
        mean_first_margin=("first_target_vs_semantic_margin", "mean"),
        median_first_margin=("first_target_vs_semantic_margin", "median"),
        first_target_beats_semantic_rate=(
            "first_target_vs_semantic_margin",
            lambda s: float((s > 0).mean()),
        ),
        mean_prefix_margin=("prefix_target_vs_semantic_margin", "mean"),
        median_prefix_margin=("prefix_target_vs_semantic_margin", "median"),
        prefix_target_beats_semantic_rate=(
            "prefix_target_vs_semantic_margin",
            lambda s: float((s > 0).mean()),
        ),
        semantic_distractor_best_rate=(
            "best_first_overall_category",
            lambda s: float((s == "semantic_distractor").mean()),
        ),
        receptacle_best_rate=(
            "best_first_overall_category",
            lambda s: float((s == "receptacle_destination_support").mean()),
        ),
        top_best_semantic_objects=("best_first_semantic_object", top_join),
    ).reset_index()
    out.insert(0, "analysis_group", name)
    out["zero_positive_score"] = out.apply(
        lambda row: score_close_to_zero_positive(
            row["median_first_margin"],
            row["mean_first_margin"],
            row["first_target_beats_semantic_rate"],
        ),
        axis=1,
    )
    out["distractor_dominated_score"] = (
        -out["median_first_margin"]
        + out["semantic_distractor_best_rate"]
        + (1.0 - out["first_target_beats_semantic_rate"])
    )
    return out.sort_values(["zero_positive_score", "rows"], ascending=[False, False])


def write_readme(
    failure_flow: pd.DataFrame,
    relation_flow: pd.DataFrame,
    handoff: pd.DataFrame,
    many_seed: pd.DataFrame,
    many_seed_tasks: pd.DataFrame,
) -> None:
    near = (
        failure_flow[failure_flow["failure_type"] != "success"]
        .sort_values("zero_positive_score", ascending=False)
        .head(4)
    )
    distractor = (
        failure_flow[failure_flow["failure_type"] != "success"]
        .sort_values("distractor_dominated_score", ascending=False)
        .head(4)
    )
    rels = (
        relation_flow[relation_flow["failure_type"] != "success"]
        .sort_values("distractor_dominated_score", ascending=False)
        .head(5)
    )
    handoff_top = handoff.head(8)
    seed_top = many_seed.head(8)
    seed_task_top = many_seed_tasks.head(8)

    def bullets(df: pd.DataFrame, cols: list[str]) -> list[str]:
        lines: list[str] = []
        for _, row in df.iterrows():
            parts = [f"{col}={row[col]}" for col in cols]
            lines.append("- " + ", ".join(parts))
        return lines

    content = [
        "# Stage 2 Failure/Flow Synthesis",
        "",
        "Inputs: `flow_binding_refined/flow_binding_refined_calls.csv`, "
        "`failure_case_selection/*`, `rollouts.csv`, `calls.csv`. "
        "No tensor files were loaded.",
        "",
        "## Key Findings",
        "",
        "Near-zero or positive target-vs-semantic margins among failures are "
        "the best candidates for causal handoff, because the refined flow "
        "signal is not simply dominated by a semantic distractor.",
        *bullets(
            near,
            [
                "failure_type",
                "rows",
                "rollouts",
                "median_first_margin",
                "first_target_beats_semantic_rate",
                "zero_positive_score",
            ],
        ),
        "",
        "Strong distractor-dominated failures have negative margins and high "
        "semantic-distractor best-object rates; use these for many-seed "
        "robustness rather than first causal handoff.",
        *bullets(
            distractor,
            [
                "failure_type",
                "rows",
                "rollouts",
                "median_first_margin",
                "semantic_distractor_best_rate",
                "distractor_dominated_score",
            ],
        ),
        "",
        "First-moved/first-lifted confusions align with refined flow margins: "
        "distractor moved/lifted groups are more distractor dominated, while "
        "target-moved groups separate approach/close failures from "
        "object-binding failures.",
        *bullets(
            rels,
            [
                "failure_type",
                "moved_target_relation",
                "lifted_target_relation",
                "rollouts",
                "median_first_margin",
                "semantic_distractor_best_rate",
            ],
        ),
        "",
        "## Top Causal Handoff Candidates",
        "",
        *bullets(
            handoff_top,
            [
                "recommendation_rank",
                "rollout_id",
                "failure_type",
                "scene_family",
                "task_id",
                "layout_id",
                "target_guess",
                "median_first_margin",
                "causal_handoff_score",
            ],
        ),
        "",
        "## Top Many-Seed Candidates",
        "",
        "Task-level groups are preferred for many-seed runs because current "
        "layout-level rows are mostly one seed per layout.",
        *bullets(
            seed_task_top,
            [
                "recommendation_rank",
                "scene_family",
                "task_id",
                "target_guess",
                "failure_type",
                "rollouts",
                "failure_layouts",
                "many_seed_task_score",
            ],
        ),
        "",
        "Layout-level rows below provide exact task/layout anchors for those runs.",
        *bullets(
            seed_top,
            [
                "recommendation_rank",
                "scene_family",
                "task_id",
                "layout_id",
                "target_guess",
                "episodes",
                "success_rate",
                "many_seed_score",
            ],
        ),
        "",
        "## Output Files",
        "",
        "- `failure_type_flow_margins.csv`: failure type ranking by refined "
        "target-vs-semantic margins.",
        "- `relation_flow_margins.csv`: first_moved/first_lifted relation "
        "groups joined to flow margins.",
        "- `task_layout_flow_candidates.csv`: ranked task/layout groups for follow-up.",
        "- `causal_handoff_candidates_ranked.csv`: rollout-level candidates "
        "with flow margins and call metadata.",
        "- `many_seed_candidates_ranked.csv`: task/layout candidates for many-seed runs.",
        "- `many_seed_task_candidates_ranked.csv`: task-level many-seed "
        "candidates aggregated across layouts.",
        "- `recommendations.csv`: compact combined ranked recommendation list.",
    ]
    (OUT / "README.md").write_text("\n".join(content) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    flow = pd.read_csv(FLOW)
    rollouts = relation_frame(pd.read_csv(ROLLOUTS))
    calls = pd.read_csv(CALLS)

    flow = flow.merge(
        rollouts[
            [
                "rollout_id",
                "moved_target_relation",
                "lifted_target_relation",
                "steps",
                "num_policy_calls",
            ]
        ],
        on="rollout_id",
        how="left",
    )

    failure_flow = aggregate_flow(flow, ["failure_type"], "failure_type")
    failure_flow.to_csv(OUT / "failure_type_flow_margins.csv", index=False)

    relation_flow = aggregate_flow(
        flow,
        ["failure_type", "moved_target_relation", "lifted_target_relation"],
        "failure_moved_lifted_relation",
    )
    relation_flow.to_csv(OUT / "relation_flow_margins.csv", index=False)

    task_layout = aggregate_flow(
        flow,
        ["scene_family", "task_id", "task_name", "layout_id", "target_guess", "failure_type"],
        "task_layout_failure",
    )
    rollout_stats = (
        rollouts.groupby(
            ["scene_family", "task_id", "task_name", "layout_id", "target_guess", "failure_type"],
            dropna=False,
        )
        .agg(
            episodes=("rollout_id", "size"),
            successes=("success", "sum"),
            success_rate=("success", "mean"),
            first_moved_objects=("first_moved_object", top_join),
            first_lifted_objects=("first_lifted_object", top_join),
        )
        .reset_index()
    )
    task_layout = task_layout.merge(
        rollout_stats,
        on=["scene_family", "task_id", "task_name", "layout_id", "target_guess", "failure_type"],
        how="left",
    )
    task_layout["candidate_score"] = (
        task_layout["zero_positive_score"]
        + task_layout["distractor_dominated_score"].clip(lower=0) * 0.25
        + (1.0 - task_layout["success_rate"].fillna(0.0))
    )
    task_layout = task_layout.sort_values(["candidate_score", "rollouts"], ascending=[False, False])
    task_layout.to_csv(OUT / "task_layout_flow_candidates.csv", index=False)

    rollout_flow = (
        flow.groupby("rollout_id", dropna=False)
        .agg(
            flow_rows=("rollout_id", "size"),
            median_first_margin=("first_target_vs_semantic_margin", "median"),
            mean_first_margin=("first_target_vs_semantic_margin", "mean"),
            min_first_margin=("first_target_vs_semantic_margin", "min"),
            first_target_beats_semantic_rate=(
                "first_target_vs_semantic_margin",
                lambda s: float((s > 0).mean()),
            ),
            semantic_distractor_best_rate=(
                "best_first_overall_category",
                lambda s: float((s == "semantic_distractor").mean()),
            ),
            phases=("phase", lambda s: ";".join(pd.Series(s).dropna().astype(str).unique())),
            best_semantic_objects=("best_first_semantic_object", top_join),
        )
        .reset_index()
    )
    call_counts = (
        calls.groupby("rollout_id", dropna=False)
        .agg(call_rows=("call_index", "size"))
        .reset_index()
    )
    rollout_join = rollouts.merge(rollout_flow, on="rollout_id", how="left").merge(
        call_counts, on="rollout_id", how="left"
    )
    failures = rollout_join[rollout_join["failure_type"] != "success"].copy()
    failures["causal_handoff_score"] = (
        failures["median_first_margin"]
        .fillna(-10)
        .map(lambda x: 1.0 - abs(x) if x >= -0.25 else 0.25 - abs(x))
        + failures["first_target_beats_semantic_rate"].fillna(0.0)
        + (failures["moved_target_relation"] == "distractor").astype(float) * 0.35
        + (failures["lifted_target_relation"] == "target").astype(float) * 0.35
        + (failures["failure_type"] == "wrong_object_moved").astype(float) * 0.5
    )
    handoff_cols = [
        "rollout_id",
        "benchmark",
        "scene_family",
        "task_id",
        "task_name",
        "layout_id",
        "seed",
        "target_guess",
        "target_object",
        "failure_type",
        "success",
        "steps",
        "first_moved_object",
        "first_lifted_object",
        "moved_target_relation",
        "lifted_target_relation",
        "first_close_step",
        "first_target_lift_step",
        "initial_target_distance",
        "min_target_distance",
        "final_target_distance",
        "target_max_lift",
        "call_rows",
        "phases",
        "median_first_margin",
        "mean_first_margin",
        "first_target_beats_semantic_rate",
        "semantic_distractor_best_rate",
        "best_semantic_objects",
        "causal_handoff_score",
    ]
    handoff = (
        failures.sort_values("causal_handoff_score", ascending=False)[handoff_cols].head(100).copy()
    )
    handoff.insert(0, "recommendation_rank", range(1, len(handoff) + 1))
    handoff.to_csv(OUT / "causal_handoff_candidates_ranked.csv", index=False)

    many_seed = task_layout[task_layout["failure_type"] != "success"].copy()
    many_seed["many_seed_score"] = (
        many_seed["distractor_dominated_score"].clip(lower=0)
        + (1.0 - many_seed["success_rate"].fillna(0.0))
        + many_seed["rollouts"].clip(upper=5) * 0.05
    )
    many_seed = (
        many_seed.sort_values(["many_seed_score", "rollouts"], ascending=[False, False])
        .head(100)
        .copy()
    )
    many_seed.insert(0, "recommendation_rank", range(1, len(many_seed) + 1))
    many_seed.to_csv(OUT / "many_seed_candidates_ranked.csv", index=False)

    many_seed_tasks = (
        flow[flow["failure_type"] != "success"]
        .groupby(
            ["scene_family", "task_id", "task_name", "target_guess", "failure_type"], dropna=False
        )
        .agg(
            rows=("rollout_id", "size"),
            rollouts=("rollout_id", "nunique"),
            failure_layouts=("layout_id", "nunique"),
            mean_first_margin=("first_target_vs_semantic_margin", "mean"),
            median_first_margin=("first_target_vs_semantic_margin", "median"),
            first_target_beats_semantic_rate=(
                "first_target_vs_semantic_margin",
                lambda s: float((s > 0).mean()),
            ),
            semantic_distractor_best_rate=(
                "best_first_overall_category",
                lambda s: float((s == "semantic_distractor").mean()),
            ),
            top_best_semantic_objects=("best_first_semantic_object", top_join),
        )
        .reset_index()
    )
    task_rollout_stats = (
        rollouts[rollouts["failure_type"] != "success"]
        .groupby(
            ["scene_family", "task_id", "task_name", "target_guess", "failure_type"], dropna=False
        )
        .agg(
            failure_episodes=("rollout_id", "size"),
            first_moved_objects=("first_moved_object", top_join),
            first_lifted_objects=("first_lifted_object", top_join),
            layouts=(
                "layout_id",
                lambda s: ";".join(
                    map(str, sorted(pd.Series(s).dropna().astype(int).unique())[:12])
                ),
            ),
        )
        .reset_index()
    )
    many_seed_tasks = many_seed_tasks.merge(
        task_rollout_stats,
        on=["scene_family", "task_id", "task_name", "target_guess", "failure_type"],
        how="left",
    )
    many_seed_tasks["many_seed_task_score"] = (
        many_seed_tasks["failure_layouts"].clip(upper=20) * 0.15
        + many_seed_tasks["rollouts"].clip(upper=20) * 0.05
        - many_seed_tasks["median_first_margin"].clip(upper=0).fillna(0.0)
        + many_seed_tasks["semantic_distractor_best_rate"].fillna(0.0)
        + (1.0 - many_seed_tasks["first_target_beats_semantic_rate"].fillna(0.0))
    )
    many_seed_tasks = (
        many_seed_tasks.sort_values(
            ["many_seed_task_score", "failure_layouts", "rollouts"], ascending=[False, False, False]
        )
        .head(100)
        .copy()
    )
    many_seed_tasks.insert(0, "recommendation_rank", range(1, len(many_seed_tasks) + 1))
    many_seed_tasks.to_csv(OUT / "many_seed_task_candidates_ranked.csv", index=False)

    recommendations = pd.concat(
        [
            handoff.head(25).assign(recommendation_type="causal_handoff"),
            many_seed.head(25).assign(recommendation_type="many_seed"),
            many_seed_tasks.head(25).assign(recommendation_type="many_seed_task"),
        ],
        ignore_index=True,
        sort=False,
    )
    recommendations.to_csv(OUT / "recommendations.csv", index=False)

    # Preserve the prior handoff ranking for traceability when present.
    prior = FAILURE / "handoff_causal_rollout_candidates.csv"
    if prior.exists():
        pd.read_csv(prior).head(100).to_csv(
            OUT / "prior_failure_selection_handoff_top100.csv", index=False
        )

    write_readme(failure_flow, relation_flow, handoff, many_seed, many_seed_tasks)


if __name__ == "__main__":
    main()
