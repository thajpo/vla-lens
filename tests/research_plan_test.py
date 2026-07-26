from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from vla_lens.research_plan import (
    check_research_plan,
    check_research_plan_file,
    format_research_plan_markdown,
    load_research_plan,
    research_plan_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "configs/campaigns/rq024_controlled_scene_to_behavior.yaml"


@pytest.fixture
def plan() -> dict:
    return dict(load_research_plan(PLAN_PATH))


def test_current_research_program_schema_is_valid_and_summary_is_honest(plan):
    check = check_research_plan_file(PLAN_PATH)

    assert check.valid, check.to_dict()
    assert check.summary["study_count"] == 16
    assert check.summary["execution_readiness_evaluated"] is False
    assert check.fingerprint == research_plan_fingerprint(plan)

    rendered = format_research_plan_markdown(plan, check)
    assert "Does PI0.5 combine the scene" in rendered
    assert "H1-relational-control" in rendered
    assert "Plan schema: `VALID`" in rendered
    assert "Execution readiness: `NOT EVALUATED`" in rendered


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda value: value["studies"][0].update({"kind": "confirmaton"}),
            "invalid_study_kind",
        ),
        (
            lambda value: value["studies"][0]["budget"].update(
                {"max_additional_persistent_gb": -1}
            ),
            "invalid_study_budget",
        ),
        (
            lambda value: value["population"]["candidate_pool"].update({"max_families": 12}),
            "candidate_pool_too_small",
        ),
        (
            lambda value: value.update({"research_question_ids": "RQ-024"}),
            "invalid_research_plan_list",
        ),
        (
            lambda value: value["authority"].update({"automatic": "everything"}),
            "invalid_research_plan_list",
        ),
    ],
)
def test_research_program_rejects_adversarial_shapes(plan, mutate, expected_code):
    malformed = deepcopy(plan)
    mutate(malformed)

    check = check_research_plan(malformed)

    assert not check.valid
    assert expected_code in _codes(check)


def test_confirmation_requires_exact_lock_fields(plan):
    malformed = deepcopy(plan)
    confirmation = next(
        study for study in malformed["studies"] if study["id"] == "GEOMETRY-CONFIRMATION"
    )
    confirmation["child_contract"]["additional_lock_fields"] = ["banana"]

    check = check_research_plan(malformed)

    assert not check.valid
    assert "confirmation_lock_incomplete" in _codes(check)


def test_research_program_rejects_unknown_entry_dependencies_and_cycles(plan):
    unknown = deepcopy(plan)
    unknown["studies"][1]["entry_conditions"]["requires_all_completed"] = ["missing-study"]
    cyclic = deepcopy(plan)
    cyclic["studies"][0]["entry_conditions"]["requires_all_completed"] = ["GEOMETRY-DISCOVERY"]

    unknown_check = check_research_plan(unknown)
    cyclic_check = check_research_plan(cyclic)

    assert "unknown_study_dependency" in _codes(unknown_check)
    assert "study_dependency_cycle" in _codes(cyclic_check)


def test_research_program_rejects_unknown_outcome_action(plan):
    malformed = deepcopy(plan)
    malformed["studies"][0]["outcome_actions"]["positive"] = "imaginary-study"

    check = check_research_plan(malformed)

    assert not check.valid
    assert "invalid_outcome_action" in _codes(check)


def test_research_program_rejects_confirmation_selection_and_unbounded_calls(plan):
    selected = deepcopy(plan)
    selected["studies"][2]["data_scope"]["selection_allowed"] = True
    unbounded = deepcopy(plan)
    del unbounded["studies"][6]["budget"]["max_model_calls"]

    selected_check = check_research_plan(selected)
    unbounded_check = check_research_plan(unbounded)

    assert "confirmation_selection_allowed" in _codes(selected_check)
    assert "study_missing_fields" in _codes(unbounded_check)


def test_research_plan_loader_rejects_duplicate_yaml_keys(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 2\n", encoding="utf-8")

    check = check_research_plan_file(path)

    assert not check.valid
    assert "invalid_research_plan_yaml" in _codes(check)


def _codes(check) -> set[str]:
    return {issue.code for issue in check.errors}
