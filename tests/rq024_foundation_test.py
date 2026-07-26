from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path

import pytest

from vla_lens.rq024_foundation import (
    PARSER_VERSION,
    SEED_DOMAINS,
    build_bundle_payloads,
    family_rank_digest,
    parse_family,
    seed_identity,
    stratum_digest,
    validate_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "configs/campaigns/rq024/foundation-r1"


@pytest.fixture(scope="module")
def source_catalog():
    return json.loads((BUNDLE_ROOT / "source_catalog.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def candidates():
    return json.loads((BUNDLE_ROOT / "candidates.json").read_text(encoding="utf-8"))


def test_checked_in_source_catalog_is_complete_and_bddl_authoritative(source_catalog):
    assert source_catalog["task_count"] == 90
    assert [task["task_id"] for task in source_catalog["tasks"]] == list(range(90))
    assert all(task["bddl_sha256"].startswith("sha256:") for task in source_catalog["tasks"])

    back_butter = source_catalog["tasks"][3]
    assert back_butter["goal_atoms"] == [
        {"predicate": "In", "arguments": ["butter_2", "wooden_cabinet_1_top_region"]},
        {"predicate": "Close", "arguments": ["wooden_cabinet_1_top_region"]},
    ]


def test_parser_preserves_qualifiers_and_rejects_every_compound_goal(source_catalog):
    back_bowl, reasons = parse_family(source_catalog["tasks"][12])
    compound, compound_reasons = parse_family(source_catalog["tasks"][3])

    assert reasons == []
    assert back_bowl is not None
    assert back_bowl["parser_version"] == PARSER_VERSION
    assert back_bowl["normalized_instructed_object"] == "black_bowl[initial=back]"
    assert back_bowl["normalized_destination"] == "plate"
    assert compound is None
    assert compound_reasons == ["goal_atom_count_is_not_one"]


def test_ranking_and_interleave_use_the_locked_byte_rules(candidates):
    rows = candidates["families"]
    assert all(row["rank_digest"] == family_rank_digest(row["canonical_family_id"]) for row in rows)
    assert all(row["stratum_digest"] == stratum_digest(row["stratum_id"]) for row in rows)

    first_round = rows[: len({row["stratum_id"] for row in rows})]
    assert [row["stratum_digest"] for row in first_round] == sorted(
        row["stratum_digest"] for row in first_round
    )


def test_bundle_partitions_every_task_and_has_fixed_assignment(source_catalog, candidates):
    rejections = json.loads((BUNDLE_ROOT / "rejections.json").read_text(encoding="utf-8"))
    accepted_ids = {row["task_id"] for row in candidates["families"]}
    rejected_ids = {row["task_id"] for row in rejections["rejections"]}
    selected = [row for row in candidates["families"] if row["selected"]]

    assert accepted_ids.isdisjoint(rejected_ids)
    assert accepted_ids | rejected_ids == set(range(90))
    assert all(row["acceptance_reason"] for row in candidates["families"])
    assert all(row["rejection_reasons"] for row in rejections["rejections"])
    assert len(selected) == 24
    assert [row["pool"] for row in selected] == [
        "discovery" if position % 2 == 0 else "confirmation" for position in range(24)
    ]
    assert Counter(row["pool"] for row in selected) == {"discovery": 12, "confirmation": 12}


def test_trial_table_has_three_rows_per_family_and_five_independent_seed_identities():
    with (BUNDLE_ROOT / "trials.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 72
    assert set(Counter(row["canonical_family_id"] for row in rows).values()) == {3}
    identities = []
    for row in rows:
        for domain in SEED_DOMAINS:
            identity, seed = seed_identity(
                row["canonical_family_id"], row["cell_id"], int(row["replicate_id"]), domain
            )
            assert row[f"{domain}_seed_identity"] == identity
            assert int(row[f"{domain}_seed"]) == seed
            identities.append(identity)
    assert len(identities) == len(set(identities))


def test_workspace_relative_destinations_resolve_the_named_secondary_object(source_catalog):
    relative_plate, _ = parse_family(source_catalog["tasks"][69])
    relative_mug, _ = parse_family(source_catalog["tasks"][34])
    relative_caddy, _ = parse_family(source_catalog["tasks"][84])

    assert relative_plate is not None
    assert relative_plate["normalized_destination"] == "plate[region=left]"
    assert relative_mug is not None
    assert relative_mug["normalized_destination"] == "white_mug[region=front]"
    assert relative_caddy is not None
    assert relative_caddy["normalized_destination"] == "caddy[region=right]"


def test_checked_in_bundle_is_byte_reproducible_and_valid(source_catalog):
    expected = build_bundle_payloads(source_catalog)
    for name, content in expected.items():
        assert (BUNDLE_ROOT / name).read_bytes() == content

    result = validate_bundle(BUNDLE_ROOT)
    assert result["valid"], result
    assert result["source_tasks"] == 90
    assert result["trial_rows"] == 72


def test_csv_is_lf_terminated_and_normal_module_has_no_capture_imports(source_catalog):
    csv_bytes = build_bundle_payloads(source_catalog)["trials.csv"]
    assert b"\r\n" not in csv_bytes
    assert len(list(csv.DictReader(io.StringIO(csv_bytes.decode())))) == 72

    source = (REPO_ROOT / "src/vla_lens/rq024_foundation.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import lerobot" not in source
    assert "import libero" not in source
