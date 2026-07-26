"""Deterministic LIBERO-90 family selection for the RQ-024 FOUNDATION study."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

PROGRAM_ID = "rq024-controlled-scene-to-behavior"
CHILD_PLAN_ID = "rq024-foundation-r1"
PARSER_VERSION = "rq024-libero90-family-v1"
EXPORTER_VERSION = "rq024-libero90-metadata-v1"
BENCHMARK = "libero_90"
UNIT_SEPARATOR = "\x1f"
SEED_DOMAINS = ("layout", "reset", "environment", "policy", "flow_noise")
SELECTED_FAMILY_COUNT = 24
REPLICATES_PER_FAMILY = 3
LIBERO_INIT_STATE_COUNT = 50

_ALIASES = {
    "akita_black_bowl": "black_bowl",
    "chefmate_8_frypan": "frying_pan",
    "desk_caddy": "caddy",
    "flat_stove": "stove",
    "new_salad_dressing": "salad_dressing",
    "porcelain_mug": "white_mug",
    "red_coffee_mug": "red_mug",
    "white_yellow_mug": "yellow_and_white_mug",
    "wooden_cabinet": "cabinet",
    "white_cabinet": "cabinet",
    "wooden_tray": "tray",
    "wooden_two_layer_shelf": "shelf",
    "black_book": "book",
    "yellow_book": "book",
}
_VERBS = {
    "Close": ("close", 1),
    "In": ("place_in", 2),
    "On": ("place_on", 2),
    "Open": ("open", 1),
    "Turnoff": ("turn_off", 1),
    "Turnon": ("turn_on", 1),
}
_POSITION_QUALIFIERS = {"back", "front", "left", "middle", "right"}
_TOKEN = re.compile(r"\(|\)|[^\s()]+")


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def parse_bddl_metadata(text: str) -> dict[str, Any]:
    """Extract only static task metadata; this does not import LIBERO or its runtime."""
    expressions = _parse_sexpressions(text)
    if len(expressions) != 1 or not isinstance(expressions[0], list):
        raise ValueError("BDDL must contain exactly one top-level expression")
    root = expressions[0]
    sections = {
        section[0]: section
        for section in root
        if isinstance(section, list) and section and isinstance(section[0], str)
    }
    required = {":language", ":fixtures", ":objects", ":obj_of_interest", ":init", ":goal"}
    missing = sorted(required - sections.keys())
    if missing:
        raise ValueError(f"BDDL is missing sections: {', '.join(missing)}")
    entities = {
        **_typed_entities(sections[":fixtures"]),
        **_typed_entities(sections[":objects"]),
    }
    goal = sections[":goal"]
    if len(goal) != 2 or not isinstance(goal[1], list):
        raise ValueError("BDDL :goal must contain one expression")
    goal_expression = goal[1]
    goal_atoms = goal_expression[1:] if goal_expression and goal_expression[0] == "And" else []
    return {
        "language": " ".join(str(item) for item in sections[":language"][1:]),
        "entities": dict(sorted(entities.items())),
        "objects_of_interest": [str(item) for item in sections[":obj_of_interest"][1:]],
        "init_atoms": [_atom(item) for item in sections[":init"][1:]],
        "goal_expression": _format_sexpression(goal_expression),
        "goal_operator": str(goal_expression[0]) if goal_expression else "",
        "goal_atoms": [_atom(item) for item in goal_atoms],
    }


def source_task_from_bddl(
    *, task_id: int, task_name: str, bddl_file: str, bddl_bytes: bytes
) -> dict[str, Any]:
    metadata = parse_bddl_metadata(bddl_bytes.decode("utf-8"))
    return {
        "task_id": task_id,
        "task_name": task_name,
        "bddl_file": bddl_file,
        "bddl_sha256": sha256_bytes(bddl_bytes),
        **metadata,
    }


def parse_family(task: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    atoms = task.get("goal_atoms")
    if task.get("goal_operator") != "And":
        reasons.append("goal_operator_is_not_and")
    if not isinstance(atoms, list) or len(atoms) != 1:
        reasons.append("goal_atom_count_is_not_one")
    if reasons:
        return None, reasons

    atom = atoms[0]
    predicate = atom.get("predicate") if isinstance(atom, Mapping) else None
    arguments = atom.get("arguments") if isinstance(atom, Mapping) else None
    if predicate not in _VERBS:
        return None, ["unsupported_goal_predicate"]
    verb, arity = _VERBS[predicate]
    if not isinstance(arguments, list) or len(arguments) != arity:
        return None, ["unsupported_goal_arity"]

    entities = task.get("entities")
    init_atoms = task.get("init_atoms")
    if not isinstance(entities, Mapping) or not isinstance(init_atoms, list):
        return None, ["missing_entity_or_init_metadata"]
    instructed = _normalize_reference(str(arguments[0]), entities, init_atoms)
    destination = (
        _normalize_reference(str(arguments[1]), entities, init_atoms) if arity == 2 else None
    )
    if instructed is None:
        return None, ["instructed_object_could_not_be_normalized"]
    if arity == 2 and destination is None:
        return None, ["destination_could_not_be_normalized"]

    normalized_instructed = _qualified_name(instructed)
    normalized_destination = "" if destination is None else _qualified_name(destination)
    canonical_family_id = UNIT_SEPARATOR.join(
        [str(task["task_id"]), verb, normalized_instructed, normalized_destination]
    )
    stratum_id = UNIT_SEPARATOR.join([verb, instructed["identity"]])
    row = {
        "parser_version": PARSER_VERSION,
        "task_id": int(task["task_id"]),
        "task_name": str(task["task_name"]),
        "language": str(task["language"]),
        "bddl_file": str(task["bddl_file"]),
        "bddl_sha256": str(task["bddl_sha256"]),
        "acceptance_reason": "single_supported_bddl_goal",
        "goal_atom": atom,
        "normalized_task_verb": verb,
        "instructed_object_identity": instructed["identity"],
        "instructed_object_qualifiers": instructed["qualifiers"],
        "normalized_instructed_object": normalized_instructed,
        "destination_identity": "" if destination is None else destination["identity"],
        "destination_qualifiers": [] if destination is None else destination["qualifiers"],
        "normalized_destination": normalized_destination,
        "canonical_family_id": canonical_family_id,
        "stratum_id": stratum_id,
    }
    row["rank_digest"] = family_rank_digest(canonical_family_id)
    row["stratum_digest"] = stratum_digest(stratum_id)
    return row, []


def family_rank_digest(canonical_family_id: str) -> str:
    encoded = (
        b"rq024-family-rank-v1\0"
        + PROGRAM_ID.encode("utf-8")
        + b"\0"
        + canonical_family_id.encode("utf-8")
    )
    return sha256_bytes(encoded)


def stratum_digest(stratum_id: str) -> str:
    return sha256_bytes(b"rq024-stratum-v1" + stratum_id.encode("utf-8"))


def seed_identity(
    canonical_family_id: str, cell_id: str, replicate_id: int, seed_domain: str
) -> tuple[str, int]:
    if seed_domain not in SEED_DOMAINS:
        raise ValueError(f"Unknown seed domain: {seed_domain}")
    fields = (
        PROGRAM_ID,
        CHILD_PLAN_ID,
        canonical_family_id,
        cell_id,
        str(replicate_id),
        seed_domain,
    )
    encoded = b"".join(len(value.encode()).to_bytes(4, "big") + value.encode() for value in fields)
    digest = hashlib.sha256(encoded).digest()
    return f"sha256:{digest.hex()}", int.from_bytes(digest[:4], "big", signed=False)


def parser_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "rq024.family_parser_contract",
        "parser_version": PARSER_VERSION,
        "authority": "BDDL :goal expression; task language is retained but does not override it",
        "acceptance": "exactly one supported atom under an And goal",
        "supported_goal_predicates": dict(sorted(_VERBS.items())),
        "aliases": dict(sorted(_ALIASES.items())),
        "qualifiers": {
            "format": "identity[key=value,key=value] with keys and values ASCII-sorted",
            "initial_position": sorted(_POSITION_QUALIFIERS),
            "region": "Every non-init BDDL region suffix is retained as region=<suffix>.",
        },
        "canonical_family_id": {
            "fields": [
                "base-10 unpadded LIBERO-90 task_id",
                "normalized_task_verb",
                "normalized_instructed_object",
                "normalized_destination_or_empty_string",
            ],
            "encoding": "UTF-8 fields joined by one ASCII unit separator byte 0x1f",
        },
        "stratum": {
            "fields": ["normalized_task_verb", "instructed_object_identity_without_qualifiers"],
            "id_encoding": "UTF-8 fields joined by one ASCII unit separator byte 0x1f",
            "digest_encoding": (
                "UTF-8 bytes of literal rq024-stratum-v1 immediately followed by "
                "stratum_id UTF-8 bytes"
            ),
        },
        "family_rank_digest": {
            "encoding": (
                "UTF-8 rq024-family-rank-v1, 0x00, program_id UTF-8, 0x00, "
                "canonical_family_id UTF-8"
            ),
            "algorithm": "sha256",
        },
        "interleave": (
            "Sort strata by (stratum_digest, stratum_id), sort each stratum by "
            "(rank_digest, canonical_family_id), then take one row from each nonempty "
            "stratum per round."
        ),
        "seed": {
            "domains": list(SEED_DOMAINS),
            "fields": [
                "program_id",
                "child_plan_id",
                "canonical_family_id",
                "cell_id",
                "base-10 replicate_id",
                "seed_domain",
            ],
            "encoding": (
                "Each field is UTF-8, prefixed by its byte length as unsigned 32-bit "
                "big-endian; framed fields are concatenated in listed order."
            ),
            "identity": "sha256:<64 lowercase hexadecimal digits> of the encoded bytes",
            "integer": (
                "unsigned big-endian integer represented by the first four digest bytes "
                "(range 0..2^32-1)"
            ),
            "layout_resolution": (
                "layout_id = layout_seed modulo 50, resolving one of the 50 LIBERO-90 "
                "pruned initial states before execution"
            ),
        },
    }


def build_bundle_payloads(source_catalog: Mapping[str, Any]) -> dict[str, bytes]:
    _check_source_catalog(source_catalog)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for task in source_catalog["tasks"]:
        family, reasons = parse_family(task)
        if family is not None:
            accepted.append(family)
        else:
            rejected.append(
                {
                    "parser_version": PARSER_VERSION,
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "language": task["language"],
                    "bddl_file": task["bddl_file"],
                    "bddl_sha256": task["bddl_sha256"],
                    "goal_expression": task["goal_expression"],
                    "goal_atoms": task["goal_atoms"],
                    "rejection_reasons": reasons,
                }
            )

    ordered = _interleave(accepted)
    for order, row in enumerate(ordered):
        row["interleave_order"] = order
        row["selected"] = order < SELECTED_FAMILY_COUNT
        row["candidate_position"] = order if row["selected"] else None
        row["pool"] = (
            ("discovery" if order % 2 == 0 else "confirmation") if row["selected"] else None
        )
        row["pool_position"] = order // 2 if row["selected"] else None

    selected = ordered[:SELECTED_FAMILY_COUNT]
    trials = _trial_rows(selected)
    candidates = {
        "schema_version": 1,
        "kind": "rq024.foundation_candidates",
        "program_id": PROGRAM_ID,
        "child_plan_id": CHILD_PLAN_ID,
        "parser_version": PARSER_VERSION,
        "accepted_family_count": len(ordered),
        "selected_family_count": len(selected),
        "families": ordered,
    }
    rejections = {
        "schema_version": 1,
        "kind": "rq024.foundation_rejections",
        "program_id": PROGRAM_ID,
        "parser_version": PARSER_VERSION,
        "rejected_task_count": len(rejected),
        "rejections": rejected,
    }
    assignment = {
        "schema_version": 1,
        "kind": "rq024.foundation_pool_assignment",
        "assignment_rule": "even zero-based positions discovery; odd positions confirmation",
        "discovery_count": 12,
        "confirmation_count": 12,
        "families": [
            {
                key: row[key]
                for key in (
                    "candidate_position",
                    "pool",
                    "pool_position",
                    "task_id",
                    "task_name",
                    "canonical_family_id",
                )
            }
            for row in selected
        ],
    }
    payloads = {
        "parser_contract.json": canonical_json_bytes(parser_contract()),
        "candidates.json": canonical_json_bytes(candidates),
        "rejections.json": canonical_json_bytes(rejections),
        "pool_assignment.json": canonical_json_bytes(assignment),
        "trials.csv": _csv_bytes(trials),
    }
    source_universe = [
        {key: task[key] for key in ("task_id", "task_name", "bddl_file", "bddl_sha256")}
        for task in source_catalog["tasks"]
    ]
    manifest = {
        "schema_version": 1,
        "kind": "rq024.foundation_catalog_selection_trial_bundle",
        "program_id": PROGRAM_ID,
        "child_plan_id": CHILD_PLAN_ID,
        "parser_version": PARSER_VERSION,
        "counts": {
            "source_tasks": 90,
            "accepted_families": len(ordered),
            "rejected_tasks": len(rejected),
            "selected_families": len(selected),
            "discovery_families": 12,
            "confirmation_families": 12,
            "trial_rows": len(trials),
            "trials_per_family": REPLICATES_PER_FAMILY,
        },
        "hashes": {
            "task_catalog_sha256": sha256_bytes(canonical_json_bytes(source_catalog)),
            "source_universe_sha256": sha256_bytes(canonical_json_bytes(source_universe)),
            "parser_contract_sha256": sha256_bytes(payloads["parser_contract.json"]),
            "parser_implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "candidate_table_sha256": sha256_bytes(payloads["candidates.json"]),
            "rejection_table_sha256": sha256_bytes(payloads["rejections.json"]),
            "pool_assignment_sha256": sha256_bytes(payloads["pool_assignment.json"]),
            "trial_table_sha256": sha256_bytes(payloads["trials.csv"]),
        },
        "files": ["source_catalog.json", *payloads.keys()],
    }
    payloads["bundle_manifest.json"] = canonical_json_bytes(manifest)
    return payloads


def write_bundle(bundle_root: Path) -> dict[str, Any]:
    source_path = bundle_root / "source_catalog.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    payloads = build_bundle_payloads(source)
    bundle_root.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        (bundle_root / name).write_bytes(content)
    return validate_bundle(bundle_root)


def validate_bundle(bundle_root: Path) -> dict[str, Any]:
    source_path = bundle_root / "source_catalog.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    expected = build_bundle_payloads(source)
    errors = []
    for name, content in expected.items():
        path = bundle_root / name
        if not path.is_file():
            errors.append(f"missing:{name}")
        elif path.read_bytes() != content:
            errors.append(f"content_mismatch:{name}")
    manifest = json.loads(expected["bundle_manifest.json"])
    return {
        "valid": not errors,
        "errors": errors,
        **manifest["counts"],
        "hashes": manifest["hashes"],
    }


def _parse_sexpressions(text: str) -> list[Any]:
    tokens = _TOKEN.findall(re.sub(r";[^\n]*", "", text))
    stack: list[list[Any]] = []
    roots: list[Any] = []
    for token in tokens:
        if token == "(":
            stack.append([])
        elif token == ")":
            if not stack:
                raise ValueError("Unexpected closing parenthesis")
            value = stack.pop()
            (stack[-1] if stack else roots).append(value)
        elif not stack:
            raise ValueError("Token outside an expression")
        else:
            stack[-1].append(token)
    if stack:
        raise ValueError("Unclosed parenthesis")
    return roots


def _typed_entities(section: Sequence[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    pending: list[str] = []
    index = 1
    while index < len(section):
        token = section[index]
        if token == "-" and index + 1 < len(section):
            entity_type = str(section[index + 1])
            result.update({name: entity_type for name in pending})
            pending = []
            index += 2
        else:
            if isinstance(token, list):
                raise ValueError("Entity declarations must be flat")
            pending.append(str(token))
            index += 1
    if pending:
        raise ValueError("Entity declaration is missing a type")
    return result


def _atom(value: Any) -> dict[str, Any]:
    if not isinstance(value, list) or not value:
        raise ValueError("Expected a nonempty predicate expression")
    return {"predicate": str(value[0]), "arguments": [str(item) for item in value[1:]]}


def _format_sexpression(value: Any) -> str:
    if isinstance(value, list):
        return "(" + " ".join(_format_sexpression(item) for item in value) + ")"
    return str(value)


def _normalize_reference(
    reference: str, entities: Mapping[str, Any], init_atoms: Sequence[Any]
) -> dict[str, Any] | None:
    if reference in entities:
        raw_type = str(entities[reference])
        qualifiers = _initial_qualifiers(reference, raw_type, init_atoms, entities)
        return {"identity": _ALIASES.get(raw_type, raw_type), "qualifiers": qualifiers}

    # A workspace region can name another entity type, such as plate_right_region.
    # Resolve that semantic reference before falling back to the workspace fixture.
    raw_types = sorted({str(value) for value in entities.values()}, key=len, reverse=True)
    for raw_type in raw_types:
        marker = f"_{raw_type}_"
        if marker in reference:
            suffix = reference.split(marker, 1)[1]
            return {
                "identity": _ALIASES.get(raw_type, raw_type),
                "qualifiers": [f"region={_trim_region(suffix)}"],
            }
    entity_prefixes = sorted(
        (str(entity) for entity in entities if reference.startswith(f"{entity}_")),
        key=len,
        reverse=True,
    )
    if entity_prefixes:
        entity = entity_prefixes[0]
        raw_type = str(entities[entity])
        suffix = reference[len(entity) + 1 :]
        return {
            "identity": _ALIASES.get(raw_type, raw_type),
            "qualifiers": [f"region={_trim_region(suffix)}"],
        }
    return None


def _initial_qualifiers(
    entity: str, raw_type: str, init_atoms: Sequence[Any], entities: Mapping[str, Any]
) -> list[str]:
    if sum(str(value) == raw_type for value in entities.values()) < 2:
        return []
    for atom in init_atoms:
        if not isinstance(atom, Mapping) or atom.get("predicate") != "On":
            continue
        arguments = atom.get("arguments")
        if not isinstance(arguments, list) or len(arguments) != 2 or arguments[0] != entity:
            continue
        marker = f"_{raw_type}_"
        if marker not in arguments[1]:
            continue
        qualifier = arguments[1].split(marker, 1)[1].removesuffix("_init_region")
        if qualifier in _POSITION_QUALIFIERS:
            return [f"initial={qualifier}"]
    return []


def _trim_region(value: str) -> str:
    return value.removesuffix("_region")


def _qualified_name(value: Mapping[str, Any]) -> str:
    qualifiers = sorted(str(item) for item in value["qualifiers"])
    return (
        str(value["identity"])
        if not qualifiers
        else f"{value['identity']}[{','.join(qualifiers)}]"
    )


def _interleave(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    strata: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["stratum_id"])].append(row)
    ordered_strata = sorted(grouped, key=lambda value: (stratum_digest(value), value))
    for stratum in ordered_strata:
        strata[stratum].extend(
            dict(row)
            for row in sorted(
                grouped[stratum],
                key=lambda item: (item["rank_digest"], item["canonical_family_id"]),
            )
        )
    result: list[dict[str, Any]] = []
    while any(strata.values()):
        for stratum in ordered_strata:
            if strata[stratum]:
                result.append(strata[stratum].popleft())
    return result


def _trial_rows(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for family in selected:
        for replicate_id in range(REPLICATES_PER_FAMILY):
            cell_id = "untouched_baseline"
            row = {
                "trial_id": f"rq024-foundation-{family['candidate_position']:02d}-r{replicate_id}",
                "child_plan_id": CHILD_PLAN_ID,
                "dataset_id": "rq024-foundation-r1",
                "benchmark": BENCHMARK,
                "task_id": family["task_id"],
                "task_name": family["task_name"],
                "split": (
                    "train_discovery_baseline"
                    if family["pool"] == "discovery"
                    else "test_confirmation_baseline"
                ),
                "capture_profile": "rollout",
                "canonical_family_id": family["canonical_family_id"],
                "candidate_position": family["candidate_position"],
                "pool": family["pool"],
                "pool_position": family["pool_position"],
                "cell_id": cell_id,
                "replicate_id": replicate_id,
            }
            for domain in SEED_DOMAINS:
                identity, seed = seed_identity(
                    str(family["canonical_family_id"]), cell_id, replicate_id, domain
                )
                row[f"{domain}_seed_identity"] = identity
                row[f"{domain}_seed"] = seed
            row["layout_id"] = int(row["layout_seed"]) % LIBERO_INIT_STATE_COUNT
            row["seed"] = row["reset_seed"]
            rows.append(row)
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("Trial table cannot be empty")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _check_source_catalog(source: Mapping[str, Any]) -> None:
    if source.get("schema_version") != 1 or source.get("benchmark") != BENCHMARK:
        raise ValueError("Unsupported source catalog schema or benchmark")
    tasks = source.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 90:
        raise ValueError("Source catalog must contain exactly 90 tasks")
    task_ids = [task.get("task_id") for task in tasks if isinstance(task, Mapping)]
    if task_ids != list(range(90)):
        raise ValueError("Source tasks must be uniquely ordered by IDs 0 through 89")
