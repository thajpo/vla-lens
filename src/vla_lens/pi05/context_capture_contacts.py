"""Control-step MuJoCo contact manifold capture for PI0.5/LIBERO rollouts."""

from __future__ import annotations

import json
import sys
from importlib import metadata as importlib_metadata
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.pi05.context_capture_common import (
    _body_name_from_value,
    _env_candidates,
    _first_existing_attr,
    _optional_int,
    _resolve_body_id,
    _Status,
)
from vla_lens.pi05.context_capture_scene import (
    _body_tree_ids,
    _scene_object_descriptors,
)

CONTACT_TELEMETRY_SCHEMA_VERSION = "mujoco_contact_manifold_v1"
CONTACT_SAMPLE_PHASE = "pre_action_control_step"
CONTACT_SAMPLING_LIMITATION = (
    "One MuJoCo contact-manifold sample is read from observation[timestep] before "
    "action[timestep]; this is not exhaustive physics-substep collision accounting and "
    "does not observe terminal post-action contact when an environment auto-resets."
)


def capture_contact_snapshot(env: Any | None, *, timestep: int) -> dict[str, Any]:
    """Read the current MuJoCo contact manifold from a nested simulator."""

    candidates = _env_candidates(env)
    sim = _first_existing_attr(candidates, ("sim", "_sim"))
    if sim is None:
        return {
            "timestep": int(timestep),
            "contacts": [],
            "capability": _unavailable_capability("nested env.sim unavailable"),
        }

    data = getattr(sim, "data", None)
    model = getattr(sim, "model", None)
    runtime = _mujoco_runtime_identity(sim)
    capability = {
        **_base_capability(),
        **runtime,
        "available": False,
        "reason": "",
    }
    if data is None or model is None:
        capability["reason"] = "env.sim.model or env.sim.data unavailable"
        return {"timestep": int(timestep), "contacts": [], "capability": capability}

    try:
        ncon = int(data.ncon)
        contact_array = data.contact
    except Exception:
        capability["reason"] = "sim.data.ncon or sim.data.contact unavailable"
        return {"timestep": int(timestep), "contacts": [], "capability": capability}

    owners = _body_owners(candidates, sim)
    force_reader, force_source = _contact_force_reader(sim)
    contacts: list[dict[str, Any]] = []
    malformed = 0
    for contact_index in range(max(0, ncon)):
        try:
            contact = contact_array[contact_index]
            record = _contact_record(
                sim,
                contact,
                contact_index=contact_index,
                timestep=timestep,
                owners=owners,
                force_reader=force_reader,
                force_source=force_source,
            )
        except Exception:
            record = None
        if record is None:
            malformed += 1
        else:
            contacts.append(record)

    capability.update(
        {
            "available": malformed == 0,
            "reason": "" if malformed == 0 else f"{malformed} malformed contact row(s)",
            "reported_contact_count": max(0, ncon),
            "captured_contact_count": len(contacts),
            "force_api_available": force_reader is not None,
            "force_source": force_source,
        }
    )
    return {
        "timestep": int(timestep),
        "contacts": contacts,
        "capability": capability,
    }


def extract_contact_context(
    contact_snapshots: Sequence[Mapping[str, Any]] | None,
    *,
    status: _Status,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize contact snapshots into manifold rows and a capability audit."""

    snapshots = [item for item in contact_snapshots or () if isinstance(item, Mapping)]
    audit = contact_capability_audit(snapshots)
    records = [
        dict(contact)
        for snapshot in snapshots
        for contact in _contact_rows(snapshot)
    ]
    contacts = pd.DataFrame.from_records(records)
    capability = pd.DataFrame.from_records([audit])

    if audit["available"]:
        status.available(
            "contact",
            "mujoco_contact_manifold",
            "nested env.sim.data.contact",
            shape=(int(audit["contact_count"]),),
        )
    else:
        status.missing("contact", "mujoco_contact_manifold", str(audit["reason"]))
    if audit["mujoco_version_exact"]:
        status.available("contact", "mujoco_version", str(audit["mujoco_version_source"]))
    else:
        status.missing("contact", "mujoco_version", "exact MuJoCo version unavailable")
    status.available("contact", "sample_phase", "capture runner pre-action hook")
    if audit["force_capture_status"] in {"available", "api_available_unobserved"}:
        status.available("contact", "force", str(audit["force_source"]))
    else:
        status.missing("contact", "force", "no robust MuJoCo contact-force API available")
    return contacts, capability


def contact_capability_audit(
    contact_snapshots: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Build a JSON-safe audit of control-step contact telemetry capability."""

    snapshots = [item for item in contact_snapshots or () if isinstance(item, Mapping)]
    if not snapshots:
        return {
            **_base_capability(),
            "available": False,
            "verdict": "unavailable_no_control_step_samples",
            "reason": "no pre-action control-step contact snapshots were supplied",
            "sample_count": 0,
            "contact_count": 0,
            "mujoco_version": "",
            "mujoco_version_exact": False,
            "mujoco_version_source": "",
            "mujoco_versions_json": "[]",
            "binding": "",
            "binding_version": "",
            "force_api_available": False,
            "force_capture_status": "unavailable",
            "force_source": "",
        }

    capabilities = [
        item.get("capability") if isinstance(item.get("capability"), Mapping) else {}
        for item in snapshots
    ]
    contacts = [contact for snapshot in snapshots for contact in _contact_rows(snapshot)]
    versions = sorted(
        {str(item.get("mujoco_version")) for item in capabilities if item.get("mujoco_version")}
    )
    all_available = all(bool(item.get("available")) for item in capabilities)
    exact_version = (
        len(versions) == 1
        and all(bool(item.get("mujoco_version_exact")) for item in capabilities)
    )
    phases_match = all(item.get("sample_phase") == CONTACT_SAMPLE_PHASE for item in capabilities)
    available = all_available and exact_version and phases_match
    if available:
        verdict = "available_control_step_contact_manifold"
        reason = ""
    elif not all_available:
        verdict = "partial_or_unavailable_contact_manifold"
        reasons = [str(item.get("reason")) for item in capabilities if item.get("reason")]
        reason = "; ".join(dict.fromkeys(reasons)) or "one or more contact samples unavailable"
    elif not exact_version:
        verdict = "partial_missing_exact_mujoco_version"
        reason = "exact, stable MuJoCo version was not resolved for every sample"
    else:
        verdict = "partial_inconsistent_sample_phase"
        reason = "contact snapshots do not all use pre_action_control_step semantics"

    force_api_available = all(bool(item.get("force_api_available")) for item in capabilities)
    force_rows_available = bool(contacts) and all(
        bool(item.get("force_available")) for item in contacts
    )
    if force_rows_available:
        force_status = "available"
    elif not contacts and force_api_available:
        force_status = "api_available_unobserved"
    else:
        force_status = "unavailable"

    bindings = sorted({str(item.get("binding")) for item in capabilities if item.get("binding")})
    binding_versions = sorted(
        {str(item.get("binding_version")) for item in capabilities if item.get("binding_version")}
    )
    version_sources = sorted(
        {
            str(item.get("mujoco_version_source"))
            for item in capabilities
            if item.get("mujoco_version_source")
        }
    )
    force_sources = sorted(
        {str(item.get("force_source")) for item in capabilities if item.get("force_source")}
    )
    return {
        **_base_capability(),
        "available": available,
        "verdict": verdict,
        "reason": reason,
        "sample_count": len(snapshots),
        "contact_count": len(contacts),
        "mujoco_version": versions[0] if len(versions) == 1 else "",
        "mujoco_version_exact": exact_version,
        "mujoco_version_source": ",".join(version_sources),
        "mujoco_versions_json": json.dumps(versions),
        "binding": bindings[0] if len(bindings) == 1 else "",
        "binding_version": binding_versions[0] if len(binding_versions) == 1 else "",
        "force_api_available": force_api_available,
        "force_capture_status": force_status,
        "force_source": ",".join(force_sources),
    }


def _contact_record(
    sim: Any,
    contact: Any,
    *,
    contact_index: int,
    timestep: int,
    owners: Mapping[int, Mapping[str, str]],
    force_reader: Any | None,
    force_source: str,
) -> dict[str, Any] | None:
    geom1_id = _optional_int(getattr(contact, "geom1", None))
    geom2_id = _optional_int(getattr(contact, "geom2", None))
    position = _exact_vector(getattr(contact, "pos", None), 3)
    frame = _exact_vector(getattr(contact, "frame", None), 9)
    try:
        distance = float(contact.dist)
    except (AttributeError, TypeError, ValueError):
        return None
    if geom1_id is None or geom2_id is None or position is None or frame is None:
        return None

    force = None
    if force_reader is not None:
        try:
            force = _exact_vector(force_reader(contact_index), 6)
        except Exception:
            force = None
    model = sim.model
    body1_id = _geom_body_id(model, geom1_id)
    body2_id = _geom_body_id(model, geom2_id)
    owner1 = _owner_record(owners, body1_id, sim)
    owner2 = _owner_record(owners, body2_id, sim)
    return {
        "schema_version": CONTACT_TELEMETRY_SCHEMA_VERSION,
        "timestep": int(timestep),
        "sample_phase": CONTACT_SAMPLE_PHASE,
        "contact_index": int(contact_index),
        "geom1_id": geom1_id,
        "geom1_name": _model_name(model, "geom", geom1_id),
        "geom2_id": geom2_id,
        "geom2_name": _model_name(model, "geom", geom2_id),
        "geom1_body_id": body1_id,
        "geom1_body_name": _body_name_from_value(sim, body1_id),
        "geom2_body_id": body2_id,
        "geom2_body_name": _body_name_from_value(sim, body2_id),
        "geom1_owner_name": owner1["name"],
        "geom1_owner_kind": owner1["kind"],
        "geom1_owner_source": owner1["source"],
        "geom2_owner_name": owner2["name"],
        "geom2_owner_kind": owner2["kind"],
        "geom2_owner_source": owner2["source"],
        "signed_distance_m": distance,
        "distance_class": _distance_class(distance),
        "physical_contact": distance <= 0.0,
        "positive_gap_proximity": distance > 0.0,
        "position_world_m": position.tolist(),
        "frame_world_row_major": frame.tolist(),
        "force_torque_contact_frame": None if force is None else force.tolist(),
        "force_available": force is not None,
        "force_source": force_source if force is not None else "",
    }


def _base_capability() -> dict[str, Any]:
    return {
        "schema_version": CONTACT_TELEMETRY_SCHEMA_VERSION,
        "telemetry_kind": "mujoco_contact_manifold",
        "sample_phase": CONTACT_SAMPLE_PHASE,
        "sample_phase_semantics": (
            "observation[timestep] immediately before action[timestep]; terminal post-action "
            "contact is not observed when the environment auto-resets"
        ),
        "exhaustive_physics_substeps": False,
        "sampling_limitation": CONTACT_SAMPLING_LIMITATION,
        "distance_semantics": (
            "signed MuJoCo contact distance: <=0 touching/penetrating; >0 is a positive-gap "
            "manifold entry within collision margin, not physical touching"
        ),
        "position_semantics": "raw MuJoCo contact.pos in world coordinates, meters",
        "frame_semantics": (
            "raw MuJoCo contact.frame 3x3 row-major contact frame in world coordinates; "
            "the first three components are the contact normal"
        ),
        "force_semantics": (
            "optional raw MuJoCo 6-vector from get_contact_force/mj_contactForce: "
            "force then torque in the contact frame"
        ),
        "legacy_interaction_metrics_relabelled": False,
        "legacy_interaction_metrics_relation": "separate_unchanged",
    }


def _unavailable_capability(reason: str) -> dict[str, Any]:
    return {
        **_base_capability(),
        "available": False,
        "reason": reason,
        "mujoco_version": "",
        "mujoco_version_exact": False,
        "mujoco_version_source": "",
        "binding": "",
        "binding_version": "",
        "reported_contact_count": 0,
        "captured_contact_count": 0,
        "force_api_available": False,
        "force_source": "",
    }


def _mujoco_runtime_identity(sim: Any) -> dict[str, Any]:
    for candidate in (sim, getattr(sim, "model", None), getattr(sim, "data", None)):
        value = getattr(candidate, "mujoco_version", None)
        if value:
            return {
                "mujoco_version": str(value),
                "mujoco_version_exact": True,
                "mujoco_version_source": f"{type(candidate).__name__}.mujoco_version",
                **_binding_identity(sim),
            }

    modules = _mujoco_modules(sim)
    for module_name, module in modules:
        for method_name in ("mj_versionString", "get_version"):
            method = getattr(module, method_name, None)
            if not callable(method):
                continue
            try:
                value = method()
            except Exception:
                continue
            if value:
                return {
                    "mujoco_version": str(value),
                    "mujoco_version_exact": True,
                    "mujoco_version_source": f"{module_name}.{method_name}",
                    **_binding_identity(sim),
                }

    identity = _binding_identity(sim)
    if identity["binding"] == "mujoco" and identity["binding_version"]:
        return {
            "mujoco_version": identity["binding_version"],
            "mujoco_version_exact": True,
            "mujoco_version_source": "package:mujoco",
            **identity,
        }
    return {
        "mujoco_version": "",
        "mujoco_version_exact": False,
        "mujoco_version_source": "",
        **identity,
    }


def _binding_identity(sim: Any) -> dict[str, str]:
    roots = {
        type(item).__module__.split(".", 1)[0]
        for item in (sim, getattr(sim, "model", None), getattr(sim, "data", None))
        if item is not None
    }
    if "mujoco" in roots:
        binding, distribution = "mujoco", "mujoco"
    elif "mujoco_py" in roots:
        binding, distribution = "mujoco_py", "mujoco-py"
    else:
        binding, distribution = "", ""
    try:
        version = importlib_metadata.version(distribution) if distribution else ""
    except importlib_metadata.PackageNotFoundError:
        version = ""
    return {"binding": binding, "binding_version": version}


def _mujoco_modules(sim: Any) -> list[tuple[str, Any]]:
    roots = []
    for item in (sim, getattr(sim, "model", None), getattr(sim, "data", None)):
        if item is None:
            continue
        root = type(item).__module__.split(".", 1)[0]
        if root in {"mujoco", "mujoco_py"} and root not in roots:
            roots.append(root)
    return [(root, sys.modules[root]) for root in roots if root in sys.modules]


def _contact_force_reader(sim: Any) -> tuple[Any | None, str]:
    data = getattr(sim, "data", None)
    method = getattr(data, "get_contact_force", None)
    if callable(method):
        return method, f"{type(data).__name__}.get_contact_force"
    for module_name, module in _mujoco_modules(sim):
        method = getattr(module, "mj_contactForce", None)
        if not callable(method):
            continue

        def read_force(contact_index: int, method: Any = method) -> np.ndarray:
            result = np.zeros(6, dtype=np.float64)
            method(sim.model, sim.data, int(contact_index), result)
            return result

        return read_force, f"{module_name}.mj_contactForce"
    return None, ""


def _body_owners(candidates: Sequence[Any], sim: Any) -> dict[int, dict[str, str]]:
    owners: dict[int, dict[str, str]] = {}
    for descriptor in _scene_object_descriptors(candidates, sim):
        body_id = _optional_int(descriptor.get("body_id"))
        if body_id is None:
            body_id = _resolve_body_id(sim, descriptor.get("body_name"))
        if body_id is None:
            continue
        owner = {
            "name": str(descriptor.get("object_name") or ""),
            "kind": str(descriptor.get("object_kind") or "object"),
            "source": str(descriptor.get("source") or ""),
        }
        for descendant in _body_tree_ids(sim.model, body_id):
            owners.setdefault(int(descendant), owner)
    return owners


def _owner_record(
    owners: Mapping[int, Mapping[str, str]], body_id: int | None, sim: Any
) -> dict[str, str]:
    if body_id is not None and body_id in owners:
        return dict(owners[body_id])
    return {
        "name": _body_name_from_value(sim, body_id),
        "kind": "body",
        "source": "mujoco.geom_bodyid",
    }


def _geom_body_id(model: Any, geom_id: int) -> int | None:
    try:
        return int(model.geom_bodyid[int(geom_id)])
    except Exception:
        return None


def _model_name(model: Any, kind: str, index: int) -> str:
    method = getattr(model, f"{kind}_id2name", None)
    if callable(method):
        try:
            return _name_text(method(int(index)))
        except Exception:
            pass
    accessor = getattr(model, kind, None)
    if callable(accessor):
        try:
            return _name_text(accessor(int(index)).name)
        except Exception:
            pass
    names = getattr(model, f"{kind}_names", None)
    try:
        return _name_text(names[int(index)])
    except Exception:
        return ""


def _name_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _exact_vector(value: Any, size: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != size or not np.all(np.isfinite(array)):
        return None
    return array.copy()


def _distance_class(distance: float) -> str:
    if distance < 0.0:
        return "penetrating"
    if distance > 0.0:
        return "positive_gap_within_contact_margin"
    return "touching"


def _contact_rows(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    contacts = snapshot.get("contacts")
    if not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes)):
        return []
    return [item for item in contacts if isinstance(item, Mapping)]


__all__ = [
    "CONTACT_SAMPLE_PHASE",
    "CONTACT_SAMPLING_LIMITATION",
    "CONTACT_TELEMETRY_SCHEMA_VERSION",
    "capture_contact_snapshot",
    "contact_capability_audit",
    "extract_contact_context",
]
