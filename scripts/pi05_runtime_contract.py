#!/usr/bin/env python
"""Build a canonical PI0.5 camera/controller/processor runtime contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lens.pi05.runtime_identity import canonical_component_identities
from vla_lens.research_io import load_research_mapping, write_bytes_create_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--obs-size", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    environment = load_research_mapping(args.environment)
    checkpoint = load_research_mapping(args.checkpoint)
    components = canonical_component_identities(
        obs_size=args.obs_size,
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        environment_receipt=environment,
        checkpoint_receipt=checkpoint,
    )
    payload = {
        "schema_version": 1,
        "kind": "vla_lens.pi05_runtime_contract",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "backend_device": args.device,
        "dtype": args.dtype,
        "obs_size": args.obs_size,
        "components": components,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if args.output is not None:
        write_bytes_create_only(args.output, encoded)
    print(encoded.decode(), end="")


if __name__ == "__main__":
    main()
