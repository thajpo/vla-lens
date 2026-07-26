#!/usr/bin/env python
"""Write or verify an immutable local PI0.5 checkpoint snapshot receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lens.pi05.runtime_identity import checkpoint_snapshot_receipt
from vla_lens.research_io import write_bytes_create_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    receipt = checkpoint_snapshot_receipt(args.model_id, args.revision, args.snapshot)
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    if args.output is not None:
        write_bytes_create_only(args.output, encoded)
    print(encoded.decode(), end="")


if __name__ == "__main__":
    main()
