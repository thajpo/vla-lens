#!/usr/bin/env python
"""Export static LIBERO-90 metadata from a dedicated PI0.5 environment."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

from vla_lens.rq024_foundation import (
    BENCHMARK,
    EXPORTER_VERSION,
    canonical_json_bytes,
    sha256_bytes,
    source_task_from_bddl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_dedicated_environment(Path(sys.prefix))
    distribution = importlib.metadata.distribution("hf_libero")
    spec = importlib.util.find_spec("libero")
    if spec is None or spec.origin is None:
        raise SystemExit("The dedicated environment has no LIBERO package")
    package_root = Path(spec.origin).parent / "libero"
    task_map_path = package_root / "benchmark" / "libero_suite_task_map.py"
    bddl_root = package_root / "bddl_files" / BENCHMARK
    task_map_bytes = task_map_path.read_bytes()
    task_names = _task_names(task_map_bytes.decode("utf-8"))
    tasks = []
    for task_id, task_name in enumerate(task_names):
        relative = f"libero/libero/bddl_files/{BENCHMARK}/{task_name}.bddl"
        tasks.append(
            source_task_from_bddl(
                task_id=task_id,
                task_name=task_name,
                bddl_file=relative,
                bddl_bytes=(bddl_root / f"{task_name}.bddl").read_bytes(),
            )
        )
    catalog = {
        "schema_version": 1,
        "kind": "rq024.libero90_source_catalog",
        "benchmark": BENCHMARK,
        "exporter_version": EXPORTER_VERSION,
        "source_distribution": "hf_libero",
        "source_distribution_version": distribution.version,
        "task_map_sha256": sha256_bytes(task_map_bytes),
        "task_count": len(tasks),
        "tasks": tasks,
    }
    if len(tasks) != 90:
        raise SystemExit(f"Expected 90 LIBERO-90 tasks, found {len(tasks)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(catalog))
    print(json.dumps({"output": str(args.output), "task_count": len(tasks)}, sort_keys=True))


def _require_dedicated_environment(prefix: Path) -> None:
    if not prefix.name.startswith(".venv-pi05-"):
        raise SystemExit(
            "Metadata export is allowed only from a dedicated .venv-pi05-<backend> environment"
        )


def _task_names(source: str) -> list[str]:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in node.targets
        ):
            task_map = ast.literal_eval(node.value)
            return [str(value) for value in task_map[BENCHMARK]]
    raise ValueError("LIBERO task map does not define libero_task_map")


if __name__ == "__main__":
    main()
