"""Log PI0.5 capture resource usage to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FIELDS = (
    "timestamp_utc",
    "elapsed_s",
    "pid",
    "pid_alive",
    "trace_count",
    "gpu_use_pct",
    "gpu_vram_used_gib",
    "gpu_vram_total_gib",
    "gpu_process_vram_gib",
    "gpu_temp_edge_c",
    "gpu_temp_junction_c",
    "gpu_temp_memory_c",
    "gpu_power_w",
    "process_cpu_pct",
    "process_mem_pct",
    "process_rss_gib",
    "process_vsz_gib",
    "system_mem_used_gib",
    "system_mem_available_gib",
    "disk_used_gib",
    "disk_free_gib",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--disk-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--stop-when-pid-exits", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists() or args.out.stat().st_size == 0
    start = time.monotonic()
    with args.out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        while True:
            row = sample(args, elapsed_s=time.monotonic() - start)
            writer.writerow(row)
            handle.flush()
            if args.stop_when_pid_exits and not row["pid_alive"]:
                return
            time.sleep(args.interval_seconds)


def sample(args: argparse.Namespace, *, elapsed_s: float) -> dict[str, Any]:
    rocm = _rocm_sample()
    process = _process_sample(args.pid)
    mem = _mem_sample()
    disk = shutil.disk_usage(args.disk_root)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "elapsed_s": round(elapsed_s, 3),
        "pid": args.pid,
        "pid_alive": bool(process),
        "trace_count": _trace_count(args.trace_root),
        "gpu_use_pct": rocm.get("gpu_use_pct"),
        "gpu_vram_used_gib": _bytes_to_gib(rocm.get("gpu_vram_used_b")),
        "gpu_vram_total_gib": _bytes_to_gib(rocm.get("gpu_vram_total_b")),
        "gpu_process_vram_gib": _bytes_to_gib(_rocm_pid_vram(args.pid)),
        "gpu_temp_edge_c": rocm.get("gpu_temp_edge_c"),
        "gpu_temp_junction_c": rocm.get("gpu_temp_junction_c"),
        "gpu_temp_memory_c": rocm.get("gpu_temp_memory_c"),
        "gpu_power_w": rocm.get("gpu_power_w"),
        "process_cpu_pct": process.get("cpu_pct") if process else None,
        "process_mem_pct": process.get("mem_pct") if process else None,
        "process_rss_gib": _kib_to_gib(process.get("rss_kib")) if process else None,
        "process_vsz_gib": _kib_to_gib(process.get("vsz_kib")) if process else None,
        "system_mem_used_gib": _kib_to_gib(mem.get("used_kib")),
        "system_mem_available_gib": _kib_to_gib(mem.get("available_kib")),
        "disk_used_gib": _bytes_to_gib(disk.used),
        "disk_free_gib": _bytes_to_gib(disk.free),
    }


def _rocm_sample() -> dict[str, Any]:
    payload = _json_command(
        [
            "rocm-smi",
            "--showuse",
            "--showmeminfo",
            "vram",
            "--showtemp",
            "--showpower",
            "--json",
        ]
    )
    card = payload.get("card0", {}) if isinstance(payload, dict) else {}
    return {
        "gpu_use_pct": _float(card.get("GPU use (%)")),
        "gpu_vram_used_b": _float(card.get("VRAM Total Used Memory (B)")),
        "gpu_vram_total_b": _float(card.get("VRAM Total Memory (B)")),
        "gpu_temp_edge_c": _float(card.get("Temperature (Sensor edge) (C)")),
        "gpu_temp_junction_c": _float(card.get("Temperature (Sensor junction) (C)")),
        "gpu_temp_memory_c": _float(card.get("Temperature (Sensor memory) (C)")),
        "gpu_power_w": _float(card.get("Average Graphics Package Power (W)")),
    }


def _rocm_pid_vram(pid: int) -> float | None:
    payload = _json_command(["rocm-smi", "--showpids", "--json"])
    system = payload.get("system", {}) if isinstance(payload, dict) else {}
    value = system.get(f"PID{pid}")
    if not isinstance(value, str):
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) < 3:
        return None
    return _float(parts[2])


def _process_sample(pid: int) -> dict[str, float] | None:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "%cpu=,%mem=,rss=,vsz="],
        check=False,
        capture_output=True,
        text=True,
    )
    text = result.stdout.strip()
    if result.returncode != 0 or not text:
        return None
    cpu, mem, rss, vsz = text.split()[:4]
    return {
        "cpu_pct": _float(cpu),
        "mem_pct": _float(mem),
        "rss_kib": _float(rss),
        "vsz_kib": _float(vsz),
    }


def _mem_sample() -> dict[str, float]:
    result = subprocess.run(
        ["free", "-k"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            return {"used_kib": _float(parts[2]), "available_kib": _float(parts[6])}
    return {}


def _trace_count(root: Path) -> int:
    if not root.exists():
        return 0
    refs = list(root.rglob("vla_lens/tables/episode_refs.parquet"))
    if refs:
        total = 0
        for path in refs:
            try:
                import pandas as pd

                total += int(len(pd.read_parquet(path)))
            except Exception:
                continue
        if total:
            return total
    return sum(1 for _ in root.rglob("*.vlatrace"))


def _json_command(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    start = result.stdout.find("{")
    if start < 0:
        return {}
    try:
        return json.loads(result.stdout[start:])
    except json.JSONDecodeError:
        return {}


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bytes_to_gib(value: float | None) -> float | None:
    return round(value / (1024**3), 3) if value is not None else None


def _kib_to_gib(value: float | None) -> float | None:
    return round(value / (1024**2), 3) if value is not None else None


if __name__ == "__main__":
    main()
