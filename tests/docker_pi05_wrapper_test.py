from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_docker_pi05_rewrites_absolute_config_output_root(tmp_path):
    output_root = tmp_path / "capture-out"
    config = tmp_path / "absolute-output.yaml"
    config.write_text(
        f"name: absolute-output\noutput_root: {output_root}\n",
        encoding="utf-8",
    )

    args = _run_wrapper(
        tmp_path,
        "--backend",
        "rocm",
        "--no-build",
        "--config",
        str(config),
        "--run",
    )

    assert _contains_pair(args, "-v", f"{config.parent}:/host-inputs/0:ro")
    assert _contains_pair(args, "-v", f"{output_root.resolve()}:/capture-output")
    assert _value_after(args, "--config") == "/host-inputs/0/absolute-output.yaml"
    assert _value_after(args, "--output-root") == "/capture-output"


def test_docker_pi05_mounts_absolute_episode_plan_inputs(tmp_path):
    plan = tmp_path / "episode-plan.csv"
    plan.write_text("trace_id\ntrace-a\n", encoding="utf-8")

    args = _run_wrapper(
        tmp_path,
        "--backend",
        "cuda",
        "--no-build",
        "--episode-plan",
        str(plan),
    )

    assert _contains_pair(args, "-v", f"{plan.parent}:/host-inputs/0:ro")
    assert _value_after(args, "--episode-plan") == "/host-inputs/0/episode-plan.csv"


def _run_wrapper(tmp_path: Path, *args: str) -> list[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    output = tmp_path / "docker-args.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$@" > "$VLA_LENS_FAKE_DOCKER_ARGS"\n',
        encoding="utf-8",
    )
    docker.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "VLA_LENS_FAKE_DOCKER_ARGS": str(output),
            "VLA_LENS_RUNS_DIR": str(tmp_path / "runs"),
            "VLA_LENS_HF_CACHE_DIR": str(tmp_path / "hf-cache"),
            "VLA_LENS_LIBERO_CACHE_DIR": str(tmp_path / "libero-cache"),
        }
    )
    subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "docker_pi05.sh"), *args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return output.read_text(encoding="utf-8").splitlines()


def _contains_pair(args: list[str], key: str, value: str) -> bool:
    return any(left == key and right == value for left, right in zip(args, args[1:], strict=False))


def _value_after(args: list[str], key: str) -> str:
    index = args.index(key)
    return args[index + 1]
