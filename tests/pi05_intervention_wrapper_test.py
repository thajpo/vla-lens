from __future__ import annotations

import subprocess
from pathlib import Path


def test_pi05_intervention_wrapper_documents_gate_and_dedicated_runtime():
    result = subprocess.run(
        ["bash", "scripts/pi05_intervene.sh", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--backend rocm|cuda|mps|cpu" in result.stdout
    assert "--max-noop-l2/--max-noop-max-abs" in result.stdout
    assert "--run-intervention" in result.stdout


def test_pi05_intervention_entrypoint_is_registered():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'vla-pi05-intervene = "vla_lens.pi05.intervention_runner:main"' in pyproject
