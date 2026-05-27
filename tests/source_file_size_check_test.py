from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_source_file_sizes.py"
_SPEC = importlib.util.spec_from_file_location("report_source_file_sizes", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

main = _MODULE.main
oversized_source_files = _MODULE.oversized_source_files


def test_source_size_check_reports_oversized_file(tmp_path, capsys):
    source = tmp_path / "src" / "demo.py"
    source.parent.mkdir()
    source.write_text("a\nb\nc\n", encoding="utf-8")

    assert main(["--root", str(tmp_path / "src"), "--max-lines", "2"]) == 0

    output = capsys.readouterr().out
    assert "source files over 2 lines" in output
    assert "demo.py" in output


def test_source_size_check_fail_mode_returns_nonzero(tmp_path):
    source = tmp_path / "src" / "demo.py"
    source.parent.mkdir()
    source.write_text("a\nb\nc\n", encoding="utf-8")

    assert main(["--root", str(tmp_path / "src"), "--max-lines", "2", "--fail"]) == 1


def test_source_size_check_ignores_generated_paths(tmp_path):
    generated = tmp_path / "frontend" / "dist" / "bundle.ts"
    generated.parent.mkdir(parents=True)
    generated.write_text("a\nb\nc\n", encoding="utf-8")

    offenders = oversized_source_files([Path("frontend")], max_lines=2, cwd=tmp_path)

    assert offenders == []


def test_refactored_core_files_stay_under_700_lines():
    repo_root = Path(__file__).resolve().parents[1]
    roots = [
        repo_root / "src" / "vla_lens" / "traces",
        repo_root / "src" / "vla_lens" / "server",
        repo_root / "frontend" / "src" / "pages" / "EpisodesPage.tsx",
        repo_root / "frontend" / "src" / "pages" / "episodes",
        repo_root / "tests",
    ]

    offenders = oversized_source_files(roots, max_lines=700, cwd=repo_root)

    assert offenders == []
