from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "serve_vla_lens_app",
    ROOT / "scripts" / "serve_vla_lens_app.py",
)
assert SPEC is not None
serve_vla_lens_app = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(serve_vla_lens_app)


def test_wait_for_backend_reports_early_process_exit():
    process = subprocess.Popen(["true"])
    process.wait(timeout=5)

    with pytest.raises(RuntimeError, match="backend exited before becoming ready"):
        serve_vla_lens_app._wait_for_backend(
            "127.0.0.1",
            9,
            process=process,
            timeout_seconds=1,
        )


def test_wait_for_backend_honors_short_timeout():
    with pytest.raises(RuntimeError, match="backend did not become ready"):
        serve_vla_lens_app._wait_for_backend("127.0.0.1", 9, timeout_seconds=0)


def test_wait_for_backend_uses_health_probe(monkeypatch):
    urls: list[tuple[str, float]] = []

    class ReadyResponse:
        status = 200

        def __enter__(self) -> ReadyResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_urlopen(url: str, timeout: float) -> ReadyResponse:
        urls.append((url, timeout))
        return ReadyResponse()

    monkeypatch.setattr(serve_vla_lens_app.urllib.request, "urlopen", fake_urlopen)

    serve_vla_lens_app._wait_for_backend("127.0.0.1", 8765, timeout_seconds=1)

    assert urls == [("http://127.0.0.1:8765/api/health", 0.25)]
