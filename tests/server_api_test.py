from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from vla_lens import create_synthetic_trace_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_script_serves_fastapi_backend(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "serve_vla_lens_dashboard.py"),
            str(dataset.root),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        payload = _wait_for_json(f"http://127.0.0.1:{port}/api/dataset", process)
        assert payload["activation_sites"] == len(dataset.model_site_index)
        assert "workbench" not in payload

        error = _http_error(f"http://127.0.0.1:{port}/api/frame")
        error_payload = json.loads(error.read().decode("utf-8"))
        assert error.code == 400
        assert error.headers["Content-Type"].startswith("application/json")
        assert error_payload["message"] == "Missing query parameter: trace_id"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_json(url: str, process: subprocess.Popen[str]) -> dict[str, object]:
    deadline = time.monotonic() + 10
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"dashboard server exited early rc={process.returncode}\n"
                f"stdout={stdout}\nstderr={stderr}"
            )
        try:
            with urllib.request.urlopen(url, timeout=0.25) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(0.1)
    raise AssertionError(f"dashboard server did not become ready: {url}") from last_error


def _http_error(url: str) -> urllib.error.HTTPError:
    try:
        urllib.request.urlopen(url, timeout=5)
    except urllib.error.HTTPError as exc:
        return exc
    raise AssertionError(f"expected HTTPError for {url}")
