from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from vla_lens import create_synthetic_trace_dataset
from vla_lens.server import TraceDashboardHandler, _dataset_signature


def test_api_client_errors_are_json(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    base_url, server, thread = _serve_dataset(dataset)
    try:
        response = _http_error(f"{base_url}/api/frame")
        payload = json.loads(response.read().decode("utf-8"))

        assert response.code == 400
        assert response.headers["Content-Type"].startswith("application/json")
        assert payload["message"] == "Missing query parameter: trace_id"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_artifact_ids_cannot_escape_dataset_root(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    base_url, server, thread = _serve_dataset(dataset)
    try:
        response = _http_error(f"{base_url}/api/artifacts/..%2F..%2Fescape")
        payload = json.loads(response.read().decode("utf-8"))

        assert response.code == 400
        assert response.headers["Content-Type"].startswith("application/json")
        assert "Invalid artifact_id" in payload["message"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _serve_dataset(dataset):
    class Handler(TraceDashboardHandler):
        pass

    Handler.dataset = dataset
    Handler.dataset_signature = _dataset_signature(dataset.root)
    Handler.dataset_signature_checked_at = 0.0
    Handler.root = dataset.root
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return f"http://{host}:{port}", server, thread


def _http_error(url: str) -> urllib.error.HTTPError:
    try:
        urllib.request.urlopen(url, timeout=5)
    except urllib.error.HTTPError as exc:
        return exc
    raise AssertionError(f"expected HTTPError for {url}")
