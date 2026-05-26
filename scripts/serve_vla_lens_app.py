"""Serve the built VLA Lens workbench and backend API through one local origin."""

from __future__ import annotations

import argparse
import http.client
import mimetypes
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]

HOP_BY_HOP_HEADERS: Final = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="LeRobot v3 dataset root, trace dataset root, or one .vlatrace bundle",
    )
    parser.add_argument(
        "--frontend-dist",
        type=Path,
        default=Path("frontend/dist"),
        help="Built React app directory produced by npm run build.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8765)
    parser.add_argument(
        "--backend-timeout-seconds",
        type=float,
        default=300.0,
        help="Seconds to wait for the backend API to open large datasets.",
    )
    parser.add_argument(
        "--public-url",
        help="User-facing URL to print instead of the bound host/port.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frontend_dist = args.frontend_dist.resolve()
    if not (frontend_dist / "index.html").exists():
        raise SystemExit(
            f"frontend build not found at {frontend_dist}; run `npm run build --prefix frontend`"
        )

    backend = _start_backend(args.root, host=args.backend_host, port=args.backend_port)
    try:
        _wait_for_backend(
            args.backend_host,
            args.backend_port,
            process=backend,
            timeout_seconds=args.backend_timeout_seconds,
        )
        _serve_gateway(
            frontend_dist=frontend_dist,
            host=args.host,
            port=args.port,
            backend_host=args.backend_host,
            backend_port=args.backend_port,
            public_url=args.public_url,
        )
    finally:
        backend.terminate()
        try:
            backend.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend.kill()
            backend.wait(timeout=5)


def _start_backend(root: Path, *, host: str, port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "serve_vla_lens_dashboard.py"),
            str(root),
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        text=True,
    )


def _wait_for_backend(
    host: str,
    port: int,
    *,
    process: subprocess.Popen[str] | None = None,
    timeout_seconds: float = 300.0,
) -> None:
    url = f"http://{host}:{port}/api/health"
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while time.monotonic() <= deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"backend exited before becoming ready: returncode={process.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=0.25) as response:
                if response.status == HTTPStatus.OK:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    raise RuntimeError(f"backend did not become ready: {url}")


def _serve_gateway(
    *,
    frontend_dist: Path,
    host: str,
    port: int,
    backend_host: str,
    backend_port: int,
    public_url: str | None,
) -> None:
    class Handler(VLALensAppHandler):
        pass

    Handler.frontend_dist = frontend_dist
    Handler.backend_host = backend_host
    Handler.backend_port = backend_port
    server = ThreadingHTTPServer((host, port), Handler)
    url = public_url or f"http://{host}:{port}"
    print(f"vla-lens dashboard: {url}", flush=True)
    server.serve_forever()


class VLALensAppHandler(BaseHTTPRequestHandler):
    """Serve static frontend assets and reverse-proxy backend API routes."""

    frontend_dist: Path
    backend_host: str
    backend_port: int

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/") or self.path == "/api":
            self._proxy()
            return
        self._serve_static(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        if self.path.startswith("/api/") or self.path == "/api":
            self._proxy(send_body=False)
            return
        self._serve_static(send_body=False)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/api/") or self.path == "/api":
            self._proxy()
            return
        self.send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {self.path}")

    def _serve_static(self, *, send_body: bool) -> None:
        path = self._static_path()
        if path is None:
            self.send_error(HTTPStatus.NOT_FOUND, f"Static asset not found: {self.path}")
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if send_body:
            self.wfile.write(data)

    def _static_path(self) -> Path | None:
        parsed = urllib.parse.urlparse(self.path)
        route = urllib.parse.unquote(parsed.path)
        if route in {"", "/"}:
            return self.frontend_dist / "index.html"
        candidate = (self.frontend_dist / route.lstrip("/")).resolve()
        if _is_relative_to(candidate, self.frontend_dist) and candidate.is_file():
            return candidate
        if "." not in Path(route).name:
            return self.frontend_dist / "index.html"
        return None

    def _proxy(self, *, send_body: bool = True) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        target = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
        body = self._read_body()
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        connection = http.client.HTTPConnection(self.backend_host, self.backend_port, timeout=120)
        try:
            connection.request(self.command, target, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() in HOP_BY_HOP_HEADERS:
                    continue
                self.send_header(key, value)
            self.end_headers()
            if send_body and payload:
                self.wfile.write(payload)
        finally:
            connection.close()

    def _read_body(self) -> bytes | None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return None
        return self.rfile.read(length)

    def log_message(self, format: str, *args: object) -> None:
        message = format % args
        print(f"{self.address_string()} - {message}", flush=True)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    main()
