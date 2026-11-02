import http.client
import socketserver
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web_app"))
import serve  # noqa: E402


@pytest.fixture(scope="module")
def port():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), serve.VEORequestHandler)
    server.RequestHandlerClass.log_message = lambda *args: None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _status(port: int, path: str, method: str = "HEAD") -> int:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(method, path)
    response = connection.getresponse()
    response.read()
    return response.status


@pytest.mark.parametrize("path", [
    "/index.html",
    "/model_lab.html",
    "/data/app_data_v2.js",
    "/assets/fontawesome/all.min.css",
    "/assets/webfonts/fa-solid-900.woff2",
    "/favicon.ico",
])
def test_site_files_are_served(port, path):
    assert _status(port, path) == 200


@pytest.mark.parametrize("path", [
    "/../README.md",
    "/assets/..%2f..%2fREADME.md",
    "/assets/..%2f..%2fpyproject.toml",
    "/%2e%2e/%2e%2e/pyproject.toml",
])
def test_paths_outside_web_root_are_refused(port, path):
    assert _status(port, path, "GET") in (403, 404)


def test_video_supports_byte_ranges(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("GET", "/veo_explainer.mp4", headers={"Range": "bytes=0-99"})
    response = connection.getresponse()
    assert response.status == 206
    assert len(response.read()) == 100


def test_server_binds_loopback_by_default(monkeypatch):
    monkeypatch.delenv("VEO_HOST", raising=False)
    import importlib

    assert importlib.reload(serve).HOST == "127.0.0.1"
