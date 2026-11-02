"""
Vision Error Observatory (VEO) - High-Performance Local HTTP Server
Features:
- Seamless byte-range support (HTTP 206 Partial Content) for smooth MP4 video playback & seeking
- MIME type mapping (mp4, css, js, json, svg, woff2, ico)
- Automatic port fallback (8000 -> 8001 -> 8080) if port is in use
- Zero external dependencies; runs on standard library
"""

import http.server
import os
import re
import socketserver
import sys
import webbrowser
from pathlib import Path
from urllib.parse import unquote

DEFAULT_PORT = 8000
DIRECTORY = Path(__file__).parent.resolve()
# Local runs stay on loopback; the container sets VEO_HOST=0.0.0.0 to accept mapped traffic
HOST = os.environ.get("VEO_HOST", "127.0.0.1")

class VEORequestHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".mp4": "video/mp4",
        ".svg": "image/svg+xml",
        ".pdf": "application/pdf",
        ".ico": "image/x-icon",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".ttf": "font/ttf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    @staticmethod
    def contained(path: Path) -> Path | None:
        # Resolve and reject anything that escapes DIRECTORY (e.g. "/assets/..%2f..%2fetc")
        resolved = path.resolve()
        try:
            resolved.relative_to(DIRECTORY)
        except ValueError:
            return None
        return resolved

    def do_HEAD(self):
        self.handle_request(head_only=True)

    def do_GET(self):
        self.handle_request(head_only=False)

    def handle_request(self, head_only: bool = False):
        # Decode clean path
        clean_path = unquote(self.path.split("?")[0].split("#")[0])

        # Route root to /index.html
        if clean_path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/index.html")
            self.end_headers()
            return

        # Handle regular files with byte-range support for videos
        target = self.contained(DIRECTORY / clean_path.lstrip("/"))
        if target is None:
            self.send_error(403, "Access denied")
            return

        if target.is_file():
            self.serve_file(target, head_only=head_only)
        else:
            if head_only:
                super().do_HEAD()
            else:
                super().do_GET()

    def serve_file(self, filepath: Path, head_only: bool = False):
        file_size = filepath.stat().st_size
        content_type = self.guess_type(str(filepath))
        range_header = self.headers.get("Range")

        if range_header and range_header.startswith("bytes="):
            # Parse Range: bytes=start-end
            range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
                end = min(end, file_size - 1)
                
                if start > end or start >= file_size:
                    self.send_response(416, "Requested Range Not Satisfiable")
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return

                length = end - start + 1
                self.send_response(206, "Partial Content")
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()

                if not head_only:
                    try:
                        with open(filepath, "rb") as f:
                            f.seek(start)
                            bytes_remaining = length
                            chunk_size = 64 * 1024
                            while bytes_remaining > 0:
                                read_len = min(bytes_remaining, chunk_size)
                                data = f.read(read_len)
                                if not data:
                                    break
                                self.wfile.write(data)
                                bytes_remaining -= len(data)
                    except (ConnectionError, OSError):
                        pass
                return

        # Full file delivery
        self.send_response(200, "OK")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

        if not head_only:
            try:
                with open(filepath, "rb") as f:
                    chunk_size = 64 * 1024
                    while True:
                        data = f.read(chunk_size)
                        if not data:
                            break
                        self.wfile.write(data)
            except (ConnectionError, OSError):
                pass

    def log_message(self, format, *args):
        # Clean terminal logging with timestamps and response codes
        code = args[1] if len(args) > 1 else "-"
        sys.stdout.write(f"[{self.log_date_time_string()}] HTTP {code} - {args[0]}\n")
        sys.stdout.flush()


def find_available_port(start_port=8000, max_attempts=10):
    import socket
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def main():
    os.chdir(DIRECTORY)
    port = find_available_port(DEFAULT_PORT)
    url = f"http://localhost:{port}/index.html"
    
    print("=" * 74)
    print("    VISION ERROR OBSERVATORY (VEO) - INTERACTIVE PLATFORM SERVER")
    print("=" * 74)
    print(f"  Root Directory : {DIRECTORY}")
    print(f"  URL            : {url}")
    print(f"  Port           : {port}")
    print("  Status         : Ready (Zero-Warning, Offline-Optimized)")
    print("=" * 74)
    print("  Serving assets: CSS (Precompiled), Chart.js (Local), FontAwesome, Video")
    print("  Opening browser automatically... Press Ctrl+C to stop.")
    print("=" * 74 + "\n")

    webbrowser.open(url)

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((HOST, port), VEORequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server. Goodbye!")


if __name__ == "__main__":
    main()
