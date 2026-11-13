"""Read-only loopback Host-header bridge (127.0.0.1:18897 -> 127.0.0.1:8897) for SSH tunnel viewing."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import gzip
import socket
import urllib.request
import urllib.error

ALLOWED_HOSTS = {"127.0.0.1:18897", "localhost:18897"}
ALLOWED_PATHS = {"/", "/app.js", "/style.css", "/api/status", "/api/ledger", "/api/equity", "/api/live_dashboard"}


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class BridgeHandler(BaseHTTPRequestHandler):
    timeout = 25

    def log_message(self, fmt, *args):
        return

    def handle(self):
        try:
            super().handle()
        except (TimeoutError, socket.timeout, BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_GET(self):
        self.close_connection = True
        if self.headers.get("Host") not in ALLOWED_HOSTS:
            try:
                self.send_response(403)
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b"Local host only")
            except (TimeoutError, socket.timeout, BrokenPipeError, ConnectionResetError):
                return
            return
        path = self.path.split("?", 1)[0]
        if path not in ALLOWED_PATHS:
            try:
                self.send_response(404)
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b"Not found")
            except (TimeoutError, socket.timeout, BrokenPipeError, ConnectionResetError):
                return
            return
        req = urllib.request.Request(
            f"http://127.0.0.1:8897{path}",
            headers={"Host": "127.0.0.1:8897", "Connection": "close"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = resp.read()
                ctype = resp.headers.get("Content-Type", "application/octet-stream")
                status = resp.status
        except urllib.error.HTTPError as exc:
            body = exc.read()
            ctype = exc.headers.get("Content-Type", "text/plain")
            status = exc.code
        except Exception:
            body = b'{"error":"UPSTREAM_UNAVAILABLE"}'
            ctype = "application/json"
            status = 503
        use_gzip = len(body) > 1024 and "gzip" in (self.headers.get("Accept-Encoding") or "").lower()
        if use_gzip:
            body = gzip.compress(body, compresslevel=5)
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            if use_gzip:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except (TimeoutError, socket.timeout, BrokenPipeError, ConnectionResetError):
            return


if __name__ == "__main__":
    BridgeServer(("127.0.0.1", 18897), BridgeHandler).serve_forever(poll_interval=0.5)
