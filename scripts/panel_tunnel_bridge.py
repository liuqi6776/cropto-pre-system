"""Read-only loopback Host-header bridge (127.0.0.1:18897 -> 127.0.0.1:8897) for SSH tunnel viewing."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request
import urllib.error

ALLOWED_HOSTS = {"127.0.0.1:18897", "localhost:18897"}
ALLOWED_PATHS = {"/", "/app.js", "/style.css", "/api/status", "/api/ledger", "/api/equity", "/api/live_dashboard"}


class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.headers.get("Host") not in ALLOWED_HOSTS:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"Local host only")
            return
        path = self.path.split("?", 1)[0]
        if path not in ALLOWED_PATHS:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return
        req = urllib.request.Request(f"http://127.0.0.1:8897{path}", headers={"Host": "127.0.0.1:8897"}, method="GET")
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
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 18897), BridgeHandler).serve_forever(poll_interval=0.5)
