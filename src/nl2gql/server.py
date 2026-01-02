"""A small JSON HTTP service on the standard library, so the container needs nothing beyond numpy and graphql-core.

    POST /translate   {"text": "...", "today": "YYYY-MM-DD", "strict": false}  -> translation result
                      ("today" and "strict" are optional; strict asks instead of filling in defaults)
    POST /check       {"query": "..."}                                    -> {"valid": bool, "issues": [...]}
    GET  /schema      the composed SDL (text/plain)
    GET  /health      {"status": "ok"}
"""

from __future__ import annotations

import datetime as dt
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nl2gql.checker import check
from nl2gql.model import Translator
from nl2gql.schema import sdl
from nl2gql.translate import translate

MAX_BODY = 64 * 1024


def make_handler(model: Translator):
    class Handler(BaseHTTPRequestHandler):
        server_version = "nl2gql/0.1"

        def _send(self, status: int, body, content_type: str = "application/json") -> None:
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json_body(self) -> dict | None:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                self._send(413 if length > MAX_BODY else 400, {"error": "expected a JSON body up to 64 KB"})
                return None
            try:
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send(400, {"error": "body is not valid JSON"})
                return None
            if not isinstance(body, dict):
                self._send(400, {"error": "body must be a JSON object"})
                return None
            return body

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/schema":
                self._send(200, sdl(), "text/plain; charset=utf-8")
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            body = self._json_body()
            if body is None:
                return
            if self.path == "/translate":
                text = body.get("text")
                if not isinstance(text, str) or not text.strip():
                    self._send(400, {"error": "'text' must be a non-empty string"})
                    return
                try:
                    today = dt.date.fromisoformat(body["today"]) if body.get("today") else dt.date.today()
                except (TypeError, ValueError):
                    self._send(400, {"error": "'today' must be YYYY-MM-DD"})
                    return
                strict = body.get("strict", False)
                if not isinstance(strict, bool):
                    self._send(400, {"error": "'strict' must be true or false"})
                    return
                self._send(200, translate(model, text.strip()[:1000], today, strict=strict).to_dict())
            elif self.path == "/check":
                query = body.get("query")
                if not isinstance(query, str):
                    self._send(400, {"error": "'query' must be a string"})
                    return
                issues = [{"code": i.code, "message": i.message} for i in check(query)]
                self._send(200, {"valid": not issues, "issues": issues})
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, fmt, *args) -> None:  # quieter default logging
            pass

    return Handler


def serve(model: Translator, host: str = "127.0.0.1", port: int = 8080) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(model))
    print(f"nl2gql listening on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
