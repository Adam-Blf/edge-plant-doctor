"""Small HTTP service: POST an image to /predict, read the top predictions as JSON.

Binds to 127.0.0.1 by default. Any other address requires a bearer token, because the
service has no other protection.
"""

import argparse
import hmac
import ipaddress
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .engine import Engine, ImageRejected

MAX_BODY = 10 * 1024 * 1024
log = logging.getLogger("plantdoctor")


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_binding(host: str, token: str | None) -> None:
    if not is_loopback(host) and not token:
        raise SystemExit(
            f"Refusing to listen on {host} without a token. Set PLANT_DOCTOR_TOKEN."
        )


def make_handler(engine: Engine, token: str | None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "PlantDoctorEdge"

        def log_message(self, fmt: str, *args) -> None:
            log.info("%s %s", self.address_string(), fmt % args)

        def reply(self, status: int, body: dict) -> None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def authorised(self) -> bool:
            if not token:
                return True
            given = self.headers.get("Authorization", "")
            return hmac.compare_digest(given, f"Bearer {token}")

        def do_GET(self) -> None:
            if self.path == "/health":
                self.reply(200, {"status": "ok", "model": engine.model_name})
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/predict":
                return self.reply(404, {"error": "not found"})
            if not self.authorised():
                return self.reply(401, {"error": "missing or wrong bearer token"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.reply(400, {"error": "bad Content-Length"})
            if length <= 0:
                return self.reply(400, {"error": "empty body"})
            if length > MAX_BODY:
                return self.reply(413, {"error": "image larger than 10 MB"})
            try:
                self.reply(200, engine.classify(self.rfile.read(length)))
            except ImageRejected as exc:
                self.reply(400, {"error": str(exc)})
            except Exception:
                log.exception("inference failed")
                self.reply(500, {"error": "inference failed"})

    return Handler


def main() -> None:
    root = Path(os.environ.get("PLANT_DOCTOR_HOME", "."))
    ap = argparse.ArgumentParser(description="Plant Doctor HTTP service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", type=Path, default=root / "models" / "plant-doctor-int8.onnx")
    ap.add_argument("--labels", type=Path, default=root / "models" / "labels.json")
    args = ap.parse_args()
    token = os.environ.get("PLANT_DOCTOR_TOKEN") or None
    validate_binding(args.host, token)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    engine = Engine(args.model, args.labels)
    with ThreadingHTTPServer((args.host, args.port), make_handler(engine, token)) as httpd:
        log.info("listening on %s:%d with %s", args.host, args.port, engine.model_name)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
