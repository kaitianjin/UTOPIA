"""Serve a results tree to the viewer page, for trees too large to inline.

``utopia view --serve`` shows the same page as the self-contained file, but its manifest says the
arrays are not inline and each one is fetched on demand as raw float32. The server is the standard
library's ``http.server``, read-only and bound to localhost by default.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .bundle import NO_RESULTS_MESSAGE, Locator, Selection, build_manifest, render_page
from .index import ModelIndex


def serve(
    models: dict[str, ModelIndex],
    selection: Selection,
    source: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Serve the viewer page and its arrays over HTTP until interrupted."""
    locator = Locator()
    manifest = build_manifest(models, selection, locator, source)
    if not manifest["models"]:
        raise ValueError(NO_RESULTS_MESSAGE)
    page = render_page(manifest, None).encode("utf-8")
    print(f"[view] {len(locator.loaders)} array(s) available; serving http://{host}:{port}/")

    class Handler(BaseHTTPRequestHandler):
        """One page and one array endpoint; everything else is a 404."""

        def do_GET(self) -> None:  # noqa: N802 -- http.server's spelling
            """Answer with the page, one array as raw float32, or a 404."""
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
                return
            if parsed.path.rstrip("/").endswith("array"):
                key = parse_qs(parsed.query).get("key", [""])[0]
                try:
                    values = locator.read(key)
                except KeyError:
                    self.send_error(404, f"no array {key!r}")
                    return
                self._send(values.tobytes(), "application/octet-stream")
                return
            self.send_error(404)

        def _send(self, body: bytes, content_type: str) -> None:
            """Send one complete 200 response."""
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            """Quieter than the default, which logs every array read."""
            if "array" not in (args[0] if args else ""):
                super().log_message(format, *args)

    with ThreadingHTTPServer((host, port), Handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[view] stopped")
