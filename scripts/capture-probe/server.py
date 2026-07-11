#!/usr/bin/env python3
"""Capture-invisibility probe for gimme-a-chance's overlays.

Serves a loud striped page and saves ONE getDisplayMedia frame POSTed back by
it. Purpose: verify whether a content-protected overlay window is truly INVISIBLE
(background/stripes show through) or renders as a BLACK rectangle under a given
screen-share capturer. Chrome's getDisplayMedia is the same path Google Meet uses,
so a Chrome "Entire Screen" capture here is a faithful proxy for a Meet full-screen
share — with NO live preview, so there is no infinity-mirror confound.

Run (from WSL or Windows):
    python3 server.py            # serves on http://localhost:8137
Then open http://localhost:8137 in the browser under test, put the overlays over
the stripes, click Capturar, pick "Entire Screen". The frame lands next to this
file as shot_<HHMMSS>.png (+ shot_latest.png). See reference in the repo notes:
the black box is tauri#14189 (hide/show degrades WDA_EXCLUDEFROMCAPTURE→WDA_MONITOR).

Windows reaches a WSL-side server via http://localhost:8137 (WSL2 localhost
forwarding). localhost is a secure context, so getDisplayMedia is allowed.
"""
import http.server
import socketserver
import os
import time

PORT = 8137
HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def do_POST(self):
        if self.path != "/shot":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(n)
        ts = time.strftime("%H%M%S")
        for name in (f"shot_{ts}.png", "shot_latest.png"):
            with open(os.path.join(HERE, name), "wb") as f:
                f.write(data)
        print(f"SAVED shot_{ts}.png ({len(data)} bytes)", flush=True)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    # Threaded: a single-thread server hangs on a Chrome keep-alive connection.
    daemon_threads = True
    allow_reuse_address = True


with Server(("0.0.0.0", PORT), Handler) as httpd:
    print(f"serving (threaded) on 0.0.0.0:{PORT} (dir={HERE})", flush=True)
    httpd.serve_forever()
