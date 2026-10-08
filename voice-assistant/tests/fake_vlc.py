#!/usr/bin/env python3
"""Поддельный VLC для тестов pc_control: HTTP-интерфейс /requests/status.json с паролем.
Аргументы командной строки и команды пишет в $FAKE_VLC_LOG (по строке JSON)."""
import base64, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

args = sys.argv[1:]
opt = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in args if a.startswith("--") and "=" in a}
port, pw = int(opt["--http-port"]), opt["--http-password"]
LOG = os.environ.get("FAKE_VLC_LOG", "/tmp/fake_vlc.log")
state = {"state": "playing", "volume": 256, "fullscreen": False, "time": 0}


def log(rec):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


log({"args": args})


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        want = "Basic " + base64.b64encode(f":{pw}".encode()).decode()
        if self.headers.get("Authorization") != want:
            self.send_response(401); self.end_headers(); return
        q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        c, val = q.get("command"), q.get("val")
        if c:
            log({"cmd": c, "val": val})
            if c == "pl_forcepause": state["state"] = "paused"
            if c == "pl_forceresume": state["state"] = "playing"
            if c == "fullscreen": state["fullscreen"] = not state["fullscreen"]
            if c == "volume":
                v = int(val)
                state["volume"] = state["volume"] + v if val[0] in "+-" else v
            if c == "seek": state["time"] += int(val.rstrip("s"))
        body = json.dumps(state).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


HTTPServer(("127.0.0.1", port), H).serve_forever()
