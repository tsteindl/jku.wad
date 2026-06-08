#!/usr/bin/env python
"""Play Doom in your browser, over SSH — records demos identically to play_doom.py.

Why this exists
    A live pygame window can't travel through a plain SSH / VS Code Remote session.
    This runs a tiny **headless** web server on the box that streams the game to a
    browser and reads your input back. It listens on localhost, so **VS Code forwards
    the port automatically** — you play "over VS Code SSH like you're working now",
    nothing to install on the laptop.

Real-time, not slow-motion
    The game runs in a **background thread at a fixed tic rate on the server** and
    records every tic (frame_skip=1, matching the eval server). The browser only
    *samples* frames and *sends input when keys change*, so a laggy link makes the
    video choppy, never the game slow.

    Shares its core with `play_doom.py` (`EpisodeRecorder`, `build_env`, ...), so the
    recording format, controls, recording toggle, discard and per-run seeds all match.

Controls
    W forward · A/D strafe · Q/E turn · Space jump · Mouse-click / Ctrl / Shift attack.
    Buttons: "Record/Explore" toggle saving · "Save run" · "Discard run" · "Stop".

Usage
    python play_doom_web.py                 # then open the printed http://localhost:PORT
    python play_doom_web.py --bots 1        # gentler start
    python play_doom_web.py --no-record     # start in explore mode (save nothing)
"""

import argparse
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image, ImageDraw

# Shared core (same env, action mapping, recording/seed logic and save format).
from play_doom import (
    ACTION_NAMES, EpisodeRecorder, build_env, make_seed_gen, resolve_action,
)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>jku.wad - browser play</title>
<style>
 body{background:#111;color:#eee;font-family:monospace;text-align:center;margin:0;padding:12px}
 img{image-rendering:pixelated;border:1px solid #333;max-width:95vw}
 #status{margin-top:8px;color:#ff0}
 button{font-family:monospace;padding:6px 12px;margin:8px 4px;cursor:pointer}
 .keys{color:#9cf;margin-top:6px}
</style></head><body>
<div><img id="screen" alt="game" draggable="false"></div>
<div class="keys">W forward &middot; A/D strafe &middot; Q/E turn &middot; Space jump &middot; click / Ctrl / Shift attack</div>
<div>
 <button id="recbtn">Record/Explore</button>
 <button id="savebtn">Save run</button>
 <button id="discardbtn">Discard run</button>
 <button id="stopbtn">Stop</button>
</div>
<div id="status">click the game, then play (one action registers per frame)</div>
<script>
const KEYMAP = {"w":"forward","a":"left","d":"right","q":"turnleft","e":"turnright",
 " ":"jump","spacebar":"jump","control":"attack","shift":"attack"};
const held = new Set();
const img = document.getElementById('screen'), status = document.getElementById('status');
let running = true, prevUrl = null;

function sendInput(){ fetch('/input?keys=' + encodeURIComponent([...held].join(','))); }
function setKey(t, down){ if(down) held.add(t); else held.delete(t); sendInput(); }
addEventListener('keydown', e=>{ const t=KEYMAP[e.key.toLowerCase()]; if(t){ e.preventDefault(); if(!e.repeat) setKey(t,true); }});
addEventListener('keyup',   e=>{ const t=KEYMAP[e.key.toLowerCase()]; if(t){ e.preventDefault(); setKey(t,false); }});
img.addEventListener('mousedown', e=>{ e.preventDefault(); setKey('attack',true); });
addEventListener('mouseup',       e=>{ if(held.has('attack')) setKey('attack',false); });
img.addEventListener('contextmenu', e=>e.preventDefault());

function show(blob){ const u=URL.createObjectURL(blob); img.src=u; if(prevUrl) URL.revokeObjectURL(prevUrl); prevUrl=u; }
async function displayLoop(){
  while(running){
    const t0 = performance.now();
    try{
      const r = await fetch('/frame');
      if(r.headers.get('X-Stopped') === '1') running = false;
      status.textContent = r.headers.get('X-Mode') + '  |  run ' + r.headers.get('X-Run')
        + '  seed ' + r.headers.get('X-Seed') + '  |  saved ' + r.headers.get('X-Saved');
      show(await r.blob());
    }catch(err){ running = false; }
    await new Promise(res => setTimeout(res, Math.max(0, 33 - (performance.now()-t0))));  // ~30fps view
  }
  status.textContent = 'stopped - recordings saved to bc_data_collection/.';
}
document.getElementById('recbtn').onclick     = () => fetch('/toggle');
document.getElementById('savebtn').onclick    = () => fetch('/save');
document.getElementById('discardbtn').onclick = () => fetch('/discard');
document.getElementById('stopbtn').onclick    = () => { fetch('/stop'); running = false; };
addEventListener('beforeunload', () => navigator.sendBeacon('/stop'));
displayLoop();
</script></body></html>"""


class GameSession:
    """Drives an EpisodeRecorder in a real-time thread; HTTP handlers only set flags."""

    def __init__(self, args):
        self.args = args
        env = build_env(args.bots, args.timeout, args.grayscale)
        self.rec = EpisodeRecorder(
            env, args.out, make_seed_gen(args),
            recording=not args.no_record, num_bots=args.bots,
        )
        self.scale = args.scale
        self.current_action = 0
        self.stopped = False
        self.active = False              # don't step/record until the browser connects
        self._toggle = self._save = self._discard = False
        self._frame_lock = threading.Lock()
        self.latest_jpeg = self._render_jpeg()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        period = 1.0 / self.args.fps
        next_t = time.perf_counter()
        while not self.stopped:
            if not self.active:
                time.sleep(0.02)
                next_t = time.perf_counter()
                continue
            if self._toggle:
                self._toggle = False
                self.rec.toggle_recording()
            if self._save:
                self._save = False
                self.rec.next_run(save=True)
                self._publish()
            if self._discard:
                self._discard = False
                self.rec.next_run(save=False)
                self._publish()
            done = self.rec.step(self.current_action)
            self._publish()
            if done:                                 # episode timed out
                self.rec.next_run(save=True)
                if self.args.episodes and self.rec.saved >= self.args.episodes:
                    self.stopped = True
                self._publish()
            next_t += period
            slack = next_t - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.perf_counter()

    def _publish(self):
        jpeg = self._render_jpeg()
        with self._frame_lock:
            self.latest_jpeg = jpeg

    def _render_jpeg(self):
        player = self.rec.env.envs[0]
        raw = player._state.screen_buffer if player._state is not None else None
        if raw is not None:
            arr = np.stack([raw] * 3, -1) if raw.ndim == 2 else np.transpose(raw, (1, 2, 0))
            img = Image.fromarray(arr.astype(np.uint8), "RGB")
            if self.scale != 1:
                img = img.resize((img.width * self.scale, img.height * self.scale), Image.NEAREST)
        else:
            img = Image.new("RGB", (256 * self.scale, 192 * self.scale))
        gv = player._game_vars
        mode = "REC" if self.rec.recording else "EXPLORE"
        hud = (
            f"{mode}  run {self.rec.run_index}  seed {self.rec.seed}  saved {self.rec.saved}\n"
            f"tic {len(self.rec.states):4d}  act {ACTION_NAMES[self.current_action]}  "
            f"reward {self.rec.reward_sum:+.0f}  frags {int(gv.get('FRAGCOUNT', 0))}  "
            f"hits {int(gv.get('HITCOUNT', 0))}  taken {int(gv.get('HITS_TAKEN', 0))}  "
            f"hp {int(gv.get('HEALTH', 0))}"
        )
        ImageDraw.Draw(img).multiline_text((5, 5), hud, fill=(255, 255, 0))
        buf = BytesIO()
        img.save(buf, "JPEG", quality=70)
        return buf.getvalue()

    # ---- HTTP-thread entry points (flags only; the loop does the work) -------
    def set_action(self, tokens):
        self.active = True
        self.current_action = resolve_action(tokens)

    def frame(self):
        self.active = True
        with self._frame_lock:
            return self.latest_jpeg

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=2.0)
        self.rec.save_current()


def make_handler(session: GameSession):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _jpeg(self, data):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Stopped", "1" if session.stopped else "0")
            self.send_header("X-Saved", str(session.rec.saved))
            self.send_header("X-Run", str(session.rec.run_index))
            self.send_header("X-Seed", str(session.rec.seed))
            self.send_header("X-Mode", "REC" if session.rec.recording else "EXPLORE")
            self.end_headers()
            self.wfile.write(data)

        def _ok(self, body=b""):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path)
            if route.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif route.path == "/input":
                q = parse_qs(route.query).get("keys", [""])[0]
                session.set_action([t for t in q.split(",") if t])
                self._ok()
            elif route.path == "/frame":
                self._jpeg(session.frame())
            elif route.path == "/toggle":
                session._toggle = True
                self._ok(b"toggled")
            elif route.path == "/save":
                session._save = True
                self._ok(b"saving")
            elif route.path == "/discard":
                session._discard = True
                self._ok(b"discarded")
            elif route.path == "/stop":
                session.stop()
                self._ok(b"stopped")
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):  # navigator.sendBeacon('/stop') on tab close
            if urlparse(self.path).path == "/stop":
                session.stop()
            self._ok()

    return Handler


def main():
    p = argparse.ArgumentParser(description="Play Doom in the browser and record BC demos.")
    p.add_argument("--bots", type=int, default=4, help="number of bots (1 for a gentle start)")
    p.add_argument("--timeout", type=int, default=2000, help="episode length in tics")
    p.add_argument("--episodes", type=int, default=0, help="auto-stop after N saved runs (0 = unlimited)")
    p.add_argument("--fps", type=int, default=35, help="server tic rate (ViZDoom ticrate is 35)")
    p.add_argument("--seed", type=int, default=None, help="first seed; then N+1, ... (default: random per run)")
    p.add_argument("--seeds", type=str, default=None, help="comma-separated seeds to cycle (overrides --seed)")
    p.add_argument("--out", type=str, default="bc_data_collection", help="output dir")
    p.add_argument("--scale", type=int, default=2, help="frame upscale factor")
    p.add_argument("--grayscale", action="store_true", help="record grayscale screen (6ch)")
    p.add_argument("--no-record", action="store_true", help="start in explore mode (save nothing)")
    p.add_argument("--port", type=int, default=8000, help="localhost port (VS Code forwards it)")
    p.add_argument("--host", type=str, default="127.0.0.1", help="bind address")
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve scenario/bots cfg paths
    session = GameSession(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    print(
        f"\n  Doom web player ready - open  http://localhost:{args.port}  in your laptop browser.\n"
        f"  (VS Code Remote forwards this port automatically; see the PORTS tab if no toast.)\n"
        f"  bots={args.bots} timeout={args.timeout} fps={args.fps} "
        f"start={'EXPLORE' if args.no_record else 'REC'} out={args.out}\n"
        f"  Ctrl-C here to quit.\n"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        session.rec.env.close()
        httpd.server_close()
        print("server stopped.")


if __name__ == "__main__":
    main()
