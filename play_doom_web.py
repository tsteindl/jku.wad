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
    *samples* frames and *sends input when keys change* — so a laggy link makes the
    video choppy, never the game slow. Inefficient but simple.

    Shares its core with `play_doom.py` (`build_env`, `resolve_action`, `save_episode`),
    so recordings are byte-for-byte the same `BCDataset` format:
    `bc_data_collection/ep_XXXX.npz`, states [T,8,128,128] f32, actions [T] i64.

Controls
    W forward · A/D strafe · Q/E turn · Space jump · Mouse-click or Ctrl attack.
    (No "backward" — it isn't in the challenge action space.)

Saving a run
    A "run" is one episode. It is written to bc_data_collection/ when it ends:
      * automatically when the episode times out,
      * when you click "Save run" (saves the current run, starts a fresh one),
      * when you close the tab.

Usage
    python play_doom_web.py                 # then open the printed http://localhost:PORT
    python play_doom_web.py --bots 1        # gentler start
    python play_doom_web.py --port 8123     # if 8000 is taken
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

# Shared core (same env, action mapping and save format as the desktop front-end).
from play_doom import ACTION_NAMES, build_env, obs_to_np, resolve_action, save_episode

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
<div class="keys">W forward &middot; A/D strafe &middot; Q/E turn &middot; Space jump &middot; click / Ctrl attack</div>
<button id="savebtn">Save run</button><button id="stopbtn">Stop &amp; save</button>
<div id="status">click the game, then play (one action registers per frame)</div>
<script>
const KEYMAP = {"w":"forward","a":"left","d":"right","q":"turnleft","e":"turnright",
 " ":"jump","spacebar":"jump","control":"attack"};
const held = new Set();
const img = document.getElementById('screen'), status = document.getElementById('status');
let running = true, prevUrl = null;

function sendInput(){ fetch('/input?keys=' + encodeURIComponent([...held].join(','))); }
function setKey(tok, down){ if(down) held.add(tok); else held.delete(tok); sendInput(); }
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
      status.textContent = 'runs saved: ' + r.headers.get('X-Saved');
      show(await r.blob());
    }catch(err){ running = false; }
    await new Promise(res => setTimeout(res, Math.max(0, 33 - (performance.now()-t0))));  // ~30fps view
  }
  status.textContent = 'stopped - recordings saved to bc_data_collection/.';
}
document.getElementById('savebtn').onclick = () => fetch('/save');
document.getElementById('stopbtn').onclick = () => { fetch('/stop'); running = false; };
addEventListener('beforeunload', () => navigator.sendBeacon('/stop'));
displayLoop();
</script></body></html>"""


class GameSession:
    """A ViZDoom game stepped in real time by a background thread.

    All env / buffer mutation happens in that one thread; HTTP handlers only set an
    atomic action int, request a save via a flag, or read the last published frame.
    """

    def __init__(self, args):
        self.args = args
        self.env = build_env(args.bots, args.timeout, args.grayscale, args.seed)
        self.scale = args.scale
        self.current_action = 0     # set by /input (atomic int write)
        self.last_action = 0
        self.saved_count = 0
        self._save_request = False  # set by /save
        self.stopped = False
        self.active = False         # gate: don't step/record until the browser connects
        self._frame_lock = threading.Lock()
        self._reset_episode()
        self.latest_jpeg = self._render_jpeg()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ---- loop-thread-only helpers -------------------------------------------
    def _reset_episode(self):
        self.obs = obs_to_np(self.env.reset()[0])
        self.states, self.actions, self.rewards = [], [], []

    def _flush_save(self):
        if not self.args.no_record and self.states:
            save_episode(self.args.out, self.states, self.actions, self.rewards)
            self.saved_count += 1
        self.states, self.actions, self.rewards = [], [], []

    def _loop(self):
        period = 1.0 / self.args.fps
        next_t = time.perf_counter()
        while not self.stopped:
            if not self.active:
                time.sleep(0.02)
                next_t = time.perf_counter()
                continue
            if self._save_request:                 # manual "Save run" boundary
                self._save_request = False
                self._flush_save()
                self._reset_episode()
            action = self.current_action
            self.last_action = action
            self.states.append(self.obs)
            self.actions.append(action)
            obs_t, rwd, done, _ = self.env.step(action)
            self.obs = obs_to_np(obs_t[0])
            self.rewards.append(float(rwd[0]))
            jpeg = self._render_jpeg()
            with self._frame_lock:
                self.latest_jpeg = jpeg
            if done:                                 # episode timed out -> save, next run
                self._flush_save()
                if self.args.episodes and self.saved_count >= self.args.episodes:
                    self.stopped = True
                else:
                    self._reset_episode()
            next_t += period
            slack = next_t - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.perf_counter()         # fell behind; resync without piling up

    def _render_jpeg(self):
        player = self.env.envs[0]
        raw = player._state.screen_buffer if player._state is not None else None
        if raw is not None:
            arr = np.stack([raw] * 3, -1) if raw.ndim == 2 else np.transpose(raw, (1, 2, 0))
            img = Image.fromarray(arr.astype(np.uint8), "RGB")
            if self.scale != 1:
                img = img.resize((img.width * self.scale, img.height * self.scale), Image.NEAREST)
        else:
            img = Image.new("RGB", (256 * self.scale, 192 * self.scale))
        gv = player._game_vars
        hud = (
            f"tic {len(self.states):4d}  act {ACTION_NAMES[self.last_action]}\n"
            f"frags {int(gv.get('FRAGCOUNT', 0))}  hp {int(gv.get('HEALTH', 0))}  "
            f"hits {int(gv.get('HITCOUNT', 0))}  taken {int(gv.get('HITS_TAKEN', 0))}"
        )
        ImageDraw.Draw(img).multiline_text((5, 5), hud, fill=(255, 255, 0))
        buf = BytesIO()
        img.save(buf, "JPEG", quality=70)
        return buf.getvalue()

    # ---- HTTP-thread entry points -------------------------------------------
    def set_action(self, tokens):
        self.active = True
        self.current_action = resolve_action(tokens)

    def request_save(self):
        self._save_request = True

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
        self._flush_save()


def make_handler(session: GameSession):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence per-request logging
            pass

        def _jpeg(self, data):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Stopped", "1" if session.stopped else "0")
            self.send_header("X-Saved", str(session.saved_count))
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
            elif route.path == "/save":
                session.request_save()
                self._ok(b"saving")
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
    p.add_argument("--seed", type=int, default=None, help="env seed (default: random)")
    p.add_argument("--out", type=str, default="bc_data_collection", help="output dir")
    p.add_argument("--scale", type=int, default=2, help="frame upscale factor")
    p.add_argument("--grayscale", action="store_true", help="record grayscale screen (6ch)")
    p.add_argument("--no-record", action="store_true", help="play without saving")
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
        f"{'(not recording)' if args.no_record else 'recording -> ' + args.out}\n"
        f"  Ctrl-C here to quit.\n"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        session.env.close()
        httpd.server_close()
        print("server stopped.")


if __name__ == "__main__":
    main()
