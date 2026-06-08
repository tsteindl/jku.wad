#!/usr/bin/env python
"""Play Doom yourself and record demonstrations for imitation learning.

This is a thin, real-time front-end around the *exact* training environment
(`doom_arena.VizdoomMPEnv`). You play with the keyboard; every tic the env is
stepped regardless of input (no key held -> NOOP), so the dynamics match what
the agent experiences during rollouts. The states we save are the env's own
transformed observations, so they line up byte-for-byte with what the agent is
fed and with the notebook's `BCDataset`.

Action space is `Discrete(8)` -> exactly ONE button per frame (the agent cannot
move+shoot at once), so held keys are resolved to a single action by priority.

Controls (interactive)
    Move / aim .. W forward · A/D strafe · Q/E turn · Space jump · Mouse/Ctrl/Shift attack
    T ........... toggle recording on/off  (REC <-> EXPLORE)
    N ........... save this run (if recording) and start a NEW one
    Backspace ... discard this run (don't save) and quit
    Esc / close . save (if recording) and quit
    (no "backward" — it isn't in the challenge action space)

Recording & seeds
    Runs are saved only when recording is ON. Start in explore mode with
    --no-record, or toggle live with R. Each run uses a seed (shown on the HUD,
    printed, and stored in the .npz) so you can skip "bot-in-your-face" spawns and
    keep the good ones:
        --seed N         seeds N, N+1, N+2, ... (reproducible, one per run)
        --seeds 5,12,37  cycle through exactly these seeds
        (default)        a fresh random seed per run

Output
    bc_data_collection/ep_XXXX.npz  with
        states  : float32 [T, C, 128, 128]  (C=8 RGB+labels+depth+automap), in [0, 1]
        actions : int64   [T]               (values 0..7)
        rewards : float32 [T]               (extra; BCDataset ignores it)
        seed    : int64                     (env seed for this run)
        num_bots: int64                     (bots this run)

Usage
    python play_doom.py                 # record vs 4 bots (R to explore, N for a new run)
    python play_doom.py --bots 1        # gentler start (curriculum)
    python play_doom.py --no-record     # just explore, save nothing
    python play_doom.py --seeds 7,19    # only use these seeds
    python play_doom.py --check         # assert the latest episode matches the BC contract
    python play_doom.py --headless-test 200   # no display: pipeline self-check over SSH
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pygame  # safe to import headless; only display.set_mode() needs a display

# ViZDoom RES_256X192 raw screen size (width, height) for display.
RAW_W, RAW_H = 256, 192

ACTION_NAMES = {
    0: "NOOP",
    1: "MOVE_FORWARD",
    2: "ATTACK",
    3: "MOVE_LEFT",
    4: "MOVE_RIGHT",
    5: "TURN_LEFT",
    6: "TURN_RIGHT",
    7: "JUMP",
}

# Conflict-resolution priority when several inputs are held at once (the agent may
# press only ONE button per tic). Shared by every front-end (desktop + web).
PRIORITY = [2, 7, 5, 6, 3, 4, 1]  # attack, jump, turn-L/R, strafe-L/R, forward
# logical input token -> discrete action
TOKEN_TO_ACTION = {
    "attack": 2,
    "jump": 7,
    "turnleft": 5,
    "turnright": 6,
    "left": 3,   # strafe
    "right": 4,  # strafe
    "forward": 1,
}


def resolve_action(tokens) -> int:
    """Resolve held logical input tokens (e.g. {'attack','forward'}) to one action."""
    active = {TOKEN_TO_ACTION[t] for t in tokens if t in TOKEN_TO_ACTION}
    for a in PRIORITY:
        if a in active:
            return a
    return 0  # NOOP


# physical pygame key -> logical token (desktop front-end bindings)
PYGAME_KEYS = {
    pygame.K_w: "forward", pygame.K_UP: "forward",
    pygame.K_a: "left",          # strafe
    pygame.K_d: "right",         # strafe
    pygame.K_q: "turnleft",
    pygame.K_e: "turnright",
    pygame.K_SPACE: "jump",
    pygame.K_LCTRL: "attack", pygame.K_RCTRL: "attack",
    pygame.K_LSHIFT: "attack", pygame.K_RSHIFT: "attack",
    # NOTE: no MOVE_BACKWARD button exists in the challenge action space, so S is unbound.
}


def keys_to_action(pressed, mouse_attack: bool = False) -> int:
    """Resolve held pygame keys (+ optional mouse attack) to one discrete action (0..7)."""
    tokens = [tok for k, tok in PYGAME_KEYS.items() if pressed[k]]
    if mouse_attack:
        tokens.append("attack")
    return resolve_action(tokens)


def build_env(bots, timeout, grayscale=False, seed=None):
    """Construct the training env with ALL buffers enabled (8-channel obs).

    Mirrors the notebook PLAYER_CONFIG (hud=none, crosshair, RGB screen) so the
    recorded states match what the agent sees. We enable labels+depth+automap so
    recordings are a superset; the BC loader selects channels per BC_TRAIN_BUFFERS.
    Shared by the desktop (pygame) and web front-ends.
    """
    from doom_arena import VizdoomMPEnv
    from doom_arena.player import ObsBuffer

    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    env = VizdoomMPEnv(
        num_players=1,
        num_bots=bots,
        bot_skill=0,
        doom_map="ROOM",
        episode_timeout=timeout,
        n_stack_frames=1,
        extra_state=[ObsBuffer.LABELS, ObsBuffer.DEPTH, ObsBuffer.AUTOMAP],
        crosshair=True,
        hud="none",
        screen_format=8 if grayscale else 0,  # 8=GRAY8, 0=CRCGCB (RGB)
        seed=seed,
    )
    return env


def obs_to_np(obs) -> np.ndarray:
    """Env obs (a CPU torch tensor) -> float32 numpy array, no torch import needed."""
    if hasattr(obs, "detach"):
        obs = obs.detach().cpu().numpy()
    return np.asarray(obs, dtype=np.float32)


# --------------------------------------------------------------------------- #
# saving                                                                       #
# --------------------------------------------------------------------------- #
def _next_index(out: str) -> int:
    nums = []
    for f in glob.glob(os.path.join(out, "ep_*.npz")):
        m = re.search(r"ep_(\d+)\.npz$", os.path.basename(f))
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 0


def save_episode(out, states, actions, rewards, seed=None, num_bots=None, stats=None) -> "str | None":
    if not states:
        print("[save] nothing recorded, skipping.")
        return None
    os.makedirs(out, exist_ok=True)
    idx = _next_index(out)
    path = os.path.join(out, f"ep_{idx:04d}.npz")
    states = np.stack(states).astype(np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float32)
    extra = {}
    if seed is not None:
        extra["seed"] = np.int64(seed)
    if num_bots is not None:
        extra["num_bots"] = np.int64(num_bots)
    if stats:
        extra.update({k: np.int64(v) for k, v in stats.items()})
    np.savez_compressed(path, states=states, actions=actions, rewards=rewards, **extra)
    uniq, counts = np.unique(actions, return_counts=True)
    hist = ", ".join(f"{ACTION_NAMES[int(a)]}={c}" for a, c in zip(uniq, counts))
    reward_line = f"reward {rewards.sum():+.1f}"
    if stats:
        reward_line += (
            f"   |   frags {stats.get('frags', 0)}  hits {stats.get('hits', 0)}"
            f"  taken {stats.get('hits_taken', 0)}  deaths {stats.get('deaths', 0)}"
        )
    print(
        f"[save] {path}  (seed={seed}, {len(actions)} frames)\n"
        f"       {reward_line}\n"
        f"       action histogram: {hist}"
    )
    return path


def make_seed_gen(args):
    """Yield one env seed per run: an explicit list, increasing from --seed, or random."""
    if getattr(args, "seeds", None):
        seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]

        def gen():
            i = 0
            while True:
                yield seeds[i % len(seeds)]
                i += 1

        return gen()
    if args.seed is not None:

        def gen():
            s = args.seed
            while True:
                yield s
                s += 1

        return gen()

    rng = np.random.default_rng()

    def gen():
        while True:
            yield int(rng.integers(0, 2**31 - 1))

    return gen()


class EpisodeRecorder:
    """Owns the env, the current run's buffers, recording mode and per-run seeds.

    Used single-threaded by both front-ends (the pygame loop and the web server's
    background loop), so it needs no locks — web HTTP handlers only set flags that
    the loop forwards to these methods.
    """

    def __init__(self, env, out, seed_gen, recording=True, num_bots=None):
        self.env = env
        self.out = out
        self.seed_gen = seed_gen
        self.recording = recording
        self.num_bots = num_bots
        self.saved = 0
        self.run_index = 0
        self.seed = None
        self.states, self.actions, self.rewards = [], [], []
        self.obs = None
        self.start_new()

    def start_new(self):
        """Begin a fresh run with the next seed (re-inits the game with that seed)."""
        self.seed = next(self.seed_gen)
        self.env.envs[0].doom_seed = self.seed  # applied by the next reset()
        self.obs = obs_to_np(self.env.reset()[0])
        self.states, self.actions, self.rewards = [], [], []
        self.reward_sum = 0.0
        self.run_index += 1
        print(f"[run {self.run_index}] seed={self.seed}  mode={'REC' if self.recording else 'EXPLORE'}")

    def step(self, action):
        """Record (state, action) then advance one tic. Returns `done`."""
        self.states.append(self.obs)
        self.actions.append(action)
        obs_t, rwd, done, _ = self.env.step(action)
        self.obs = obs_to_np(obs_t[0])
        r = float(rwd[0])
        self.rewards.append(r)
        self.reward_sum += r
        return done

    def episode_stats(self):
        """Cumulative episode counters from the game (frags/hits/taken/deaths)."""
        gv = self.env.envs[0]._game_vars
        return {
            "frags": int(gv.get("FRAGCOUNT", 0)),
            "hits": int(gv.get("HITCOUNT", 0)),
            "hits_taken": int(gv.get("HITS_TAKEN", 0)),
            "deaths": int(gv.get("DEATHCOUNT", 0)),
        }

    def save_current(self):
        """Write the current run iff recording and non-empty. Returns whether it saved."""
        if self.recording and self.states:
            save_episode(self.out, self.states, self.actions, self.rewards,
                         seed=self.seed, num_bots=self.num_bots, stats=self.episode_stats())
            self.saved += 1
            return True
        return False

    def next_run(self, save):
        """Finish the current run (saving iff `save`), then start a fresh one."""
        if save:
            self.save_current()
        elif self.states:
            print(f"[run {self.run_index}] discarded {len(self.states)} frames (seed={self.seed})")
        self.start_new()

    def toggle_recording(self):
        self.recording = not self.recording
        print(f"[mode] {'REC' if self.recording else 'EXPLORE'}")
        return self.recording


# --------------------------------------------------------------------------- #
# interactive play                                                             #
# --------------------------------------------------------------------------- #
def _frame_to_surface(frame, last_surface):
    """RGB frame (3,H,W) uint8 -> pygame Surface (unscaled)."""
    if frame is None:
        return last_surface
    arr = np.ascontiguousarray(np.transpose(frame, (2, 1, 0)))  # (3,H,W) -> (W,H,3)
    return pygame.surfarray.make_surface(arr)


def run_interactive(args):
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print(
            "No display found (DISPLAY / WAYLAND_DISPLAY unset).\n"
            "Run this on the machine's own desktop (log into GNOME and open a terminal),\n"
            "or validate the data pipeline without a window via:\n"
            "    python play_doom.py --headless-test 200 && python play_doom.py --check",
            file=sys.stderr,
        )
        sys.exit(2)

    pygame.init()
    scale = args.scale
    win = pygame.display.set_mode((RAW_W * scale, RAW_H * scale))
    pygame.display.set_caption("jku.wad — human play")
    font = pygame.font.SysFont("monospace", 14, bold=True)
    clock = pygame.time.Clock()

    print(__doc__.split("Output")[0])  # controls blurb
    env = build_env(args.bots, args.timeout, args.grayscale)
    rec = EpisodeRecorder(
        env, args.out, make_seed_gen(args),
        recording=not args.no_record, num_bots=args.bots,
    )
    last_surface = None
    running = True
    try:
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_n:          # save (if recording) + new run
                        rec.next_run(save=True)
                        last_surface = None
                    elif e.key == pygame.K_BACKSPACE:  # discard + quit
                        rec.next_run(save=False)
                        running = False
                    elif e.key == pygame.K_t:          # toggle REC <-> EXPLORE
                        rec.toggle_recording()
            if not running:
                break

            action = keys_to_action(
                pygame.key.get_pressed(), pygame.mouse.get_pressed()[0]
            )

            # draw the CURRENT frame (the state the action is taken in)
            player = env.envs[0]
            screen = rec.obs[:1] if args.grayscale else rec.obs[:3]  # float32 [0,1], (C,H,W)
            if args.grayscale:
                screen = np.repeat(screen, 3, axis=0)  # (1,H,W) -> (3,H,W)
            frame = (screen * 255).clip(0, 255).astype(np.uint8)
            last_surface = _frame_to_surface(frame, last_surface)
            if last_surface is not None:
                win.blit(pygame.transform.scale(last_surface, win.get_size()), (0, 0))
            gv = player._game_vars
            mode = "REC ●" if rec.recording else "EXPLORE"
            lines = [
                f"{mode}   run {rec.run_index}  seed {rec.seed}  saved {rec.saved}",
                f"tic {len(rec.states):4d}   action {ACTION_NAMES[action]}   reward {rec.reward_sum:+.0f}",
                f"frags {int(gv.get('FRAGCOUNT', 0))}  hits {int(gv.get('HITCOUNT', 0))}"
                f"  taken {int(gv.get('HITS_TAKEN', 0))}  hp {int(gv.get('HEALTH', 0))}",
                "T record  N save+new  Bksp discard+quit  Esc save+quit",
            ]
            for i, text in enumerate(lines):
                col = (255, 90, 90) if (i == 0 and rec.recording) else (255, 255, 0)
                win.blit(font.render(text, True, col), (6, 6 + i * 16))
            pygame.display.flip()

            if rec.step(action):                       # episode timed out
                rec.next_run(save=True)                # auto-save iff recording
                last_surface = None
            clock.tick(args.fps)

            if args.episodes and rec.saved >= args.episodes:
                running = False
    finally:
        rec.save_current()  # keep the in-progress run if recording
        env.close()
        pygame.quit()


# --------------------------------------------------------------------------- #
# headless pipeline test (no display) + format check                          #
# --------------------------------------------------------------------------- #
def run_headless_test(args):
    print(f"[headless] stepping {args.headless_test} tics with random actions (no window)...")
    env = build_env(args.bots, args.timeout, args.grayscale)
    rec = EpisodeRecorder(
        env, args.out, make_seed_gen(args),
        recording=not args.no_record, num_bots=args.bots,
    )
    rng = np.random.default_rng(args.seed)
    n_actions = env.action_space.n
    try:
        for _ in range(args.headless_test):
            if rec.step(int(rng.integers(0, n_actions))):
                break
    finally:
        rec.save_current()
        env.close()


def run_check(out, check_path):
    path = check_path
    if path in (None, "__LATEST__"):
        files = sorted(glob.glob(os.path.join(out, "ep_*.npz")))
        if not files:
            print(f"[check] no ep_*.npz in {out!r}", file=sys.stderr)
            sys.exit(1)
        path = files[-1]

    d = np.load(path)
    assert "states" in d and "actions" in d, "missing 'states'/'actions' keys"
    s = d["states"].astype(np.float32)
    a = d["actions"].astype(np.int64)

    assert s.ndim == 4, f"states must be [T,C,H,W], got {s.shape}"
    T, C, H, W = s.shape
    assert (H, W) == (128, 128), f"expected 128x128, got {H}x{W}"
    assert C in (6, 8), f"expected 6 (gray) or 8 (rgb) channels, got {C}"
    assert a.ndim == 1 and a.shape[0] == T, f"actions must be [T], got {a.shape} vs T={T}"
    assert 0.0 <= float(s.minF()) and float(s.max()) <= 1.0001, (
        f"states must be normalised to [0,1], got [{s.min():.3f},{s.max():.3f}]"
    )
    assert int(a.min()) >= 0 and int(a.max()) <= 7, (
        f"actions must be in 0..7, got [{a.min()},{a.max()}]"
    )

    print(
        f"[check] OK  {path}\n"
        f"        states {s.shape} {s.dtype} in [{s.min():.3f},{s.max():.3f}]\n"
        f"        actions {a.shape} {a.dtype} in [{a.min()},{a.max()}]  "
        f"({'RGB' if C == 8 else 'GRAY'}+labels+depth+automap)\n"
        f"        -> matches BCDataset contract (states/actions); extra keys: "
        f"{[k for k in d.files if k not in ('states', 'actions')]}"
    )


def main():
    p = argparse.ArgumentParser(description="Play Doom and record BC demonstrations.")
    p.add_argument("--bots", type=int, default=4, help="number of bots (1 for a gentle start)")
    p.add_argument("--timeout", type=int, default=2000, help="episode length in tics")
    p.add_argument("--episodes", type=int, default=0, help="auto-quit after N saved runs (0 = unlimited)")
    p.add_argument("--fps", type=int, default=35, help="real-time pacing (ticrate is 35)")
    p.add_argument("--seed", type=int, default=None, help="first seed; then N+1, N+2, ... (default: random per run)")
    p.add_argument("--seeds", type=str, default=None, help="comma-separated seeds to cycle, e.g. 7,19,42 (overrides --seed)")
    p.add_argument("--out", type=str, default="bc_data_collection", help="output dir")
    p.add_argument("--scale", type=int, default=3, help="window upscale factor")
    p.add_argument("--grayscale", action="store_true", help="record grayscale screen (6ch)")
    p.add_argument("--no-record", action="store_true", help="play without saving")
    p.add_argument(
        "--headless-test", type=int, default=0, metavar="N",
        help="no window: step N tics with random actions and save (pipeline check over SSH)",
    )
    p.add_argument(
        "--check", nargs="?", const="__LATEST__", default=None, metavar="PATH",
        help="validate an episode against the BC contract (default: latest in --out)",
    )
    args = p.parse_args()

    # Run from the repo root so relative paths (scenario cfg, bots cfg) resolve.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.check is not None:
        run_check(args.out, args.check)
    elif args.headless_test > 0:
        run_headless_test(args)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()
