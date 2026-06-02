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

Controls (interactive mode)
    W / Up ............. move forward
    A / D .............. strafe left / right
    Q / E .............. turn left / right
    Space .............. jump
    Mouse / Ctrl ....... attack
    Esc or close ....... stop & save
    (no "backward" — it isn't in the challenge action space)

Output
    bc_data_collection/ep_XXXX.npz  with
        states  : float32 [T, C, 128, 128]  (C=8 RGB+labels+depth+automap), values in [0, 1]
        actions : int64   [T]               (values 0..7)
        rewards : float32 [T]               (extra; BCDataset ignores it)

Usage
    # interactive play + record (run on a machine WITH a display, e.g. the box's own desktop)
    python play_doom.py                 # 1 episode vs 4 bots
    python play_doom.py --bots 1        # gentler start (curriculum)
    python play_doom.py --no-record     # just explore, save nothing

    # validate the data pipeline over SSH WITHOUT a display (vizdoom runs headless):
    python play_doom.py --headless-test 200
    python play_doom.py --check         # assert the latest episode matches the BC contract
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


def save_episode(out: str, states, actions, rewards) -> str | None:
    if not states:
        print("[save] nothing recorded, skipping.")
        return None
    os.makedirs(out, exist_ok=True)
    idx = _next_index(out)
    path = os.path.join(out, f"ep_{idx:04d}.npz")
    states = np.stack(states).astype(np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float32)
    np.savez_compressed(path, states=states, actions=actions, rewards=rewards)
    uniq, counts = np.unique(actions, return_counts=True)
    hist = ", ".join(f"{ACTION_NAMES[int(a)]}={c}" for a, c in zip(uniq, counts))
    print(
        f"[save] {path}\n"
        f"       states {states.shape} {states.dtype} in "
        f"[{states.min():.3f},{states.max():.3f}] | "
        f"actions {actions.shape} | reward sum {rewards.sum():.1f}\n"
        f"       action histogram: {hist}"
    )
    return path


# --------------------------------------------------------------------------- #
# interactive play                                                             #
# --------------------------------------------------------------------------- #
def _frame_to_surface(frame, last_surface):
    """Raw vizdoom screen_buffer (C,H,W) or (H,W) -> pygame Surface (unscaled)."""
    if frame is None:
        return last_surface
    if frame.ndim == 2:  # grayscale -> rgb
        frame = np.stack([frame] * 3, axis=0)
    arr = np.ascontiguousarray(np.transpose(frame, (2, 1, 0)))  # (C,H,W) -> (W,H,3)
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
    pygame.display.set_caption("jku.wad — human play (Esc to stop & save)")
    font = pygame.font.SysFont("monospace", 14, bold=True)
    clock = pygame.time.Clock()

    print(__doc__.split("Output")[0])  # controls blurb
    env = build_env(args.bots, args.timeout, args.grayscale, args.seed)
    running = True
    try:
        for ep in range(args.episodes):
            if not running:
                break
            obs = obs_to_np(env.reset()[0])
            states, actions, rewards = [], [], []
            last_surface = None
            done = False
            print(f"[play] episode {ep + 1}/{args.episodes} — go!")
            while not done and running:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT or (
                        e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE
                    ):
                        running = False
                action = keys_to_action(
                    pygame.key.get_pressed(), pygame.mouse.get_pressed()[0]
                )

                # draw the CURRENT frame (the state the action is taken in)
                player = env.envs[0]
                frame = player._state.screen_buffer if player._state is not None else None
                last_surface = _frame_to_surface(frame, last_surface)
                if last_surface is not None:
                    win.blit(pygame.transform.scale(last_surface, win.get_size()), (0, 0))
                gv = player._game_vars
                lines = [
                    f"tic {len(states):4d}   action {ACTION_NAMES[action]}",
                    f"frags {int(gv.get('FRAGCOUNT', 0))}  health {int(gv.get('HEALTH', 0))}",
                    f"hits {int(gv.get('HITCOUNT', 0))}  taken {int(gv.get('HITS_TAKEN', 0))}",
                ]
                for i, text in enumerate(lines):
                    surf = font.render(text, True, (255, 255, 0))
                    win.blit(surf, (6, 6 + i * 16))
                pygame.display.flip()

                # record (state, action) BEFORE stepping, then advance one tic
                states.append(obs)
                actions.append(action)
                obs_t, rwd, done, _ = env.step(action)
                obs = obs_to_np(obs_t[0])
                rewards.append(float(rwd[0]))
                clock.tick(args.fps)

            if not args.no_record:
                save_episode(args.out, states, actions, rewards)
            else:
                print(f"[play] --no-record: discarded {len(states)} frames.")
    finally:
        env.close()
        pygame.quit()


# --------------------------------------------------------------------------- #
# headless pipeline test (no display) + format check                          #
# --------------------------------------------------------------------------- #
def run_headless_test(args):
    print(f"[headless] stepping {args.headless_test} tics with random actions (no window)...")
    env = build_env(args.bots, args.timeout, args.grayscale, args.seed)
    rng = np.random.default_rng(args.seed)
    n_actions = env.action_space.n
    obs = obs_to_np(env.reset()[0])
    states, actions, rewards = [], [], []
    done = False
    try:
        while not done and len(states) < args.headless_test:
            a = int(rng.integers(0, n_actions))
            states.append(obs)
            actions.append(a)
            obs_t, rwd, done, _ = env.step(a)
            obs = obs_to_np(obs_t[0])
            rewards.append(float(rwd[0]))
    finally:
        env.close()
    save_episode(args.out, states, actions, rewards)


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
    assert 0.0 <= float(s.min()) and float(s.max()) <= 1.0001, (
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
    p.add_argument("--episodes", type=int, default=1, help="episodes to record back-to-back")
    p.add_argument("--fps", type=int, default=35, help="real-time pacing (ticrate is 35)")
    p.add_argument("--seed", type=int, default=None, help="env seed (default: random)")
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
