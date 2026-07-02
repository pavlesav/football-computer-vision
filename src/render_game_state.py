"""
Render a fully-annotated review video from the persisted game state.

This is the "judge the entire work" artifact: it draws everything the pipeline
believes onto the source footage so a human can verify (or fault) each layer at
a glance — homography overlay, team-coloured player ellipses, the Kalman ball
with its source, live event banners/arrows, an event ticker, a minimap, and a
per-frame trust readout explaining *why* events are or aren't being emitted.

Reads only the artifact (``output/game_state/{slug}/``) + the source video —
no GPU, no YOLO, no PnLCalib. Rendering ~6000 frames takes a few minutes.

Colour language:
  players   team0 = yellow, team1 = blue, other = gray (kit-matched for sut-mla)
  ball      green = detected, cyan-blue = bridged (hindsight), orange = Kalman coast
  overlay   white pitch lines = trusted frame (events active),
            dim red = projection exists but NOT trusted (events paused)

Usage::

    python -m src.render_game_state --match sut-mla
    python -m src.render_game_state --match sut-mla --scale 0.6667   # 720p
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import cv2
import numpy as np

from .config import Config
from .game_state import GameState, P_COLS
from .ball_tracker import track_ball
from .events import detect_events
from .game_state import trusted_frame_mask, adaptive_conf_min

# Team colours (BGR): team0 yellow, team1 blue, other gray. Labels are
# neutral — which club is team0 varies per match (the classifier assigns
# indices arbitrarily), so naming clubs here would mislead half the time.
TEAM_BGR = {0: (0, 215, 255), 1: (255, 130, 60), 2: (170, 170, 170)}
TEAM_NAME = {0: "T0", 1: "T1", 2: "?"}
BALL_BGR = {"detected": (80, 255, 80), "bridged": (255, 200, 0),
            "kalman": (0, 165, 255)}
EVENT_SHOW_AFTER = 30      # frames an event stays highlighted after it fires
EVENT_SHOW_BEFORE = 5

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Pitch-line geometry (natural metres, origin at a corner) ─────────────────

def _pitch_segments(step: float = 1.0) -> list[np.ndarray]:
    """Sampled polylines for the main pitch markings."""
    segs = []

    def line(x0, y0, x1, y1):
        n = max(int(np.hypot(x1 - x0, y1 - y0) / step), 2)
        t = np.linspace(0, 1, n)
        segs.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=1))

    line(0, 0, 105, 0); line(105, 0, 105, 68)
    line(105, 68, 0, 68); line(0, 68, 0, 0)
    line(52.5, 0, 52.5, 68)
    for x0, s in ((0, 1), (105, -1)):                     # penalty + 6yd boxes
        line(x0, 13.84, x0 + s * 16.5, 13.84)
        line(x0, 54.16, x0 + s * 16.5, 54.16)
        line(x0 + s * 16.5, 13.84, x0 + s * 16.5, 54.16)
        line(x0, 24.84, x0 + s * 5.5, 24.84)
        line(x0, 43.16, x0 + s * 5.5, 43.16)
        line(x0 + s * 5.5, 24.84, x0 + s * 5.5, 43.16)
    th = np.linspace(0, 2 * np.pi, 64)
    segs.append(np.stack([52.5 + 9.15 * np.cos(th), 34 + 9.15 * np.sin(th)], axis=1))
    return segs


PITCH_SEGS = _pitch_segments()


def _project(pts_xy: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Nx2 pitch metres -> Nx3 (u, v, valid) image px via the z=0 plane of P."""
    n = len(pts_xy)
    world = np.column_stack([pts_xy[:, 0] - 52.5, pts_xy[:, 1] - 34.0,
                             np.zeros(n), np.ones(n)])
    img = (P @ world.T).T
    valid = img[:, 2] > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = img[:, :2] / img[:, 2:3]
    return np.column_stack([uv, valid])


def draw_pitch_overlay(frame, P, colour, fw, fh):
    for seg in PITCH_SEGS:
        proj = _project(seg, P)
        run = []
        for u, v, ok in proj:
            inside = ok and -200 <= u <= fw + 200 and -200 <= v <= fh + 200
            if inside:
                run.append((int(u), int(v)))
            elif len(run) >= 2:
                cv2.polylines(frame, [np.array(run)], False, colour, 2, cv2.LINE_AA)
                run = []
            else:
                run = []
        if len(run) >= 2:
            cv2.polylines(frame, [np.array(run)], False, colour, 2, cv2.LINE_AA)


def pitch_to_img(x, y, P):
    p = P @ np.array([x - 52.5, y - 34.0, 0.0, 1.0])
    if p[2] <= 1e-6:
        return None
    return int(p[0] / p[2]), int(p[1] / p[2])


# ── Minimap ──────────────────────────────────────────────────────────────────

MM_W, MM_H, MM_PAD = 340, 224, 10


def draw_minimap(frame, players_rows, ball_row, fw, fh):
    x0, y0 = fw - MM_W - 18, fh - MM_H - 18
    mm = frame[y0:y0 + MM_H, x0:x0 + MM_W]
    mm[:] = (mm * 0.25 + np.array((60, 90, 60)) * 0.75).astype(np.uint8)

    def m2p(px, py):
        return (int(MM_PAD + px / 105 * (MM_W - 2 * MM_PAD)),
                int(MM_PAD + py / 68 * (MM_H - 2 * MM_PAD)))

    white = (240, 240, 240)
    cv2.rectangle(mm, m2p(0, 0), m2p(105, 68), white, 1)
    cv2.line(mm, m2p(52.5, 0), m2p(52.5, 68), white, 1)
    cv2.circle(mm, m2p(52.5, 34), int(9.15 / 105 * (MM_W - 2 * MM_PAD)), white, 1)
    for gx, s in ((0, 1), (105, -1)):
        cv2.rectangle(mm, m2p(gx, 13.84), m2p(gx + s * 16.5, 54.16), white, 1)

    for r in players_rows:
        if r["pitch_x"] == r["pitch_x"]:
            cv2.circle(mm, m2p(r["pitch_x"], r["pitch_y"]), 4,
                       TEAM_BGR.get(r["team_id"], TEAM_BGR[2]), -1)
    if ball_row is not None and ball_row["x"] == ball_row["x"]:
        cv2.circle(mm, m2p(ball_row["x"], ball_row["y"]), 4,
                   BALL_BGR.get(ball_row["source"], (255, 255, 255)), -1)
        cv2.circle(mm, m2p(ball_row["x"], ball_row["y"]), 6, (255, 255, 255), 1)
    frame[y0:y0 + MM_H, x0:x0 + MM_W] = mm


# ── Text helpers ─────────────────────────────────────────────────────────────

def _text(frame, s, org, scale=0.62, colour=(255, 255, 255), thick=2):
    cv2.putText(frame, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def _event_label(e) -> str:
    m, s = divmod(int(e.time_sec), 60)
    base = f"{m:02d}:{s:02d} {TEAM_NAME.get(e.team, '?')} {e.type}"
    if e.type == "Pass":
        out = e.details.get("outcome", "")
        base += f" #{e.player}->#{e.details.get('recipient', '?')}" \
            if out == "complete" else f" #{e.player} ({out})"
    elif e.type == "Carry":
        base += f" #{e.player} {e.details.get('length', 0):.0f}m"
    elif e.type == "Shot":
        base += f" #{e.player}!"
    return base


# ── Main render ──────────────────────────────────────────────────────────────

def render(slug: str, out_path=None, scale: float = 1.0,
           start_sec: Optional[float] = None,
           duration_sec: Optional[float] = None,
           period: Optional[int] = None):
    """``start_sec``/``duration_sec`` (relative to the artifact window) render
    just a clip — for quick QC without re-rendering a whole half."""
    gs = GameState.load(slug, period=period)
    ball = track_ball(gs).set_index("frame")
    events, summary = detect_events(gs)
    trusted = dict(zip(gs.frames["frame"].astype(int),
                       trusted_frame_mask(gs).tolist()))

    frames_ix = gs.frames.set_index("frame")
    players_by_frame = defaultdict(list)
    for r in gs.players.to_dict("records"):
        players_by_frame[int(r["frame"])].append(r)
    events_by_frame = defaultdict(list)
    for e in events:
        events_by_frame[int(e.frame)].append(e)

    start = int(gs.meta["start_frame"])
    end = int(gs.meta["end_frame"])
    tag = ""
    if start_sec is not None:
        start = int(gs.meta["start_frame"]) + int(start_sec * gs.fps)
        tag = f"_{int(start_sec)}s"
    if duration_sec is not None:
        end = min(end, start + int(duration_sec * gs.fps))
    cap = cv2.VideoCapture(str(Config.MATCH_VIDEOS[slug]))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ow, oh = int(fw * scale), int(fh * scale)
    if out_path is None:
        out_path = (Config.OUTPUT_DIR / "qc" / slug /
                    f"{slug}_p{gs.period}_annotated_review{tag}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         gs.fps, (ow, oh))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ticker: list = []            # last few event labels
    active: list = []            # (event, expires_at_frame)

    for f in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        row = frames_ix.loc[f] if f in frames_ix.index else None
        is_trusted = trusted.get(f, False)

        # 1. homography overlay
        P = None
        if row is not None and bool(row["has_P"]):
            P = row[P_COLS].to_numpy(dtype=float).reshape(3, 4)
            colour = (255, 255, 255) if is_trusted else (60, 60, 200)
            draw_pitch_overlay(frame, P, colour, fw, fh)

        # 2. players
        prs = players_by_frame.get(f, [])
        for r in prs:
            c = TEAM_BGR.get(r["team_id"], TEAM_BGR[2])
            w = max(r["x2"] - r["x1"], 14)
            cv2.ellipse(frame, ((r["x1"] + r["x2"]) // 2, r["y2"]),
                        (int(w * 0.55), max(int(w * 0.19), 6)),
                        0, -40, 220, c, 2, cv2.LINE_AA)
            _text(frame, str(r["track_id"]),
                  (r["x1"], max(r["y1"] - 6, 14)), 0.45, c, 1)

        # 3. ball (projected from pitch estimate with this frame's P)
        brow = ball.loc[f] if f in ball.index else None
        if brow is not None and brow["x"] == brow["x"] and P is not None:
            pt = pitch_to_img(brow["x"], brow["y"], P)
            if pt is not None:
                c = BALL_BGR.get(brow["source"], (255, 255, 255))
                cv2.circle(frame, pt, 12, c, 3, cv2.LINE_AA)
                cv2.putText(frame, brow["source"][0].upper(),
                            (pt[0] + 15, pt[1] + 5), FONT, 0.5, c, 2)

        # 4. events: activate, draw arrows for active passes/shots
        for e in events_by_frame.get(f + EVENT_SHOW_BEFORE, []):
            active.append((e, f + EVENT_SHOW_BEFORE + EVENT_SHOW_AFTER))
            ticker.append(_event_label(e))
            ticker[:] = ticker[-4:]
        active = [(e, exp) for e, exp in active if f <= exp]
        for e, _ in active:
            if P is None:
                continue
            p0 = pitch_to_img(*e.location, P)
            if e.type in ("Pass", "Shot") and "end_location" in e.details:
                p1 = pitch_to_img(*e.details["end_location"], P)
                if p0 and p1:
                    col = (0, 255, 255) if e.type == "Pass" else (0, 0, 255)
                    cv2.arrowedLine(frame, p0, p1, col, 3, cv2.LINE_AA,
                                    tipLength=0.04)
            if p0:
                cv2.circle(frame, p0, 20, (0, 255, 255), 2, cv2.LINE_AA)

        # 5. status bar + ticker + minimap
        t = float(row["time_sec"]) if row is not None else 0.0
        m, s = divmod(int(t), 60)
        src = row["homog_source"] if row is not None else "-"
        conf = float(row["homog_conf"]) if row is not None else 0.0
        status = f"{m:02d}:{s:02d}  f{f}  homog:{src} conf:{conf:.2f}"
        _text(frame, status, (24, fh - 60), 0.62)
        if is_trusted:
            _text(frame, "TRUSTED - events active", (24, fh - 28), 0.62,
                  (80, 255, 80))
        else:
            why = ("close-up" if row is not None and not bool(row["is_wide_shot"])
                   else "low homography confidence")
            _text(frame, f"NOT TRUSTED ({why}) - events paused",
                  (24, fh - 28), 0.62, (60, 60, 230))
        for i, line in enumerate(reversed(ticker)):
            _text(frame, line, (fw - 620, 40 + 30 * i), 0.6,
                  (255, 255, 255) if i == 0 else (190, 190, 190),
                  2 if i == 0 else 1)
        draw_minimap(frame, prs, brow, fw, fh)

        if scale != 1.0:
            frame = cv2.resize(frame, (ow, oh))
        vw.write(frame)
        if (f - start) % 1000 == 0:
            print(f"  rendered {f - start}/{end - start}")

    cap.release()
    vw.release()
    print(f"saved -> {out_path}")
    print(f"events drawn: {len(events)} | summary possession "
          f"{summary['possession_pct']}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Render annotated review video from a game state")
    ap.add_argument("--match", default="sut-mla")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="output resolution scale (1.0 = source)")
    ap.add_argument("--start_sec", type=float, default=None,
                    help="seconds into the artifact window to start (clip mode)")
    ap.add_argument("--duration_sec", type=float, default=None,
                    help="clip length in seconds")
    ap.add_argument("--half", type=int, default=None, choices=[1, 2],
                    help="period to render (default: the only stored one)")
    args = ap.parse_args()
    render(args.match, scale=args.scale, start_sec=args.start_sec,
           duration_sec=args.duration_sec, period=args.half)


if __name__ == "__main__":
    main()
