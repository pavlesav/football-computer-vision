"""
Scoreboard goal-oracle: read the broadcast score graphics and derive *certain*
goals from score changes.

Why: rule-based shot detection from tracking data is unreliable (the 7 shots it
once produced were all nearest-goal-guess artifacts, and it now yields 0 until
validated), but the broadcast itself tells us the score. A score change on the
graphics is a goal with certainty; the only question is *when exactly* — the
answer is "between the last reading of the old score and the first reading of
the new one", which this module narrows as far as the graphics allow.

1.CFL broadcasts (sut-mla verified) show two score graphics:

* **Top-left scoreboard** — intermittent (appears for stretches, then hides):
  ``MM:SS | SUT 1 0 MLD``. Gives (match clock, score) pairs. Lives inside
  ``Config.CLOCK_SEARCH_ROI`` — the same region the period detector OCRs.
* **Bottom banner** — event-driven (kickoff / goal / halftime / full-time):
  red caption bar + black score bar ``FK SUTJESKA 1 | 0 OFK MLADOST``. A goal
  banner appears within seconds of the goal, so its first frame is the best
  time anchor we have.

Pipeline: one sequential sweep flags frames where either graphic is present
(cheap colour tests, no OCR); easyocr (CPU) then reads score digits only on
flagged frames; readings merge into a monotonic score timeline whose steps are
the goals. Output: ``output/events/{slug}_goal_oracle.json`` with the goal
list (bracketed frames, period, period-relative time) + every reading kept as
an audit trail.

Team mapping: the graphics identify scorers as home/away (home is always the
left/first team on 1.CFL graphics and in our slugs), but classifier team ids
are arbitrary per match — pass ``--home_team`` (0 or 1). For sut-mla the
kit-hue audit fixed team1 = Sutjeska (blue, home), team0 = Mladost (yellow).

Usage::

    python -m src.score_ocr --match sut-mla --home_team 1
    python -m src.score_ocr --match sut-mla --home_team 1 --no_cache
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import Config

# ── Graphic geometry (1920x1080 1.CFL broadcasts, sut-mla layout) ────────────
# Bottom banner: red caption bar over a black score bar, horizontally centred.
BANNER_RED_ROI = (400, 840, 1500, 1000)     # x1, y1, x2, y2 — presence check
BANNER_OCR_ROI = (200, 840, 1750, 1010)     # caption + score bar, for OCR
BANNER_RED_MIN = 0.10                       # red fraction ⇒ banner present

# Top-left scoreboard. Two layout families across 1.CFL broadcasts:
#   * score as one text token in a dark box  — "JED 0-0 ARS 28:10"
#     (jed-ars, jez-jed, dec-mla, mor-bud): generic OCR reads it directly.
#   * score as two digits in separate red boxes — "25:24 PET [0][1] MOR"
#     (sut-mla, pet-mor): generic OCR misses the white-on-red digits, so the
#     red boxes are auto-located and re-read with a digit allowlist.
# Visibility is intermittent on every match, so the scoreboard is sampled on
# a fixed cadence across the whole video rather than gated by a presence
# test (a reading either parses or it doesn't).
SB_RED_MIN_AREA = 500    # px^2 — smallest credible score box component
SB_RED_MAX_W = 120       # px — largest credible score box width

SWEEP_STEP = 12          # frames between banner sweep samples (~0.5 s @ 25fps)
OCR_PER_RUN = 3          # banner frames OCR'd per contiguous run
SB_OCR_EVERY_S = 10.0    # seconds between scoreboard OCR samples


def _red_fraction(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 168))
           & (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 60))
    return float(red.mean())


# ── Step 1: sweep for graphic-present frames ─────────────────────────────────

def sweep_graphics(video_path: str, step: int = SWEEP_STEP,
                   start: int = 0, end: Optional[int] = None) -> dict:
    """Sequential pass over the video (grab-only between samples — fast).
    Returns {"banner": [(frame, red_frac)...]} for samples where the bottom
    banner's red bar is present."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(end or total, total)
    bx1, by1, bx2, by2 = BANNER_RED_ROI

    banner = []
    f = start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    while f < end:
        ok = cap.grab()
        if not ok:
            break
        if (f - start) % step == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            rb = _red_fraction(img[by1:by2, bx1:bx2])
            if rb >= BANNER_RED_MIN:
                banner.append((f, round(rb, 3)))
        f += 1
    cap.release()
    return {"banner": banner, "step": step, "start": start, "end": end}


def _runs(samples: list, step: int, max_gap_steps: int = 3) -> list:
    """Group flagged sample frames into contiguous runs [(f0, f1), ...]."""
    runs = []
    for f, _ in samples:
        if runs and f - runs[-1][1] <= step * max_gap_steps:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    return [tuple(r) for r in runs]


# ── Step 2: OCR flagged frames ───────────────────────────────────────────────

_READER = None


def _reader():
    """easyocr on CPU: the GPU belongs to perception runs, and a handful of
    small crops doesn't need it."""
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _READER


_CLOCK_RE = re.compile(r"^(\d{1,3})[:.;](\d{2})$")


def _read_clock(img: np.ndarray) -> Optional[int]:
    """Match clock (seconds) from CLOCK_SEARCH_ROI, or None."""
    for text, _ in _ocr_tokens(img, Config.CLOCK_SEARCH_ROI):
        t = text.strip().replace(" ", "")
        m = _CLOCK_RE.match(t)
        if m and int(m.group(2)) < 60:
            return int(m.group(1)) * 60 + int(m.group(2))
    return None


_SCORE_TOKEN_RE = re.compile(r"^(\d{1,2})\s*[-:|.]\s*(\d{1,2})$")


def _locate_red_boxes(img: np.ndarray) -> Optional[tuple]:
    """Bounding region of the red score boxes inside CLOCK_SEARCH_ROI, or
    None. Auto-located so the red-box layouts (sut-mla, pet-mor) work
    wherever the graphic sits."""
    x1, y1, x2, y2 = Config.CLOCK_SEARCH_ROI
    roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 168))
           & (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 60)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= SB_RED_MIN_AREA and w <= SB_RED_MAX_W and h >= 20:
            boxes.append((x, y, w, h))
    if not boxes:
        return None
    bx1 = min(b[0] for b in boxes) - 5
    by1 = min(b[1] for b in boxes) - 5
    bx2 = max(b[0] + b[2] for b in boxes) + 5
    by2 = max(b[1] + b[3] for b in boxes) + 5
    return (x1 + max(bx1, 0), y1 + max(by1, 0), x1 + bx2, y1 + by2)


def _read_score_boxes(img: np.ndarray) -> Optional[tuple]:
    """(home, away) from auto-located red score boxes, or None. Digit
    allowlist + 4x upscale: generic OCR misses the white-on-red digits."""
    roi = _locate_red_boxes(img)
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    big = cv2.resize(img[y1:y2, x1:x2], None, fx=4.0, fy=4.0,
                     interpolation=cv2.INTER_CUBIC)
    res = _reader().readtext(big, allowlist="0123456789")
    digits = []
    for bb, text, conf in res:
        t = text.strip()
        if not t or conf < 0.4:
            continue
        x = float((bb[0][0] + bb[1][0]) / 2)
        if len(t) == 1:
            digits.append((int(t), x))
        elif len(t) == 2:           # both boxes read as one token
            digits.extend([(int(t[0]), x - 1), (int(t[1]), x + 1)])
    if len(digits) != 2:
        return None
    digits.sort(key=lambda d: d[1])
    home, away = digits[0][0], digits[1][0]
    if home > 15 or away > 15:      # OCR garbage; football scores are small
        return None
    return home, away


def _parse_scoreboard_frame(img: np.ndarray) -> Optional[dict]:
    """Score + clock from one frame, trying both layout families:
    (1) score as one OCR token ("0-0", "0 : 1"); (2) score digits in
    auto-located red boxes. Requires a clock for family (1) — a lone "0-0"
    string elsewhere on screen must not mint a reading."""
    tokens = _ocr_tokens(img, Config.CLOCK_SEARCH_ROI)
    clock_cands, score_cands = [], []
    for text, _x in tokens:
        t = text.strip().replace(" ", "")
        m = _CLOCK_RE.match(t)
        if m and int(m.group(2)) < 60 and int(m.group(1)) <= 130:
            clock_cands.append((int(m.group(1)), int(m.group(2))))
        m2 = _SCORE_TOKEN_RE.match(t)
        if m2 and int(m2.group(1)) <= 15 and int(m2.group(2)) <= 15:
            score_cands.append((int(m2.group(1)), int(m2.group(2))))
    # A colon-separated score ("0 : 1") also matches the clock pattern; the
    # real clock has the larger minutes figure. Ambiguity in the opening
    # minutes is tolerated — the majority filter downstream absorbs it.
    clock = None
    if clock_cands:
        mm, ss = max(clock_cands)
        clock = mm * 60 + ss
    score = None
    for h, a in score_cands:
        if clock is not None and h * 60 + a == clock and len(clock_cands) == 1:
            continue        # that token WAS the clock, not a score
        score = (h, a)
        break
    if score is None:
        score = _read_score_boxes(img)
    if score is None:
        return None
    return {"home": score[0], "away": score[1], "clock_sec": clock}


def _parse_banner(tokens: list) -> Optional[dict]:
    """Banner tokens: caption words + team names + two lone digits."""
    digits = []
    caption = []
    for text, x in tokens:
        t = text.strip()
        if re.fullmatch(r"\d{1,2}", t):
            digits.append((int(t), x))
        else:
            m2 = re.fullmatch(r"(\d)\s*[|:l\-]\s*(\d)", t)
            if m2:
                digits.extend([(int(m2.group(1)), x - 1),
                               (int(m2.group(2)), x + 1)])
            elif t:
                caption.append(t.upper())
    if len(digits) < 2:
        return None
    digits.sort(key=lambda d: d[1])
    home, away = digits[0][0], digits[-1][0]
    if home > 15 or away > 15:
        return None
    return {"home": home, "away": away, "caption": " ".join(caption)}


def _ocr_tokens(img: np.ndarray, roi: tuple) -> list:
    x1, y1, x2, y2 = roi
    crop = img[y1:y2, x1:x2]
    # 2x upscale helps easyocr on the small scoreboard digits
    crop = cv2.resize(crop, None, fx=2.0, fy=2.0,
                      interpolation=cv2.INTER_CUBIC)
    res = _reader().readtext(crop)
    toks = [(text, float((bb[0][0] + bb[1][0]) / 2))
            for bb, text, conf in res if conf > 0.25]
    toks.sort(key=lambda t: t[1])
    return toks


def read_graphics(video_path: str, sweep: dict, fps: float = 25.0) -> list:
    """OCR flagged frames → readings
    [{frame, source, home, away, clock_sec?, caption?}, ...]."""
    cap = cv2.VideoCapture(video_path)
    readings = []

    def grab(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        return img if ok else None

    step = sweep["step"]
    for f0, f1 in _runs(sweep["banner"], step):
        span = max(f1 - f0, 1)
        picks = sorted({f0 + int(span * k / (OCR_PER_RUN - 1))
                        for k in range(OCR_PER_RUN)} if span > 1 else {f0})
        for f in picks:
            img = grab(f)
            if img is None:
                continue
            parsed = _parse_banner(_ocr_tokens(img, BANNER_OCR_ROI))
            if parsed:
                readings.append({"frame": int(f), "source": "banner", **parsed})

    # Scoreboard: fixed cadence across the whole video — visibility is
    # intermittent and layout-dependent, so presence isn't pre-tested; a
    # sample either parses into (score, clock) or is discarded.
    sb_every = int(SB_OCR_EVERY_S * fps)
    n_tried = 0
    for f in range(sweep["start"], sweep["end"], sb_every):
        img = grab(f)
        if img is None:
            continue
        n_tried += 1
        parsed = _parse_scoreboard_frame(img)
        if parsed:
            readings.append({"frame": int(f), "source": "scoreboard",
                             **parsed})
    print(f"  scoreboard OCR: {len([r for r in readings if r['source'] == 'scoreboard'])}"
          f"/{n_tried} samples parsed")
    cap.release()
    readings.sort(key=lambda r: r["frame"])
    return readings


# ── Step 3: score timeline → goals ───────────────────────────────────────────

def _majority_filter(readings: list) -> list:
    """Drop isolated OCR misreads: a reading survives if a neighbour (±2 in
    sequence) agrees with its (home, away), or it is a banner reading (large
    digits, essentially never misread)."""
    keep = []
    for i, r in enumerate(readings):
        if r["source"] == "banner":
            keep.append(r)
            continue
        score = (r["home"], r["away"])
        agree = any((readings[j]["home"], readings[j]["away"]) == score
                    for j in range(max(0, i - 2), min(len(readings), i + 3))
                    if j != i)
        if agree:
            keep.append(r)
    return keep


def derive_goals(readings: list, period_info: dict) -> dict:
    """Walk the reading timeline; each monotonic score step is a goal.
    Non-monotonic readings (OCR noise that survived filtering) are dropped
    with a warning rather than minting phantom goals.

    PERSISTENCE CHECK: a step only counts if the new score sticks. dec-mla's
    graphic showed 3-3 for ~80s then reverted to 3-2 for the rest of the
    broadcast (disallowed goal or operator error, corrected on air; real
    result 3-2) — under a pure monotonic walk the revert readings look like
    'noise' and the phantom goal survives. A step whose OLD score outnumbers
    the NEW one in subsequent readings is recorded as reverted, not a goal."""
    fps = float(period_info["fps"])
    fh, sh = (period_info["first_half_start_frame"],
              period_info["second_half_start_frame"])

    def frame_period_time(frame: int) -> tuple:
        if frame >= sh:
            return 2, (frame - sh) / fps
        return 1, max(frame - fh, 0) / fps

    def persists(i: int, new: tuple, old: tuple) -> bool:
        n_new = n_old = 0
        for r2 in readings[i + 1:]:
            s = (int(r2["home"]), int(r2["away"]))
            if s == new:
                n_new += 1
            elif s == old:
                n_old += 1
        return n_new >= n_old

    goals, dropped, reverted = [], [], []
    cur = (0, 0)
    last_frame = 0
    for i, r in enumerate(readings):
        score = (int(r["home"]), int(r["away"]))
        if score == cur:
            last_frame = r["frame"]
            continue
        dh, da = score[0] - cur[0], score[1] - cur[1]
        if dh < 0 or da < 0 or dh + da != 1:
            dropped.append(r)
            continue
        if not persists(i, score, cur):
            reverted.append({"frame": int(r["frame"]),
                             "transient_score": {"home": score[0],
                                                 "away": score[1]}})
            continue
        # Anchor choice (pixel-verified on the sut-mla goal): when the
        # scoreboard is visible through the goal the bracket is seconds wide
        # and the operator updates it AFTER the goal — the last old-score
        # reading sits within ~1s of the ball crossing the line. When the
        # bracket is wide (graphic hidden; score change seen on a goal
        # banner / scoreboard reappearance minutes later), the FIRST sighting
        # of the new score is the near end of the goal, so anchor there.
        tight = (r["frame"] - last_frame) <= int(30 * fps)
        anchor = int(last_frame) if (tight and last_frame) else int(r["frame"])
        period, tsec = frame_period_time(anchor)
        goals.append({
            "scorer_side": "home" if dh == 1 else "away",
            "score_after": {"home": score[0], "away": score[1]},
            "first_seen_frame": int(r["frame"]),
            "last_old_score_frame": int(last_frame),
            "first_seen_source": r["source"],
            "anchor_frame": anchor,
            "period": period,
            "time_sec": round(tsec, 1),
            "clock_sec": r.get("clock_sec"),
        })
        cur = score
        last_frame = r["frame"]
    return {"goals": goals, "final_score": {"home": cur[0], "away": cur[1]},
            "dropped_readings": dropped, "reverted_steps": reverted}


# ── Orchestration ────────────────────────────────────────────────────────────

def oracle_path(slug: str) -> Path:
    return Config.OUTPUT_EVENTS_DIR / f"{slug}_goal_oracle.json"


def run_oracle(slug: str, home_team: Optional[int] = None,
               use_cache: bool = True) -> dict:
    """Full oracle for one match; caches the sweep+readings (the slow part)
    so re-derivation is instant."""
    video = str(Config.MATCH_VIDEOS[slug])
    periods = json.loads((Config.PROJECT_ROOT / "data" /
                          "period_detection_results.json").read_text())
    period_info = next(r for r in periods if r["slug"] == slug)

    cache = Config.OUTPUT_EVENTS_DIR / f"{slug}_score_readings.json"
    if use_cache and cache.exists():
        readings = json.loads(cache.read_text())["readings"]
        print(f"[{slug}] {len(readings)} cached readings from {cache.name}")
    else:
        print(f"[{slug}] sweeping graphics (step {SWEEP_STEP})...")
        sweep = sweep_graphics(video)
        print(f"[{slug}] banner samples flagged: {len(sweep['banner'])}")
        print(f"[{slug}] OCR on flagged frames (CPU)...")
        readings = read_graphics(video, sweep, fps=period_info["fps"])
        Config.OUTPUT_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"readings": readings}, indent=1))
        print(f"[{slug}] {len(readings)} parsed readings -> {cache.name}")

    filtered = _majority_filter(readings)
    result = derive_goals(filtered, period_info)
    result.update({
        "slug": slug,
        "home_team_id": home_team,
        "n_readings": len(readings),
        "n_used": len(filtered),
        "method": "score-graphic OCR: a score change is a certain goal; "
                  "timing bracketed [last_old_score_frame, first_seen_frame]",
    })
    out = oracle_path(slug)
    out.write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description="Scoreboard goal-oracle (score-digit OCR)")
    ap.add_argument("--match", default="sut-mla")
    ap.add_argument("--home_team", type=int, default=None, choices=[0, 1],
                    help="classifier team id of the HOME side (left on the "
                         "graphics). sut-mla: 1 (Sutjeska, blue)")
    ap.add_argument("--no_cache", action="store_true",
                    help="re-run the sweep + OCR instead of using cached readings")
    args = ap.parse_args()

    r = run_oracle(args.match, home_team=args.home_team,
                   use_cache=not args.no_cache)
    print(f"\nfinal score (OCR): {r['final_score']['home']}-"
          f"{r['final_score']['away']}")
    for g in r["goals"]:
        m, s = divmod(int(g["time_sec"]), 60)
        print(f"  GOAL {g['scorer_side']} -> {g['score_after']['home']}-"
              f"{g['score_after']['away']} | period {g['period']} "
              f"{m:02d}:{s:02d} | frames [{g['last_old_score_frame']}, "
              f"{g['first_seen_frame']}] ({g['first_seen_source']})")
    if r["dropped_readings"]:
        print(f"  ({len(r['dropped_readings'])} non-monotonic readings dropped)")
    print(f"saved -> {oracle_path(args.match)}")


if __name__ == "__main__":
    main()
