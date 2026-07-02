"""
Perception pipeline → persisted :mod:`src.game_state`.

Runs the full per-frame perception stack once over a match window and writes the
game-state artifact that :mod:`src.events` (and the correction UI / validation)
consume. This is the batch, cache-once counterpart to the interactive per-frame
loops in :mod:`src.run_demo` and :mod:`src.run_pnlcalib_video` — it *imports*
their proven pieces rather than duplicating them:

  * detection + tracking      — ``yolo.track(persist=True)`` (as run_demo)
  * homography fallback chain  — ``predict_one_frame`` → optical-flow propagate →
    ``refine_projection_to_lines`` → ``ProjectionSmoother`` (all from
    run_pnlcalib_video), gated by ``check_projection_sanity`` /
    ``check_line_alignment`` and the ``is_gameplay_frame`` scoreboard check.
  * foot projection           — ``project_foot`` / ``is_wide_shot`` (run_demo)
  * team assignment           — sampled ResNet embeddings + the saved
    ``TeamClassifier`` (track-level majority vote, joined back after the pass).

Usage::

    python -m src.pipeline --match sut-mla --offset_min 10 --duration_sec 120

Then build events with ``python -m src.events --match sut-mla``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.camera_motion import CameraMotionTracker
from src.manual_calibration import load_match_seeds
from src.team_classifier import TeamClassifier
import pandas as pd

from src.game_state import (
    PlayerState, BallState, FrameState, write_game_state, GameState,
    BALL_CAND_COLS,
)
from src.run_pnlcalib_video import (
    load_models,
    predict_one_frame,
    refine_projection_to_lines,
    ProjectionSmoother,
    check_projection_sanity,
    check_line_alignment,
    is_gameplay_frame,
    _line_alignment_score,
)
from src.run_demo import project_foot, is_wide_shot

BALL_CONF_THRESHOLD = 0.08
TEAM_SAMPLE_EVERY = 15          # frames between ResNet embedding samples per pass


def _load_period(slug: str) -> dict:
    periods = json.loads(
        (PROJECT_ROOT / "data" / "period_detection_results.json").read_text())
    for r in periods:
        if r["slug"] == slug:
            return r
    raise KeyError(f"No period entry for '{slug}'")


class PerceptionPipeline:
    """Stateful per-match runner. Construct once, call :meth:`run`."""

    def __init__(self, slug: str, device: str = "cuda:0", alpha: float = 0.3,
                 use_flow: bool = True, use_manual_seeds: bool = True,
                 team_sample_every: int = TEAM_SAMPLE_EVERY,
                 max_flow_streak: int = 60, pnl_stride: int = 1):
        self.slug = slug
        # pnl_stride > 1: run the two HRNet passes only every Nth wide frame
        # and let gated optical flow carry the projection between (speed
        # lever for full-half runs; stride 1 = PnLCalib on every wide frame).
        self.pnl_stride = max(int(pnl_stride), 1)
        self.period_info = _load_period(slug)
        self.fps = float(self.period_info["fps"])
        self.team_sample_every = team_sample_every
        self.max_flow_streak = max_flow_streak
        self.scoreboard_roi = Config.SCOREBOARD_ROI

        self.video_path = str(Config.MATCH_VIDEOS[slug])
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self.fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"[{slug}] loading models...")
        self.yolo = YOLO(Config.resolve_yolo_model())
        self.pnl_models = load_models(device)

        clf_path = Config.OUTPUT_CLASSIFIERS_DIR / f"{slug}_classifier.pkl"
        if not clf_path.exists():
            raise FileNotFoundError(
                f"No team classifier for {slug} at {clf_path}. "
                f"Run 01_team_classification.ipynb first.")
        self.clf = TeamClassifier.load(clf_path)

        self.manual_seeds = load_match_seeds(slug) if use_manual_seeds else {}
        self.alpha = alpha
        self.use_flow = use_flow

    # ── homography (mirrors run_pnlcalib_video main loop, minus rendering) ────

    def _calibrate_scoreboard(self, cap, start_frame, span) -> float:
        densities = []
        x1, y1, x2, y2 = self.scoreboard_roi
        for off in range(0, min(250, span), 25):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + off)
            ok, f = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            densities.append(float(np.mean(cv2.Canny(gray, 50, 150) > 0)))
        return float(np.median(densities)) * 0.4 if densities else 0.08

    def _homography(self, frame, abs_frame, player_boxes, wide):
        """Return (P_smooth_or_None, source, conf) for one frame.

        Homography is only attempted on wide gameplay frames. On close-ups the
        stale/manual-seed fallbacks would otherwise carry a wide-shot P onto a
        zoomed-in frame — a confidently-wrong projection (verified in QC). We
        skip them entirely so the artifact never stores those garbage overlays.
        """
        gameplay = is_gameplay_frame(frame, self.scoreboard_roi, self._sb_thr)
        if not wide or not gameplay:
            if self.motion_tracker is not None:
                self.motion_tracker.reset()
            self.flow_streak = 0
            # Feed None so the smoother decays its stale hold rather than
            # freezing a projection over a replay / close-up.
            self.smoother.update(None)
            return None, ("non_gameplay" if not gameplay else "close_up"), 0.0

        P_raw, source = None, "none"
        flow_failed = False

        def _try_flow():
            P_flow = self.motion_tracker.propagate(frame)
            if (P_flow is not None
                    and check_projection_sanity(P_flow, player_boxes)
                    and check_line_alignment(P_flow, frame, min_score=0.12)):
                return P_flow
            return None

        # Off-stride frame: cheap gated optical flow carries the projection
        # so PnLCalib (2x HRNet-W48) only runs every pnl_stride-th frame.
        if (self.pnl_stride > 1 and self.motion_tracker is not None
                and self.motion_tracker.is_ready()
                and self.flow_streak < self.max_flow_streak
                and abs_frame % self.pnl_stride != 0):
            P_flow = _try_flow()
            if P_flow is not None:
                P_raw, source = P_flow, "flow"
                self.flow_streak += 1
            else:
                flow_failed = True

        if P_raw is None:
            P_pnl, status = predict_one_frame(
                frame, self.pnl_models, self.fw, self.fh,
                manual_seeds=self.manual_seeds, abs_frame=abs_frame,
                player_boxes=player_boxes)
            if P_pnl is not None:
                P_raw, source = P_pnl, status
                if self.motion_tracker is not None:
                    self.motion_tracker.seed(frame, P_raw)
                    self.flow_streak = 0
            elif self.motion_tracker is not None:
                if flow_failed or self.flow_streak >= self.max_flow_streak:
                    # Flow already failed this frame (or has carried too
                    # long) and PnLCalib failed too — drop the trail.
                    self.motion_tracker.reset()
                    self.flow_streak = 0
                else:
                    P_flow = _try_flow()
                    if P_flow is not None:
                        P_raw, source = P_flow, "flow"
                        self.flow_streak += 1
                    else:
                        self.motion_tracker.reset()
                        self.flow_streak = 0

        if P_raw is not None:
            P_raw = refine_projection_to_lines(P_raw, frame)

        P_smooth = self.smoother.update(P_raw)
        if P_raw is not None and P_smooth is not None:
            # Fresh-beats-stale escape hatch: P_raw passed the sanity + line
            # gates on THIS frame's pixels. If the smoother's output scores
            # clearly worse against the same pixels, its state is stale or a
            # cross-shot blend (confirmation deadlock on fast-panning
            # cameras — see ProjectionSmoother.reset_to) — snap to fresh.
            s_raw = _line_alignment_score(P_raw, frame)
            s_smooth = _line_alignment_score(P_smooth, frame)
            if s_raw > s_smooth * 1.5 + 0.02:
                self.smoother.reset_to(P_raw)
                P_smooth = P_raw
                s_smooth = s_raw
            return P_smooth, source, s_smooth
        if P_smooth is None:
            return None, "none", 0.0
        return P_smooth, "stale", _line_alignment_score(P_smooth, frame)

    # ── main pass ────────────────────────────────────────────────────────────

    def run(self, start_frame: int, end_frame: int, period: int = 1) -> GameState:
        cap = cv2.VideoCapture(self.video_path)
        span = end_frame - start_frame
        self._sb_thr = self._calibrate_scoreboard(cap, start_frame, span)

        self.smoother = ProjectionSmoother(
            alpha=self.alpha, max_stale_frames=int(self.fps * 3),
            reset_threshold=0.3, outlier_threshold=0.08, confirm_frames=3)
        self.motion_tracker = CameraMotionTracker() if self.use_flow else None
        self.flow_streak = 0

        # Reset BoT-SORT global ID counter so track_ids start clean.
        if getattr(self.yolo, "predictor", None) is not None:
            self.yolo.predictor = None
        from ultralytics.trackers.basetrack import BaseTrack
        BaseTrack._count = 0

        # time_sec is anchored to THIS half's kickoff, so period-2 artifacts
        # start at 00:00 (StatsBomb timestamps restart every period).
        half_anchor = (self.period_info["first_half_start_frame"] if period == 1
                       else self.period_info["second_half_start_frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_states: list[FrameState] = []
        track_embeddings: dict[int, list] = defaultdict(list)
        ball_cand_rows: list[dict] = []   # every raw ball detection, all frames
        src_counts: dict[str, int] = defaultdict(int)

        for i in tqdm(range(span), desc=f"{self.slug} perception"):
            ok, frame = cap.read()
            if not ok:
                break
            abs_frame = start_frame + i

            results = self.yolo.track(
                frame, conf=min(Config.PLAYER_CONF_THRESHOLD, BALL_CONF_THRESHOLD),
                tracker=f"{Config.TRACKER_TYPE}.yaml", persist=True, verbose=False,
            )[0]
            boxes = results.boxes

            players, player_boxes = [], []
            best_ball, best_ball_conf = None, -1.0
            for j in range(len(boxes)):
                cls_id = int(boxes.cls[j])
                bbox = boxes.xyxy[j].cpu().numpy().astype(int).tolist()
                conf = float(boxes.conf[j])
                tid = int(boxes.id[j]) if boxes.id is not None else -1
                if cls_id == 0 and conf >= Config.PLAYER_CONF_THRESHOLD:
                    players.append({"bbox": bbox, "conf": conf, "track_id": tid})
                    player_boxes.append(bbox)
                elif cls_id == 1 and conf >= BALL_CONF_THRESHOLD:
                    ball_cand_rows.append({
                        "frame": abs_frame, "x1": bbox[0], "y1": bbox[1],
                        "x2": bbox[2], "y2": bbox[3], "conf": conf})
                    if conf > best_ball_conf:
                        best_ball, best_ball_conf = bbox, conf

            wide = is_wide_shot(players, self.fh)
            P, source, conf = self._homography(frame, abs_frame, player_boxes, wide)
            src_counts[source] += 1

            # Players → PlayerState (team filled in post-pass).
            pstates = []
            for p in players:
                tid = p["track_id"]
                if tid < 0:
                    continue
                bbox = p["bbox"]
                foot = ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))
                pitch = None
                if P is not None:
                    xy = project_foot(bbox, P)
                    pitch = (float(xy[0]), float(xy[1])) if xy is not None else None
                pstates.append(PlayerState(
                    frame=abs_frame, track_id=tid, bbox=tuple(bbox),
                    foot_xy_img=foot, conf=p["conf"], pitch_xy=pitch))

            # Ball: best raw detection only. Gap bridging is an analysis step
            # (src.ball_tracker consumes ball.parquet) — perception stays raw.
            ball = self._resolve_ball(best_ball, best_ball_conf, P)

            # Sample embeddings for team assignment.
            if i % self.team_sample_every == 0 and players:
                embs = self.clf.collect_embeddings(frame, players)
                for p, e in zip(players, embs):
                    if p["track_id"] >= 0 and np.linalg.norm(e) > 0:
                        track_embeddings[p["track_id"]].append(e)

            frame_states.append(FrameState(
                frame=abs_frame,
                time_sec=(abs_frame - half_anchor) / self.fps,
                period=period, is_gameplay=source != "non_gameplay",
                is_wide_shot=wide, P=P, homog_source=source, homog_conf=conf,
                players=pstates, ball=ball))

        cap.release()

        # Post-pass: track-level team assignment, joined back onto every row.
        print(f"[{self.slug}] classifying {len(track_embeddings)} tracks...")
        track_teams = self.clf.classify_tracks(track_embeddings)

        # Persist mean appearance embeddings per track: identity consolidation
        # (src.identity) is kinematics-only without them, which fragments on
        # close-up-heavy halves. Saved so ReID upgrades need no perception re-run.
        from src.game_state import game_state_dir
        out_dir_early = game_state_dir(self.slug, period)
        out_dir_early.mkdir(parents=True, exist_ok=True)
        emb_ids = sorted(track_embeddings)
        emb_mat = np.stack([np.mean(track_embeddings[t], axis=0)
                            for t in emb_ids]) if emb_ids else np.zeros((0, 1))
        np.savez_compressed(out_dir_early / "embeddings.npz",
                            track_ids=np.array(emb_ids), embeddings=emb_mat)
        for fs in frame_states:
            for p in fs.players:
                p.team_id = int(track_teams.get(p.track_id,
                                                TeamClassifier.TEAM_OTHER))

        meta = {
            "video": Path(self.video_path).name, "fps": self.fps,
            "period": period, "start_frame": start_frame, "end_frame": end_frame,
            "homog_source_counts": dict(src_counts),
            "n_tracks": len(track_teams),
            "n_ball_candidates": len(ball_cand_rows),
        }
        ball_cands = pd.DataFrame(ball_cand_rows, columns=BALL_CAND_COLS)
        out_dir = write_game_state(self.slug, frame_states, meta,
                                   ball_candidates=ball_cands)
        print(f"[{self.slug}] wrote game state -> {out_dir}")
        self._print_summary(frame_states, src_counts, track_teams)
        return GameState.load(self.slug, period=period)

    def _resolve_ball(self, best_ball, best_ball_conf, P):
        """Raw best detection for the frames-table convenience view. The old
        2-frame velocity extrapolation is gone: image-space coasting is wrong
        while the camera pans, and its damped ghost positions were exactly what
        merged passes into one long touch (QC finding). The Kalman tracker in
        :mod:`src.ball_tracker` owns gap bridging now."""
        if best_ball is None:
            return None
        xy_img = ((best_ball[0] + best_ball[2]) / 2.0,
                  (best_ball[1] + best_ball[3]) / 2.0)
        pitch = project_foot(best_ball, P) if P is not None else None
        return BallState(bbox=tuple(best_ball), xy_img=xy_img,
                         pitch_xy=(tuple(pitch) if pitch is not None else None),
                         conf=best_ball_conf, source="detected")

    def _print_summary(self, frame_states, src_counts, track_teams):
        n = len(frame_states)
        with_p = sum(1 for fs in frame_states if fs.P is not None)
        gameplay = sum(1 for fs in frame_states if fs.is_gameplay)
        teamcount = defaultdict(int)
        for t in track_teams.values():
            teamcount[t] += 1
        print(f"  frames: {n} | gameplay: {gameplay} | with P: {with_p} "
              f"({100*with_p/max(n,1):.1f}%)")
        print(f"  homography sources: {dict(src_counts)}")
        print(f"  tracks by team (A/B/Other): "
              f"{teamcount.get(0,0)}/{teamcount.get(1,0)}/{teamcount.get(2,0)}")


def main():
    ap = argparse.ArgumentParser(description="Build persisted game state for a match window")
    ap.add_argument("--match", default="sut-mla")
    ap.add_argument("--offset_min", type=float, default=10.0,
                    help="Minutes into the half to start")
    ap.add_argument("--duration_sec", type=int, default=120)
    ap.add_argument("--half", type=int, default=1, choices=[1, 2])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--team_sample_every", type=int, default=TEAM_SAMPLE_EVERY)
    ap.add_argument("--no_flow", action="store_true")
    ap.add_argument("--no_manual_seeds", action="store_true")
    ap.add_argument("--pnl_stride", type=int, default=1,
                    help="Run PnLCalib every Nth wide frame, flow carries "
                         "between (1 = every frame)")
    args = ap.parse_args()

    pipe = PerceptionPipeline(
        args.match, device=args.device, alpha=args.alpha,
        use_flow=not args.no_flow, use_manual_seeds=not args.no_manual_seeds,
        team_sample_every=args.team_sample_every, pnl_stride=args.pnl_stride)

    p = pipe.period_info
    fps = pipe.fps
    if args.half == 1:
        half_start = p["first_half_start_frame"]
    else:
        half_start = p["second_half_start_frame"]
    start = half_start + int(args.offset_min * 60 * fps)
    end = start + int(args.duration_sec * fps)
    print(f"Window: frames {start}–{end} "
          f"({args.duration_sec}s at {args.offset_min}min into half {args.half})")
    pipe.run(start, end, period=args.half)


if __name__ == "__main__":
    main()
