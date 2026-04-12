"""
Broadcast video analysis: camera cut detection and scoreboard presence.

Detects shot boundaries and classifies segments as gameplay vs replay/pre-game
by checking for the scoreboard overlay in the top-left corner.

Typical 1.CFL broadcast structure:
    [pre-game: no scoreboard] → [1st half: scoreboard present, replay interruptions]
    → [halftime: full-screen graphic] → [2nd half: scoreboard present] → [post-game]
"""

import re
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .video_utils import open_video, get_video_info


# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class Segment:
    """A continuous camera shot between two cuts."""
    start_frame: int
    end_frame: int
    has_scoreboard: Optional[bool] = None

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def duration_seconds(self, fps: float) -> float:
        return self.duration_frames / fps

    def time_range(self, fps: float) -> str:
        """Human-readable time range, e.g. '05:21 - 05:34'."""
        return f"{_fmt_time(self.start_frame / fps)} - {_fmt_time(self.end_frame / fps)}"

    def __repr__(self):
        sb = {True: "scoreboard", False: "no-scoreboard", None: "unchecked"}[self.has_scoreboard]
        return f"Segment({self.start_frame}-{self.end_frame}, {self.duration_frames}f, {sb})"


@dataclass
class GamePeriod:
    """A contiguous period of gameplay (e.g. first half, second half)."""
    start_frame: int
    end_frame: int
    label: str  # "first_half", "second_half", etc.

    def duration_seconds(self, fps: float) -> float:
        return (self.end_frame - self.start_frame) / fps

    def time_range(self, fps: float) -> str:
        return f"{_fmt_time(self.start_frame / fps)} - {_fmt_time(self.end_frame / fps)}"


@dataclass
class BroadcastAnalysis:
    """Complete analysis of a broadcast video."""
    video_path: Path
    fps: float
    total_frames: int
    cut_frames: List[int]
    diff_scores: np.ndarray
    segments: List[Segment]
    scoreboard_roi: Optional[Tuple[int, int, int, int]] = None
    edge_densities: Optional[np.ndarray] = None
    scoreboard_threshold: Optional[float] = None

    def gameplay_segments(self) -> List[Segment]:
        """Segments where scoreboard is present (likely gameplay)."""
        return [s for s in self.segments if s.has_scoreboard is True]

    def non_gameplay_segments(self) -> List[Segment]:
        """Segments without scoreboard (replays, pre-game, halftime, etc.)."""
        return [s for s in self.segments if s.has_scoreboard is False]

    def summary(self) -> str:
        total_sec = self.total_frames / self.fps
        gp = self.gameplay_segments()
        ngp = self.non_gameplay_segments()
        gp_frames = sum(s.duration_frames for s in gp)
        ngp_frames = sum(s.duration_frames for s in ngp)
        unchecked = sum(1 for s in self.segments if s.has_scoreboard is None)

        lines = [
            f"Video: {self.video_path.name}",
            f"Duration: {_fmt_time(total_sec)} ({self.total_frames:,} frames @ {self.fps} fps)",
            f"Cuts detected: {len(self.cut_frames)}",
            f"Segments: {len(self.segments)}",
        ]
        if unchecked < len(self.segments):
            lines += [
                f"  With scoreboard:    {len(gp):>4}  ({gp_frames / self.fps / 60:.1f} min)",
                f"  Without scoreboard: {len(ngp):>4}  ({ngp_frames / self.fps / 60:.1f} min)",
            ]
            if unchecked:
                lines.append(f"  Unchecked:          {unchecked:>4}")
        return "\n".join(lines)


# ── Cut detection ─────────────────────────────────────────────────────


def detect_cuts(
    video_path: Path,
    threshold: Optional[float] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    progress_interval: int = 5000,
    adaptive_k: float = 6.0,
    adaptive_floor: float = 0.15,
) -> Tuple[List[int], np.ndarray]:
    """
    Detect camera cuts by comparing HSV histograms of consecutive frames.

    Args:
        video_path: Path to video file.
        threshold: Difference score above which a cut is detected (0-1).
                   None (default) enables adaptive mode: the threshold is
                   chosen from the robust statistics of the per-frame diff
                   distribution, specifically
                       max(adaptive_floor, median + adaptive_k * MAD).
                   This adapts to broadcasts with different production styles
                   (hard-cut older feeds vs softer modern ones). Pass a
                   scalar to force a fixed threshold.
        start_frame / end_frame: Range to analyze (None = full video).
        progress_interval: Print progress every N frames.
        adaptive_k: Multiplier on the MAD in adaptive mode. Larger = stricter.
        adaptive_floor: Minimum adaptive threshold, prevents within-shot
                        motion from registering as cuts on very static videos.

    Returns:
        (cut_frames, diff_scores): Frame indices of cuts, and the raw
        per-frame difference array (for plotting / threshold tuning).
    """
    cap = open_video(video_path)
    info = get_video_info(cap)
    total = end_frame or info["total_frames"]

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return [], np.array([])

    prev_hist = _frame_hist(prev_frame)
    diff_scores = []

    for frame_idx in range(start_frame + 1, total):
        ret, frame = cap.read()
        if not ret:
            break

        curr_hist = _frame_hist(frame)
        # Correlation: 1.0 = identical, -1.0 = inverse
        corr = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
        diff = 1.0 - max(corr, 0.0)
        diff_scores.append(diff)
        prev_hist = curr_hist

        if frame_idx % progress_interval == 0:
            pct = (frame_idx - start_frame) / (total - start_frame) * 100
            print(f"  Scanning cuts: frame {frame_idx:,}/{total:,} ({pct:.1f}%)")

    cap.release()
    diff_arr = np.array(diff_scores)

    if threshold is None:
        median = float(np.median(diff_arr))
        mad = float(np.median(np.abs(diff_arr - median)))
        thr = max(adaptive_floor, median + adaptive_k * mad)
        print(f"  Adaptive threshold: median={median:.4f}, MAD={mad:.4f}, "
              f"threshold={thr:.4f}")
    else:
        thr = float(threshold)

    cut_frames = [start_frame + 1 + i
                  for i, d in enumerate(diff_scores)
                  if d >= thr]
    print(f"  Done. {len(cut_frames)} cuts in {len(diff_scores):,} frames.")
    return cut_frames, diff_arr


# ── Segment building ──────────────────────────────────────────────────


def build_segments(
    cut_frames: List[int],
    total_frames: int,
    start_frame: int = 0,
    min_segment_frames: int = 3,
) -> List[Segment]:
    """
    Convert cut indices into Segment objects.

    Segments shorter than min_segment_frames are absorbed into the
    previous segment (these are usually flash frames from graphic
    transitions before replays).
    """
    boundaries = [start_frame] + cut_frames + [total_frames]
    raw = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e > s:
            raw.append(Segment(start_frame=s, end_frame=e))

    if not raw:
        return raw

    merged = [raw[0]]
    for seg in raw[1:]:
        if seg.duration_frames < min_segment_frames and merged:
            merged[-1] = Segment(
                start_frame=merged[-1].start_frame,
                end_frame=seg.end_frame,
            )
        else:
            merged.append(seg)

    return merged


# ── Scoreboard detection ──────────────────────────────────────────────


def check_scoreboard(
    video_path: Path,
    segments: List[Segment],
    roi: Tuple[int, int, int, int],
    ref_frame_idx: Optional[int] = None,
    threshold: Optional[float] = None,
    samples_per_segment: int = 3,
    progress_interval: int = 200,
) -> Tuple[List[Segment], np.ndarray, float]:
    """
    Detect scoreboard presence in each segment via edge density in the ROI.

    The scoreboard overlay (team names, score, clock) creates consistent
    edges that are absent during replays and pre-game footage.

    Uses Otsu auto-thresholding on edge densities across all segments to
    separate "scoreboard present" from "absent". If ref_frame_idx is given,
    uses 50% of its edge density as threshold instead.

    Args:
        video_path: Path to video.
        segments: Segments to check (modified in place).
        roi: (x1, y1, x2, y2) of the scoreboard region.
        ref_frame_idx: Frame where scoreboard is definitely visible.
                       If None, auto-thresholds via Otsu.
        threshold: Manual override for edge density threshold.
        samples_per_segment: Frames to sample per segment.
        progress_interval: Print progress every N segments.

    Returns:
        (segments, edge_densities, threshold_used).
    """
    cap = open_video(video_path)
    x1, y1, x2, y2 = roi

    # Collect edge densities per segment
    densities = []
    for i, seg in enumerate(segments):
        seg_densities = []
        for j in range(samples_per_segment):
            t = (j + 1) / (samples_per_segment + 1)
            sample_idx = int(seg.start_frame + t * seg.duration_frames)

            cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            crop = frame[y1:y2, x1:x2]
            seg_densities.append(_edge_density(crop))

        avg = float(np.mean(seg_densities)) if seg_densities else 0.0
        densities.append(avg)

        if (i + 1) % progress_interval == 0:
            print(f"  Checking scoreboard: {i + 1}/{len(segments)} segments")

    densities = np.array(densities)

    # Determine threshold
    if threshold is not None:
        thr = threshold
    elif ref_frame_idx is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
        ret, ref_frame = cap.read()
        if ret:
            ref_density = _edge_density(ref_frame[y1:y2, x1:x2])
            thr = ref_density * 0.5
            print(f"  Reference edge density: {ref_density:.4f}, threshold: {thr:.4f}")
        else:
            thr = _otsu_threshold(densities)
    else:
        thr = _otsu_threshold(densities)
        print(f"  Auto-threshold (Otsu): {thr:.4f}")

    # Classify
    for seg, d in zip(segments, densities):
        seg.has_scoreboard = bool(d >= thr)

    cap.release()
    n_with = sum(1 for s in segments if s.has_scoreboard)
    print(f"  Done. {n_with}/{len(segments)} segments have scoreboard.")
    return segments, densities, thr


# ── Match clock OCR ───────────────────────────────────────────────────


_ocr_reader = None


def _get_ocr_reader():
    """Lazy-load the easyocr reader (single global instance)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    return _ocr_reader


# MM[sep]SS where separator is colon, period, or absent (OCR occasionally
# drops it). MM must be 2 digits to reject 3-digit false positives like the
# score "0-1" being read as "021".
_CLOCK_PATTERN = re.compile(r'^(\d{2})[:.]?(\d{2})$')


def read_clock_seconds(
    frame: np.ndarray,
    search_roi: Tuple[int, int, int, int],
    reader=None,
    min_conf: float = 0.4,
) -> Optional[int]:
    """
    OCR the match clock from a single frame.

    Searches a wide region of the frame (covering all known 1.CFL scoreboard
    layouts) and returns the highest-confidence text that matches the MM:SS
    pattern. The colon is sometimes read as a period or dropped entirely —
    the regex handles all three cases.

    Args:
        frame: BGR frame from the broadcast.
        search_roi: (x1, y1, x2, y2) wide enough to contain the clock
                    regardless of which layout the broadcast uses. See
                    Config.CLOCK_SEARCH_ROI for a default that covers
                    Layouts A/B/C.
        reader: An easyocr.Reader instance (lazy-loaded if None).
        min_conf: Minimum OCR confidence to accept a candidate.

    Returns:
        Match clock in seconds (0-5999), or None if unreadable.
    """
    if reader is None:
        reader = _get_ocr_reader()

    x1, y1, x2, y2 = search_roi
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # 2x upscale for OCR. No thresholding — easyocr handles the coloured
    # scoreboard background better than a fixed threshold.
    big = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                     interpolation=cv2.INTER_CUBIC)

    results = reader.readtext(big, detail=1)
    best_seconds = None
    best_conf = 0.0
    for _bbox, text, conf in results:
        if conf < min_conf:
            continue
        m = _CLOCK_PATTERN.match(text.strip())
        if not m:
            continue
        mm, ss = int(m.group(1)), int(m.group(2))
        if mm > 99 or ss >= 60:
            continue
        if conf > best_conf:
            best_conf = conf
            best_seconds = mm * 60 + ss

    return best_seconds


def detect_period_starts(
    video_path: Path,
    segments: List[Segment],
    search_roi: Tuple[int, int, int, int],
    fps: float,
    n_samples: int = 60,
    verbose: bool = True,
) -> Dict[str, Optional[int]]:
    """
    Auto-detect first/second half kickoff broadcast frames by reading the clock.

    Samples frames uniformly across the entire broadcast (ignoring segment
    structure), reads the match clock from each, and extrapolates:

        first_half_start  = broadcast_time - clock_time            (clock < 40:00)
        second_half_start = broadcast_time - (clock_time - 45:00)  (clock >= 50:00)

    Readings between 40:00 and 50:00 are ambiguous (could be late first half
    stoppage or early second half) and are skipped. The median is used to
    collapse the readings into a single robust estimate per half.

    Uniform sampling (rather than per-segment sampling) is more robust on
    broadcasts with very few or very many cuts. OCR naturally returns None
    on non-scoreboard frames (replays, halftime graphics), so no pre-filter
    is needed.

    Args:
        video_path: Path to broadcast video.
        segments: Used only to infer total_frames (last segment's end_frame).
        search_roi: Wide (x1, y1, x2, y2) region where the clock may appear.
                    See Config.CLOCK_SEARCH_ROI for a default covering all
                    known 1.CFL scoreboard layouts.
        fps: Video frame rate.
        n_samples: Number of frames to sample evenly across the broadcast.
        verbose: Print progress and per-reading debug info.

    Returns:
        {"first_half_start_frame": int or None,
         "second_half_start_frame": int or None,
         "validation": dict from validate_period_starts}
    """
    reader = _get_ocr_reader()
    cap = open_video(video_path)

    if not segments:
        cap.release()
        return {
            "first_half_start_frame": None,
            "second_half_start_frame": None,
            "validation": validate_period_starts(None, None, fps, 0),
        }

    total_frames = segments[-1].end_frame
    sample_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)

    if verbose:
        print(f"  Reading clock from {len(sample_indices)} uniformly-spaced frames...")

    fh_offsets = []
    sh_offsets = []
    n_success = 0

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        clock_sec = read_clock_seconds(frame, search_roi, reader=reader)
        if clock_sec is None:
            continue

        n_success += 1
        broadcast_sec = int(idx) / fps

        if clock_sec < 40 * 60:
            fh_offsets.append(broadcast_sec - clock_sec)
        elif clock_sec >= 50 * 60:
            sh_offsets.append(broadcast_sec - (clock_sec - 45 * 60))
        # 40:00-50:00 is ambiguous, skip

    cap.release()

    fh_start_frame = None
    sh_start_frame = None
    if fh_offsets:
        fh_start_sec = float(np.median(fh_offsets))
        fh_start_frame = int(round(fh_start_sec * fps))
    if sh_offsets:
        sh_start_sec = float(np.median(sh_offsets))
        sh_start_frame = int(round(sh_start_sec * fps))

    if verbose:
        print(f"  Clock read OK in {n_success}/{len(sample_indices)} sampled frames")
        if fh_offsets:
            print(f"  First half:  {len(fh_offsets):>2} readings, "
                  f"start = {_fmt_time(fh_start_frame / fps)}  "
                  f"(stdev {np.std(fh_offsets):.1f}s)")
        else:
            print(f"  First half:  no readings")
        if sh_offsets:
            print(f"  Second half: {len(sh_offsets):>2} readings, "
                  f"start = {_fmt_time(sh_start_frame / fps)}  "
                  f"(stdev {np.std(sh_offsets):.1f}s)")
        else:
            print(f"  Second half: no readings")

    validation = validate_period_starts(
        fh_start_frame, sh_start_frame, fps, total_frames,
    )
    if verbose:
        for e in validation["errors"]:
            print(f"  [ERROR] {e}")
        for w in validation["warnings"]:
            print(f"  [warn]  {w}")
        if validation["ok"] and not validation["warnings"]:
            print(f"  Sanity checks: PASS")

    return {
        "first_half_start_frame": fh_start_frame,
        "second_half_start_frame": sh_start_frame,
        "validation": validation,
    }


def validate_period_starts(
    first_half_start_frame: Optional[int],
    second_half_start_frame: Optional[int],
    fps: float,
    total_frames: int,
) -> Dict[str, object]:
    """
    Sanity-check detected kickoff frames against football reality.

    Errors indicate almost-certainly wrong detection; warnings are plausible
    but unusual and worth a human glance.

    Checks:
      - Both halves detected
      - first_half_start < second_half_start
      - 45 <= gap between halves <= 75 minutes (allows stoppage + halftime)
      - first_half_start within first 20 min of broadcast
      - second_half_start at least 40 min before broadcast end
    """
    errors: List[str] = []
    warnings: List[str] = []

    if first_half_start_frame is None:
        errors.append("first_half_start not detected")
    if second_half_start_frame is None:
        errors.append("second_half_start not detected")

    if first_half_start_frame is not None and second_half_start_frame is not None:
        if first_half_start_frame >= second_half_start_frame:
            errors.append(
                f"first_half_start ({_fmt_time(first_half_start_frame / fps)}) "
                f">= second_half_start ({_fmt_time(second_half_start_frame / fps)})"
            )
        else:
            gap_min = (second_half_start_frame - first_half_start_frame) / fps / 60
            if gap_min < 45 or gap_min > 75:
                warnings.append(
                    f"gap between halves = {gap_min:.1f} min (expected 45-75)"
                )

    if first_half_start_frame is not None:
        fh_start_min = first_half_start_frame / fps / 60
        if fh_start_min < 0:
            errors.append(f"first_half_start negative ({fh_start_min:.1f} min)")
        elif fh_start_min > 20:
            warnings.append(
                f"first_half_start at {fh_start_min:.1f} min (expected < 20)"
            )

    if second_half_start_frame is not None and total_frames > 0:
        sh_end_gap_min = (total_frames - second_half_start_frame) / fps / 60
        if sh_end_gap_min < 40:
            warnings.append(
                f"second_half_start is only {sh_end_gap_min:.1f} min before end "
                f"(expected > 40)"
            )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ── Game period detection ─────────────────────────────────────────────


def find_game_periods(
    segments: List[Segment],
    fps: float,
    first_half_start_frame: Optional[int] = None,
    second_half_start_frame: Optional[int] = None,
    halftime_frame: Optional[int] = None,
    min_period_minutes: float = 15.0,
    max_gap_seconds: float = 120.0,
) -> List[GamePeriod]:
    """
    Group gameplay segments into game periods (first half, second half).

    Preferred path: pass `first_half_start_frame` and `second_half_start_frame`
    from `detect_period_starts()`. Each period is bracketed by the kickoff
    frame and the last gameplay segment that ends before the next kickoff.

    Fallback paths (legacy):
      - `halftime_frame`: forces a split at a given frame.
      - Neither: groups consecutive scoreboard segments by time gaps. This
        is unreliable because halftime graphics often have text/edges in
        the scoreboard ROI and bridge the halftime gap.

    Args:
        segments: Segments with has_scoreboard set.
        fps: Video frame rate.
        first_half_start_frame: Kickoff frame for the first half (from OCR).
        second_half_start_frame: Kickoff frame for the second half (from OCR).
        halftime_frame: Legacy manual split point.
        min_period_minutes: Minimum length for a group to count as a period.
        max_gap_seconds: Maximum gap between gameplay segments before
                         starting a new period (fallback path only).

    Returns:
        List of GamePeriod objects, typically 2 (first_half, second_half).
    """
    gameplay = [s for s in segments if s.has_scoreboard]
    if not gameplay:
        return []

    # Preferred path: explicit kickoff frames from clock OCR
    if first_half_start_frame is not None or second_half_start_frame is not None:
        periods = []
        last_frame = gameplay[-1].end_frame

        if first_half_start_frame is not None and second_half_start_frame is not None:
            # FH ends where SH starts. Halftime graphics fall inside this
            # range but aren't gameplay segments, so downstream filters by
            # has_scoreboard still get the right data.
            periods.append(GamePeriod(
                start_frame=first_half_start_frame,
                end_frame=second_half_start_frame - 1,
                label="first_half",
            ))
            periods.append(GamePeriod(
                start_frame=second_half_start_frame,
                end_frame=last_frame,
                label="second_half",
            ))
        elif first_half_start_frame is not None:
            periods.append(GamePeriod(
                start_frame=first_half_start_frame,
                end_frame=last_frame,
                label="first_half",
            ))
        else:
            periods.append(GamePeriod(
                start_frame=second_half_start_frame,
                end_frame=last_frame,
                label="second_half",
            ))
        return periods

    # Legacy path: manual halftime split
    if halftime_frame is not None:
        before = [s for s in gameplay if s.end_frame <= halftime_frame]
        after = [s for s in gameplay if s.start_frame >= halftime_frame]
        halves = [before, after]
    else:
        # Gap-based auto-detection (unreliable — halftime graphics bridge the gap)
        groups_of_segs = []
        current_group = [gameplay[0]]

        for seg in gameplay[1:]:
            gap_seconds = (seg.start_frame - current_group[-1].end_frame) / fps
            if gap_seconds <= max_gap_seconds:
                current_group.append(seg)
            else:
                groups_of_segs.append(current_group)
                current_group = [seg]

        groups_of_segs.append(current_group)
        halves = groups_of_segs

    min_frames = min_period_minutes * 60 * fps
    periods = []
    labels = ["first_half", "second_half", "extra_time_1", "extra_time_2"]

    for i, group in enumerate(halves):
        if not group:
            continue
        start = group[0].start_frame
        end = group[-1].end_frame
        if (end - start) >= min_frames:
            label = labels[i] if i < len(labels) else f"period_{i + 1}"
            periods.append(GamePeriod(start_frame=start, end_frame=end, label=label))

    return periods


# ── Convenience wrapper ───────────────────────────────────────────────


def analyze_broadcast(
    video_path: Path,
    cut_threshold: Optional[float] = 0.4,
    scoreboard_roi: Optional[Tuple[int, int, int, int]] = None,
    scoreboard_ref_frame: Optional[int] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> BroadcastAnalysis:
    """
    Full broadcast analysis: detect cuts → build segments → check scoreboard.

    Args:
        video_path: Path to broadcast video.
        cut_threshold: Histogram difference for cut detection (0-1). Pass
                       None to enable adaptive mode (see detect_cuts()).
        scoreboard_roi: (x1, y1, x2, y2) of the scoreboard region.
                        If None, scoreboard check is skipped.
        scoreboard_ref_frame: Frame where scoreboard is visible (for threshold).
        start_frame / end_frame: Range to analyze.

    Returns:
        BroadcastAnalysis with all results.
    """
    cap = open_video(video_path)
    info = get_video_info(cap)
    cap.release()
    total = end_frame or info["total_frames"]

    print(f"Analyzing: {Path(video_path).name}")
    print(f"  {info['width']}x{info['height']} @ {info['fps']} fps, "
          f"{total:,} frames ({total / info['fps'] / 60:.1f} min)")

    # Step 1: cuts
    print("\nStep 1: Detecting camera cuts...")
    cut_frames, diff_scores = detect_cuts(
        video_path, threshold=cut_threshold,
        start_frame=start_frame, end_frame=end_frame,
    )

    # Step 2: segments
    print("\nStep 2: Building segments...")
    segments = build_segments(cut_frames, total, start_frame)
    print(f"  {len(segments)} segments (after merging flash frames)")

    # Step 3: scoreboard
    edge_densities = None
    sb_threshold = None
    if scoreboard_roi is not None:
        print("\nStep 3: Checking scoreboard presence...")
        segments, edge_densities, sb_threshold = check_scoreboard(
            video_path, segments, scoreboard_roi,
            ref_frame_idx=scoreboard_ref_frame,
        )

    analysis = BroadcastAnalysis(
        video_path=Path(video_path),
        fps=info["fps"],
        total_frames=total,
        cut_frames=cut_frames,
        diff_scores=diff_scores,
        segments=segments,
        scoreboard_roi=scoreboard_roi,
        edge_densities=edge_densities,
        scoreboard_threshold=sb_threshold,
    )

    print(f"\n{analysis.summary()}")
    return analysis


# ── Internal helpers ──────────────────────────────────────────────────


def _frame_hist(frame: np.ndarray) -> np.ndarray:
    """HSV color histogram for cut detection. Downscales for speed."""
    small = cv2.resize(frame, (320, 180))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _edge_density(crop: np.ndarray) -> float:
    """Fraction of edge pixels in a crop (Canny). Higher = more text/graphics."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0))


def _otsu_threshold(values: np.ndarray) -> float:
    """Auto-threshold a 1D array using Otsu's method (via OpenCV)."""
    # Scale to uint8 range for cv2.threshold
    v_min, v_max = values.min(), values.max()
    if v_max - v_min < 1e-9:
        return float(v_max)

    scaled = ((values - v_min) / (v_max - v_min) * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Map back to original scale
    return float(v_min + (otsu_val / 255.0) * (v_max - v_min))


def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
