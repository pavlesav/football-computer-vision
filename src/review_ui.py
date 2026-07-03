"""
Human review UI for player identity: a self-contained HTML page per
(match, half) with crop galleries for the players that matter, where the
reviewer types shirt numbers (and optionally names), then exports one JSON
that folds back into the pipeline.

Why this design: the correction workload measured so far is identity-shaped —
jersey OCR is precision-safe but recall-starved (numbers rarely attach to the
metas that actually appear in events), and cross-half unification needs
numbers on BOTH halves' event participants. A human fills exactly that gap in
~10 minutes per half: the page shows the top event-participating meta-tracks
with their best crops (numbers usually readable), pre-filled with the OCR
suggestion where one exists.

No server, no dependencies: crops are embedded as base64 JPEG, inputs
auto-save to localStorage, "Export" downloads the review JSON.

Workflow::

    python -m src.review_ui --match sut-mla --half 1     # -> output/review/*.html
    # open the page, type numbers/names, click Export
    python -m src.review_ui --apply output/review/sut-mla_p1_review.json
    python -m src.run_match --match sut-mla --home_team 1   # rebuild events+report
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .game_state import GameState
from . import identity as identity_mod
from .jersey_ocr import load_jersey_numbers
from .roles import infer_attack_direction, identify_goalkeepers

TOP_N = 28              # event-ranked metas shown (plus all jersey-confident)
CROPS_PER_META = 4
CROP_MAX_H = 190        # px, embedded image height
JPEG_QUALITY = 82


def review_dir() -> Path:
    d = Config.OUTPUT_DIR / "review"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Selection & data collection ──────────────────────────────────────────────

def _event_counts_per_meta(gs, track_meta: dict) -> dict:
    """{meta_id: n_events} from a fresh detect_events pass (the ranking must
    reflect who actually appears in the deliverable, not just who is on
    screen a lot)."""
    from .events import detect_events
    events, _ = detect_events(gs)
    counts: dict = defaultdict(int)
    for e in events:
        tids = [e.player]
        r = e.details.get("recipient")
        if r is not None:
            tids.append(r)
        for tid in tids:
            if tid is None or int(tid) < 0:
                continue
            counts[track_meta.get(int(tid), int(tid))] += 1
    return dict(counts)


def select_metas(gs, jersey: dict, top_n: int = TOP_N) -> tuple:
    """Returns (ordered meta ids, info per meta). Selection = top-N by event
    participation (frames as tiebreak) UNION all jersey-confident metas."""
    row_counts = gs.players.groupby("track_id").size()
    meta_of = identity_mod.meta_map(gs)
    track_meta = {int(t): int(meta_of.get(int(t), int(t)))
                  for t in row_counts.index}
    frames_per_meta: dict = defaultdict(int)
    members: dict = defaultdict(list)
    for tid, mid in track_meta.items():
        frames_per_meta[mid] += int(row_counts[tid])
        members[mid].append(tid)

    ev_counts = _event_counts_per_meta(gs, track_meta)
    cons_team = identity_mod.meta_teams(gs)
    gk_map = identify_goalkeepers(gs, infer_attack_direction(gs))
    gk_metas = {track_meta.get(int(t), int(t)): tm for t, tm in gk_map.items()}

    ranked = sorted(ev_counts, key=lambda m: (-ev_counts[m],
                                              -frames_per_meta.get(m, 0)))
    selected = ranked[:top_n]
    for mid in jersey or {}:
        if mid not in selected:
            selected.append(mid)

    info = {}
    for mid in selected:
        jn = (jersey or {}).get(mid)
        team = jn.get("team") if jn else cons_team.get(mid)
        info[mid] = {
            "team": team,
            "events": ev_counts.get(mid, 0),
            "frames": frames_per_meta.get(mid, 0),
            "members": sorted(members.get(mid, [mid]),
                              key=lambda t: -row_counts.get(t, 0)),
            "ocr": (jn or {}).get("number"),
            "ocr_votes": (jn or {}).get("votes"),
            "is_gk": mid in gk_metas,
        }
    return selected, info


def _best_crop_rows(gs, member_tids: list, k: int = CROPS_PER_META):
    pl = gs.players[gs.players.track_id.isin(member_tids)].copy()
    if pl.empty:
        return []
    pl["h"] = pl.y2 - pl.y1
    pl = pl.sort_values("h", ascending=False)
    picks, seen_frames = [], []
    for r in pl.itertuples(index=False):
        if any(abs(int(r.frame) - f) < 40 for f in seen_frames):
            continue
        picks.append(r)
        seen_frames.append(int(r.frame))
        if len(picks) >= k:
            break
    return picks


def collect_crops(gs, selected: list, info: dict) -> dict:
    """{meta_id: [jpeg-base64, ...]} — decoded in frame order (one pass)."""
    jobs = []
    for mid in selected:
        for r in _best_crop_rows(gs, info[mid]["members"]):
            jobs.append((int(r.frame), mid, int(r.x1), int(r.y1),
                         int(r.x2), int(r.y2)))
    jobs.sort()
    cap = cv2.VideoCapture(str(Config.MATCH_VIDEOS[gs.slug]))
    out: dict = defaultdict(list)
    for frame, mid, x1, y1, x2, y2 in jobs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            continue
        pad = 12
        h, w = img.shape[:2]
        crop = img[max(y1 - pad, 0):min(y2 + pad, h),
                   max(x1 - pad, 0):min(x2 + pad, w)]
        if crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        if ch > CROP_MAX_H:
            crop = cv2.resize(crop, (int(cw * CROP_MAX_H / ch), CROP_MAX_H))
        ok2, buf = cv2.imencode(".jpg", crop,
                                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok2:
            out[mid].append(base64.b64encode(buf.tobytes()).decode("ascii"))
    cap.release()
    return out


# ── Page rendering ───────────────────────────────────────────────────────────

_CSS = """
body{font-family:'Segoe UI',system-ui,sans-serif;background:#14171c;color:#e8e8e8;
     margin:0;padding:0 0 120px 0}
header{position:sticky;top:0;background:#1c2129;padding:14px 26px;z-index:10;
       border-bottom:2px solid #2e3745;display:flex;align-items:center;gap:24px}
h1{font-size:18px;margin:0} .sub{color:#9ab;font-size:13px}
#exportBtn{margin-left:auto;background:#2e7d32;color:#fff;border:0;
           padding:12px 26px;font-size:15px;border-radius:8px;cursor:pointer}
#exportBtn:hover{background:#388e3c}
#progress{color:#9ab;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));
      gap:16px;padding:20px 26px}
.card{background:#1c2129;border:1px solid #2e3745;border-radius:10px;
      padding:12px 14px}
.card.done{border-color:#2e7d32}
.meta-head{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:13px}
.chip{padding:2px 9px;border-radius:10px;font-weight:600;font-size:12px}
.t0{background:#5a4d00;color:#ffe14d}.t1{background:#0d3a66;color:#7fc4ff}
.tn{background:#444;color:#ccc}.gk{background:#5e2a75;color:#eab6ff}
.stats{color:#9ab}
.crops{display:flex;gap:6px;overflow-x:auto;margin-bottom:10px}
.crops img{height:170px;border-radius:6px}
.inputs{display:flex;gap:10px;align-items:center}
.inputs label{font-size:13px;color:#9ab}
input[type=text]{background:#12151a;color:#fff;border:1px solid #3a4656;
     border-radius:6px;padding:8px 10px;font-size:16px}
.num{width:64px;text-align:center;font-weight:700}
.name{flex:1}
.ocr-hint{font-size:12px;color:#7fa;white-space:nowrap}
"""

_JS = """
const KEY = 'review_' + document.body.dataset.slug + '_p' + document.body.dataset.period;
function stateLoad(){ try{return JSON.parse(localStorage.getItem(KEY))||{}}catch(e){return{}} }
function stateSave(s){ localStorage.setItem(KEY, JSON.stringify(s)); }
function refresh(){
  const s = stateLoad(); let done = 0;
  document.querySelectorAll('.card').forEach(c => {
    const mid = c.dataset.mid, e = s[mid]||{};
    c.querySelector('.num').value = e.number||'';
    c.querySelector('.name').value = e.name||'';
    const filled = (e.number||'').trim() || (e.name||'').trim();
    c.classList.toggle('done', !!filled); if(filled) done++;
  });
  document.getElementById('progress').textContent =
    done + ' / ' + document.querySelectorAll('.card').length + ' filled';
}
document.addEventListener('input', ev => {
  const c = ev.target.closest('.card'); if(!c) return;
  const s = stateLoad();
  s[c.dataset.mid] = {number: c.querySelector('.num').value.trim(),
                      name: c.querySelector('.name').value.trim()};
  stateSave(s); refresh();
});
function exportJson(){
  const s = stateLoad(), entries = {};
  document.querySelectorAll('.card').forEach(c => {
    const e = s[c.dataset.mid]||{};
    if((e.number||'').trim() || (e.name||'').trim())
      entries[c.dataset.mid] = {number: e.number.trim()? parseInt(e.number):null,
                                name: e.name.trim()||null};
  });
  const payload = {slug: document.body.dataset.slug,
                   period: parseInt(document.body.dataset.period),
                   entries: entries};
  const blob = new Blob([JSON.stringify(payload, null, 2)],
                        {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = payload.slug + '_p' + payload.period + '_review.json';
  a.click();
}
window.addEventListener('load', refresh);
"""


def build_page(slug: str, period: int, top_n: int = TOP_N) -> Path:
    gs = GameState.load(slug, period=period)
    jersey = load_jersey_numbers(slug, period) or {}
    print(f"[{slug} p{period}] ranking metas by event participation...")
    selected, info = select_metas(gs, jersey, top_n)
    print(f"[{slug} p{period}] collecting crops for {len(selected)} metas...")
    crops = collect_crops(gs, selected, info)

    cards = []
    for mid in selected:
        i = info[mid]
        if not crops.get(mid):
            continue
        team = i["team"]
        chip = (f'<span class="chip t{team}">team {team}</span>'
                if team in (0, 1) else '<span class="chip tn">team ?</span>')
        if i["is_gk"]:
            chip += ' <span class="chip gk">GK</span>'
        ocr = (f'<span class="ocr-hint">OCR: #{i["ocr"]} '
               f'({i["ocr_votes"]} votes)</span>' if i["ocr"] else "")
        imgs = "".join(f'<img src="data:image/jpeg;base64,{b}">'
                       for b in crops[mid])
        cards.append(f"""
<div class="card" data-mid="{mid}">
  <div class="meta-head">{chip}
    <span class="stats">m{mid} &middot; {i['events']} events &middot;
      {i['frames']} frames</span> {ocr}</div>
  <div class="crops">{imgs}</div>
  <div class="inputs">
    <label>#</label><input type="text" class="num" inputmode="numeric"
      placeholder="{i['ocr'] or ''}">
    <label>name</label><input type="text" class="name"
      placeholder="optional">
  </div>
</div>""")

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{html.escape(slug)} p{period} — player review</title>
<style>{_CSS}</style></head>
<body data-slug="{html.escape(slug)}" data-period="{period}">
<header>
  <div><h1>{html.escape(slug)} — half {period}</h1>
  <div class="sub">Type the shirt number you see (name optional, from the
  public lineup). Filled cards turn green; inputs auto-save in the browser.
  When done: Export, then run
  <code>python -m src.review_ui --apply &lt;downloaded file&gt;</code></div></div>
  <div id="progress"></div>
  <button id="exportBtn" onclick="exportJson()">Export JSON</button>
</header>
<div class="grid">{''.join(cards)}</div>
<script>{_JS}</script>
</body></html>"""

    out = review_dir() / f"{slug}_p{period}_review.html"
    out.write_text(page, encoding="utf-8")
    print(f"[{slug} p{period}] {len(cards)} cards -> {out}")
    return out


# ── Apply ────────────────────────────────────────────────────────────────────

def apply_review(json_path: str) -> Path:
    """Fold an exported review JSON into ``data/identities/{slug}_p{N}.json``
    (the identity-file schema the export already consumes with top
    priority). Numbers become period-independent identities; names override
    everything."""
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    slug, period = d["slug"], int(d["period"])
    entries = {int(k): v for k, v in d["entries"].items()}
    if not entries:
        raise SystemExit("review file has no filled entries")

    gs = GameState.load(slug, period=period)
    row_counts = gs.players.groupby("track_id").size()
    meta_of = identity_mod.meta_map(gs)
    meta_of_track = {int(t): int(meta_of.get(int(t), int(t)))
                     for t in row_counts.index}

    payload = {
        "slug": slug,
        "period": period,
        "source": "review_ui",
        "meta_of_track": {str(t): m for t, m in meta_of_track.items()},
        "players": {str(mid): {"name": v.get("name"),
                               "number": v.get("number")}
                    for mid, v in entries.items()},
    }
    out = identity_mod.identity_path(slug, period)
    out.write_text(json.dumps(payload, indent=2))
    print(f"applied {len(entries)} entries -> {out}")
    print("rebuild the deliverables with: python -m src.run_match "
          f"--match {slug} --home_team <0|1>")
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate / apply the player review page")
    ap.add_argument("--match")
    ap.add_argument("--half", type=int, choices=[1, 2])
    ap.add_argument("--top_n", type=int, default=TOP_N)
    ap.add_argument("--apply", default=None,
                    help="exported review JSON to fold into data/identities/")
    args = ap.parse_args()
    if args.apply:
        apply_review(args.apply)
        return
    if not args.match or not args.half:
        raise SystemExit("--match and --half required (or --apply FILE)")
    build_page(args.match, args.half, top_n=args.top_n)


if __name__ == "__main__":
    main()
