"""
Verification-first human review UI — the workflow for the AUTOMATIC identity
era. One self-contained HTML page per (match, half); no server, no installs:
open in a browser, work, click Export, feed the JSON back with ``--apply``.

Why a new page next to :mod:`src.review_ui`: that UI was built when a human
labeled ~40 anonymous tracks from scratch. The pipeline now PROPOSES most of
the roster on its own (VLM shirt numbers + ReID propagation resolve ~82% of
the SofaScore XI, ~60% of pass events). The human job flips from "label
everything" to "verify what the machine claims + name what it missed" —
faster (~5-10 min/half), and simple enough to hand to a paid student:
every card is "do these photos show ONE player, and is the claimed shirt
number right?"

Two sections:

* **Verify** — one card per automatically-resolved identity (team + shirt
  number) that appears in events. Crops are sampled across DIFFERENT tracked
  fragments of that identity, so a wrong ReID/propagation merge is visible
  as two different people on one card. Buttons: Correct / Wrong (then type
  the right number, or leave blank = just kill the claim).
* **Name the rest** — the top event-bearing tracks that got NO number.
  Type the number (name auto-fills from the lineup) or click the lineup.

Apply semantics (all merged into ``data/identities/{slug}_p{N}.json``,
which outranks every automatic source in the export):

* confirmed  -> the identity's tracks become human entries (locks the label
  and attaches the real player name from the lineup, so reports show
  "M. Perošević", not "#44")
* corrected  -> same tracks, the corrected number/team/name
* wrong      -> (team, number) recorded in ``rejected_numbers`` — the
  export stops minting that identity (see jersey_ocr._rejected_numbers)
* unattributed cards -> human entries exactly like review_ui

Workflow::

    python -m src.verify_ui --match sut-pet --half 1
    # open output/review/sut-pet_p1_verify.html, review, Export
    python -m src.verify_ui --apply output/review/sut-pet_p1_verify.json
    python -m src.events --match sut-pet && python -m src.report --match sut-pet
"""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import Config
from .game_state import GameState
from . import identity as identity_mod
from .jersey_ocr import load_jersey_numbers, jersey_path
from .roles import infer_attack_direction, identify_goalkeepers
from .review_ui import (_event_counts_per_tid, collect_crops, load_lineup,
                        review_dir, _lineup_assets, _minimap)

MAX_IDENTITY_CARDS = 40
MAX_UNATTRIBUTED_CARDS = 18
CROPS_PER_CARD = 8


# ── Identity grouping (mirrors events.resolve_player priority) ─────────────

def resolve_groups(gs, slug: str, period: int) -> tuple:
    """Group event-bearing tracks the way the EXPORT will resolve them.
    Returns (identity_groups, unattributed, meta_read_frames):
    identity_groups = {(team, number): {"tids", "events", "source"}};
    unattributed = [(tid, n_events, team)] sorted by events desc."""
    from .events import _resolve_idmap
    ev = _event_counts_per_tid(gs)
    idmap = _resolve_idmap(slug, period) or {}
    jersey = load_jersey_numbers(slug, period) or {}
    meta_of = identity_mod.meta_map(gs)
    meta_team = identity_mod.meta_teams(gs)
    gk_tids = set(identify_goalkeepers(gs, infer_attack_direction(gs)))

    jersey_team_num = {int(m): (r.get("team"), int(r["number"]))
                       for m, r in jersey.items() if r.get("team") in (0, 1)}

    groups: dict = defaultdict(lambda: {"tids": [], "events": 0,
                                        "source": set()})
    unattributed = []
    team_mode = gs.players.groupby("track_id")["team_id"].agg(
        lambda s: s.mode().iat[0])
    for tid, n in sorted(ev.items(), key=lambda kv: -kv[1]):
        if tid in gk_tids:
            continue                        # GK identity is role-based, solid
        info = idmap.get(int(tid))
        key = None
        src = None
        if info and info.get("number") is not None:
            team = info.get("team")
            if team not in (0, 1):
                mid = int(meta_of.get(int(tid), tid))
                jrec = jersey.get(mid)
                team = (jrec or {}).get("team")
                if team not in (0, 1):
                    team = meta_team.get(mid)
            if team in (0, 1):
                key = (int(team), int(info["number"]))
                src = "human" if info.get("name") else "auto"
        if key is None:
            mid = int(meta_of.get(int(tid), tid))
            tn = jersey_team_num.get(mid)
            if tn and tn[0] in (0, 1):
                key = (int(tn[0]), int(tn[1]))
                src = "jersey"
        if key is None:
            t = team_mode.get(tid)
            unattributed.append((int(tid), n,
                                 int(t) if t in (0, 1) else None))
            continue
        g = groups[key]
        g["tids"].append(int(tid))
        g["events"] += n
        g["source"].add(src)

    read_frames: dict = defaultdict(set)
    p = jersey_path(slug, period)
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        for tid, v in d.get("track_reads", {}).items():
            for r in v.get("reads", []):
                read_frames[int(tid)].add(int(r["frame"]))
    return dict(groups), unattributed, dict(read_frames)


# ── Page ────────────────────────────────────────────────────────────────────

_CSS = """
:root{--bg:#12151b;--panel:#1a1f27;--line:#2c3644;--txt:#e9edf2;--dim:#8fa3b8;
      --ok:#34a853;--warn:#f0a832;--bad:#e05656;--accent:#4d9fff}
*{box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);
     color:var(--txt);margin:0;padding:0 0 140px}
header{position:sticky;top:0;background:var(--panel);z-index:20;
       padding:12px 24px;border-bottom:2px solid var(--line);
       display:flex;align-items:center;gap:20px}
h1{font-size:17px;margin:0}
#progress{font-size:14px;color:var(--dim);min-width:150px}
#bar{height:6px;background:#28303c;border-radius:3px;width:180px;overflow:hidden}
#bar>div{height:100%;background:var(--ok);width:0%}
#exportBtn{margin-left:auto;background:var(--ok);color:#fff;border:0;
           padding:12px 26px;font-size:15px;border-radius:8px;cursor:pointer;
           font-weight:600}
#exportBtn:hover{filter:brightness(1.1)}
details.help{margin:14px 24px 0;background:var(--panel);border:1px solid
             var(--line);border-radius:10px;padding:10px 16px;color:var(--dim);
             font-size:14px;line-height:1.55}
details.help b{color:var(--txt)}
h2.section{margin:26px 24px 4px;font-size:15px;color:var(--dim);
           text-transform:uppercase;letter-spacing:.08em}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));
      gap:14px;padding:12px 24px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
      padding:12px 14px;position:relative}
.card.v-ok{border-color:var(--ok)} .card.v-fix{border-color:var(--accent)}
.card.v-bad{border-color:var(--bad)} .card.done{}
.head{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:13px}
.bignum{font-size:24px;font-weight:800;min-width:52px;text-align:center;
        background:#0e1116;border-radius:8px;padding:2px 8px}
.chip{padding:2px 9px;border-radius:10px;font-weight:600;font-size:12px}
.t0{background:#5a4d00;color:#ffe14d}.t1{background:#0d3a66;color:#7fc4ff}
.tn{background:#444;color:#ccc}
.pname{font-weight:600}
.stats{color:var(--dim);margin-left:auto;text-align:right;font-size:12px}
.crops-wrap{display:flex;gap:8px;align-items:flex-start;margin-bottom:10px}
.crops{display:flex;gap:6px;overflow-x:auto;flex:1}
.crops img{height:165px;border-radius:6px}
.mini{flex:0 0 140px;border-radius:6px}
.verdict{display:flex;gap:10px;align-items:center}
.vbtn{border:1px solid var(--line);background:#222a35;color:var(--txt);
      padding:10px 20px;border-radius:8px;font-size:14px;cursor:pointer;
      font-weight:600}
.vbtn.ok.active{background:var(--ok);border-color:var(--ok);color:#fff}
.vbtn.bad.active{background:var(--bad);border-color:var(--bad);color:#fff}
.fix{display:none;gap:8px;align-items:center;margin-top:10px}
.card.v-bad .fix,.card.v-fix .fix{display:flex}
.fix label{font-size:13px;color:var(--dim)}
input[type=text]{background:#0e1116;color:#fff;border:1px solid #3a4656;
     border-radius:6px;padding:8px 10px;font-size:16px}
.num{width:64px;text-align:center;font-weight:700}
.name{flex:1;min-width:120px}
.inputs{display:flex;gap:10px;align-items:center}
.hint{font-size:12px;color:var(--dim)}
body.has-lineup .grid,body.has-lineup details.help,body.has-lineup h2.section
  {margin-right:300px}
#lineup{position:fixed;top:64px;right:0;bottom:0;width:290px;overflow-y:auto;
        background:#181c22;border-left:2px solid var(--line);padding:10px 12px;
        font-size:13px}
#lineup h3{font-size:14px;margin:10px 0 6px;display:flex;gap:8px;align-items:center}
.lp-row{padding:4px 8px;border-radius:6px;cursor:pointer;display:flex;gap:8px}
.lp-row:hover{background:#26303d}
.lp-row.used{opacity:0.35;text-decoration:line-through}
.lp-num{width:26px;text-align:right;font-weight:700;color:#ffd75e}
.lp-sub{opacity:0.6}
.lp-pos{margin-left:auto;color:#68809a}
.lp-hint{color:#68809a;font-size:12px;margin-bottom:4px}
#exportModal{position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:50;
             display:none;align-items:center;justify-content:center}
#exportModal.open{display:flex}
#exportBox{background:var(--panel);border:1px solid var(--line);
           border-radius:12px;width:min(720px,92vw);max-height:86vh;
           display:flex;flex-direction:column;padding:16px 18px;gap:10px}
#exportBox h3{margin:0;font-size:16px}
#exportBox .hint{line-height:1.5}
#exportTa{flex:1;min-height:280px;background:#0e1116;color:#cfe0f0;
          border:1px solid #3a4656;border-radius:8px;padding:10px;
          font:12px/1.45 Consolas,monospace;white-space:pre;resize:vertical}
#exportBox .row{display:flex;gap:10px;align-items:center}
#copyBtn{background:var(--accent);color:#fff;border:0;padding:10px 22px;
         border-radius:8px;font-size:14px;font-weight:600;cursor:pointer}
#closeBtn{background:#222a35;color:var(--txt);border:1px solid var(--line);
          padding:10px 18px;border-radius:8px;font-size:14px;cursor:pointer}
#copyMsg{color:var(--ok);font-size:13px;font-weight:600}
"""

_JS = """
const KEY = 'verify_' + document.body.dataset.slug + '_p' + document.body.dataset.period;
function stateLoad(){ try{return JSON.parse(localStorage.getItem(KEY))||{}}catch(e){return{}} }
function stateSave(s){ localStorage.setItem(KEY, JSON.stringify(s)); }
let lastCard = null;

function refresh(){
  const s = stateLoad(); let done = 0; const used = {};
  const cards = document.querySelectorAll('.card');
  cards.forEach(c => {
    const e = s[c.dataset.key]||{};
    c.classList.remove('v-ok','v-fix','v-bad');
    if(c.dataset.kind === 'identity'){
      if(e.verdict === 'ok'){ c.classList.add('v-ok'); done++; }
      else if(e.verdict === 'bad'){
        c.classList.add((e.number||'').trim() ? 'v-fix' : 'v-bad'); done++;
      }
      c.querySelectorAll('.vbtn.ok').forEach(b=>b.classList.toggle('active', e.verdict==='ok'));
      c.querySelectorAll('.vbtn.bad').forEach(b=>b.classList.toggle('active', e.verdict==='bad'));
    } else {
      const filled = (e.number||'').trim() || (e.name||'').trim() || e.mixed;
      if(filled){ c.classList.add((e.mixed)?'v-bad':'v-fix'); done++; }
    }
    const numEl = c.querySelector('.num'); if(numEl) numEl.value = e.number||'';
    const nameEl = c.querySelector('.name'); if(nameEl) nameEl.value = e.name||'';
    const mix = c.querySelector('.mix'); if(mix) mix.checked = !!e.mixed;
    const n = (e.number||'').trim() || (c.dataset.kind==='identity' && e.verdict==='ok' ? c.dataset.num : '');
    if(n) used[(e.team!=null?e.team:c.dataset.team) + '_' + n] = 1;
  });
  document.querySelectorAll('.lp-row').forEach(r => {
    r.classList.toggle('used', !!used[r.dataset.team + '_' + r.dataset.num]);
  });
  document.getElementById('progress').textContent = done + ' / ' + cards.length + ' reviewed';
  document.querySelector('#bar>div').style.width = (100*done/Math.max(cards.length,1)) + '%';
}
function entryTeam(c, n){
  if(c.dataset.chosenteam) return parseInt(c.dataset.chosenteam);
  const teams = Object.keys(LINEUP).filter(t => LINEUP[t][n]);
  if(n && teams.length === 1) return parseInt(teams[0]);
  return c.dataset.team === '' ? null : parseInt(c.dataset.team);
}
function save(c, patch){
  const s = stateLoad();
  const e = s[c.dataset.key] || {};
  Object.assign(e, patch);
  const numEl = c.querySelector('.num');
  if(numEl){ e.number = numEl.value.trim();
             e.team = entryTeam(c, e.number || c.dataset.num); }
  const nameEl = c.querySelector('.name');
  if(nameEl) e.name = nameEl.value.trim();
  const mix = c.querySelector('.mix'); if(mix) e.mixed = mix.checked;
  s[c.dataset.key] = e; stateSave(s); refresh();
}
document.addEventListener('click', ev => {
  const b = ev.target.closest('.vbtn');
  if(b){
    const c = b.closest('.card'); lastCard = c;
    save(c, {verdict: b.classList.contains('ok') ? 'ok' : 'bad'});
    return;
  }
  const r = ev.target.closest('.lp-row');
  if(r){
    if(!lastCard){ alert('Click a player card first, then the lineup name.'); return; }
    lastCard.querySelector('.num').value = r.dataset.num;
    lastCard.querySelector('.name').value = r.dataset.name;
    lastCard.dataset.chosenteam = r.dataset.team;
    save(lastCard, {});
    return;
  }
  const c = ev.target.closest('.card'); if(c) lastCard = c;
});
document.addEventListener('input', ev => {
  const c = ev.target.closest('.card'); if(!c) return;
  if(ev.target.classList.contains('num')){
    const n = ev.target.value.trim();
    const nameEl = c.querySelector('.name');
    const teams = Object.keys(LINEUP).filter(t => LINEUP[t][n]);
    const t = c.dataset.chosenteam || (teams.length === 1 ? teams[0] : c.dataset.team);
    const hit = (LINEUP[t]||{})[n];
    if(nameEl && (!nameEl.value.trim() || c.dataset.autoname === '1')){
      nameEl.value = hit || '';
      c.dataset.autoname = hit ? '1' : '';
    }
  }
  if(ev.target.classList.contains('name')) c.dataset.autoname = '';
  save(c, {});
});
document.addEventListener('focusin', ev => {
  const c = ev.target.closest('.card'); if(c) lastCard = c;
});
function exportJson(){
  const s = stateLoad();
  const identities = {}, unattributed = {};
  document.querySelectorAll('.card').forEach(c => {
    const e = s[c.dataset.key]||{};
    if(c.dataset.kind === 'identity'){
      if(!e.verdict) return;
      identities[c.dataset.key] = {
        team: parseInt(c.dataset.team), number: parseInt(c.dataset.num),
        verdict: e.verdict,
        corr_number: (e.number||'').trim() ? parseInt(e.number) : null,
        corr_name: (e.name||'').trim() || null,
        corr_team: (e.team === 0 || e.team === 1) ? e.team : null};
    } else {
      const has = (e.number||'').trim() || (e.name||'').trim() || e.mixed;
      if(!has) return;
      unattributed[c.dataset.key.replace('u_','')] = {
        number: (e.number||'').trim() ? parseInt(e.number) : null,
        name: (e.name||'').trim() || null,
        team: (e.team === 0 || e.team === 1) ? e.team : null,
        mixed: !!e.mixed};
    }
  });
  const payload = {slug: document.body.dataset.slug,
                   period: parseInt(document.body.dataset.period),
                   mode: 'verify', identities, unattributed};
  const fname = payload.slug + '_p' + payload.period + '_verify.json';
  const text = JSON.stringify(payload, null, 2);
  // A real download works when the page is opened as a local file, but
  // sandboxed previews (claude.ai artifacts, some viewers) silently block
  // it — so ALWAYS also open the copy-paste modal with the same JSON.
  try {
    const blob = new Blob([text], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = fname;
    document.body.appendChild(a); a.click(); a.remove();
  } catch(e) {}
  document.getElementById('exportFname').textContent = fname;
  document.getElementById('exportTa').value = text;
  document.getElementById('copyMsg').textContent = '';
  document.getElementById('exportModal').classList.add('open');
}
function copyExport(){
  const ta = document.getElementById('exportTa');
  ta.focus(); ta.select();
  const done = () => document.getElementById('copyMsg').textContent = 'Copied!';
  if (navigator.clipboard && navigator.clipboard.writeText)
    navigator.clipboard.writeText(ta.value).then(done, () => {
      document.execCommand('copy'); done(); });
  else { document.execCommand('copy'); done(); }
}
function closeExport(){
  document.getElementById('exportModal').classList.remove('open');
}
window.addEventListener('load', refresh);
"""

_MODAL = """
<div id="exportModal">
  <div id="exportBox">
    <h3>Export — your review as JSON</h3>
    <div class="hint">If a file download didn't start automatically (blocked
    in preview windows), copy the text below into a file named
    <b id="exportFname"></b> and send it back / run
    <code>python -m src.verify_ui --apply &lt;file&gt;</code>.</div>
    <textarea id="exportTa" readonly></textarea>
    <div class="row">
      <button id="copyBtn" onclick="copyExport()">Copy to clipboard</button>
      <span id="copyMsg"></span>
      <button id="closeBtn" onclick="closeExport()" style="margin-left:auto">Close</button>
    </div>
  </div>
</div>
"""

_HELP = """
<details class="help" open><summary><b>How to review (read once — 2 min)</b></summary>
<p><b>Section 1 — Verify.</b> Each card is one player the computer identified
by shirt number. Look at the photos (they come from different moments of the
half — the time is stamped on each) and answer one question: <b>is this ONE
player, wearing the claimed number?</b></p>
<ul>
<li>Photos show one player and the number is right → click <b>✓ Correct</b>.</li>
<li>The number/team is wrong but you can tell who it really is → click
<b>✗ Wrong</b>, then type the right number (the name fills itself), or click
the player in the lineup on the right.</li>
<li>Photos mix two different people, or you can't tell → click <b>✗ Wrong</b>
and leave the correction empty.</li>
</ul>
<p><b>Section 2 — Name the rest.</b> These players got no number
automatically. Type the shirt number if you can read/recognize it (use the
mini-pitch position and the lineup panel). Tick <b>2+ players mixed</b> when
one card shows two different people. Leave blank if you can't tell.</p>
<p>Everything auto-saves in the browser. When done press
<b>Export JSON</b> and send the downloaded file back.</p></details>
"""


def build_page(slug: str, period: int) -> Path:
    gs = GameState.load(slug, period=period)
    lineup = load_lineup(slug)
    panel, lineup_js, club_of = _lineup_assets(lineup)
    lineup_names = {}
    if lineup:
        for side in ("home", "away"):
            s = lineup[side]
            for p in s["players"]:
                if p.get("number") is not None:
                    lineup_names[(int(s["classifier_team"]),
                                  int(p["number"]))] = p["name"]

    print(f"[{slug} p{period}] resolving identity groups "
          f"(the export's own view)...")
    groups, unattributed, read_frames = resolve_groups(gs, slug, period)

    ordered = sorted(groups.items(), key=lambda kv: -kv[1]["events"])
    ordered = [(k, g) for k, g in ordered if g["events"] > 0]
    ordered = ordered[:MAX_IDENTITY_CARDS]
    un = [u for u in unattributed if u[1] >= 2][:MAX_UNATTRIBUTED_CARDS]

    # crop collection (reuse review_ui's frame-ordered single video pass)
    pos_df = (gs.players[np.isfinite(gs.players.pitch_x)]
              .groupby("track_id")[["pitch_x", "pitch_y"]].mean())

    def avg_pos(tids):
        rows = pos_df[pos_df.index.isin(tids)]
        if rows.empty:
            return None
        return (float(rows.pitch_x.mean()), float(rows.pitch_y.mean()))

    sel, info, prefer = [], {}, {}
    for key, g in ordered:
        cid = f"i_{key[0]}_{key[1]}"
        sel.append(cid)
        info[cid] = {"members": g["tids"]}
        pf = set()
        for t in g["tids"]:
            pf |= read_frames.get(t, set())
        prefer[cid] = pf
    for tid, n, team in un:
        cid = f"u_{tid}"
        sel.append(cid)
        info[cid] = {"members": [tid]}
        prefer[cid] = read_frames.get(tid, set())
    print(f"[{slug} p{period}] collecting crops for {len(sel)} cards...")
    crops = collect_crops(gs, sel, info, prefer=prefer)

    id_cards = []
    for key, g in ordered:
        team, num = key
        cid = f"i_{team}_{num}"
        if not crops.get(cid):
            continue
        club = club_of.get(team)
        chip = (f'<span class="chip t{team}">'
                f'{html.escape(club) if club else f"team {team}"}</span>')
        pname = lineup_names.get((team, num))
        name_html = (f'<span class="pname">{html.escape(pname)}</span>'
                     if pname else
                     '<span class="pname hint">not in lineup!</span>')
        src = ("verified before" if "human" in g["source"] else
               "auto (shirt-number OCR + appearance match)")
        imgs = "".join(f'<img src="data:image/jpeg;base64,{b}">'
                       for b in crops[cid])
        id_cards.append(f"""
<div class="card" data-key="{cid}" data-kind="identity"
     data-team="{team}" data-num="{num}">
  <div class="head"><span class="bignum">#{num}</span>{chip}{name_html}
    <span class="stats">{g['events']} events · {len(g['tids'])} fragments<br>
    {src}</span></div>
  <div class="crops-wrap"><div class="crops">{imgs}</div>{_minimap(avg_pos(g['tids']))}</div>
  <div class="verdict">
    <button class="vbtn ok">✓ Correct</button>
    <button class="vbtn bad">✗ Wrong</button>
    <span class="hint">wrong → give the right number below, or leave empty
    if you can't tell</span>
  </div>
  <div class="fix">
    <label>#</label><input type="text" class="num" inputmode="numeric">
    <label>name</label><input type="text" class="name" placeholder="auto">
  </div>
</div>""")

    un_cards = []
    for tid, n, team in un:
        cid = f"u_{tid}"
        if not crops.get(cid):
            continue
        chip = (f'<span class="chip t{team}">'
                f'{html.escape(club_of.get(team, f"team {team}"))}</span>'
                if team in (0, 1) else '<span class="chip tn">team ?</span>')
        imgs = "".join(f'<img src="data:image/jpeg;base64,{b}">'
                       for b in crops[cid])
        un_cards.append(f"""
<div class="card" data-key="{cid}" data-kind="track"
     data-team="{team if team in (0, 1) else ''}">
  <div class="head">{chip}
    <span class="stats">{n} events · fragment {tid}</span></div>
  <div class="crops-wrap"><div class="crops">{imgs}</div>{_minimap(avg_pos([tid]))}</div>
  <div class="inputs">
    <label>#</label><input type="text" class="num" inputmode="numeric">
    <label>name</label><input type="text" class="name" placeholder="optional">
    <label class="hint"><input type="checkbox" class="mix"> 2+ players mixed</label>
  </div>
</div>""")

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(slug)} p{period} — verify players</title>
<style>{_CSS}</style></head>
<body data-slug="{html.escape(slug)}" data-period="{period}"
      class="{'has-lineup' if panel else ''}">
<header>
  <h1>{html.escape(slug)} — half {period}</h1>
  <div>
    <div id="progress"></div>
    <div id="bar"><div></div></div>
  </div>
  <button id="exportBtn" onclick="exportJson()">Export JSON</button>
</header>
{_HELP}
<h2 class="section">1 · Verify — is each card one player with this number?</h2>
<div class="grid">{''.join(id_cards)}</div>
<h2 class="section">2 · Name the rest — players the computer couldn't number</h2>
<div class="grid">{''.join(un_cards)}</div>
{panel}
{_MODAL}
<script>{lineup_js}
{_JS}</script>
</body></html>"""

    out = review_dir() / f"{slug}_p{period}_verify.html"
    out.write_text(page, encoding="utf-8")
    print(f"[{slug} p{period}] {len(id_cards)} identity cards + "
          f"{len(un_cards)} unattributed cards -> {out}")
    return out


# ── Apply ────────────────────────────────────────────────────────────────────

def apply_verify(json_path: str) -> Path:
    """Merge an exported verify JSON into ``data/identities/{slug}_p{N}.json``.
    Confirmations and corrections become human track entries (locking the
    label + attaching real names); rejections go to ``rejected_numbers`` so
    the export stops minting those identities."""
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    slug, period = d["slug"], int(d["period"])
    gs = GameState.load(slug, period=period)
    groups, _, _ = resolve_groups(gs, slug, period)
    lineup = load_lineup(slug)
    lineup_names = {}
    if lineup:
        for side in ("home", "away"):
            s = lineup[side]
            for p in s["players"]:
                if p.get("number") is not None:
                    lineup_names[(int(s["classifier_team"]),
                                  int(p["number"]))] = p["name"]

    out_path = identity_mod.identity_path(slug, period)
    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        payload = {"slug": slug, "period": period, "source": "verify_ui",
                   "meta_of_track": {}, "players": {}, "mixed_metas": []}
    payload.setdefault("rejected_numbers", [])
    players = payload["players"]
    mot = payload["meta_of_track"]
    mixed = set(int(m) for m in payload.get("mixed_metas", []))
    rejected = {(int(t), int(n)) for t, n in payload["rejected_numbers"]}

    n_conf = n_corr = n_rej = n_un = 0
    for key, v in d.get("identities", {}).items():
        team, num = int(v["team"]), int(v["number"])
        g = groups.get((team, num))
        tids = g["tids"] if g else []
        if v["verdict"] == "ok":
            name = lineup_names.get((team, num))
            for t in tids:
                mot[str(t)] = int(t)
                players[str(t)] = {"name": name, "number": num, "team": team}
            n_conf += 1
        elif v["verdict"] == "bad":
            cn = v.get("corr_number")
            if cn:
                ct = v.get("corr_team")
                ct = ct if ct in (0, 1) else team
                name = v.get("corr_name") or lineup_names.get((ct, int(cn)))
                for t in tids:
                    mot[str(t)] = int(t)
                    players[str(t)] = {"name": name, "number": int(cn),
                                       "team": ct}
                n_corr += 1
            else:
                rejected.add((team, num))
                for t in tids:                # strip stale entries if any
                    players.pop(str(t), None)
                n_rej += 1

    for tid, v in d.get("unattributed", {}).items():
        tid = int(tid)
        if v.get("mixed"):
            mixed.add(tid)
            continue
        if not (v.get("number") or v.get("name")):
            continue
        mot[str(tid)] = tid
        players[str(tid)] = {"name": v.get("name"), "number": v.get("number"),
                             "team": v.get("team")}
        n_un += 1

    payload["players"] = players
    payload["meta_of_track"] = mot
    payload["mixed_metas"] = sorted(mixed)
    payload["rejected_numbers"] = sorted([list(k) for k in rejected])
    payload["source"] = "verify_ui"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"applied: {n_conf} confirmed, {n_corr} corrected, {n_rej} rejected"
          f" identities, {n_un} newly named tracks -> {out_path}")
    print(f"rebuild: python -m src.events --match {slug}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Verification-first review page")
    ap.add_argument("--match")
    ap.add_argument("--half", type=int, choices=[1, 2])
    ap.add_argument("--apply", default=None)
    args = ap.parse_args()
    if args.apply:
        apply_verify(args.apply)
        return
    if not args.match or not args.half:
        raise SystemExit("--match and --half required (or --apply FILE)")
    build_page(args.match, args.half)


if __name__ == "__main__":
    main()
