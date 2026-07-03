"""
One-page match report from the merged StatsBomb-lite events JSON.

This is the artifact a 1.CFL analyst actually opens: team pass maps, a match
momentum strip, the shot/goal map, top passers, and the headline numbers —
rendered with mplsoccer from ``output/events/{slug}_events.json`` (the
match-level export of :mod:`src.events`). No video, no GPU, seconds to run.

Provenance is printed on the page: homography-trusted coverage, event counts,
and the goal-oracle method note, so nobody mistakes estimated locations for
measured ones.

Usage::

    python -m src.events --match sut-mla          # build the merged JSON
    python -m src.report --match sut-mla          # render the report
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from .config import Config

# Club names by slug prefix (1.CFL 25/26). Diacritics render fine in the
# figure; keep prints ASCII.
CLUB_NAMES = {
    "ars": "FK Arsenal", "dec": "FK Dečić", "bok": "FK Bokelj",
    "jed": "FK Jedinstvo", "bud": "FK Budućnost", "sut": "FK Sutjeska",
    "jez": "FK Jezero", "pet": "OFK Petrovac", "mor": "OFK Mornar",
    "mla": "OFK Mladost DG",
}

# Display colours per classifier team id — kit-matched for sut-mla (team0
# Mladost yellow, team1 Sutjeska blue), same convention as render_game_state.
TEAM_HEX = {0: "#D9B800", 1: "#1F6FCC"}
PITCH_KW = dict(pitch_type="statsbomb", pitch_color="#f8f9f6",
                line_color="#b0b8b0", linewidth=1.2)


def _club_names(slug: str) -> tuple:
    parts = slug.split("-")
    home = CLUB_NAMES.get(parts[0], parts[0].upper())
    away = CLUB_NAMES.get(parts[1], parts[1].upper())
    return home, away


def _team_label(team_id: int, home_id, home: str, away: str) -> str:
    if home_id is None:
        return f"Team {team_id}"
    return home if int(team_id) == int(home_id) else away


def load_match(slug: str) -> dict:
    p = Config.OUTPUT_EVENTS_DIR / f"{slug}_events.json"
    d = json.loads(p.read_text())
    # oracle output (if any) carries the home-team mapping
    home_id = None
    op = Config.OUTPUT_EVENTS_DIR / f"{slug}_goal_oracle.json"
    if op.exists():
        home_id = json.loads(op.read_text()).get("home_team_id")
    d["home_team_id"] = home_id
    return d


# ── Panel builders ───────────────────────────────────────────────────────────

def _passes(events: list, team: int) -> tuple:
    """(complete, incomplete) pass arrays [(x, y, ex, ey), ...] for a team."""
    comp, inc = [], []
    for e in events:
        if e["type"]["name"] != "Pass" or e["team"]["id"] != team:
            continue
        x, y = e["location"]
        ex, ey = e["pass"]["end_location"]
        (comp if "outcome" not in e["pass"] else inc).append((x, y, ex, ey))
    return np.array(comp).reshape(-1, 4), np.array(inc).reshape(-1, 4)


def _draw_pass_map(ax, pitch, events, team, name, colour):
    comp, inc = _passes(events, team)
    if len(inc):
        pitch.arrows(inc[:, 0], inc[:, 1], inc[:, 2], inc[:, 3], ax=ax,
                     width=1.2, headwidth=4, headlength=4,
                     color="#c9c9c9", alpha=0.55, zorder=1)
    if len(comp):
        pitch.arrows(comp[:, 0], comp[:, 1], comp[:, 2], comp[:, 3], ax=ax,
                     width=1.6, headwidth=4.5, headlength=4.5,
                     color=colour, alpha=0.75, zorder=2)
    tot = len(comp) + len(inc)
    pct = 100 * len(comp) / max(tot, 1)
    ax.set_title(f"{name} — {tot} passes, {pct:.0f}% complete  (attacking →)",
                 fontsize=11, pad=6)


def _draw_momentum(ax, events, home_id, home, away, goals, bin_min=3.0):
    """Diverging pass-count strip per time bin: above zero = first team."""
    t0 = int(home_id) if home_id is not None else 0
    t1 = 1 - t0
    max_min = max((e["minute"] for e in events), default=90) + 1
    edges = np.arange(0, max_min + bin_min, bin_min)
    v0 = np.zeros(len(edges) - 1)
    v1 = np.zeros(len(edges) - 1)
    for e in events:
        if e["type"]["name"] != "Pass":
            continue
        b = min(int(e["minute"] / bin_min), len(v0) - 1)
        if e["team"]["id"] == t0:
            v0[b] += 1
        else:
            v1[b] += 1
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, v0, width=bin_min * 0.9, color=TEAM_HEX[t0], alpha=0.85,
           label=home if home_id is not None else f"Team {t0}")
    ax.bar(centers, -v1, width=bin_min * 0.9, color=TEAM_HEX[t1], alpha=0.85,
           label=away if home_id is not None else f"Team {t1}")
    ax.axhline(0, color="#888", lw=0.8)
    ht = next((e["minute"] for e in events if e["period"] == 2), None)
    if ht is not None:
        ax.axvline(45, color="#888", lw=1.0, ls="--")
        ax.text(45, ax.get_ylim()[1] * 0.9, " HT", fontsize=8, color="#666")
    for g in goals:
        gm = g["minute"]
        sign = 1 if g["team_id"] == t0 else -1
        ymax = max(v0.max() if len(v0) else 1, v1.max() if len(v1) else 1)
        ax.plot([gm], [sign * ymax * 0.75], marker="*", ms=18,
                color=TEAM_HEX[g["team_id"]], mec="#333", zorder=5)
        ax.text(gm, sign * ymax * 0.95, g["score"], ha="center", fontsize=9,
                fontweight="bold", color="#333")
    ax.set_xlabel("match minute", fontsize=9)
    ax.set_ylabel("passes / 3 min", fontsize=9)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_title("Match momentum (pass volume) with goals", fontsize=11)


def _draw_shot_map(ax, pitch, events, home_id, home, away):
    n = 0
    for e in events:
        if e["type"]["name"] != "Shot":
            continue
        n += 1
        x, y = e["location"]
        team = e["team"]["id"]
        is_goal = e["shot"]["outcome"]["name"] == "Goal"
        if is_goal:
            pitch.scatter([x], [y], ax=ax, marker="*", s=600,
                          color=TEAM_HEX[team], edgecolors="#222",
                          linewidth=1.2, zorder=4)
            lbl = _team_label(team, home_id, home, away)
            ax.annotate(f"GOAL {e['minute']}'  {lbl}", (x, y),
                        textcoords="offset points", xytext=(0, -18),
                        ha="center", fontsize=9, fontweight="bold")
        else:
            pitch.scatter([x], [y], ax=ax, marker="o", s=140,
                          color=TEAM_HEX[team], alpha=0.7,
                          edgecolors="#333", zorder=3)
    title = "Shots & goals  (attacking →, both teams)"
    if any(e["type"]["name"] == "Shot" and e["shot"].get("oracle")
           for e in events):
        title += "\ngoal timing from scoreboard OCR; location estimated"
    ax.set_title(title, fontsize=10, pad=6)
    if n == 0:
        ax.text(60, 40, "no shots detected", ha="center", fontsize=11,
                color="#777")


def _draw_top_passers(ax, events, home_id, home, away, top_n=6):
    counts: dict = {}
    for e in events:
        if e["type"]["name"] != "Pass" or "outcome" in e["pass"]:
            continue
        key = (e["team"]["id"], e["player"]["name"])
        counts[key] = counts.get(key, 0) + 1
    rows = sorted(counts.items(), key=lambda kv: kv[1])
    t0 = int(home_id) if home_id is not None else 0
    top0 = [r for r in rows if r[0][0] == t0][-top_n:]
    top1 = [r for r in rows if r[0][0] == 1 - t0][-top_n:]
    entries = top1 + [((None, ""), 0)] + top0        # away block below home
    labels = [k[1] for k, _ in entries]
    vals = [v for _, v in entries]
    colours = [TEAM_HEX.get(k[0], "#ffffff") for k, _ in entries]
    ax.barh(range(len(entries)), vals, color=colours, alpha=0.85)
    ax.set_yticks(range(len(entries)), labels, fontsize=8)
    ax.set_xlabel("completed passes", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_title("Top passers (completed)", fontsize=11)


def _stats_rows(summary: dict, goals: list, home_id) -> list:
    t0 = int(home_id) if home_id is not None else 0
    t1 = 1 - t0

    def g(d, team):
        return d.get(team, d.get(str(team), 0))

    ng0 = sum(1 for x in goals if x["team_id"] == t0)
    ng1 = sum(1 for x in goals if x["team_id"] == t1)
    return [
        ("Goals (scoreboard oracle)", ng0, ng1),
        ("Possession %", g(summary["possession_pct"], t0),
         g(summary["possession_pct"], t1)),
        ("Passes", g(summary["passes"], t0), g(summary["passes"], t1)),
        ("Pass completion %", g(summary["pass_completion_pct"], t0),
         g(summary["pass_completion_pct"], t1)),
        ("Carries", g(summary["carries"], t0), g(summary["carries"], t1)),
        ("Possession changes", summary.get("possession_changes", 0), ""),
    ]


# ── Page assembly ────────────────────────────────────────────────────────────

def render_report(slug: str, out_path: Path = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch

    d = load_match(slug)
    events, summary = d["events"], d["summary"]
    home_id = d["home_team_id"]
    home, away = _club_names(slug)

    goals = [{"minute": e["minute"], "team_id": e["team"]["id"],
              "score": (f"{e['shot']['oracle']['score_after']['home']}–"
                        f"{e['shot']['oracle']['score_after']['away']}")
              if e["shot"].get("oracle") else ""}
             for e in events
             if e["type"]["name"] == "Shot"
             and e["shot"]["outcome"]["name"] == "Goal"]

    t0 = int(home_id) if home_id is not None else 0
    ng_home = sum(1 for x in goals if x["team_id"] == t0)
    ng_away = len(goals) - ng_home

    fig = plt.figure(figsize=(16.5, 11.5), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.45, 1.25],
                          hspace=0.34, wspace=0.14,
                          left=0.05, right=0.97, top=0.86, bottom=0.05)

    # header
    if home_id is not None:
        title = f"{home}  {ng_home} – {ng_away}  {away}"
    else:
        title = f"{slug}: Team 0 vs Team 1"
    fig.text(0.05, 0.955, title, fontsize=26, fontweight="bold")
    periods = summary.get("periods", [1])
    fig.text(0.05, 0.925,
             f"Montenegro 1.CFL  |  match slug {slug}  |  periods "
             f"{periods}  |  report generated {date.today().isoformat()}",
             fontsize=11, color="#555")
    fig.text(0.05, 0.905,
             f"pipeline: {summary['n_events']} events, homography trusted on "
             f"{summary.get('homography_trusted_pct', '?')}% of frames; "
             f"unnamed players are tracker ids",
             fontsize=9, color="#888")

    # stats block (top-left)
    ax_stats = fig.add_subplot(gs[0, 0])
    ax_stats.axis("off")
    rows = _stats_rows(summary, goals, home_id)
    lbl0 = home if home_id is not None else "Team 0"
    lbl1 = away if home_id is not None else "Team 1"
    ax_stats.text(0.16, 1.0, lbl0, ha="center", fontsize=11, fontweight="bold",
                  color=TEAM_HEX[t0], transform=ax_stats.transAxes)
    ax_stats.text(0.84, 1.0, lbl1, ha="center", fontsize=11, fontweight="bold",
                  color=TEAM_HEX[1 - t0], transform=ax_stats.transAxes)
    for i, (name, v0, v1) in enumerate(rows):
        y = 0.82 - i * 0.155
        ax_stats.text(0.5, y, name, ha="center", fontsize=10, color="#444",
                      transform=ax_stats.transAxes)
        ax_stats.text(0.16, y, str(v0), ha="center", fontsize=12,
                      fontweight="bold", transform=ax_stats.transAxes)
        ax_stats.text(0.84, y, str(v1), ha="center", fontsize=12,
                      fontweight="bold", transform=ax_stats.transAxes)

    # momentum (top-right)
    ax_mom = fig.add_subplot(gs[0, 1])
    _draw_momentum(ax_mom, events, home_id, home, away, goals)

    # pass maps (middle row)
    pitch = Pitch(**PITCH_KW)
    ax_p0 = fig.add_subplot(gs[1, 0])
    ax_p1 = fig.add_subplot(gs[1, 1])
    pitch.draw(ax=ax_p0)
    pitch.draw(ax=ax_p1)
    _draw_pass_map(ax_p0, pitch, events, t0, lbl0, TEAM_HEX[t0])
    _draw_pass_map(ax_p1, pitch, events, 1 - t0, lbl1, TEAM_HEX[1 - t0])

    # shot map + top passers (bottom row)
    ax_shot = fig.add_subplot(gs[2, 0])
    pitch.draw(ax=ax_shot)
    _draw_shot_map(ax_shot, pitch, events, home_id, home, away)
    ax_top = fig.add_subplot(gs[2, 1])
    _draw_top_passers(ax_top, events, home_id, home, away)

    if out_path is None:
        out_dir = Config.OUTPUT_DIR / "reports" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slug}_match_report.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Render one-page match report")
    ap.add_argument("--match", default="sut-mla")
    args = ap.parse_args()
    p = render_report(args.match)
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
