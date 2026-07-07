"""
Shot action-spotting on raw broadcast frames (the SoccerNet-style approach the
geometric pipeline pointed to — see shot_candidates.py for why geometry can't
do this).

Design, tuned to what we measured:
* **Candidates = wide→close-up camera transitions.** That single signal already
  catches ~86% of shots (the director zooms in on every shot); the job is to
  rank/filter them by *visual content* rather than the too-generic geometric
  context (box occupancy etc.) that failed.
* **Features = ResNet18 embeddings of frames around each cut** — a few wide
  frames just before (the shot itself, ball leaving the foot) and several
  close-up frames just after (the aftermath: keeper retrieving, players
  reacting, goal/crowd in frame). This is where a shot's visual signature
  lives, and it's exactly what box-occupancy couldn't see.
* **Labels** from SofaScore shot timestamps via the validated clock map.
* **Posture: over-detect.** A missed shot is invisible to the human; a false
  candidate is one click to reject. So we optimise recall at a
  human-reviewable candidate budget, not precision.

Two stages so model iteration is cheap:
    python -m src.shot_spotting --extract     # ResNet feats -> cache (slow, GPU)
    python -m src.shot_spotting --train       # fit + leave-one-match-out CV
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .game_state import GameState, available_periods
from .shot_candidates import _sofa_shots

# frame offsets relative to the cut: negatives = wide build-up (the shot),
# positives = close-up aftermath.
OFFSETS = [-30, -15, -6, 4, 12, 24, 45]
LABEL_S = 20.0
CACHE = Config.OUTPUT_DIR / "shot_spotting"
MATCHES = ["sut-mla", "jez-jed", "mla-bud-2", "jez-ars", "dec-mla", "jed-ars"]


def _transitions(gs):
    """(frame, time_sec) for each wide→close-up transition in a period."""
    f = gs.frames.sort_values("frame")
    wide = f.is_wide_shot.fillna(False).to_numpy()
    tsec = f.time_sec.to_numpy(); fr = f.frame.to_numpy()
    idx = np.where((wide[:-1]) & (~wide[1:]))[0]
    return [(int(fr[i + 1]), float(tsec[i + 1])) for i in idx]


def _embed_frames(frames_bgr, cnn, dev):
    """ResNet18 avgpool (512-d) for a list of BGR frames, batched on GPU."""
    import torch
    import torchvision.transforms as T
    tf = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch = torch.stack([tf(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                         for f in frames_bgr]).to(dev)
    with torch.no_grad():
        out = cnn(batch)
    return out.cpu().numpy().astype(np.float16)


def extract_match(slug: str, cnn, dev):
    CACHE.mkdir(parents=True, exist_ok=True)
    trans, labels, times, periods = [], [], [], []
    for per in available_periods(slug):
        gs = GameState.load(slug, period=per)
        shots = [ts for sp, ts, st in _sofa_shots(slug) if sp == per]
        tr = _transitions(gs)
        cap = cv2.VideoCapture(str(Config.MATCH_VIDEOS[slug]))
        # collect all (frame, trans_idx, off_idx) sorted for sequential reads
        need = []
        base = len(trans)
        for ti, (frm, ts) in enumerate(tr):
            for oi, off in enumerate(OFFSETS):
                need.append((frm + off, base + ti, oi))
            labels.append(1 if any(abs(ts - s) <= LABEL_S for s in shots) else 0)
            times.append(ts); periods.append(per)
            trans.append(np.zeros((len(OFFSETS), 512), np.float16))
        need.sort()
        # sequential decode
        want = {}
        for frmi, tj, oi in need:
            want.setdefault(frmi, []).append((tj, oi))
        frame_ids = sorted(want)
        buf_frames, buf_targets = [], []
        pos = 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fid in frame_ids:
            if fid < 0 or fid >= total:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, img = cap.read()
            if not ok:
                continue
            buf_frames.append(img)
            buf_targets.append(want[fid])
            if len(buf_frames) >= 128:
                embs = _embed_frames(buf_frames, cnn, dev)
                for e, tgts in zip(embs, buf_targets):
                    for tj, oi in tgts:
                        trans[tj][oi] = e
                buf_frames, buf_targets = [], []
        if buf_frames:
            embs = _embed_frames(buf_frames, cnn, dev)
            for e, tgts in zip(embs, buf_targets):
                for tj, oi in tgts:
                    trans[tj][oi] = e
        cap.release()
        print(f"  {slug} p{per}: {len(tr)} transitions embedded", flush=True)
    arr = np.stack(trans)             # (n, n_off, 512) float16
    np.savez_compressed(CACHE / f"{slug}_feats.npz", feats=arr,
                        labels=np.array(labels), times=np.array(times),
                        periods=np.array(periods), offsets=np.array(OFFSETS))
    print(f"  saved {slug}: {arr.shape}, {int(np.sum(labels))} positive", flush=True)


def extract_all():
    from .segmentation import _get_cnn
    cnn, dev = _get_cnn()
    for slug in MATCHES:
        extract_match(slug, cnn, dev)


def extract_geometric(slug: str):
    """Appearance-INVARIANT play features per transition, in the SAME order as
    the appearance cache (they transfer across the day/night domain gap that
    sinks raw ResNet features). Cached to {slug}_geom.npz."""
    from .events import ball_series, detect_restarts, dead_ball_frames
    rows = []
    for per in available_periods(slug):
        gs = GameState.load(slug, period=per)
        fps = gs.fps
        b = ball_series(gs); dead = dead_ball_frames(b)
        rst = [(r["f_dead0"], r["type"]) for r in detect_restarts(b, dead, fps)]
        bidx = b.set_index("frame")
        pl = gs.players[gs.players.pitch_x.notna()]
        w = int(4.0 * fps)
        for frm, ts in _transitions(gs):
            bw = bidx[(bidx.index >= frm - w) & (bidx.index < frm)]
            ball_third = ball_tg = 0.0
            if len(bw):
                bx = bw["bx"].to_numpy(); vx = bw["vx"].to_numpy(); sp = bw["speed"].to_numpy()
                fin = (bx > 70) | (bx < 35)
                ball_third = float(np.nanmean(fin.astype(float))) if len(fin) else 0.0
                m = (sp > 8) & fin
                ball_tg = float(np.nanmax(np.where(m, np.abs(vx), 0))) if m.any() else 0.0
            pw = pl[(pl.frame >= frm - w) & (pl.frame < frm)]
            occ = 0
            if len(pw):
                for xlo, xhi in ((85, 110), (-5, 20)):
                    mm = pw[(pw.pitch_x >= xlo) & (pw.pitch_x <= xhi)
                            & (pw.pitch_y > 10) & (pw.pitch_y < 58)]
                    occ = max(occ, mm.track_id.nunique())
            nxt = [typ for rf, typ in rst if 0 <= (rf - frm) <= 25 * fps]
            gk = 1.0 if nxt[:1] and nxt[0] in ("goal_kick", "corner") else 0.0
            cor = 1.0 if nxt[:1] == ["corner"] else 0.0
            rows.append([occ, ball_tg, ball_third, gk, cor])
    np.savez_compressed(CACHE / f"{slug}_geom.npz", geom=np.array(rows, np.float32))
    print(f"  {slug}: {len(rows)} geometric rows", flush=True)


# ── Training / evaluation ────────────────────────────────────────────────────

# Best config found by leave-one-match-out sweep (2026-07-07): appearance
# ResNet (wide-mean ++ close-mean) → PCA to combat the day/night overfit,
# concatenated with the appearance-invariant geometric block, balanced LR.
# GB tied LR; LR is simpler. Result: ~92% of REACHABLE shots (86% of all
# shots have a camera cut nearby) at ~250 candidates/match — a 2.6x cut from
# the raw 650 camera cuts. The over-detect+human-QC operating point.
PCA_N = 64
LR_C = 0.1
DEFAULT_BUDGET = 250
MODEL_PATH = CACHE / "shot_model.joblib"


def _aggregate(feats):
    off = np.array(OFFSETS)
    wide = feats[:, off < 0, :].mean(axis=1)
    close = feats[:, off >= 0, :].mean(axis=1)
    return np.concatenate([wide, close], axis=1).astype(np.float32)


def _load():
    data = {}
    for slug in MATCHES:
        fp = CACHE / f"{slug}_feats.npz"
        gp = CACHE / f"{slug}_geom.npz"
        if fp.exists():
            d = np.load(fp)
            app = _aggregate(d["feats"])
            geom = np.load(gp)["geom"] if gp.exists() else np.zeros((len(app), 5), np.float32)
            X = np.concatenate([app, geom], axis=1)
            data[slug] = (X, d["labels"], d["times"], d["periods"], app.shape[1])
    return data


def _fit(X, y, napp):
    """Scaler + (PCA on appearance block only) + balanced LR. Returns a
    callable prob(X) and the fitted pieces for persistence."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    pca = PCA(n_components=min(PCA_N, napp)).fit(Xs[:, :napp])

    def reduce(Xraw):
        z = sc.transform(Xraw)
        return np.concatenate([pca.transform(z[:, :napp]), z[:, napp:]], axis=1)
    clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=LR_C)
    clf.fit(reduce(X), y)
    return reduce, clf, (sc, pca, clf, napp)


def train_eval():
    data = _load()
    if len(data) < 2:
        print("need >=2 matches extracted; run --extract first")
        return
    print(f"loaded {len(data)} matches (LOMO CV)\n")
    scored = {}
    for tm in data:
        others = [m for m in data if m != tm]
        Xtr = np.vstack([data[m][0] for m in others])
        ytr = np.concatenate([data[m][1] for m in others])
        napp = data[tm][4]
        reduce, clf, _ = _fit(Xtr, ytr, napp)
        X, y, times, per, _ = data[tm]
        scored[tm] = (clf.predict_proba(reduce(X))[:, 1], times, per)

    for K in (150, 200, 250, 300):
        HIT = REACH = ALL = 0
        for slug, (prob, times, per) in scored.items():
            shots = _sofa_shots(slug)
            keep = set(np.argsort(-prob)[:K])
            kept = [(per[i], times[i]) for i in keep]
            allt = list(zip(per, times))
            for sp, ts, st in shots:
                ALL += 1
                if any(p == sp and abs(t - ts) <= 20 for p, t in allt):
                    REACH += 1
                    if any(p == sp and abs(t - ts) <= 20 for p, t in kept):
                        HIT += 1
        print(f"top-{K}/match: {HIT}/{REACH} of reachable = {HIT/max(REACH,1)*100:.0f}% "
              f"| {HIT}/{ALL} of ALL shots = {HIT/max(ALL,1)*100:.0f}% "
              f"(reachable ceiling {REACH/ALL*100:.0f}%)")


def train_final():
    """Train on ALL cached matches and persist — for scoring a NEW match."""
    import joblib
    data = _load()
    X = np.vstack([data[m][0] for m in data])
    y = np.concatenate([data[m][1] for m in data])
    napp = next(iter(data.values()))[4]
    _, _, pieces = _fit(X, y, napp)
    joblib.dump({"sc": pieces[0], "pca": pieces[1], "clf": pieces[2],
                 "napp": pieces[3], "offsets": OFFSETS}, MODEL_PATH)
    print(f"saved final model -> {MODEL_PATH} (trained on {len(data)} matches)")


def predict_match(slug: str, budget: int = DEFAULT_BUDGET) -> list:
    """Ranked shot-candidate times for a match (needs its feats+geom cached).
    Returns the top-`budget` (period, time_sec, score) for human review."""
    import joblib
    m = joblib.load(MODEL_PATH)
    d = np.load(CACHE / f"{slug}_feats.npz")
    app = _aggregate(d["feats"])
    gp = CACHE / f"{slug}_geom.npz"
    geom = np.load(gp)["geom"] if gp.exists() else np.zeros((len(app), 5), np.float32)
    X = np.concatenate([app, geom], axis=1)
    z = m["sc"].transform(X)
    Xr = np.concatenate([m["pca"].transform(z[:, :m["napp"]]), z[:, m["napp"]:]], axis=1)
    prob = m["clf"].predict_proba(Xr)[:, 1]
    order = np.argsort(-prob)[:budget]
    out = sorted([(int(d["periods"][i]), float(d["times"][i]), float(prob[i]))
                  for i in order], key=lambda r: (r[0], r[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--geom", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--fit_final", action="store_true")
    ap.add_argument("--predict", default=None)
    args = ap.parse_args()
    if args.extract:
        extract_all()
    if args.geom:
        for s in MATCHES:
            extract_geometric(s)
    if args.train:
        train_eval()
    if args.fit_final:
        train_final()
    if args.predict:
        for per, ts, sc in predict_match(args.predict):
            print(f"  p{per} {int(ts//60):>2}:{int(ts%60):02d}  score {sc:.2f}")


if __name__ == "__main__":
    main()
