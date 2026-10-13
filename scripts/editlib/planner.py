"""Edit planning: beat-aligned cuts, effect choices, speed ramps and source-moment selection.

The plan is plain data (JSON-serialisable) so it can be logged, inspected and rendered at any
resolution/frame rate. Times are in seconds; frames are only computed when rendering.
"""
from __future__ import annotations

import math
import random

from .common import EditError, log, require_numpy

np = require_numpy()

ACCENTS = ("whip_blur", "chroma_hit", "zoom_snap")


def rank_hits(segs: list[dict], sensitivity: float, drop_rel: float) -> None:
    """Mark the clips that start on the hardest hits of THIS edit.

    Ranking (instead of a fixed threshold) keeps effects selective in loud songs where almost
    every beat is strong: about 15-50 % of clips are 'hits' and 5-15 % are 'big' hits,
    depending on beats.beat_sensitivity."""
    n = len(segs)
    if not n:
        return
    score = [s["strength"] + 0.15 * s["downbeat"] + 0.1 * s["energy"] for s in segs]
    order = sorted(range(n), key=lambda i: -score[i])
    n_hit = int(math.ceil((0.15 + 0.35 * sensitivity) * n))
    n_big = int(math.ceil((0.05 + 0.10 * sensitivity) * n))
    drop_i = min(range(n), key=lambda i: abs(segs[i]["t0"] - drop_rel))
    has_drop = abs(segs[drop_i]["t0"] - drop_rel) < 0.08
    for rank, i in enumerate(order):
        s = segs[i]
        s["is_drop"] = bool(has_drop and i == drop_i)
        loud_enough = s["strength"] >= 0.3
        s["hit"] = bool(s["is_drop"] or (rank < n_hit and loud_enough))
        s["big"] = bool(s["is_drop"] or (rank < n_big and loud_enough))


# ---------------------------------------------------------------------------
# 1) Cuts on beats
# ---------------------------------------------------------------------------

def make_segments(cfg: dict, music: dict, start: float, length: float, rng: random.Random) -> list[dict]:
    bcfg = cfg["beats"]
    bt = np.array(music["beat_times"])
    strength = np.array(music["beat_strength"])
    energy = np.array(music["beat_energy"])
    down = np.array(music["downbeat"], dtype=bool)

    idx = np.flatnonzero((bt >= start - 1e-3) & (bt < start + length - 0.05))
    rel = [float(x) for x in bt[idx] - start]
    ids = [int(i) for i in idx]
    if not rel or rel[0] > 0.05:            # always start a clip at time 0
        nearest = int(np.argmin(np.abs(bt - start)))
        rel.insert(0, 0.0)
        ids.insert(0, nearest)
    rel[0] = 0.0

    base = int(bcfg["cut_every_beats"])
    min_c, max_c = float(bcfg["min_clip_seconds"]), float(bcfg["max_clip_seconds"])
    half = max(1, base // 2)
    is_down = [bool(down[k]) for k in ids]

    # Cuts follow bar patterns chosen at every downbeat (like an editor counting "1-2-3-4"):
    #   normal: every `base` beats   busy (loud parts): e.g. 1+1+2   calm (quiet parts): 2x base
    # A cut never jumps over a bar line, so cuts stay locked to the downbeats.
    cut_pos = [0]
    i = 0
    steps: list[int] = []
    while i < len(rel) - 1:
        if is_down[i] or i == 0:
            e = float(energy[ids[i]])
            r = rng.random() if bcfg["adaptive_cuts"] else 0.5
            if bcfg["adaptive_cuts"] and r < 0.3 * e * e and base >= 2:
                steps = rng.choice([[half, half, base], [base, half, half]])
            elif bcfg["adaptive_cuts"] and r > 1.0 - 0.6 * (1.0 - e) ** 2:
                steps = [base * 2]
            else:
                steps = [base]
        n = steps.pop(0) if steps else base
        if not steps:
            steps = [n]
        j = i + n
        nxt_down = next((k for k in range(i + 1, min(j, len(rel))) if is_down[k]), None)
        if nxt_down is not None and n < 4:
            j = nxt_down
        while j < len(rel) and rel[j] - rel[i] < min_c:
            j += 1
        while j - 1 > i and j < len(rel) and rel[j] - rel[i] > max_c:
            j -= 1
        if j >= len(rel):
            break
        cut_pos.append(j)
        i = j

    times = [rel[p] for p in cut_pos] + [length]
    starts = [ids[p] for p in cut_pos]
    if len(times) > 2 and times[-1] - times[-2] < min_c:   # tiny last clip -> merge
        del times[-2]
        del starts[-1]

    segs = []
    for k in range(len(times) - 1):
        t0, t1 = times[k], times[k + 1]
        bi = int(starts[k])
        inner = [r - t0 for r in rel if t0 - 1e-6 <= r < t1 - 0.05]
        pieces = max(1, int(math.ceil((t1 - t0) / max_c - 1e-9)))  # split over-long gaps (no beats)
        for p in range(pieces):
            a = t0 + (t1 - t0) * p / pieces
            b = t0 + (t1 - t0) * (p + 1) / pieces
            segs.append({
                "t0": round(a, 4), "t1": round(b, 4),
                "beats": [round(x - (a - t0), 4) for x in inner if a - t0 - 1e-6 <= x < b - t0 - 0.05] or [0.0],
                "strength": float(strength[bi]) if p == 0 else 0.3,
                "energy": float(energy[bi]),
                "downbeat": bool(down[bi]) if p == 0 else False,
            })
    for n, s in enumerate(segs):
        s["index"] = n
    return segs


# ---------------------------------------------------------------------------
# 2) Effects
# ---------------------------------------------------------------------------

def assign_effects(segs: list[dict], cfg: dict, rng: random.Random, drop_rel: float) -> None:
    E = cfg["effects"]
    rank_hits(segs, cfg["beats"]["beat_sensitivity"], drop_rel)
    Iv, Is, Iz = E["velocity_intensity"], E["shake_intensity"], E["zoom_intensity"]
    G, F, Fmax, T = E["glow_strength"], E["flash_intensity"], E["max_flashes_per_second"], E["transition_rate"]
    min_flash_gap = 1.0 / Fmax if Fmax > 0 else float("inf")
    last_flash = -1e9
    last_accent = None

    for s in segs:
        D = s["t1"] - s["t0"]
        st, en = s["strength"], s["energy"]
        hit, big, is_drop = s["hit"], s["big"], s["is_drop"]

        # --- velocity (speed ramp) --------------------------------------------------
        if Iv <= 0 or D < 0.4:
            s["speed"] = {"mode": "steady", "v": 1.0 if D < 0.4 else round(1.0 + 0.2 * Iv * en, 3)}
        elif big:
            s["speed"] = {"mode": "impact", "slow": max(0.25, 1 - 0.85 * Iv), "fast": 1 + 1.6 * Iv}
        elif rng.random() < 0.25 + 0.5 * en:
            s["speed"] = {"mode": "velocity", "slow": max(0.3, 1 - 0.65 * Iv * (0.6 + 0.4 * st)),
                          "fast": 1 + 1.4 * Iv * (0.5 + 0.5 * en)}
        else:
            s["speed"] = {"mode": "steady", "v": round(1.0 + 0.3 * Iv * (en - 0.5), 3)}

        # --- zoom -------------------------------------------------------------------
        zoom = {}
        if Iz > 0:
            if hit:
                zoom = {"type": "punch", "amount": Iz * 0.28 * (0.6 + 0.4 * st), "decay": min(0.45, 0.6 * D)}
            elif D >= 0.6 and rng.random() < 0.35:
                zoom = {"type": rng.choice(["push", "pull"]), "amount": Iz * 0.14}
        s["zoom"] = zoom

        # --- shake ------------------------------------------------------------------
        shake = {}
        if Is > 0 and hit and (big or rng.random() < 0.6):
            shake = {"amp": min(0.06, Is * 0.05 * (0.6 + 0.4 * st)),       # fraction of frame width
                     "rot_deg": Is * 1.8 * (0.5 + 0.5 * st),
                     "decay": min(0.5, 0.8 * D),
                     "fx": rng.uniform(11, 16), "fy": rng.uniform(9, 14), "fr": rng.uniform(6, 10),
                     "px": rng.uniform(0, 6.28), "py": rng.uniform(0, 6.28), "pr": rng.uniform(0, 6.28)}
        s["shake"] = shake

        # --- flash (rate-limited for photosensitivity safety) ---------------------------
        flash = 0.0
        if F > 0 and Fmax > 0 and big and s["index"] > 0 and s["t0"] - last_flash >= min_flash_gap:
            flash = F if is_drop else F * (0.6 + 0.4 * st)
            last_flash = s["t0"]
        s["flash"] = round(flash, 3)

        # --- glow --------------------------------------------------------------------
        s["glow"] = round(min(1.0, G * (0.55 + 0.45 * en) + (0.15 * G if hit else 0.0)), 3)

        # --- selective transition accents ---------------------------------------------
        accent = None
        if T > 0 and hit and s["index"] > 0 and rng.random() < T:
            options = [a for a in ACCENTS if a != last_accent]
            accent = rng.choice(options)
            last_accent = accent
        s["accent"] = accent
    if segs:
        segs[-1]["fade_out"] = min(0.4, 0.5 * (segs[-1]["t1"] - segs[-1]["t0"]))


# ---------------------------------------------------------------------------
# 3) Speed ramps -> time maps
# ---------------------------------------------------------------------------

def speed_curve(t: "np.ndarray", seg: dict) -> "np.ndarray":
    sp = seg["speed"]
    if sp["mode"] == "steady":
        return np.full_like(t, sp["v"])
    beats = seg["beats"] or [0.0]
    D = seg["t1"] - seg["t0"]
    gaps = np.diff(beats + [D])
    w = float(np.clip(0.33 * (np.median(gaps) if gaps.size else D), 0.08, 0.4))
    dip = np.zeros_like(t)
    for n, b in enumerate(beats):
        x = t - b
        g = np.where(x >= 0, np.exp(-(x / w) ** 2), np.exp(-(x / (0.35 * w)) ** 2))
        dip = np.maximum(dip, g * (1.0 if n == 0 else 0.7))
    return sp["fast"] - (sp["fast"] - sp["slow"]) * dip


def build_time_map(seg: dict, scale: float = 1.0) -> tuple[list, float]:
    """Return knots [(output_t, source_t), ...] and the total source seconds needed."""
    D = seg["t1"] - seg["t0"]
    t = np.linspace(0, D, max(8, int(D * 240)) + 1)
    v = speed_curve(t, seg) * scale
    src = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(t))])
    k = int(np.clip(math.ceil(D / 0.06), 2, 20))
    pick = np.linspace(0, t.size - 1, k + 1).round().astype(int)
    knots = [(round(float(t[i]), 5), round(float(src[i]), 5)) for i in pick]
    return knots, float(src[-1])


# ---------------------------------------------------------------------------
# 4) Choosing source moments
# ---------------------------------------------------------------------------

def _prep_videos(videos: list[dict], cfg: dict) -> list[dict]:
    out = []
    for vi, v in enumerate(videos):
        info, mo = v["info"], v["motion"]
        dur = info["duration"]
        lo, hi = cfg["edit"]["skip_start_seconds"], dur - cfg["edit"]["skip_end_seconds"] - 0.05
        if hi - lo < 1.0:
            lo, hi = 0.0, dur - 0.05
        a = np.asarray(mo["action"], dtype=np.float64)
        l = np.asarray(mo["luma"], dtype=np.float64)
        out.append({
            "vi": vi, "info": info, "fps": float(mo["fps"]), "lo": lo, "hi": hi,
            "cs_a": np.concatenate([[0.0], np.cumsum(a)]), "cs_l": np.concatenate([[0.0], np.cumsum(l)]),
            "n": a.size, "cuts": np.asarray(mo["cut_times"], dtype=np.float64),
            "centroid": np.asarray(mo["centroid"], dtype=np.float64), "action": a,
        })
    offset = 0.0
    for v in out:
        v["goff"] = offset
        offset += v["hi"] - v["lo"]
    for v in out:
        v["gtotal"] = offset
    return out


def _window_scores(v: dict, L: float, used: list, allow_reuse: bool, target_g: float | None,
                   radius: float) -> tuple["np.ndarray", "np.ndarray"]:
    fps = v["fps"]
    if v["hi"] - v["lo"] < L:
        return np.array([]), np.array([])
    starts = np.arange(v["lo"], v["hi"] - L + 1e-9, 1.0 / fps)
    i0 = np.clip(np.round(starts * fps).astype(int), 0, v["n"] - 1)
    nL = max(1, int(round(L * fps)))
    nE = max(1, int(round(0.35 * nL)))
    i1 = np.clip(i0 + nL, 0, v["n"])
    iE = np.clip(i0 + nE, 0, v["n"])
    cnt = np.maximum(i1 - i0, 1)
    mean_all = (v["cs_a"][i1] - v["cs_a"][i0]) / cnt
    mean_early = (v["cs_a"][iE] - v["cs_a"][i0]) / np.maximum(iE - i0, 1)
    luma = (v["cs_l"][i1] - v["cs_l"][i0]) / cnt
    score = 0.55 * mean_all + 0.45 * mean_early
    score -= np.where(luma < 0.07, 0.5, 0.0)
    for c in v["cuts"]:
        score -= np.where((c > starts + 0.08) & (c < starts + L - 0.08), 0.3, 0.0)
    for (us, ue) in used:
        ov = np.clip(np.minimum(starts + L, ue) - np.maximum(starts, us), 0, None) / L
        score -= (1.2 * ov) if allow_reuse else np.where(ov > 0.02, 1e6, 0.0)
    if target_g is not None:  # chronological: stay near this clip's place in the story
        g = v["goff"] + (starts - v["lo"])
        score -= 0.35 * np.clip(np.abs(g - target_g) - radius, 0, None) / max(radius, 1e-6)
    return starts, score


def select_sources(segs: list[dict], videos: list[dict], cfg: dict) -> None:
    vids = _prep_videos(videos, cfg)
    max_len = max(v["hi"] - v["lo"] for v in vids)
    allow_reuse = bool(cfg["edit"]["allow_clip_reuse"])
    order_mode = cfg["edit"]["clip_order"]
    edit_len = segs[-1]["t1"] if segs else 1.0
    total = vids[0]["gtotal"]
    used = {v["vi"]: [] for v in vids}
    crop_strength = float(cfg["output"]["smart_crop_strength"])

    for s in segs:  # speed maps first: they decide how much source time each clip needs
        knots, L = build_time_map(s)
        if L > max_len * 0.95:
            scale = max_len * 0.95 / L
            knots, L = build_time_map(s, scale)
            log.debug("Segment %d: source too short, speed scaled by %.2f", s["index"], scale)
        s["time_map"], s["src_len"] = knots, round(L, 4)

    def importance(s):
        return s["strength"] * (0.5 + s["energy"]) + (2 if s["is_drop"] else 0) + (1 if s["big"] else 0)

    if order_mode == "best_moments":
        order = sorted(segs, key=lambda s: -importance(s))
    elif order_mode == "hybrid":   # drop + biggest hits first get the best moments, rest in story order
        bigs = sorted([s for s in segs if s["big"]], key=lambda s: -importance(s))
        order = bigs + [s for s in segs if not s["big"]]
    else:
        order = segs

    for s in order:
        L = s["src_len"]
        target = None
        radius = 0.0
        if order_mode == "chronological" or (order_mode == "hybrid" and not s["big"]):
            target = (s["t0"] / max(edit_len, 1e-6)) * max(total - L, 0)
            radius = max(2.0, 1.5 * total / max(len(segs), 1))
        best = (-1e18, None, None)
        for v in vids:
            starts, scores = _window_scores(v, L, used[v["vi"]], allow_reuse, target, radius)
            if scores.size:
                j = int(np.argmax(scores))
                if scores[j] > best[0]:
                    best = (float(scores[j]), v, float(starts[j]))
        score, v, st = best
        if v is None or score < -1e5:
            raise EditError("Not enough unique footage for this edit. Add more clips, shorten "
                            "edit.duration_seconds, or set edit.allow_clip_reuse to true.")
        used[v["vi"]].append((st, st + L))
        fps = v["fps"]
        a, b = int(st * fps), max(int(st * fps) + 1, int((st + L) * fps))
        w = v["action"][a:b] + 1e-3
        cx = float((v["centroid"][a:b] * w).sum() / w.sum()) if b > a else 0.5
        s["source"] = {"video": v["vi"], "name": v["info"]["name"], "path": v["info"]["path"],
                       "start": round(st, 3), "score": round(score, 3)}
        s["crop_x"] = round(0.5 + crop_strength * (cx - 0.5), 3)


# ---------------------------------------------------------------------------
# 5) Whole plan
# ---------------------------------------------------------------------------

def plan_edit(cfg: dict, music: dict, music_start: float, length: float, videos: list[dict]) -> dict:
    rng = random.Random(int(cfg["edit"]["random_seed"]))
    segs = make_segments(cfg, music, music_start, length, rng)
    if not segs:
        raise EditError("Could not create any clips from the beat map (is the song long enough?).")
    assign_effects(segs, cfg, rng, music["drop_time"] - music_start)
    select_sources(segs, videos, cfg)
    plan = {
        "music": {"path": music["path"], "name": music["name"], "start": round(music_start, 3),
                  "bpm": music["bpm"], "drop_rel": round(music["drop_time"] - music_start, 3)},
        "length": round(length, 4),
        "segments": segs,
        "stats": {
            "clips": len(segs),
            "avg_clip_seconds": round(length / len(segs), 3),
            "ramped": sum(s["speed"]["mode"] != "steady" for s in segs),
            "shakes": sum(bool(s["shake"]) for s in segs),
            "zooms": sum(bool(s["zoom"]) for s in segs),
            "flashes": sum(s["flash"] > 0 for s in segs),
            "accents": sum(s["accent"] is not None for s in segs),
        },
    }
    return plan


def preview_range(plan: dict, seconds: float, section: str) -> tuple[int, int]:
    """Indices [a, b) of the segments that make up the preview."""
    segs = plan["segments"]
    p0 = 0.0
    if section == "drop":
        p0 = max(0.0, plan["music"]["drop_rel"] - 0.3 * seconds)
        p0 = min(p0, max(0.0, plan["length"] - seconds))
    a = next((i for i, s in enumerate(segs) if s["t1"] > p0 + 1e-6), 0)
    b = a
    while b < len(segs) and segs[b]["t0"] < segs[a]["t0"] + seconds - 1e-6:
        b += 1
    return a, max(b, a + 1)
