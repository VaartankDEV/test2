"""Simple visual-motion analysis to find action moments.

FFmpeg decodes each clip to tiny grayscale frames (e.g. 128 px wide, 10 fps), streamed through
a pipe so memory stays small even for full episodes. For every frame we measure:
  * motion      - mean absolute difference to the previous frame
  * column mix  - where in the frame (left/right) the motion happens (used for smart cropping)
  * brightness  - to skip black frames and to spot white impact flashes
  * histogram change - to detect hard scene cuts (which are not "action")
Camera pans are separated from on-screen action with phase correlation (global-motion
compensation), and flat/blank frames (empty sky, white screens) are marked as low-detail.
These are combined into an action score between 0 and 1.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path

from .common import EditError, find_tool, log, require_numpy

np = require_numpy()
MOTION_VERSION = 3
COLUMN_BINS = 32


def _cache_key(path: str, extra: dict) -> str:
    st = Path(path).stat()
    raw = json.dumps([str(Path(path).resolve()), st.st_size, int(st.st_mtime), MOTION_VERSION, extra],
                     ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _smooth(x: "np.ndarray", size: int) -> "np.ndarray":
    size = max(1, int(size))
    if size == 1 or x.size == 0:
        return x.copy()
    k = np.ones(size) / size
    return np.convolve(np.pad(x, (size // 2, size - 1 - size // 2), mode="edge"), k, mode="valid")


def analyze_video(info: dict, cfg: dict, cache_dir: Path | None = None, progress=None) -> dict:
    """Return per-frame arrays (numpy) + a JSON-friendly summary for one clip."""
    fps_a = float(cfg["analysis"]["motion_fps"])
    w = int(cfg["analysis"]["motion_width"]) // 2 * 2
    dw, dh = max(info["display_width"], 2), max(info["display_height"], 2)
    h = max(16, int(round(w * dh / dw / 2)) * 2)
    key = _cache_key(info["path"], {"fps": fps_a, "w": w, "h": h})
    cache_file = cache_dir / f"video_{key}.npz" if cache_dir else None
    if cache_file and cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            log.debug("Motion analysis loaded from cache %s", cache_file.name)
            return _finish(data, info, fps_a)
        except (OSError, ValueError):
            pass

    vf = f"fps={fps_a:g},scale={w}:{h}:flags=area,format=gray"
    if info.get("interlaced"):
        vf = "bwdif=mode=send_frame," + vf
    cmd = [find_tool("ffmpeg", cfg), "-hide_banner", "-nostdin", "-v", "error", "-i", info["path"],
           "-map", "0:v:0", "-an", "-sn", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    log.debug("RUN: %s", " ".join(cmd))
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        data, done, stderr = _read_frames(proc, w, h, fps_a, info, progress)
    if proc.returncode != 0 or done == 0:
        raise EditError(f"Motion analysis failed for '{info['name']}': "
                        f"{stderr.splitlines()[-1] if stderr else 'no frames decoded'}")
    if stderr:
        log.debug("Motion decode warnings for %s: %s", info["name"], stderr[:500])
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_name(cache_file.stem + "_tmp.npz")
        np.savez_compressed(tmp, **data)
        tmp.replace(cache_file)
    return _finish(data, info, fps_a)


def _read_frames(proc, w: int, h: int, fps_a: float, info: dict, progress):
    """Read the raw grayscale frames from FFmpeg and compute per-frame statistics."""
    err_chunks: list[bytes] = []
    t_err = threading.Thread(target=lambda: err_chunks.append(proc.stderr.read()), daemon=True)
    t_err.start()

    fsize = w * h
    expected = max(1, int(info["duration"] * fps_a))
    motion, luma, detail, pan, hist_d, cols = [], [], [], [], [], []
    prev = None
    prev_hist = None
    done = 0
    batch = 256
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    while True:
        buf = proc.stdout.read(fsize * batch)
        if not buf:
            break
        n = len(buf) // fsize
        if n == 0:
            break
        frames = np.frombuffer(buf[: n * fsize], dtype=np.uint8).reshape(n, h, w).astype(np.float32) / 255.0
        luma.extend(frames.mean(axis=(1, 2)).tolist())
        detail.extend(frames.std(axis=(1, 2)).tolist())
        hists = np.stack([np.histogram(f, bins=16, range=(0, 1))[0] for f in frames]).astype(np.float32)
        hists /= hists.sum(axis=1, keepdims=True)
        stack = frames if prev is None else np.concatenate([prev[None], frames])
        hstack = hists if prev_hist is None else np.concatenate([prev_hist[None], hists])
        res, cam, col = _compensated_diff(stack, win)
        hd = np.abs(np.diff(hstack, axis=0)).sum(axis=1)
        if prev is None:  # first frame has no predecessor
            res, cam, hd = np.concatenate([[0.0], res]), np.concatenate([[0.0], cam]), np.concatenate([[0.0], hd])
            col = np.concatenate([np.zeros((1, w), np.float32), col])
        motion.extend(res.tolist())
        pan.extend(cam.tolist())
        hist_d.extend(hd.tolist())
        cols.append(col.reshape(n, COLUMN_BINS, -1).mean(axis=2) if w % COLUMN_BINS == 0
                    else np.stack([np.interp(np.linspace(0, w - 1, COLUMN_BINS), np.arange(w), c) for c in col]))
        prev, prev_hist = frames[-1], hists[-1]
        done += n
        if progress:
            progress(min(1.0, done / expected))
    proc.wait()
    t_err.join(timeout=5)
    stderr = b"".join(err_chunks).decode("utf-8", errors="replace").strip()
    if done == 0:
        return {}, 0, stderr
    data = {
        "motion": np.array(motion, dtype=np.float32),
        "pan": np.array(pan, dtype=np.float32),
        "luma": np.array(luma, dtype=np.float32),
        "detail": np.array(detail, dtype=np.float32),
        "hist_diff": np.array(hist_d, dtype=np.float32),
        "columns": np.concatenate(cols).astype(np.float32),
    }
    return data, done, stderr


def _compensated_diff(stack: "np.ndarray", win: "np.ndarray"):
    """For consecutive frames: residual motion after removing the camera pan (phase correlation),
    the pan size (fraction of the frame width) and the per-column residual motion."""
    k, h, w = stack.shape
    spec = np.fft.rfft2(stack * win)
    cross = spec[1:] * np.conj(spec[:-1])
    cross /= np.abs(cross) + 1e-9
    corr = np.fft.irfft2(cross, s=(h, w))
    flat = corr.reshape(k - 1, -1).argmax(axis=1)
    dys, dxs = np.unravel_index(flat, (h, w))
    dys = np.where(dys > h // 2, dys - h, dys)
    dxs = np.where(dxs > w // 2, dxs - w, dxs)
    res = np.empty(k - 1)
    cols = np.zeros((k - 1, w), np.float32)
    for i in range(k - 1):
        dy, dx = int(dys[i]), int(dxs[i])
        if abs(dy) > h // 4 or abs(dx) > w // 4:   # implausible shift (cut/flash): no compensation
            dy = dx = 0
        moved = np.roll(stack[i], (dy, dx), axis=(0, 1))
        d = np.abs(stack[i + 1] - moved)
        y0, y1 = max(0, dy), h + min(0, dy)        # ignore the wrapped-around border
        x0, x1 = max(0, dx), w + min(0, dx)
        valid = d[y0:y1, x0:x1]
        res[i] = valid.mean() if valid.size else d.mean()
        cols[i, x0:x1] = valid.mean(axis=0) if valid.size else 0
    cam = np.hypot(dys, dxs) / w
    return res, cam, cols


def _finish(data: dict, info: dict, fps_a: float) -> dict:
    motion = data["motion"].astype(np.float64)
    luma = data["luma"].astype(np.float64)
    pan = data["pan"].astype(np.float64)
    detail = data["detail"].astype(np.float64)
    hd = data["hist_diff"].astype(np.float64)
    n = motion.size

    # Hard cuts: a big change in the brightness histogram (adaptive threshold).
    med = np.median(hd)
    mad = np.median(np.abs(hd - med)) + 1e-6
    is_cut = (hd > max(0.45, med + 8 * mad)) & (motion > np.percentile(motion, 60))
    is_cut[0] = False
    raw_cuts = np.flatnonzero(is_cut)

    # Several "cuts" in a row are really a fade/dissolve: keep one cut time, mark the region.
    clusters: list[list[int]] = []
    for i in raw_cuts:
        if clusters and i - clusters[-1][-1] <= max(1, int(0.6 * fps_a)):
            clusters[-1].append(int(i))
        else:
            clusters.append([int(i)])
    cut_idx = np.array([c[0] for c in clusters], dtype=np.int64)
    transition = np.zeros(n, dtype=bool)
    for c in clusters:
        transition[c[0]:c[-1] + 1] = True

    # A cut/fade is a big frame difference but not action - replace it with the calmer neighbours.
    m = motion.copy()
    for c in clusters:
        a, b = c[0], c[-1]
        nb = np.concatenate([motion[max(0, a - 3):a], motion[b + 1:b + 4]])
        m[a:b + 1] = np.min(nb) if nb.size else 0.0
    is_cut = transition

    scale = np.percentile(m, 95) if n > 10 else m.max()
    m_norm = np.clip(m / (scale + 1e-6), 0, 1.2) / 1.2

    # Impact flashes: sudden brightness jumps (common in anime hits).
    base = np.array([np.median(luma[max(0, i - 6):i]) if i > 0 else luma[0] for i in range(n)])
    impact = np.clip((luma - base - 0.08) / 0.2, 0, 1)
    impact[is_cut] = 0

    dark = np.clip((0.07 - luma) / 0.05, 0, 1)
    flat = np.clip((0.06 - detail) / 0.04, 0, 1)                 # empty sky, blank/white screens
    fast_pan = np.clip(pan / 0.06, 0, 1) * (~is_cut)            # whip pans still feel energetic
    action = np.clip(0.85 * _smooth(m_norm, 3) + 0.15 * _smooth(fast_pan, 3) + 0.35 * _smooth(impact, 3)
                     - 0.7 * dark - 0.5 * flat, 0, 1)

    cols = data["columns"].astype(np.float64)
    weight = cols ** 2
    pos = (np.arange(COLUMN_BINS) + 0.5) / COLUMN_BINS
    tot = weight.sum(axis=1)
    centroid = np.where(tot > 1e-9, (weight * pos).sum(axis=1) / np.maximum(tot, 1e-12), 0.5)

    # Top moments for the report: peaks of 1-second smoothed action, at least 3 s apart.
    sm = _smooth(action, int(fps_a))
    order = np.argsort(sm)[::-1]
    picks: list[int] = []
    for i in order:
        if len(picks) >= 5 or sm[i] <= 0.05:
            break
        if all(abs(i - p) > 3 * fps_a for p in picks):
            picks.append(int(i))
    summary = {
        "frames": int(n),
        "fps": fps_a,
        "mean_action": round(float(action.mean()), 3) if n else 0.0,
        "scene_cuts": [round(float(i / fps_a), 2) for i in cut_idx],
        "impacts": int((impact > 0.5).sum()),
        "dark_fraction": round(float((luma < 0.07).mean()), 3) if n else 0.0,
        "top_moments": [{"time": round(i / fps_a, 2), "score": round(float(sm[i]), 3)} for i in sorted(picks)],
    }
    return {"summary": summary, "action": action, "luma": luma, "centroid": centroid,
            "cut_times": cut_idx / fps_a, "fps": fps_a}
