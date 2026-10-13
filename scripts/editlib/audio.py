"""Music analysis with numpy only: onset envelope, tempo, beat tracking, beat strength,
downbeats, energy curve and drop detection.

Method (well-known MIR techniques, re-implemented so no librosa is needed):
  1. Decode to mono 22.05 kHz with FFmpeg.
  2. Log-frequency spectrogram -> positive spectral flux = onset strength envelope.
  3. Tempo from the autocorrelation of the envelope with a soft prior around 120 BPM.
  4. Beat positions with Ellis' dynamic-programming beat tracker (2007).
  5. Beat strength from low-frequency (kick/bass) flux, energy from RMS loudness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .common import EditError, find_tool, log, require_numpy, run

np = require_numpy()

SR = 22050
N_FFT = 2048
HOP = 512
ENV_FPS = SR / HOP          # ~43.07 onset frames per second
ANALYSIS_VERSION = 3


def load_audio_mono(path: str, cfg: dict, start: float = 0.0, duration: float | None = None) -> "np.ndarray":
    cmd = [find_tool("ffmpeg", cfg), "-hide_banner", "-nostdin", "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    proc = run(cmd, what="audio decode")
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if y.size < SR:
        raise EditError(f"Could not decode enough audio from '{Path(path).name}'.")
    return y


def _frames(y: "np.ndarray") -> "np.ndarray":
    pad = N_FFT // 2
    y = np.pad(y, (pad, pad), mode="reflect" if y.size > pad else "constant")
    n = 1 + (y.size - N_FFT) // HOP
    return np.lib.stride_tricks.as_strided(y, shape=(n, N_FFT), strides=(y.strides[0] * HOP, y.strides[0]))


def _log_bands(n_bins: int, n_bands: int = 64, fmin: float = 30.0, fmax: float = 11000.0):
    """Triangular filters spaced logarithmically (similar to a mel filterbank)."""
    freqs = np.linspace(0, SR / 2, n_bins)
    edges = np.geomspace(fmin, fmax, n_bands + 2)
    w = np.zeros((n_bins, n_bands), dtype=np.float32)
    for b in range(n_bands):
        lo, mid, hi = edges[b], edges[b + 1], edges[b + 2]
        up = (freqs - lo) / max(mid - lo, 1e-6)
        down = (hi - freqs) / max(hi - mid, 1e-6)
        w[:, b] = np.maximum(0, np.minimum(up, down))
        if w[:, b].sum() == 0:  # band narrower than one FFT bin
            w[np.argmin(np.abs(freqs - mid)), b] = 1.0
        w[:, b] /= w[:, b].sum()
    centers = edges[1:-1]
    return w, centers


def onset_features(y: "np.ndarray") -> dict:
    """Return onset envelope (full + low band) and RMS loudness, one value per hop."""
    frames = _frames(y)
    win = np.hanning(N_FFT).astype(np.float32)
    weights, centers = _log_bands(N_FFT // 2 + 1)
    n = frames.shape[0]
    band_db = np.empty((n, weights.shape[1]), dtype=np.float32)
    rms = np.empty(n, dtype=np.float32)
    chunk = 2048
    for s in range(0, n, chunk):
        fr = frames[s:s + chunk]
        rms[s:s + chunk] = np.sqrt(np.mean(fr.astype(np.float32) ** 2, axis=1) + 1e-12)
        mag2 = np.abs(np.fft.rfft(fr * win, axis=1)) ** 2
        band_db[s:s + chunk] = 10 * np.log10(mag2 @ weights + 1e-10)
    band_db = np.maximum(band_db, band_db.max() - 80.0)
    diff = np.maximum(0.0, np.diff(band_db, axis=0))
    flux = np.concatenate([[0.0], diff.mean(axis=1)])
    low = centers < 160.0
    # linear-magnitude low-band flux: proportional to how hard the kick/bass actually hits
    low_mag = np.sqrt(np.maximum(10 ** (band_db[:, low] / 10), 0)).sum(axis=1)
    low_flux = np.concatenate([[0.0], np.maximum(0.0, np.diff(low_mag))])
    return {"flux": flux.astype(np.float64), "low_flux": low_flux.astype(np.float64), "rms": rms.astype(np.float64)}


def _moving_average(x: "np.ndarray", size: int) -> "np.ndarray":
    size = max(1, int(size))
    if size == 1:
        return x.copy()
    kernel = np.ones(size) / size
    padded = np.pad(x, (size // 2, size - 1 - size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def estimate_tempo(env: "np.ndarray", bpm_min=60.0, bpm_max=200.0, prior_bpm=120.0) -> tuple[float, float]:
    """Return (bpm, confidence 0..1) from the onset-envelope autocorrelation."""
    x = env - _moving_average(env, int(ENV_FPS * 2))
    x = x - x.mean()
    n = x.size
    spec = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    if ac[0] <= 0:
        return prior_bpm, 0.0
    ac = ac / ac[0]
    bpms = np.arange(bpm_min, bpm_max + 0.01, 0.25)
    lags = 60.0 * ENV_FPS / bpms
    idx = np.arange(n)
    vals = np.interp(lags, idx, ac)
    vals2 = np.interp(lags * 2, idx, ac)
    score = (vals + 0.5 * vals2) * np.exp(-0.5 * (np.log2(bpms / prior_bpm) / 1.0) ** 2)
    best = int(np.argmax(score))
    # confidence: how much the winning lag stands out from typical autocorrelation values
    conf = float(np.clip((vals[best] - np.median(vals)) / (np.abs(vals).max() + 1e-9), 0, 1))
    return float(bpms[best]), conf


def track_beats(env: "np.ndarray", bpm: float, tightness: float = 100.0) -> "np.ndarray":
    """Ellis (2007) dynamic-programming beat tracker. Returns beat frame indices."""
    period = 60.0 * ENV_FPS / bpm
    std = env.std()
    norm = env / std if std > 0 else env
    half = int(round(period))
    g = np.exp(-0.5 * (np.arange(-half, half + 1) * 32.0 / period) ** 2)
    local = np.convolve(norm, g, mode="same")
    n = local.size
    lo_off, hi_off = -int(round(2 * period)), -int(round(period / 2))
    offsets = np.arange(lo_off, hi_off + 1)
    txcost = -tightness * np.log(-offsets / period) ** 2
    cum = np.zeros(n)
    back = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        start, end = i + lo_off, i + hi_off
        if end < 0:
            cum[i] = local[i]
            continue
        s = max(start, 0)
        cand = cum[s:end + 1] + txcost[s - start:]
        j = int(np.argmax(cand))
        cum[i] = local[i] + cand[j]
        back[i] = s + j
    # last beat: the final local maximum of the cumulative score that is reasonably strong
    peaks = np.flatnonzero((cum[1:-1] > cum[:-2]) & (cum[1:-1] >= cum[2:])) + 1
    if peaks.size == 0:
        return np.array([], dtype=np.int64)
    med = np.median(cum[peaks])
    last = int(peaks[cum[peaks] >= 0.5 * med][-1])
    beats = [last]
    while back[beats[-1]] >= 0:
        beats.append(int(back[beats[-1]]))
    beats = np.array(beats[::-1], dtype=np.int64)
    # trim weak leading/trailing beats (silence at the start/end of the song)
    strength = local[beats]
    thresh = 0.5 * np.sqrt(np.mean(strength ** 2))
    good = np.flatnonzero(strength >= thresh)
    if good.size:
        beats = beats[good[0]:good[-1] + 1]
    return beats


# The positive log-spectral flux peaks slightly before the perceived onset (the analysis
# window "sees" the hit early). Measured on synthetic drum tracks; added back here.
ONSET_LATENCY_S = 0.010


def refine_beat_times(env: "np.ndarray", beat_frames: "np.ndarray") -> "np.ndarray":
    """Move each beat to the onset peak within +-2 frames, with sub-frame (parabolic) precision."""
    out = []
    n = env.size
    for b in beat_frames:
        lo, hi = max(1, b - 2), min(n - 2, b + 2)
        if hi <= lo:
            out.append(float(b))
            continue
        m = lo + int(np.argmax(env[lo:hi + 1]))
        y0, y1, y2 = env[m - 1], env[m], env[m + 1]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom < 0 else 0.0
        peak = m + float(np.clip(frac, -0.5, 0.5))
        # only trust the peak if it is a real onset, otherwise keep the tracker's grid position
        out.append(peak if y1 > 0.5 * np.median(env[max(0, b - 40):b + 40]) + 1e-9 else float(b))
    return np.array(out) / ENV_FPS + ONSET_LATENCY_S


def _grid_bpm(beat_times: "np.ndarray", fallback: float) -> float:
    """Tempo from a straight-line fit through the beats (more precise than one interval)."""
    if beat_times.size < 4:
        return fallback
    ibi = np.diff(beat_times)
    med = np.median(ibi)
    # use the longest run of regular beats (ignore tempo glitches/phase jumps)
    ok = np.abs(ibi - med) < 0.25 * med
    idx = np.concatenate([[0], np.cumsum(np.where(ok, 1.0, 1.0 + np.round(ibi / med) - 1))])
    slope = np.polyfit(idx, beat_times, 1)[0]
    return float(60.0 / slope) if slope > 0 else fallback


def _cache_key(path: str, extra: dict) -> str:
    st = Path(path).stat()
    raw = json.dumps([str(Path(path).resolve()), st.st_size, int(st.st_mtime), ANALYSIS_VERSION, extra],
                     ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def analyze_music(path: str, cfg: dict, cache_dir: Path | None = None) -> dict:
    """Full beat map for one song. Cached in work/cache because it never changes for the same file."""
    bpm_override = cfg["beats"]["bpm_override"]
    key = _cache_key(path, {"bpm": bpm_override})
    cache_file = cache_dir / f"music_{key}.json" if cache_dir else None
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            log.debug("Music analysis loaded from cache %s", cache_file.name)
            return data
        except (OSError, json.JSONDecodeError):
            pass

    y = load_audio_mono(path, cfg)
    duration = y.size / SR
    feats = onset_features(y)
    env = feats["flux"]
    if bpm_override:
        bpm, conf = float(bpm_override), 1.0
    else:
        bpm, conf = estimate_tempo(env)
    beat_frames = track_beats(env, bpm)
    if beat_frames.size < 4:
        raise EditError(f"Could not find a steady beat in '{Path(path).name}'. "
                        "Try a song with clear drums, or set beats.bpm_override in config.json.")
    beat_times = refine_beat_times(env, beat_frames)
    refined_bpm = _grid_bpm(beat_times, bpm)

    # --- loudness / energy curve ----------------------------------------------------------
    db = 20 * np.log10(feats["rms"] + 1e-6)
    smooth = _moving_average(db, int(ENV_FPS * 1.5))
    lo, hi = np.percentile(smooth, 5), np.percentile(smooth, 95)
    energy = np.clip((smooth - lo) / max(hi - lo, 1e-6), 0, 1)
    beat_energy = energy[np.minimum(beat_frames, energy.size - 1)]

    # --- beat strength: how hard each beat hits (kick/bass punch, transient, loudness) -----
    low = feats["low_flux"] / (np.percentile(feats["low_flux"], 95) + 1e-9)
    full = env / (np.percentile(env, 95) + 1e-9)
    combo = 0.6 * low + 0.4 * full
    raw = np.array([combo[max(0, b - 2):b + 3].max() for b in beat_frames])
    raw = np.sqrt(np.clip(raw, 0, None)) * (0.35 + 0.65 * beat_energy)
    strength = np.clip(raw / (np.percentile(raw, 90) + 1e-9), 0, 1)

    # --- downbeats: choose the 4/4 bar phase with the strongest low-end hits --------------
    low_at_beats = np.array([low[max(0, b - 2):b + 3].max() for b in beat_frames])
    phase_scores = [low_at_beats[p::4].mean() if low_at_beats[p::4].size else 0 for p in range(4)]
    phase = int(np.argmax(phase_scores))
    downbeat = [(i - phase) % 4 == 0 for i in range(beat_frames.size)]

    # --- drop: start of a loud section with the biggest energy rise (8 beats after vs before)
    drop_time = float(beat_times[0])
    best_rise = -1.0
    loud = np.percentile(beat_energy, 90)
    for i in range(beat_frames.size):
        before = beat_energy[max(0, i - 8):i]
        after = beat_energy[i:i + 8]
        if before.size < 4 or after.size < 4 or after.mean() < 0.8 * loud:
            continue
        rise = after.mean() - before.mean() + (0.05 if downbeat[i] else 0.0)
        if rise > best_rise:
            best_rise, drop_time = rise, float(beat_times[i])

    env10 = energy[:: max(1, int(round(ENV_FPS / 10)))]
    result = {
        "path": str(path),
        "name": Path(path).name,
        "duration": float(duration),
        "bpm": round(refined_bpm, 2),
        "bpm_initial": round(bpm, 2),
        "tempo_confidence": round(conf, 3),
        "beat_times": [round(float(t), 4) for t in beat_times],
        "beat_strength": [round(float(s), 3) for s in strength],
        "beat_energy": [round(float(e), 3) for e in beat_energy],
        "downbeat": downbeat,
        "drop_time": round(drop_time, 3),
        "drop_rise": round(float(best_rise), 3),
        "energy_10hz": [round(float(e), 3) for e in env10],
        "version": ANALYSIS_VERSION,
    }
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f)
    return result


def choose_music_window(music: dict, wanted: float, start_setting) -> tuple[float, float]:
    """Pick where in the song the edit starts, and the final edit duration (ending on a beat)."""
    beats = np.array(music["beat_times"])
    down = np.array(music["downbeat"], dtype=bool)
    dur = music["duration"]
    energy = np.array(music["energy_10hz"])

    if start_setting != "auto":
        start = float(start_setting)
        if start >= dur - 3:
            raise EditError(f"edit.music_start_seconds ({start}) is past the end of the song ({dur:.1f} s).")
        # snap to the nearest beat so the first cut lands on the music
        near = beats[np.argmin(np.abs(beats - start))]
        if abs(near - start) < 0.25:
            start = float(near)
    else:
        cands = beats[down] if down.any() else beats
        cands = cands[cands + min(wanted, dur * 0.9) <= dur + 0.01]
        if cands.size == 0:
            start = float(beats[0])
        else:
            best, start = -1e9, float(cands[0])
            for t in cands:
                a, b = int(t * 10), int(min(dur, t + wanted) * 10)
                seg = energy[a:max(b, a + 1)]
                score = float(seg.mean()) if seg.size else 0.0
                rel = (music["drop_time"] - t) / max(wanted, 1e-6)
                if 0.1 <= rel <= 0.6:
                    score += 0.35  # include the build-up and the drop
                if score > best + 1e-6:
                    best, start = score, float(t)

    available = dur - start
    length = min(wanted, available)
    if length < wanted - 0.01:
        log.warning("The song only has %.1f s left after the start point; the edit will be %.1f s long.",
                    available, length)
    # end on a downbeat (or beat) near the requested length, so the edit finishes on the music
    end_target = start + length
    ends = beats[(beats > start + 1.0) & (beats <= start + available + 1e-6)]
    downs = beats[down & (beats > start + 1.0) & (beats <= start + available + 1e-6)]
    pick = None
    for pool, tol in ((downs, 1.6), (ends, 0.6)):
        if pool.size:
            cand = pool[np.argmin(np.abs(pool - end_target))]
            if abs(cand - end_target) <= tol and cand <= start + available:
                pick = cand
                break
    if pick is not None:
        length = float(pick - start)
    return float(start), float(length)
