"""Create royalty-free demo media so the pipeline can be tried without your own files.

  python scripts/make_demo_media.py                 -> music/Demo Phonk 140 BPM (original).mp3
  python scripts/make_demo_media.py --bpm 150       -> different tempo
  python scripts/make_demo_media.py --test-video    -> also a synthetic motion-test clip in input/

The song is synthesized from scratch (808 kick, clap, hi-hats, phonk cowbell, sub bass),
so it is 100% original and free to use. Existing files are never overwritten.
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from editlib.common import (EditError, find_tool, fix_console_encoding,  # noqa: E402
                            load_config, require_numpy, resolve_dir, run)

np = require_numpy()
SR = 44100


def _env(n, decay_s, sr=SR):
    return np.exp(-np.arange(n) / (decay_s * sr))


def kick(sr=SR, boom=0.35, level=1.0):
    n = int(sr * 0.6)
    t = np.arange(n) / sr
    freq = 45 + 110 * np.exp(-t / 0.04)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    body = np.sin(phase) * _env(n, boom, sr)
    click = np.random.default_rng(1).standard_normal(n) * _env(n, 0.004, sr) * 0.4
    return level * np.tanh(1.6 * (body + click))


def clap(sr=SR, level=0.7, rng=None):
    rng = rng or np.random.default_rng(2)
    n = int(sr * 0.3)
    noise = rng.standard_normal(n)
    noise = noise - np.concatenate([[0], noise[:-1]]) * 0.6           # brighten
    envl = np.zeros(n)
    for off in (0.0, 0.011, 0.022):                                     # 3 quick bursts = clap
        o = int(off * sr)
        envl[o:] += _env(n - o, 0.012 if off < 0.02 else 0.11, sr)
    tone = np.sin(2 * np.pi * 190 * np.arange(n) / sr) * _env(n, 0.05, sr) * 0.4
    return level * (noise * envl * 0.5 + tone)


def hat(sr=SR, level=0.25, decay=0.03, rng=None):
    rng = rng or np.random.default_rng(3)
    n = int(sr * 0.12)
    noise = rng.standard_normal(n)
    hp = noise - np.concatenate([[0], noise[:-1]])                       # crude high-pass
    return level * hp * _env(n, decay, sr) * 0.5


def cowbell(freq, sr=SR, level=0.35, decay=0.16):
    n = int(sr * 0.45)
    t = np.arange(n) / sr
    sq = np.sign(np.sin(2 * np.pi * freq * t)) + np.sign(np.sin(2 * np.pi * freq * 1.48 * t))
    sq = np.convolve(sq, np.ones(4) / 4, mode="same")                    # soften
    return level * 0.5 * sq * _env(n, decay, sr)


def sub(freq, length_s, sr=SR, level=0.55):
    n = int(sr * length_s)
    t = np.arange(n) / sr
    e = np.minimum(1, t / 0.005) * np.exp(-t / (length_s * 0.9))
    return level * np.tanh(1.5 * np.sin(2 * np.pi * freq * t)) * e


def pad_chord(freqs, length_s, sr=SR, level=0.12):
    n = int(sr * length_s)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for f in freqs:
        for det in (-0.003, 0.003):
            out += 2 * ((t * f * (1 + det)) % 1.0) - 1                    # detuned saws
    out = np.convolve(out, np.ones(24) / 24, mode="same")                # low-pass
    e = np.minimum(1, t / 0.3) * np.minimum(1, (length_s - t) / 0.3)
    return level * out / (2 * len(freqs)) * e


def note(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def synth_track(bpm: float = 140.0, bars: int = 28, seed: int = 11, sr: int = SR):
    """Return (stereo float array, beat_times, downbeat_times, drop_time)."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    total = bars * 4 * beat + 1.5
    L = np.zeros(int(total * sr) + sr)
    R = np.zeros_like(L)

    def add(sig, t, pan=0.0, gain=1.0):
        i = int(round(t * sr))
        j = min(L.size, i + sig.size)
        if i >= L.size:
            return
        L[i:j] += sig[: j - i] * gain * (1 - max(0, pan))
        R[i:j] += sig[: j - i] * gain * (1 + min(0, pan))

    # A minor phonk cowbell riff (16th grid, None = rest)
    riff = [81, None, 84, None, 76, None, 79, 81, None, 76, None, 74, 72, None, 74, None]
    roots = [45, 41, 43, 40]  # A, F, G, E (sub bass, one per bar)
    chords = [[57, 60, 64], [53, 57, 60], [55, 59, 62], [52, 56, 59]]
    k_drop, k_soft = kick(sr), kick(sr, boom=0.18, level=0.55)
    cl = clap(sr, rng=rng)
    crash = rng.standard_normal(int(sr * 2.0)) * _env(int(sr * 2.0), 0.6, sr) * 0.25

    sections = []
    for bar in range(bars):
        if bar < 4:
            sections.append("intro")
        elif bar < 8:
            sections.append("build")
        elif bar < 16:
            sections.append("drop")
        elif bar < 20:
            sections.append("break")
        else:
            sections.append("drop")

    for bar, sec in enumerate(sections):
        t0 = bar * 4 * beat
        root, chord = roots[bar % 4], chords[bar % 4]
        if sec in ("intro", "break"):
            add(pad_chord([note(n) for n in chord], 4 * beat, sr), t0, gain=1.0)
        for step in range(16):
            t = t0 + step * beat / 4
            b, sub_step = divmod(step, 4)
            # cowbell riff
            n = riff[step]
            if n is not None:
                lvl = {"intro": 0.45, "build": 0.6, "drop": 1.0, "break": 0.5}[sec]
                add(cowbell(note(n), sr), t, pan=0.15, gain=lvl)
            # hats
            if sec in ("build", "drop") and sub_step in (0, 2):
                add(hat(sr, rng=rng), t, pan=-0.3, gain=1.0 if sub_step == 0 else 0.6)
            if sec == "drop" and step in (13, 15):
                add(hat(sr, level=0.18, rng=rng), t + beat / 8, pan=-0.3)
            if sec == "intro" and sub_step == 2:
                add(hat(sr, level=0.12, rng=rng), t, pan=-0.3)
        for b in range(4):
            t = t0 + b * beat
            if sec == "build":
                add(k_soft, t)
            elif sec == "drop":
                add(k_drop, t)
                add(sub(note(root), beat * 0.95, sr), t)
                if b in (1, 3):
                    add(cl, t, gain=0.9)
            elif sec == "break":
                if b == 0:
                    add(k_drop, t, gain=0.8)
                    add(sub(note(root), beat * 1.9, sr), t)
                if b == 2:
                    add(cl, t, gain=0.7)
        if sec == "build" and bar == 7:                       # snare roll + riser into the drop
            for k in range(16):
                add(cl, t0 + k * beat / 4, gain=0.15 + 0.5 * k / 15)
            n = int(4 * beat * sr)
            riser = rng.standard_normal(n) * np.linspace(0, 0.25, n)
            add(riser - np.concatenate([[0], riser[:-1]]) * 0.9, t0)
        if sec == "drop" and (bar == 0 or sections[bar - 1] != "drop"):
            add(crash, t0, gain=1.0)

    mix = np.stack([L, R], axis=1)[: int(total * sr)]
    mix = np.tanh(mix * 1.4)                                  # gentle saturation / glue
    mix *= 0.89 / (np.abs(mix).max() + 1e-9)                  # peak at about -1 dBFS
    beats = [i * beat for i in range(bars * 4)]
    downs = beats[::4]
    drop_time = 8 * 4 * beat
    return mix.astype(np.float32), beats, downs, drop_time


def write_wav(path: Path, stereo: "np.ndarray", sr: int = SR) -> None:
    pcm = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 1000):
        cand = path.with_name(f"{path.stem} ({i}){path.suffix}")
        if not cand.exists():
            return cand
    raise EditError(f"Too many existing files named like {path.name}")


def make_music(out_dir: Path, bpm: float, name: str | None, cfg: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stereo, beats, _, drop = synth_track(bpm=bpm)
    base = name or f"Demo Phonk {bpm:g} BPM (original)"
    wav = unique_path(out_dir / f"{base}.wav")
    write_wav(wav, stereo)
    mp3 = unique_path(out_dir / f"{base}.mp3")
    proc = run([find_tool("ffmpeg", cfg), "-hide_banner", "-nostdin", "-v", "error", "-i", str(wav),
                "-c:a", "libmp3lame", "-q:a", "2", str(mp3)], check=False, what="mp3 encode")
    if proc.returncode == 0 and mp3.exists():
        wav.unlink()  # remove the temporary WAV this script just created
        out = mp3
    else:
        out = wav
    print(f"Created demo song: {out}  ({len(beats)} beats at {bpm:g} BPM, drop at {drop:.2f} s)")
    return out


def make_test_video(out_dir: Path, cfg: dict) -> Path:
    """4 s still picture, 4 s fast motion, 4 s still colour - used to check the motion analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = unique_path(out_dir / "motion test clip.mp4")
    graph = ("testsrc2=s=1280x720:r=24:d=1,trim=end_frame=1,loop=loop=95:size=1,setpts=N/24/TB,format=yuv420p[a];"
             "mandelbrot=s=1280x720:r=24:end_scale=0.01:start_scale=3,trim=duration=4,format=yuv420p[b];"
             "color=c=0x203040:s=1280x720:r=24:d=4,format=yuv420p[c];"
             "[a][b][c]concat=n=3:v=1:a=0[v]")
    run([find_tool("ffmpeg", cfg), "-hide_banner", "-nostdin", "-v", "error", "-filter_complex", graph,
         "-map", "[v]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)],
        what="test video")
    print(f"Created motion test clip: {out}  (fast motion between 4 s and 8 s)")
    return out


def main() -> int:
    fix_console_encoding()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bpm", type=float, default=140.0)
    ap.add_argument("--name", help="file name without extension")
    ap.add_argument("--test-video", action="store_true", help="also create a synthetic motion test clip")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    try:
        cfg, _ = load_config(Path(args.config) if args.config else None)
        make_music(resolve_dir(cfg, "music_dir"), args.bpm, args.name, cfg)
        if args.test_video:
            make_test_video(resolve_dir(cfg, "input_dir"), cfg)
    except EditError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
