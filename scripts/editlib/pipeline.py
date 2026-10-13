"""Glue used by the command-line scripts: scan -> analyze -> plan -> render."""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import audio, motion, planner, probe
from .common import (EditError, PROJECT_ROOT, ffmpeg_version, find_tool, fmt_time, load_config, log,
                     resolve_dir, setup_logging, timestamp)


def start(script: str, config_path: str | None) -> tuple[dict, Path]:
    cfg, warnings = load_config(Path(config_path) if config_path else None)
    log_path = setup_logging(script, resolve_dir(cfg, "log_dir"))
    log.debug("Config file: %s", cfg["_config_path"])
    log.debug("Effective config: %s", json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")},
                                                  ensure_ascii=False))
    for w in warnings:
        log.warning("Config: %s", w)
    find_tool("ffmpeg", cfg)
    find_tool("ffprobe", cfg)
    log.debug("FFmpeg: %s (%s)", ffmpeg_version(cfg), find_tool("ffmpeg", cfg))
    return cfg, log_path


def profile(cfg: dict, draft: bool) -> dict:
    src = cfg["draft"] if draft else cfg["output"]
    crf = int(src["video_crf"])
    return {"name": "draft" if draft else "final", "width": int(src["width"]), "height": int(src["height"]),
            "fps": float(src["fps"]), "crf": crf, "preset": src["x264_preset"],
            # intermediates are re-encoded once more at the end, so keep them near-lossless
            "intermediate_crf": max(0, min(crf - 6, 16))}


def scan(cfg: dict) -> dict:
    in_dir, mu_dir = resolve_dir(cfg, "input_dir"), resolve_dir(cfg, "music_dir")
    for d in (in_dir, mu_dir):
        d.mkdir(parents=True, exist_ok=True)
    log.info("Scanning videos in: %s", in_dir)
    videos, vnotes = probe.scan_folder(in_dir, "video", cfg)
    log.info("Scanning music in:  %s", mu_dir)
    music, mnotes = probe.scan_folder(mu_dir, "music", cfg)
    for n in vnotes + mnotes:
        log.warning("  note: %s", n)
    _report(videos, music)
    probe.require_inputs(videos, music, in_dir, mu_dir)
    return {"videos": videos, "music": music, "input_dir": in_dir, "music_dir": mu_dir}


def _report(videos, music) -> None:
    log.info("")
    log.info("VIDEOS (%d found)", len(videos))
    for v in videos:
        status = "OK " if v.ok else "BAD"
        if v.ok:
            extra = (f"{fmt_time(v.duration)}  {v.display_width}x{v.display_height}  {v.fps:.3f} fps  "
                     f"{v.vcodec}/{v.pix_fmt}  audio: {v.acodec or 'none'}")
        else:
            extra = ""
        log.info("  [%s] %s  %s", status, v.name, extra)
        for p in v.problems:
            log.error("        problem: %s", p)
        for w in v.warnings:
            log.warning("        note: %s", w)
    log.info("MUSIC (%d found)", len(music))
    for m in music:
        status = "OK " if m.ok else "BAD"
        extra = f"{fmt_time(m.duration)}  {m.acodec}  {m.sample_rate} Hz  {m.channels} ch" if m.ok else ""
        log.info("  [%s] %s  %s", status, m.name, extra)
        for p in m.problems:
            log.error("        problem: %s", p)
    log.info("")


def pick_music(cfg: dict, music: list) -> probe.MediaInfo:
    good = [m for m in music if m.ok]
    wanted = str(cfg["edit"]["music_file"]).strip()
    if wanted.lower() == "auto":
        if len(good) > 1:
            log.info("Several songs found; using '%s'. Set edit.music_file in config.json to choose another.",
                     good[0].name)
        return good[0]
    for m in good:
        if m.name.lower() == wanted.lower() or Path(m.name).stem.lower() == wanted.lower():
            return m
    raise EditError(f"edit.music_file is '{wanted}' but no usable song with that name is in the music folder.\n"
                    "Available: " + ", ".join(m.name for m in good))


def analyze(cfg: dict, media: dict) -> dict:
    cache = resolve_dir(cfg, "work_dir") / "cache"
    song = pick_music(cfg, media["music"])
    log.info("Detecting beats in '%s'...", song.name)
    t0 = time.time()
    mus = audio.analyze_music(song.path, cfg, cache)
    log.info("  tempo %.1f BPM, %d beats, drop/energy peak at %s  (%.1f s)", mus["bpm"], len(mus["beat_times"]),
             fmt_time(mus["drop_time"]), time.time() - t0)
    if mus["tempo_confidence"] < 0.15:
        log.warning("  Beat detection is unsure about this song (confidence %.2f). If cuts feel off-beat, "
                    "set beats.bpm_override in config.json.", mus["tempo_confidence"])

    vids = []
    for v in [v for v in media["videos"] if v.ok]:
        log.info("Analyzing motion in '%s' (%s)...", v.name, fmt_time(v.duration))
        t0 = time.time()
        last = [-1]

        def prog(frac, last=last):
            pct = int(frac * 100) // 20 * 20
            if pct > last[0] and pct < 100:
                last[0] = pct
                print(f"    {pct}%", flush=True)

        mo = motion.analyze_video(v.to_dict(), cfg, cache, progress=prog)
        s = mo["summary"]
        tops = ", ".join(f"{fmt_time(m['time'])} ({m['score']:.2f})" for m in s["top_moments"][:5])
        log.info("  %d scene cuts, mean action %.2f, best moments: %s  (%.1f s)", len(s["scene_cuts"]),
                 s["mean_action"], tops or "-", time.time() - t0)
        vids.append({"info": v.to_dict(), "motion": mo})
    if not vids:
        raise EditError("No usable videos after analysis.")

    summary = {
        "created": timestamp(),
        "music": {k: v for k, v in mus.items() if k not in ("energy_10hz",)},
        "videos": [{"info": v["info"], "motion": v["motion"]["summary"]} for v in vids],
        "skipped": [m.to_dict() for m in media["videos"] + media["music"] if not m.ok],
    }
    out = resolve_dir(cfg, "work_dir") / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    log.debug("Analysis summary written to %s", out)
    return {"music": mus, "videos": vids, "song": song}


def make_plan(cfg: dict, data: dict, duration: float | None = None) -> dict:
    mus = data["music"]
    wanted = float(duration or cfg["edit"]["duration_seconds"])
    start, length = audio.choose_music_window(mus, wanted, cfg["edit"]["music_start_seconds"])
    plan = planner.plan_edit(cfg, mus, start, length, data["videos"])
    st = plan["stats"]
    log.info("Edit plan: %.2f s of music from %s, %d clips (avg %.2f s), %d speed ramps, %d zooms, "
             "%d shakes, %d flashes, %d transition accents", length, fmt_time(start), st["clips"],
             st["avg_clip_seconds"], st["ramped"], st["zooms"], st["shakes"], st["flashes"], st["accents"])
    for s in plan["segments"]:
        log.debug("  clip %02d %6.2f-%6.2f  src %-28s @%7.2f len %.2f  speed=%s zoom=%s shake=%s flash=%.2f "
                  "glow=%.2f accent=%s crop_x=%.2f", s["index"] + 1, s["t0"], s["t1"], s["source"]["name"][:28],
                  s["source"]["start"], s["src_len"], s["speed"]["mode"], (s["zoom"] or {}).get("type"),
                  bool(s["shake"]), s["flash"], s["glow"], s["accent"], s.get("crop_x", 0.5))
    return plan


def output_path(cfg: dict, kind: str) -> Path:
    out_dir = resolve_dir(cfg, "output_dir")
    for key in ("input_dir", "music_dir"):
        src = resolve_dir(cfg, key).resolve()
        o = out_dir.resolve()
        if o == src or src in o.parents:
            raise EditError(f"paths.output_dir must not be inside paths.{key} (originals are never written to).")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["edit"]["output_name"]
    suffix = "" if kind == "final" else f"_{kind}"
    path = out_dir / f"{name}_{timestamp()}{suffix}.mp4"
    n = 2
    while path.exists():  # never overwrite an earlier render
        path = out_dir / f"{name}_{timestamp()}{suffix}_{n}.mp4"
        n += 1
    return path


def save_plan(plan: dict, log_path: Path) -> Path:
    p = log_path.with_name(log_path.stem + "_plan.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=1, ensure_ascii=False)
    return p


def report_result(res: dict, path: Path) -> bool:
    log.info("")
    log.info("Output: %s", path)
    log.info("  %s, %s, %.2f s, %d frames (expected %d), %.1f MB, rendered in %.0f s", res["video"], res["audio"],
             res["duration"], res["frames"], res["expected_frames"], res["size_mb"], res["render_seconds"])
    if "loudness_lufs" in res:
        log.info("  loudness %.1f LUFS, peak %s dBFS", res["loudness_lufs"], res.get("true_peak_dbfs", "?"))
    log.debug("Verification: %s", json.dumps(res, ensure_ascii=False))
    for p in res["problems"]:
        log.warning("  CHECK FAILED: %s", p)
    return not res["problems"]


def relpath(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)
