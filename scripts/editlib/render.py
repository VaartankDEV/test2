"""Rendering with FFmpeg.

Each planned clip is rendered to its own short intermediate file with its own filter chain:

  trim/seek -> deinterlace/square pixels -> 9:16 framing -> speed ramp (setpts) -> fps
  -> color grade -> glow -> camera (zoom/shake/rotation via 'perspective') -> accents
  -> flash -> tags

Then all clips are joined (concat demuxer) and encoded once together with the music, which is
faded, loudness-normalised (2-pass EBU R128 loudnorm) and peak-limited so it never clips.
Exact frame counts per clip keep picture and music in sync.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .common import EditError, find_tool, fmt_time, log, run

# Contrast/saturation are applied with 'eq'; the color tint uses per-channel RGB 'curves'
# (lookup tables, much faster than 'colorbalance' at 1080x1920).
GRADES = {
    "anime_punch": {"contrast": 1.12, "saturation": 1.30, "gamma": 1.0,      # teal shadows, warm highlights
                    "curves": "r='0/0 0.25/0.225 0.75/0.775 1/1':g='0/0 0.25/0.245 0.75/0.755 1/1'"
                              ":b='0/0.02 0.25/0.285 0.75/0.725 1/0.97'"},
    "super_saiyan_gold": {"contrast": 1.15, "saturation": 1.25, "gamma": 1.0,  # warm gold
                          "curves": "r='0/0.01 0.25/0.27 0.5/0.56 0.75/0.82 1/1':g='0/0 0.25/0.25 0.5/0.52 "
                                    "0.75/0.78 1/1':b='0/0 0.25/0.21 0.5/0.43 0.75/0.67 1/0.92'"},
    "ninja_orange": {"contrast": 1.10, "saturation": 1.22, "gamma": 1.0,     # orange mids, teal shadows
                     "curves": "r='0/0 0.25/0.22 0.5/0.55 0.75/0.8 1/1':g='0/0 0.5/0.51 1/1'"
                               ":b='0/0.03 0.25/0.28 0.5/0.45 0.75/0.7 1/0.95'"},
    "cold_night": {"contrast": 1.12, "saturation": 0.90, "gamma": 0.95,      # blue night
                   "curves": "r='0/0 0.5/0.46 1/0.95':g='0/0 0.5/0.49 1/0.98':b='0/0.04 0.5/0.56 1/1'"},
    "natural": {"contrast": 1.05, "saturation": 1.08, "gamma": 1.0, "curves": ""},
    "none": {"contrast": 1.0, "saturation": 1.0, "gamma": 1.0, "curves": ""},
}


def num(x: float, digits: int = 5) -> str:
    """Format a number for FFmpeg expressions (never scientific notation)."""
    s = f"{float(x):.{digits}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


# ---------------------------------------------------------------------------
# Filter pieces
# ---------------------------------------------------------------------------

def remap_expr(knots: list) -> str:
    """setpts expression mapping source time T -> output time (piecewise linear, monotonic)."""
    pieces = []
    for (o0, s0), (o1, s1) in zip(knots[:-1], knots[1:]):
        k = (o1 - o0) / max(s1 - s0, 1e-6)
        pieces.append((s1, f"{num(o0)}+(T-{num(s0)})*{num(k, 6)}"))
    expr = pieces[-1][1]  # beyond the last knot: continue with the last slope
    for s1, e in reversed(pieces[:-1]):
        expr = f"if(lt(T,{num(s1)}),{e},{expr})"
    return f"({expr})/TB"


def camera_filter(seg: dict, W: int, H: int, F: float) -> str | None:
    """Zoom punch / push, accent zoom snap and impact shake (incl. a little rotation) in one
    'perspective' filter. The visible window always stays inside the frame (auto margin zoom)."""
    D = seg["t1"] - seg["t0"]
    t = f"(in/{num(F)})"
    zt_terms, active = [], 0.0
    z = seg.get("zoom") or {}
    if z.get("type") == "punch":
        zt_terms.append(f"{num(z['amount'])}*pow(max(0,1-{t}/{num(z['decay'])}),2.5)")
        active = max(active, z["decay"])
    elif z.get("type") == "push":
        zt_terms.append(f"{num(z['amount'])}*{t}/{num(D)}")
        active = D + 1
    elif z.get("type") == "pull":
        zt_terms.append(f"{num(z['amount'])}*max(0,1-{t}/{num(D)})")
        active = D + 1
    if seg.get("accent") == "zoom_snap":
        dec = min(0.2, 0.5 * D)
        zt_terms.append(f"0.22*pow(max(0,1-{t}/{num(dec)}),3)")
        active = max(active, dec)
    zexpr = f"(1+{'+'.join(zt_terms)})" if zt_terms else "1"

    sh = seg.get("shake") or {}
    if sh:
        env = f"pow(max(0,1-{t}/{num(sh['decay'])}),2)"
        ax, ay = sh["amp"] * W, sh["amp"] * W * 0.8
        rot = math.radians(sh["rot_deg"])
        dx = f"({num(ax, 3)}*{env}*sin({num(2 * math.pi * sh['fx'])}*{t}+{num(sh['px'])}))"
        dy = f"({num(ay, 3)}*{env}*sin({num(2 * math.pi * sh['fy'])}*{t}+{num(sh['py'])}))"
        th = f"({num(rot, 6)}*{env}*sin({num(2 * math.pi * sh['fr'])}*{t}+{num(sh['pr'])}))"
        zx = (W * math.cos(rot) + H * math.sin(rot)) / (W - 2 * ax)
        zy = (H * math.cos(rot) + W * math.sin(rot)) / (H - 2 * ay)
        margin = max(zx, zy) - 1 + 0.003
        zexpr = f"max({zexpr},1+{num(margin)}*{env})"
        active = max(active, sh["decay"])
    else:
        dx = dy = th = None
    if active <= 0:
        return None

    hw, hh = f"(W/(2*{zexpr}))", f"(H/(2*{zexpr}))"
    cx = f"(W/2+{dx})" if dx else "(W/2)"
    cy = f"(H/2+{dy})" if dy else "(H/2)"
    coords = []
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        if th:
            x = f"{cx}+({sx})*{hw}*cos({th})-({sy})*{hh}*sin({th})"
            y = f"{cy}+({sx})*{hw}*sin({th})+({sy})*{hh}*cos({th})"
        else:
            x = f"{cx}+({sx})*{hw}"
            y = f"{cy}+({sy})*{hh}"
        coords += [x, y]
    names = ["x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"]
    opts = ":".join(f"{n}='{c}'" for n, c in zip(names, coords))
    enable = "" if active > D else f":enable='lte(t,{num(active + 1.5 / F)})'"
    return f"perspective={opts}:interpolation=linear:eval=frame{enable}"


def framing_chain(seg: dict, W: int, H: int, mode: str, matrix: str) -> tuple[list, list, list]:
    """Return (pre, post, extra): 'pre' runs on source frames before the speed remap (cheap crop),
    'post' runs on output frames and produces exactly WxH with SAR 1:1 without stretching;
    'extra' holds graph statements for blur_fill, which needs a split."""
    sc = f"flags=bicubic:in_color_matrix={matrix}:out_color_matrix=bt709:out_range=tv"
    A = W / H
    if mode in ("smart_crop", "center_crop"):
        cx = seg.get("crop_x", 0.5) if mode == "smart_crop" else 0.5
        crop = (f"crop=w='trunc(min(iw,ih*{num(A, 6)})/2)*2':h='trunc(min(ih,iw/{num(A, 6)})/2)*2'"
                f":x='(iw-ow)*{num(min(max(cx, 0), 1), 4)}':y='(ih-oh)*0.4'")
        return [crop], [f"scale={W}:{H}:{sc}", "setsar=1"], []
    if mode == "fit":
        return [], [f"scale={W}:{H}:force_original_aspect_ratio=decrease:{sc}",
                    f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black", "setsar=1"], []
    # blur_fill: blurred, darkened copy fills the frame behind the whole (un-cropped) picture
    w4, h4 = even(W / 4), even(H / 4)
    return [], ["split[fa][fb]"], [
        f"[fa]scale={w4}:{h4}:force_original_aspect_ratio=increase:{sc},crop={w4}:{h4},"
        f"gblur=sigma=6,eq=brightness=-0.12:saturation=1.15,scale={W}:{H}:flags=bilinear,setsar=1[bg]",
        f"[fb]scale={W}:{H}:force_original_aspect_ratio=decrease:{sc},setsar=1[fg]",
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1",
    ]


def segment_filtergraph(seg: dict, info: dict, prof: dict, cfg: dict, is_first: bool) -> str:
    W, H, F = prof["width"], prof["height"], prof["fps"]
    E = cfg["effects"]
    D = seg["t1"] - seg["t0"]
    matrix = "bt709" if (info.get("color_space") in ("bt709",) or
                         (not info.get("color_space") and info["height"] >= 700)) else "bt601"
    if info.get("color_space") in ("bt470bg", "smpte170m"):
        matrix = "bt601"

    head = ["setpts=PTS-STARTPTS"]
    if info.get("interlaced"):
        head.append("bwdif=mode=send_frame")
    if info.get("sar", "1:1") != "1:1":
        head += ["scale=trunc(iw*sar/2)*2:ih", "setsar=1"]

    pre, post, extra = framing_chain(seg, W, H, cfg["output"]["framing"], matrix)
    timing = []
    if E["slowmo_smoothing"] == "blend" and seg["speed"]["mode"] != "steady":
        timing.append(f"framerate=fps={num(max(F, 2 * (info.get('fps') or 24)))}")
    timing.append(f"setpts='{remap_expr(seg['time_map'])}'")
    timing.append(f"fps={num(F)}")
    timing.append("tpad=stop_mode=clone:stop_duration=2")
    # crop -> speed remap -> fps -> scale: only frames that end up in the output get scaled
    first = "[0:v]" + ",".join(head + pre + timing + post)
    stmts = []
    if extra:  # blur_fill: split -> bg/fg -> overlay -> continue
        stmts.append(first)
        stmts += extra[:-1]
        body_start = extra[-1] + ","
    else:
        body_start = first + ","

    body = []
    g = GRADES[E["color_grade"]]
    contrast = g["contrast"] * E["contrast"]
    saturation = g["saturation"] * E["saturation"]
    if abs(contrast - 1) > 1e-3 or abs(saturation - 1) > 1e-3 or abs(E["brightness"]) > 1e-3 or g["gamma"] != 1:
        body.append(f"eq=contrast={num(contrast)}:saturation={num(saturation)}:"
                    f"brightness={num(E['brightness'])}:gamma={num(g['gamma'])}")

    # ---- RGB section (accurate color math for tint/glow/accents) ----
    body.append("scale=in_color_matrix=bt709:in_range=tv,format=gbrp")
    if g["curves"]:
        body.append(f"curves={g['curves']}")
    chain = body_start + ",".join(body)

    label = 0

    def nxt():
        nonlocal label
        label += 1
        return f"s{label}"

    glow = seg.get("glow", 0.0)
    if glow > 0.01:
        a, b, c, o = nxt(), nxt(), nxt(), nxt()
        w4, h4 = even(W / 4), even(H / 4)
        stmts.append(f"{chain},split[{a}][{b}]")
        stmts.append(f"[{b}]scale={w4}:{h4}:flags=area,colorlevels=rimin=0.68:gimin=0.68:bimin=0.68,"
                     f"gblur=sigma={num(max(2.0, H / 4 * 0.012))},scale={W}:{H}:flags=bilinear[{c}]")
        stmts.append(f"[{a}][{c}]blend=all_mode=screen:all_opacity={num(min(0.8, glow * 0.8))}[{o}]")
        chain = f"[{o}]"
        sep = ""
    else:
        sep = ","

    post = []
    if E["vignette"]:
        post.append("vignette=angle=0.45")
    cam = camera_filter(seg, W, H, F)
    if cam:
        post.append(cam)
    if post:
        chain = chain + sep + ",".join(post)
        sep = ","

    accent = seg.get("accent")
    if accent in ("whip_blur", "chroma_hit"):
        k = max(3, int(round(0.12 * F)))
        m, wv, ov, o = nxt(), nxt(), nxt(), nxt()
        if accent == "whip_blur":
            fx = f"avgblur=sizeX={max(2, int(W * 0.045))}:sizeY=1"
        else:
            s = max(2, int(W * 0.012))
            fx = f"rgbashift=rh=-{s}:bh={s}:edge=smear"
        stmts.append(f"{chain}{sep}split[{m}][{wv}]")
        stmts.append(f"[{wv}]trim=end_frame={k},{fx},format=gbrap,"
                     f"fade=t=out:st=0:d={num(k / F)}:alpha=1[{ov}]")
        stmts.append(f"[{m}][{ov}]overlay=eof_action=pass:format=gbrp[{o}]")
        chain, sep = f"[{o}]", ""

    tail = ["scale=out_color_matrix=bt709:out_range=tv", "format=yuv420p"]
    if seg.get("flash", 0) > 0:
        tail.append(f"eq=brightness='{num(seg['flash'] * 0.75)}*pow(max(0,1-t/{num(min(0.14, D * 0.5))}),2)'"
                    f":contrast='1-{num(seg['flash'] * 0.3)}*pow(max(0,1-t/{num(min(0.14, D * 0.5))}),2)'"
                    ":eval=frame")
    if seg.get("fade_out"):
        tail.append(f"fade=t=out:st={num(D - seg['fade_out'])}:d={num(seg['fade_out'])}")
    if is_first:
        tail.append(f"fade=t=in:st=0:d={num(min(0.12, D / 3))}")
    tail.append("setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv")
    stmts.append(f"{chain}{sep}{','.join(tail)}[vout]")
    return ";".join(stmts)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def frame_bounds(seg: dict, F: float, offset: float = 0.0) -> tuple[int, int]:
    return int(round((seg["t0"] - offset) * F)), int(round((seg["t1"] - offset) * F))


def render_segment(seg: dict, info: dict, prof: dict, cfg: dict, out: Path, nframes: int,
                   is_first: bool) -> dict:
    graph = segment_filtergraph(seg, info, prof, cfg, is_first)
    src = seg["source"]
    need = seg["src_len"] + 0.6
    cmd = [find_tool("ffmpeg", cfg), "-hide_banner", "-nostdin", "-v", "error", "-y",
           "-ss", num(src["start"], 3), "-t", num(need, 3), "-i", src["path"],
           "-filter_complex", graph, "-map", "[vout]", "-frames:v", str(nframes), "-an",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(prof["intermediate_crf"]),
           "-pix_fmt", "yuv420p", "-r", num(prof["fps"]), str(out)]
    t0 = time.time()
    run(cmd, what=f"rendering clip {seg['index'] + 1}")
    return {"index": seg["index"], "seconds": round(time.time() - t0, 2), "cmd_len": sum(len(c) for c in cmd)}


def count_frames(path: Path, cfg: dict) -> int:
    proc = run([find_tool("ffprobe", cfg), "-v", "error", "-select_streams", "v:0", "-count_packets",
                "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)], what="ffprobe")
    try:
        return int(proc.stdout_text.strip().split(",")[0])
    except ValueError:
        return -1


def measure_loudness(music_path: str, start: float, length: float, afilter: str, cfg: dict) -> dict | None:
    proc = run([find_tool("ffmpeg", cfg), "-hide_banner", "-nostdin", "-v", "info", "-ss", num(start, 3),
                "-t", num(length, 3), "-i", music_path, "-vn", "-af",
                afilter + ",loudnorm=I=-14:TP=-1:LRA=20:print_format=json", "-f", "null", "-"],
               check=False, what="loudness measurement")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr_text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        if not all(math.isfinite(float(data[k])) for k in ("input_i", "input_tp", "input_lra", "input_thresh")):
            return None
        return data
    except (ValueError, KeyError):
        return None


def audio_filter(length: float, cfg: dict, measured: dict | None) -> str:
    A = cfg["audio"]
    fi = min(A["fade_in_seconds"], length / 4)
    fo = min(A["fade_out_seconds"], length / 3)
    base = "aformat=sample_fmts=fltp:channel_layouts=stereo"
    if fi > 0:
        base += f",afade=t=in:st=0:d={num(fi)}"
    if fo > 0:
        base += f",afade=t=out:st={num(length - fo)}:d={num(fo)}"
    tp = A["true_peak_db"]
    if measured is not None:
        base += (f",loudnorm=I={num(A['target_lufs'])}:TP={num(tp)}:LRA=20"
                 f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
                 f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
                 f":offset={measured['target_offset']}:linear=true")
    limit = 10 ** (tp / 20)
    # latency=1 compensates the limiter's look-ahead so the music stays frame-accurate.
    return (f"{base},aresample=48000,volume={num(A['output_gain_db'])}dB,"
            f"alimiter=limit={num(limit, 4)}:attack=5:release=50:level=0:latency=1")


def run_with_progress(cmd: list[str], total: float, label: str) -> None:
    """Run ffmpeg showing a simple percentage (parses -progress output)."""
    full = cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:]
    log.debug("RUN: %s", " ".join(full))
    err: list[bytes] = []
    with subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        th = threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)
        th.start()
        last = -10
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    sec = int(line.split("=")[1]) / 1e6
                except ValueError:
                    continue
                pct = int(min(100, 100 * sec / max(total, 1e-6)))
                if pct >= last + 10:
                    last = pct - pct % 10
                    print(f"  {label}: {last}%", flush=True)
        proc.wait()
        th.join(timeout=10)
    stderr = b"".join(err).decode("utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-15:])
        log.debug("FAILED: %s", tail)
        raise EditError(f"{label} failed (exit code {proc.returncode}).\nLast FFmpeg messages:\n{tail}")


def render_plan(plan: dict, seg_range: tuple[int, int], videos: list[dict], prof: dict, cfg: dict,
                out_path: Path, work_dir: Path) -> dict:
    """Render segments [a, b) of the plan to out_path. Returns verification info."""
    a, b = seg_range
    segs = plan["segments"][a:b]
    F = prof["fps"]
    offset = segs[0]["t0"]
    length = segs[-1]["t1"] - offset
    total_frames = int(round(length * F))
    tmp = work_dir / f"render_{out_path.stem}"
    tmp.mkdir(parents=True, exist_ok=True)

    jobs = []
    for s in segs:
        f0, f1 = frame_bounds(s, F, offset)
        jobs.append((s, f1 - f0, tmp / f"seg_{s['index']:04d}.mp4"))
    assert sum(j[1] for j in jobs) == total_frames, "frame accounting error"

    print(f"Rendering {len(jobs)} clips at {prof['width']}x{prof['height']} {num(F)} fps "
          f"({prof['name']} quality)...", flush=True)
    t_start = time.time()
    done = 0
    workers = int(cfg["output"]["parallel_jobs"])
    infos = {i: v["info"] for i, v in enumerate(videos)}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(render_segment, s, infos[s["source"]["video"]], prof, cfg, p, n,
                          s["index"] == 0): s for s, n, p in jobs if n > 0}
        for fut in cf.as_completed(futs):
            s = futs[fut]
            try:
                res = fut.result()
            except EditError as exc:
                for f in futs:
                    f.cancel()
                raise EditError(f"Clip {s['index'] + 1} (from '{s['source']['name']}' at "
                                f"{fmt_time(s['source']['start'])}) failed to render.\n{exc}") from exc
            done += 1
            log.debug("Segment %d rendered in %.2fs (command %d chars)", res["index"], res["seconds"], res["cmd_len"])
            if done % max(1, len(futs) // 10) == 0 or done == len(futs):
                print(f"  clips: {done}/{len(futs)}", flush=True)

    bad = []
    for s, n, p in jobs:
        if n <= 0:
            continue
        got = count_frames(p, cfg)
        if got != n:
            bad.append((s["index"], n, got))
    if bad:
        log.warning("Frame count mismatch in clips (index, expected, got): %s", bad)

    list_file = tmp / "concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for s, n, p in jobs:
            if n > 0:
                f.write(f"file '{p.name}'\n")

    music = plan["music"]
    m_start = music["start"] + offset
    A = cfg["audio"]
    fades = "aformat=sample_fmts=fltp:channel_layouts=stereo"
    if A["fade_in_seconds"] > 0:
        fades += f",afade=t=in:st=0:d={num(min(A['fade_in_seconds'], length / 4))}"
    if A["fade_out_seconds"] > 0:
        fo = min(A["fade_out_seconds"], length / 3)
        fades += f",afade=t=out:st={num(length - fo)}:d={num(fo)}"
    measured = measure_loudness(music["path"], m_start, length, fades, cfg)
    if measured is None:
        log.warning("Could not measure music loudness; skipping loudness normalisation.")
    afilt = audio_filter(length, cfg, measured)
    log.debug("Audio filter: %s", afilt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [find_tool("ffmpeg", cfg), "-hide_banner", "-nostdin", "-v", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", str(list_file),
           "-ss", num(m_start, 3), "-t", num(length + 0.5, 3), "-i", music["path"],
           "-filter_complex", f"[1:a]{afilt}[aout]",
           "-map", "0:v:0", "-map", "[aout]",
           "-c:v", "libx264", "-preset", prof["preset"], "-crf", str(prof["crf"]),
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-r", num(F), "-g", str(int(round(2 * F))),
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-c:a", "aac", "-b:a", f"{int(cfg['output']['audio_bitrate_kbps'])}k", "-ar", "48000",
           "-frames:v", str(total_frames), "-t", num(length, 4),
           "-movflags", "+faststart", str(out_path)]
    run_with_progress(cmd, length, "Final encode")
    elapsed = time.time() - t_start

    verify = verify_output(out_path, prof, total_frames, length, cfg)
    verify["render_seconds"] = round(elapsed, 1)
    verify["frame_mismatches"] = bad
    if not cfg["output"]["keep_temp_files"]:
        shutil.rmtree(tmp, ignore_errors=True)  # only the temporary folder this run created
    else:
        log.info("Temporary clips kept in %s", tmp)
    return verify


def verify_output(path: Path, prof: dict, frames: int, length: float, cfg: dict) -> dict:
    proc = run([find_tool("ffprobe", cfg), "-v", "error", "-count_packets", "-show_entries",
                "stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_packets,duration,sample_rate,channels"
                ":format=duration,size", "-of", "json", str(path)], what="ffprobe (verify)")
    data = json.loads(proc.stdout_text)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), {})
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), {})
    res = {
        "file": str(path),
        "size_mb": round(int(data["format"].get("size", 0)) / 1e6, 2),
        "duration": round(float(data["format"].get("duration", 0)), 3),
        "video": f"{v.get('codec_name')} {v.get('width')}x{v.get('height')} @ {v.get('r_frame_rate')}",
        "frames": int(v.get("nb_read_packets", -1)),
        "expected_frames": frames,
        "video_duration": round(float(v.get("duration", 0) or 0), 3),
        "audio": f"{a.get('codec_name')} {a.get('sample_rate')} Hz {a.get('channels')} ch" if a else "MISSING",
        "audio_duration": round(float(a.get("duration", 0) or 0), 3) if a else 0.0,
    }
    problems = []
    if v.get("width") != prof["width"] or v.get("height") != prof["height"]:
        problems.append("resolution differs from the target")
    if res["frames"] != frames:
        problems.append(f"frame count {res['frames']} != expected {frames}")
    if not a:
        problems.append("no audio track")
    elif abs(res["audio_duration"] - length) > 0.1:
        problems.append(f"audio length {res['audio_duration']} s differs from video length {length:.3f} s")
    res["problems"] = problems
    lp = run([find_tool("ffmpeg", cfg), "-hide_banner", "-nostdin", "-v", "info", "-i", str(path), "-vn",
              "-af", "ebur128=peak=true", "-f", "null", "-"], check=False, what="loudness check")
    m_i = re.findall(r"I:\s+(-?[\d.]+) LUFS", lp.stderr_text)
    m_p = re.findall(r"Peak:\s+(-?[\d.]+|-inf) dBFS", lp.stderr_text)
    if m_i:
        res["loudness_lufs"] = float(m_i[-1])
    if m_p:
        try:
            res["true_peak_dbfs"] = float(m_p[-1])
        except ValueError:
            pass
    return res
