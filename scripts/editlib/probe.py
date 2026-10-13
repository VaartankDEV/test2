"""Find input media, inspect it with ffprobe, and verify FFmpeg can actually decode it."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from pathlib import Path

from .common import (AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, EditError, find_tool, log, run)

IGNORED_NAMES = {"desktop.ini", "thumbs.db", ".ds_store", ".gitkeep"}

CONVERT_HINT = (
    "Fix: convert it to a standard H.264 MP4, e.g.\n"
    '  ffmpeg -i "{name}" -c:v libx264 -crf 16 -preset medium -c:a aac "{stem}_h264.mp4"\n'
    "(or use HandBrake with the 'Fast 1080p30' preset), then put the converted copy in the folder.\n"
    "Alternatively install a 'full' FFmpeg build (https://www.gyan.dev/ffmpeg/builds/)."
)


@dataclass
class MediaInfo:
    path: str
    name: str
    kind: str                      # "video" or "music"
    ok: bool = False
    problems: list = field(default_factory=list)   # fatal: file is skipped
    warnings: list = field(default_factory=list)   # non-fatal notes
    duration: float = 0.0
    format_name: str = ""
    size_bytes: int = 0
    # video
    vcodec: str = ""
    vprofile: str = ""
    pix_fmt: str = ""
    width: int = 0                 # coded size
    height: int = 0
    display_width: int = 0         # after rotation + sample-aspect-ratio correction
    display_height: int = 0
    sar: str = "1:1"
    rotation: int = 0
    fps: float = 0.0
    vfr: bool = False
    interlaced: bool = False
    color_space: str = ""
    color_transfer: str = ""
    hdr: bool = False
    # audio
    audio_streams: int = 0
    acodec: str = ""
    sample_rate: int = 0
    channels: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def list_media_files(folder: Path, kind: str) -> tuple[list[Path], list[str]]:
    """Return (candidate files, notes about ignored files). Searches sub-folders too."""
    exts = VIDEO_EXTENSIONS if kind == "video" else (AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)
    notes: list[str] = []
    if not folder.exists():
        return [], notes
    files = []
    for p in sorted(folder.rglob("*"), key=_natural_key):
        if not p.is_file():
            continue
        low = p.name.lower()
        if low in IGNORED_NAMES or low.startswith(".") or low.endswith((".txt", ".md")):
            continue
        if p.suffix.lower() in exts:
            files.append(p)
        elif kind == "video" and p.suffix.lower() in AUDIO_EXTENSIONS:
            notes.append(f"'{p.name}' looks like a music file - move it to the music folder.")
        else:
            notes.append(f"Ignored '{p.name}' (unsupported file type '{p.suffix or 'none'}').")
    return files, notes


def _fraction(text: str | None) -> float:
    try:
        f = Fraction(str(text))
        return float(f) if f.denominator else 0.0
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.0


def _decoder_names(cfg) -> set[str]:
    if not hasattr(_decoder_names, "_cache"):
        proc = run([find_tool("ffmpeg", cfg), "-hide_banner", "-decoders"], check=False, what="ffmpeg -decoders")
        names = set()
        for line in proc.stdout_text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
                names.add(parts[1])
        _decoder_names._cache = names
    return _decoder_names._cache


def probe(path: Path, kind: str, cfg: dict) -> MediaInfo:
    info = MediaInfo(path=str(path), name=path.name, kind=kind)
    try:
        info.size_bytes = path.stat().st_size
    except OSError as exc:
        info.problems.append(f"Cannot read the file: {exc}")
        return info
    if info.size_bytes == 0:
        info.problems.append("The file is empty (0 bytes). Was the download/copy interrupted?")
        return info

    proc = run([find_tool("ffprobe", cfg), "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path)], check=False, what="ffprobe")
    if proc.returncode != 0:
        msg = proc.stderr_text.strip().splitlines()[-1:] or ["unknown error"]
        info.problems.append(
            f"FFprobe cannot read this file ({msg[0]}). It may be damaged, incomplete, "
            "DRM-protected, or not a media file. Try playing it in VLC; if it plays, re-save/convert it.")
        return info
    try:
        data = json.loads(proc.stdout_text or "{}")
    except json.JSONDecodeError:
        info.problems.append("FFprobe returned unreadable information for this file.")
        return info

    fmt = data.get("format", {})
    info.format_name = fmt.get("format_name", "")
    info.duration = _fraction(fmt.get("duration"))
    streams = data.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    info.audio_streams = len(audios)
    if audios:
        a = audios[0]
        info.acodec = a.get("codec_name", "") or "unknown"
        info.sample_rate = int(a.get("sample_rate", 0) or 0)
        info.channels = int(a.get("channels", 0) or 0)

    if kind == "video":
        _fill_video(info, videos, cfg)
    else:
        _fill_music(info, audios, cfg)
    info.ok = not info.problems
    return info


def _fill_video(info: MediaInfo, videos: list, cfg: dict) -> None:
    if not videos:
        info.problems.append("No video track found (audio-only file or cover art only). "
                             "If this is music, move it to the music folder.")
        return
    v = videos[0]
    info.vcodec = v.get("codec_name", "") or "unknown"
    info.vprofile = v.get("profile", "") or ""
    info.pix_fmt = v.get("pix_fmt", "") or ""
    info.width = int(v.get("width", 0) or 0)
    info.height = int(v.get("height", 0) or 0)
    stream_duration = _fraction(v.get("duration"))
    if stream_duration > 0.5:  # the video track can be shorter than the container (audio runs longer)
        info.duration = stream_duration

    sar = v.get("sample_aspect_ratio", "1:1")
    sar_f = _fraction(sar.replace(":", "/")) if sar and sar not in ("0:1", "N/A") else 1.0
    if sar_f <= 0:
        sar_f = 1.0
    info.sar = sar if sar_f != 1.0 else "1:1"

    rot = 0
    for sd_item in v.get("side_data_list", []) or []:
        if "rotation" in sd_item:
            rot = int(round(float(sd_item["rotation"])))
    if not rot and "rotate" in (v.get("tags") or {}):
        rot = int(v["tags"]["rotate"])
    info.rotation = rot % 360
    dw, dh = int(round(info.width * sar_f)), info.height
    if info.rotation in (90, 270):
        dw, dh = dh, dw
    info.display_width, info.display_height = dw, dh

    avg = _fraction(v.get("avg_frame_rate"))
    rfr = _fraction(v.get("r_frame_rate"))
    info.fps = avg if 1 <= avg <= 300 else rfr
    info.vfr = bool(avg and rfr and abs(avg - rfr) / max(rfr, 1e-6) > 0.01)
    fo = (v.get("field_order") or "").lower()
    info.interlaced = fo in ("tt", "bb", "tb", "bt")
    info.color_space = v.get("color_space", "") or ""
    info.color_transfer = v.get("color_transfer", "") or ""
    info.hdr = info.color_transfer in ("smpte2084", "arib-std-b67")

    if info.width <= 0 or info.height <= 0:
        info.problems.append("The video track has no valid resolution.")
        return
    if info.duration <= 0.5:
        info.problems.append(f"Video is too short ({info.duration:.2f} s); at least 1 second is needed.")
        return
    if info.fps <= 0:
        info.warnings.append("Frame rate is unknown; assuming 24 fps.")
        info.fps = 24.0

    decoders = _decoder_names(cfg)
    known_alt = {"av1": {"libdav1d", "libaom-av1", "av1"}, "vp9": {"vp9", "libvpx-vp9"},
                 "vp8": {"vp8", "libvpx"}}
    if info.vcodec not in decoders and not (known_alt.get(info.vcodec, set()) & decoders):
        info.problems.append(f"Your FFmpeg has no decoder for video codec '{info.vcodec}'. "
                             + CONVERT_HINT.format(name=info.name, stem=Path(info.name).stem))
        return
    _decode_test(info, cfg, "v")
    if info.problems:
        return

    if info.hdr:
        info.warnings.append("HDR video detected. Colors may look washed out because HDR->SDR tone "
                             "mapping is not applied. Use an SDR version of the clip for best results.")
    if info.interlaced:
        info.warnings.append("Interlaced video detected; it will be deinterlaced automatically (bwdif).")
    if info.vfr:
        info.warnings.append(f"Variable frame rate detected (avg {info.fps:.3f} fps); handled automatically.")
    if min(info.display_width, info.display_height) < 480:
        info.warnings.append(f"Low resolution ({info.display_width}x{info.display_height}); "
                             "the 1080x1920 export will look soft.")


def _fill_music(info: MediaInfo, audios: list, cfg: dict) -> None:
    if not audios:
        info.problems.append("No audio track found in this file.")
        return
    if info.duration <= 0:
        d = _fraction(audios[0].get("duration"))
        info.duration = d
    if info.duration < 3:
        info.problems.append(f"Music is too short ({info.duration:.1f} s); at least 3 seconds is needed.")
        return
    if info.acodec not in _decoder_names(cfg):
        info.problems.append(f"Your FFmpeg has no decoder for audio codec '{info.acodec}'. "
                             "Convert the song to MP3 or WAV (e.g. with Audacity or "
                             f'ffmpeg -i "{info.name}" -c:a libmp3lame -q:a 2 "{Path(info.name).stem}.mp3").')
        return
    _decode_test(info, cfg, "a")


def _decode_test(info: MediaInfo, cfg: dict, stream: str) -> None:
    """Really decode a little bit from the middle of the file to catch broken/unsupported data."""
    start = max(0.0, info.duration * 0.5 - 0.5)
    sel = "0:v:0" if stream == "v" else "0:a:0"
    limit = ["-frames:v", "3"] if stream == "v" else ["-t", "1"]
    proc = run([find_tool("ffmpeg", cfg), "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{start:.3f}",
                "-i", info.path, "-map", sel, *limit, "-f", "null", "-"], check=False, timeout=120,
               what="decode test")
    err = proc.stderr_text.strip()
    if proc.returncode != 0:
        last = err.splitlines()[-1] if err else "unknown error"
        info.problems.append(f"FFmpeg could not decode this file ({last}). "
                             + CONVERT_HINT.format(name=info.name, stem=Path(info.name).stem))
    elif err:
        log.debug("Decode test warnings for %s: %s", info.name, err[:500])
        if stream == "v":
            info.warnings.append("FFmpeg reported minor decoding errors; the file may be partly damaged.")


def scan_folder(folder: Path, kind: str, cfg: dict) -> tuple[list[MediaInfo], list[str]]:
    files, notes = list_media_files(folder, kind)
    infos = []
    for f in files:
        info = probe(f, kind, cfg)
        log.debug("Probe %s: %s", f.name, json.dumps(info.to_dict(), ensure_ascii=False))
        infos.append(info)
    return infos, notes


def require_inputs(videos: list[MediaInfo], music: list[MediaInfo], input_dir: Path, music_dir: Path) -> None:
    good_v = [v for v in videos if v.ok]
    good_m = [m for m in music if m.ok]
    msgs = []
    if not good_v:
        if videos:
            msgs.append(f"None of the {len(videos)} file(s) in the video folder can be used (see problems above).")
        msgs.append(f"Add anime clips (MP4/MKV/MOV/WebM/AVI...) to:\n    {input_dir}")
    if not good_m:
        if music:
            msgs.append(f"None of the {len(music)} file(s) in the music folder can be used (see problems above).")
        msgs.append(f"Add a song (MP3/WAV/FLAC/M4A/OGG...) to:\n    {music_dir}")
    if msgs:
        raise EditError("No valid input media.\n" + "\n".join(msgs) +
                        "\nTip: run  python scripts/make_demo_media.py  to create a royalty-free demo song.")
