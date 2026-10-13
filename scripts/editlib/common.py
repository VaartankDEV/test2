"""Shared helpers: project paths, config loading/validation, logging, FFmpeg discovery.

Everything here is cross-platform (Windows, macOS, Linux) and only uses the
Python standard library.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv",
    ".ts", ".m2ts", ".mts", ".mpg", ".mpeg", ".vob", ".ogv", ".3gp",
}
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma",
    ".aiff", ".aif", ".alac", ".mka",
}

log = logging.getLogger("anime_edit")
CURRENT_LOG: list = []  # path of the log file of this run (set by setup_logging)


class EditError(Exception):
    """An error with a message meant for the user (no traceback needed)."""


# ---------------------------------------------------------------------------
# Console / logging
# ---------------------------------------------------------------------------

def fix_console_encoding() -> None:
    """Make print() safe for Japanese/Chinese/accented filenames on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def setup_logging(script_name: str, log_dir: Path) -> Path:
    """Log everything to a timestamped file in logs/, and INFO+ to the console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{script_name}_{stamp}.log"

    log.handlers.clear()
    log.setLevel(logging.DEBUG)
    log.propagate = False

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    log.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)

    log.debug("=== %s started ===", script_name)
    log.debug("Python %s on %s (%s)", sys.version.replace("\n", " "), platform.platform(), platform.machine())
    log.debug("Project root: %s", PROJECT_ROOT)
    log.debug("CPU count: %s", os.cpu_count())
    CURRENT_LOG[:] = [log_path]
    return log_path


def run_cli(body) -> int:
    """Run a script's main body with friendly error messages instead of tracebacks."""
    fix_console_encoding()
    try:
        return int(body() or 0)
    except EditError as exc:
        if log.handlers:
            log.error("\nERROR: %s", exc)
        else:
            print(f"\nERROR: {exc}")
        code = 2
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 130
    except Exception:
        if log.handlers:
            log.exception("\nUnexpected error")
        else:
            import traceback
            traceback.print_exc()
        code = 1
    if CURRENT_LOG:
        print(f"Details are in the log file: {CURRENT_LOG[0]}")
    return code


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "paths": {
        "input_dir": "input",
        "music_dir": "music",
        "output_dir": "output",
        "work_dir": "work",
        "log_dir": "logs",
        "ffmpeg": "auto",
        "ffprobe": "auto",
    },
    "edit": {
        "output_name": "anime_edit",
        "duration_seconds": 30.0,
        "music_file": "auto",
        "music_start_seconds": "auto",
        "clip_order": "hybrid",
        "skip_start_seconds": 0.0,
        "skip_end_seconds": 0.0,
        "allow_clip_reuse": True,
        "random_seed": 7,
    },
    "output": {
        "width": 1080,
        "height": 1920,
        "fps": 60,
        "framing": "smart_crop",
        "smart_crop_strength": 0.6,
        "video_crf": 18,
        "x264_preset": "medium",
        "audio_bitrate_kbps": 192,
        "parallel_jobs": 2,
        "keep_temp_files": False,
    },
    "draft": {
        "width": 540,
        "height": 960,
        "fps": 30,
        "video_crf": 26,
        "x264_preset": "veryfast",
    },
    "preview": {
        "seconds": 10.0,
        "section": "drop",
    },
    "beats": {
        "cut_every_beats": 2,
        "beat_sensitivity": 0.6,
        "adaptive_cuts": True,
        "min_clip_seconds": 0.35,
        "max_clip_seconds": 2.5,
        "bpm_override": None,
    },
    "effects": {
        "velocity_intensity": 0.6,
        "slowmo_smoothing": "none",
        "shake_intensity": 0.5,
        "zoom_intensity": 0.5,
        "glow_strength": 0.4,
        "flash_intensity": 0.35,
        "max_flashes_per_second": 0.5,
        "transition_rate": 0.3,
        "color_grade": "anime_punch",
        "contrast": 1.0,
        "saturation": 1.0,
        "brightness": 0.0,
        "vignette": True,
    },
    "audio": {
        "target_lufs": -14.0,
        "true_peak_db": -1.0,
        "output_gain_db": 0.0,
        "fade_in_seconds": 0.2,
        "fade_out_seconds": 1.5,
    },
    "analysis": {
        "motion_fps": 10,
        "motion_width": 128,
    },
}

# (min, max) safety ranges. Values outside are clamped with a warning.
RANGES = {
    ("edit", "duration_seconds"): (3.0, 600.0),
    ("output", "width"): (144, 4096),
    ("output", "height"): (144, 4096),
    ("output", "fps"): (12, 120),
    ("output", "smart_crop_strength"): (0.0, 1.0),
    ("output", "video_crf"): (0, 51),
    ("output", "parallel_jobs"): (1, 16),
    ("draft", "width"): (144, 4096),
    ("draft", "height"): (144, 4096),
    ("draft", "fps"): (12, 120),
    ("draft", "video_crf"): (0, 51),
    ("preview", "seconds"): (2.0, 60.0),
    ("beats", "cut_every_beats"): (1, 16),
    ("beats", "beat_sensitivity"): (0.0, 1.0),
    ("beats", "min_clip_seconds"): (0.15, 10.0),
    ("beats", "max_clip_seconds"): (0.3, 30.0),
    ("effects", "velocity_intensity"): (0.0, 1.0),
    ("effects", "shake_intensity"): (0.0, 1.0),
    ("effects", "zoom_intensity"): (0.0, 1.0),
    ("effects", "glow_strength"): (0.0, 1.0),
    # Flash limits follow photosensitivity guidance (no more than 3 flashes per second).
    ("effects", "flash_intensity"): (0.0, 0.8),
    ("effects", "max_flashes_per_second"): (0.0, 3.0),
    ("effects", "transition_rate"): (0.0, 1.0),
    ("effects", "contrast"): (0.5, 2.0),
    ("effects", "saturation"): (0.0, 3.0),
    ("effects", "brightness"): (-0.5, 0.5),
    ("audio", "target_lufs"): (-30.0, -6.0),
    ("audio", "true_peak_db"): (-9.0, 0.0),
    ("audio", "output_gain_db"): (-20.0, 10.0),
    ("audio", "fade_in_seconds"): (0.0, 10.0),
    ("audio", "fade_out_seconds"): (0.0, 10.0),
    ("analysis", "motion_fps"): (2, 30),
    ("analysis", "motion_width"): (32, 320),
}

CHOICES = {
    ("edit", "clip_order"): {"hybrid", "chronological", "best_moments"},
    ("output", "framing"): {"smart_crop", "center_crop", "blur_fill", "fit"},
    ("output", "x264_preset"): {"ultrafast", "superfast", "veryfast", "faster", "fast",
                                 "medium", "slow", "slower", "veryslow"},
    ("draft", "x264_preset"): {"ultrafast", "superfast", "veryfast", "faster", "fast",
                                "medium", "slow", "slower", "veryslow"},
    ("preview", "section"): {"drop", "start"},
    ("effects", "slowmo_smoothing"): {"none", "blend"},
    ("effects", "color_grade"): {"anime_punch", "super_saiyan_gold", "ninja_orange",
                                  "cold_night", "natural", "none"},
}


def _deep_merge(base: dict, override: dict, warnings: list, prefix: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key.startswith("_"):  # comments such as "_help"
            continue
        if key not in base:
            warnings.append(f"Unknown config setting '{prefix}{key}' was ignored (check the spelling).")
            continue
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                warnings.append(f"Config section '{prefix}{key}' should be a group of settings; using defaults.")
                continue
            out[key] = _deep_merge(base[key], value, warnings, prefix + key + ".")
        else:
            out[key] = value
    return out


def _validate(cfg: dict, warnings: list) -> None:
    for (section, key), (lo, hi) in RANGES.items():
        val = cfg[section][key]
        default = DEFAULT_CONFIG[section][key]
        try:
            num = float(val)
        except (TypeError, ValueError):
            warnings.append(f"{section}.{key} = {val!r} is not a number; using default {default}.")
            cfg[section][key] = default
            continue
        clamped = min(max(num, lo), hi)
        if clamped != num:
            warnings.append(f"{section}.{key} = {val} is outside the safe range {lo}..{hi}; using {clamped}.")
        cfg[section][key] = int(round(clamped)) if isinstance(default, int) and not isinstance(default, bool) else clamped

    for (section, key), allowed in CHOICES.items():
        val = cfg[section][key]
        if val not in allowed:
            default = DEFAULT_CONFIG[section][key]
            warnings.append(f"{section}.{key} = {val!r} is not one of {sorted(allowed)}; using {default!r}.")
            cfg[section][key] = default

    for section in ("output", "draft"):
        for key in ("width", "height"):  # x264 + yuv420p need even sizes
            cfg[section][key] = int(cfg[section][key]) // 2 * 2

    b = cfg["beats"]
    if b["max_clip_seconds"] < b["min_clip_seconds"]:
        warnings.append("beats.max_clip_seconds was smaller than min_clip_seconds; swapped them.")
        b["max_clip_seconds"], b["min_clip_seconds"] = b["min_clip_seconds"], b["max_clip_seconds"]
    if b["bpm_override"] is not None:
        try:
            bpm = float(b["bpm_override"])
            if not 40 <= bpm <= 250:
                raise ValueError
            b["bpm_override"] = bpm
        except (TypeError, ValueError):
            warnings.append("beats.bpm_override must be null or a BPM between 40 and 250; ignoring it.")
            b["bpm_override"] = None

    ms = cfg["edit"]["music_start_seconds"]
    if ms != "auto":
        try:
            cfg["edit"]["music_start_seconds"] = max(0.0, float(ms))
        except (TypeError, ValueError):
            warnings.append("edit.music_start_seconds must be \"auto\" or a number of seconds; using \"auto\".")
            cfg["edit"]["music_start_seconds"] = "auto"
    for key in ("skip_start_seconds", "skip_end_seconds"):
        try:
            cfg["edit"][key] = max(0.0, float(cfg["edit"][key]))
        except (TypeError, ValueError):
            warnings.append(f"edit.{key} must be a number; using 0.")
            cfg["edit"][key] = 0.0
    name = str(cfg["edit"]["output_name"]).strip()
    safe = "".join(c for c in name if c not in '<>:"/\\|?*' and ord(c) >= 32).strip(" .")
    cfg["edit"]["output_name"] = safe or "anime_edit"


def load_config(path: Path | None = None) -> tuple[dict, list[str]]:
    """Load config.json, fill in defaults, clamp unsafe values. Returns (config, warnings)."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    warnings: list[str] = []
    user_cfg: dict = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                user_cfg = json.load(f)
        except json.JSONDecodeError as exc:
            raise EditError(
                f"config.json has a formatting error on line {exc.lineno}, column {exc.colno}: {exc.msg}.\n"
                "Tip: every setting needs a comma after it except the last one in a group, "
                "and text values need \"double quotes\"."
            ) from exc
        if not isinstance(user_cfg, dict):
            raise EditError("config.json must contain a JSON object ({ ... }).")
    else:
        warnings.append(f"No config file at {path}; using built-in defaults.")
    cfg = _deep_merge(DEFAULT_CONFIG, user_cfg, warnings)
    _validate(cfg, warnings)
    cfg["_config_path"] = str(path)
    return cfg, warnings


def resolve_dir(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# FFmpeg discovery and execution
# ---------------------------------------------------------------------------

_TOOL_CACHE: dict[str, str] = {}


def find_tool(name: str, cfg: dict | None = None) -> str:
    """Find ffmpeg/ffprobe: config path -> project tools folder -> PATH."""
    if name in _TOOL_CACHE:
        return _TOOL_CACHE[name]
    exe = name + (".exe" if os.name == "nt" else "")
    candidates: list[Path] = []
    if cfg:
        configured = str(cfg["paths"].get(name, "auto"))
        if configured and configured.lower() != "auto":
            p = Path(configured)
            candidates.append(p if p.is_absolute() else PROJECT_ROOT / p)
    for sub in ("tools/ffmpeg/bin", "ffmpeg/bin", "tools", "bin"):
        candidates.append(PROJECT_ROOT / sub / exe)
    for c in candidates:
        if c.is_file():
            _TOOL_CACHE[name] = str(c)
            return str(c)
    found = shutil.which(name)
    if found:
        _TOOL_CACHE[name] = found
        return found
    raise EditError(
        f"Could not find '{name}'.\n"
        "Install FFmpeg and make sure it is on your PATH, for example:\n"
        "  Windows:  winget install Gyan.FFmpeg   (then open a NEW terminal window)\n"
        "            or download the 'full' build from https://www.gyan.dev/ffmpeg/builds/ and put\n"
        f"            ffmpeg.exe and ffprobe.exe into: {PROJECT_ROOT / 'tools' / 'ffmpeg' / 'bin'}\n"
        "  macOS:    brew install ffmpeg\n"
        "  Linux:    sudo apt install ffmpeg\n"
        f"You can also set paths.{name} in config.json to the full path of {exe}."
    )


def run(cmd: list[str], *, check: bool = True, capture: bool = True, timeout: float | None = None,
        what: str = "ffmpeg") -> subprocess.CompletedProcess:
    """Run a command (no shell, so spaces/unicode in paths are safe) and log it."""
    log.debug("RUN: %s", " ".join(_quote(c) for c in cmd))
    try:
        proc = subprocess.run(cmd, capture_output=capture, timeout=timeout)
    except FileNotFoundError as exc:
        raise EditError(f"Could not start {cmd[0]}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise EditError(f"{what} timed out after {timeout} s.") from exc
    if capture:
        proc.stdout_text = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
        proc.stderr_text = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
    else:
        proc.stdout_text = proc.stderr_text = ""
    if check and proc.returncode != 0:
        tail = "\n".join(proc.stderr_text.strip().splitlines()[-15:])
        log.debug("FAILED (exit %s). stderr tail:\n%s", proc.returncode, tail)
        raise EditError(f"{what} failed (exit code {proc.returncode}).\nLast FFmpeg messages:\n{tail}")
    return proc


def _quote(s: str) -> str:
    return f'"{s}"' if (" " in s or not s) else s


def ffmpeg_version(cfg: dict | None = None) -> str:
    proc = run([find_tool("ffmpeg", cfg), "-hide_banner", "-version"], check=False)
    first = (proc.stdout_text or proc.stderr_text).splitlines()
    return first[0] if first else "unknown"


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:05.2f}"


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def require_numpy():
    try:
        import numpy  # noqa: F401
        return numpy
    except ImportError as exc:
        raise EditError(
            "The Python package 'numpy' is not installed. It is needed for beat and motion analysis.\n"
            "Install it with:   python -m pip install -r requirements.txt\n"
            "(run_edit.bat offers to do this for you.)"
        ) from exc
