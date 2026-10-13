"""Step 3: render the finished edit.

  python scripts/create_edit.py              full-quality export (default 1080x1920, 60 fps)
  python scripts/create_edit.py --draft      full length, but fast low-resolution draft
  python scripts/create_edit.py --duration 20 --music "my song.mp3"

The MP4 is written to the output/ folder with a timestamp in its name (earlier renders are
never overwritten). Your original clips and music are only read, never modified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from editlib.common import log, resolve_dir, run_cli  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.json (default: the one in the project folder)")
    ap.add_argument("--draft", action="store_true", help="fast low-resolution render (settings in 'draft')")
    ap.add_argument("--duration", type=float, help="edit length in seconds (overrides config)")
    ap.add_argument("--music", help="file name of the song in the music folder (overrides config)")
    args = ap.parse_args()

    from editlib import pipeline, render
    cfg, log_path = pipeline.start("create_edit", args.config)
    if args.duration:
        cfg["edit"]["duration_seconds"] = min(600.0, max(3.0, args.duration))
    if args.music:
        cfg["edit"]["music_file"] = args.music
    media = pipeline.scan(cfg)
    data = pipeline.analyze(cfg, media)
    plan = pipeline.make_plan(cfg, data)
    pipeline.save_plan(plan, log_path)

    prof = pipeline.profile(cfg, draft=args.draft)
    out = pipeline.output_path(cfg, "draft" if args.draft else "final")
    res = render.render_plan(plan, (0, len(plan["segments"])), data["videos"], prof, cfg, out,
                             resolve_dir(cfg, "work_dir"))
    ok = pipeline.report_result(res, out)
    log.info("")
    log.info("DONE! Your %s edit is here:\n  %s", "draft" if args.draft else "finished", out)
    log.info("Log file: %s", pipeline.relpath(log_path))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
