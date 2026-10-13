"""Step 1: inspect your clips and music.

  python scripts/analyze_media.py

Checks every file in input/ and music/ with FFprobe (duration, frame rate, resolution, codecs,
audio streams), test-decodes it, detects the beats of the song and finds action moments in the
clips. Results are cached in work/ so later steps are fast. Nothing in input/ or music/ is changed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from editlib.common import log, run_cli  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.json (default: the one in the project folder)")
    ap.add_argument("--no-plan", action="store_true", help="skip building an example edit plan")
    args = ap.parse_args()

    from editlib import pipeline
    cfg, log_path = pipeline.start("analyze", args.config)
    media = pipeline.scan(cfg)
    data = pipeline.analyze(cfg, media)
    if not args.no_plan:
        plan = pipeline.make_plan(cfg, data)
        log.info("Example edit plan saved to %s", pipeline.relpath(pipeline.save_plan(plan, log_path)))
    log.info("")
    log.info("Analysis OK. Next: python scripts/preview_edit.py   (or double-click run_edit.bat)")
    log.info("Log file: %s", pipeline.relpath(log_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
