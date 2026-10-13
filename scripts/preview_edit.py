"""Step 2: render a short, fast, low-resolution preview of the edit.

  python scripts/preview_edit.py               ~10 s around the drop (settings in config 'preview')
  python scripts/preview_edit.py --seconds 15  longer preview
  python scripts/preview_edit.py --open        open the preview in your video player afterwards

The preview uses exactly the same edit plan as the final export (same cuts, clips and effects),
just at the 'draft' resolution and frame rate, so you can check the timing quickly.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from editlib.common import log, resolve_dir, run_cli  # noqa: E402


def open_file(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        log.warning("Could not open the video player automatically (%s).", exc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.json (default: the one in the project folder)")
    ap.add_argument("--seconds", type=float, help="preview length (overrides config preview.seconds)")
    ap.add_argument("--section", choices=["drop", "start"], help="preview around the drop or from the start")
    ap.add_argument("--open", action="store_true", help="open the preview when it is ready")
    args = ap.parse_args()

    from editlib import pipeline, planner, render
    cfg, log_path = pipeline.start("preview_edit", args.config)
    seconds = args.seconds or cfg["preview"]["seconds"]
    section = args.section or cfg["preview"]["section"]
    media = pipeline.scan(cfg)
    data = pipeline.analyze(cfg, media)
    plan = pipeline.make_plan(cfg, data)
    pipeline.save_plan(plan, log_path)

    a, b = planner.preview_range(plan, seconds, section)
    segs = plan["segments"]
    log.info("Preview: clips %d-%d (%.2f s - %.2f s of the edit)", a + 1, b, segs[a]["t0"], segs[b - 1]["t1"])
    prof = pipeline.profile(cfg, draft=True)
    out = pipeline.output_path(cfg, "preview")
    res = render.render_plan(plan, (a, b), data["videos"], prof, cfg, out, resolve_dir(cfg, "work_dir"))
    ok = pipeline.report_result(res, out)
    log.info("")
    log.info("Preview ready:\n  %s", out)
    log.info("If you like it, run the full-quality export:  python scripts/create_edit.py")
    log.info("Log file: %s", pipeline.relpath(log_path))
    if args.open:
        open_file(out)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
