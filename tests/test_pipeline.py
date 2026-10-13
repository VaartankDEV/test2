"""Automated checks for the anime edit generator.

Run from the project folder:   python -m unittest discover -s tests -v
Everything is generated in a temporary folder (synthetic music + video), so your own
input/music/output folders are not touched. Needs FFmpeg and numpy.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import make_demo_media  # noqa: E402
from editlib import audio, motion, planner, probe, render  # noqa: E402
from editlib.common import EditError, load_config  # noqa: E402


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", *args], check=True)


class Workspace:
    """A temporary project with its own config.json and folders."""

    def __init__(self, **overrides):
        self.dir = Path(tempfile.mkdtemp(prefix="anime_edit_test_"))
        cfg = {"paths": {k: str(self.dir / k) for k in ("input", "music", "output", "work", "logs")}}
        cfg["paths"] = {"input_dir": cfg["paths"]["input"], "music_dir": cfg["paths"]["music"],
                        "output_dir": cfg["paths"]["output"], "work_dir": cfg["paths"]["work"],
                        "log_dir": cfg["paths"]["logs"]}
        for section, values in overrides.items():
            cfg.setdefault(section, {}).update(values)
        self.config_path = self.dir / "config.json"
        self.config_path.write_text(json.dumps(cfg), encoding="utf-8")
        for k in ("input", "music"):
            (self.dir / k).mkdir()
        self.cfg, self.warnings = load_config(self.config_path)

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def make_music(path: Path, bpm=128.0, bars=16):
    stereo, beats, downs, drop = make_demo_media.synth_track(bpm=bpm, bars=bars)
    make_demo_media.write_wav(path, stereo)
    return beats, downs, drop


def make_motion_clip(path: Path):
    """4 s still test pattern, 4 s fast fractal zoom, 4 s static colour."""
    ffmpeg("-filter_complex",
           "testsrc2=s=640x360:r=24:d=1,trim=end_frame=1,loop=loop=95:size=1,setpts=N/24/TB,format=yuv420p[a];"
           "mandelbrot=s=640x360:r=24:end_scale=0.01:start_scale=3,trim=duration=4,format=yuv420p[b];"
           "color=c=0x203040:s=640x360:r=24:d=4,format=yuv420p[c];[a][b][c]concat=n=3:v=1:a=0[v]",
           "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(path))


class ConfigTests(unittest.TestCase):
    def test_unsafe_values_are_clamped(self):
        ws = Workspace(effects={"flash_intensity": 5, "max_flashes_per_second": 10, "shake_intensity": -1,
                                "color_grade": "rainbow"}, edit={"typo_setting": 1})
        try:
            e = ws.cfg["effects"]
            self.assertEqual(e["flash_intensity"], 0.8)
            self.assertEqual(e["max_flashes_per_second"], 3.0)   # photosensitivity limit
            self.assertEqual(e["shake_intensity"], 0.0)
            self.assertEqual(e["color_grade"], "anime_punch")
            self.assertTrue(any("typo_setting" in w for w in ws.warnings))
        finally:
            ws.cleanup()

    def test_broken_json_gives_line_number(self):
        d = Path(tempfile.mkdtemp())
        try:
            p = d / "config.json"
            p.write_text('{\n "edit": {"duration_seconds": 30,}\n}', encoding="utf-8")
            with self.assertRaises(EditError) as ctx:
                load_config(p)
            self.assertIn("line 2", str(ctx.exception))
        finally:
            shutil.rmtree(d)

    def test_shipped_config_is_valid(self):
        cfg, warnings = load_config(ROOT / "config.json")
        self.assertEqual(warnings, [])
        self.assertEqual((cfg["output"]["width"], cfg["output"]["height"]), (1080, 1920))


class BeatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()

    @classmethod
    def tearDownClass(cls):
        cls.ws.cleanup()

    def check_track(self, bpm):
        path = self.ws.dir / "music" / f"beat test {bpm:g} ビート.wav"
        beats, downs, _ = make_music(path, bpm=bpm, bars=16)
        res = audio.analyze_music(str(path), self.ws.cfg)
        self.assertAlmostEqual(res["bpm"], bpm, delta=bpm * 0.015)
        period = 60.0 / bpm
        det = np.array(res["beat_times"])
        after_intro = det[det > 16 * period + 0.3]     # intro has off-beat hi-hats (ambiguous)
        err = np.array([abs(d - round(d / period) * period) for d in after_intro])
        self.assertGreater((err < 0.05).mean(), 0.95, f"beats off by {np.round(err, 3)}")
        return res

    def test_tempo_and_beat_positions_128(self):
        self.check_track(128)

    def test_tempo_and_beat_positions_150(self):
        self.check_track(150)

    def test_music_window_ends_on_beat(self):
        path = self.ws.dir / "music" / "window.wav"
        make_music(path, bpm=140, bars=20)
        res = audio.analyze_music(str(path), self.ws.cfg)
        start, length = audio.choose_music_window(res, 12.0, "auto")
        beats = np.array(res["beat_times"])
        self.assertLess(np.min(np.abs(beats - start)), 0.002)
        self.assertLess(np.min(np.abs(beats - (start + length))), 0.002)
        self.assertLess(abs(length - 12.0), 1.7)


class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()
        cls.clip = cls.ws.dir / "input" / "motion test 動き clip.mp4"
        make_motion_clip(cls.clip)

    @classmethod
    def tearDownClass(cls):
        cls.ws.cleanup()

    def test_probe_unicode_filename(self):
        info = probe.probe(self.clip, "video", self.ws.cfg)
        self.assertTrue(info.ok, info.problems)
        self.assertEqual((info.display_width, info.display_height), (640, 360))
        self.assertAlmostEqual(info.fps, 24.0, places=2)
        self.assertAlmostEqual(info.duration, 12.0, delta=0.1)

    def test_broken_and_wrong_files_are_rejected_with_advice(self):
        bad = self.ws.dir / "input" / "broken.mkv"
        bad.write_bytes(random.Random(1).randbytes(50000))
        info = probe.probe(bad, "video", self.ws.cfg)
        self.assertFalse(info.ok)
        self.assertIn("damaged", info.problems[0])
        audio_only = self.ws.dir / "input" / "song.m4a"
        ffmpeg("-f", "lavfi", "-i", "sine=f=330:d=4", str(audio_only))
        info = probe.probe(audio_only, "video", self.ws.cfg)
        self.assertFalse(info.ok)
        self.assertIn("No video track", info.problems[0])
        empty = self.ws.dir / "input" / "empty.mp4"
        empty.write_bytes(b"")
        self.assertIn("empty", probe.probe(empty, "video", self.ws.cfg).problems[0])
        for p in (bad, audio_only, empty):
            p.unlink()

    def test_no_inputs_message(self):
        with self.assertRaises(EditError) as ctx:
            probe.require_inputs([], [], self.ws.dir / "input", self.ws.dir / "music")
        msg = str(ctx.exception)
        self.assertIn("Add anime clips", msg)
        self.assertIn("Add a song", msg)

    def test_motion_finds_fast_section_and_cuts(self):
        info = probe.probe(self.clip, "video", self.ws.cfg).to_dict()
        res = motion.analyze_video(info, self.ws.cfg)
        a, fps = res["action"], res["fps"]
        fast = a[int(4.5 * fps):int(7.5 * fps)].mean()
        calm = a[int(8.5 * fps):int(11.5 * fps)].mean()
        pattern = a[int(0.5 * fps):int(3.5 * fps)].mean()
        self.assertGreater(fast, pattern + 0.15)
        self.assertGreater(fast, calm + 0.3)
        cuts = res["cut_times"]
        self.assertTrue(any(abs(c - 4.0) < 0.15 for c in cuts) and any(abs(c - 8.0) < 0.15 for c in cuts), cuts)


class PlanTests(unittest.TestCase):
    """Plan invariants with synthetic analysis data (fast, no rendering)."""

    def make_plan(self, **effects):
        ws = Workspace(effects=effects)
        self.addCleanup(ws.cleanup)
        path = ws.dir / "music" / "plan.wav"
        make_music(path, bpm=140, bars=24)
        mus = audio.analyze_music(str(path), ws.cfg)
        rng = np.random.default_rng(0)
        videos = []
        for i, dur in enumerate((40.0, 25.0)):
            n = int(dur * 10)
            videos.append({"info": {"name": f"v{i}.mp4", "path": f"/x/v{i}.mp4", "duration": dur},
                           "motion": {"action": rng.random(n), "luma": np.full(n, 0.5), "fps": 10.0,
                                      "cut_times": np.array([10.0, 20.0]), "centroid": rng.random(n)}})
        start, length = audio.choose_music_window(mus, 25.0, "auto")
        return ws, mus, planner.plan_edit(ws.cfg, mus, start, length, videos), videos

    def test_cuts_are_on_beats_and_contiguous(self):
        ws, mus, plan, videos = self.make_plan()
        segs = plan["segments"]
        beats = np.array(mus["beat_times"]) - plan["music"]["start"]
        self.assertEqual(segs[0]["t0"], 0.0)
        self.assertAlmostEqual(segs[-1]["t1"], plan["length"], places=3)
        for a, b in zip(segs[:-1], segs[1:]):
            self.assertAlmostEqual(a["t1"], b["t0"], places=4)
        for s in segs[1:]:
            self.assertLess(np.min(np.abs(beats - s["t0"])), 0.002, f"cut at {s['t0']} not on a beat")
        mn = ws.cfg["beats"]["min_clip_seconds"]
        self.assertTrue(all(s["t1"] - s["t0"] >= mn - 1e-6 for s in segs))

    def test_effects_are_selective_and_flashes_rate_limited(self):
        ws, _, plan, _ = self.make_plan(max_flashes_per_second=0.5)
        segs = plan["segments"]
        flashes = [s["t0"] for s in segs if s["flash"] > 0]
        self.assertTrue(all(b - a >= 2.0 - 1e-6 for a, b in zip(flashes, flashes[1:])), flashes)
        self.assertLess(sum(bool(s["shake"]) for s in segs), 0.6 * len(segs))
        self.assertLess(sum(s["accent"] is not None for s in segs), 0.4 * len(segs))

    def test_effects_can_be_switched_off(self):
        _, _, plan, _ = self.make_plan(velocity_intensity=0, shake_intensity=0, zoom_intensity=0,
                                       flash_intensity=0, glow_strength=0, transition_rate=0)
        for s in plan["segments"]:
            self.assertEqual(s["speed"]["mode"], "steady")
            self.assertFalse(s["shake"] or s["zoom"] or s["flash"] or s["accent"] or s["glow"])

    def test_time_maps_and_source_windows(self):
        _, _, plan, videos = self.make_plan()
        for s in plan["segments"]:
            knots = s["time_map"]
            outs = [k[0] for k in knots]
            srcs = [k[1] for k in knots]
            self.assertTrue(all(b > a for a, b in zip(outs, outs[1:])))
            self.assertTrue(all(b > a for a, b in zip(srcs, srcs[1:])))
            self.assertAlmostEqual(outs[-1], s["t1"] - s["t0"], places=3)
            dur = videos[s["source"]["video"]]["info"]["duration"]
            self.assertGreaterEqual(s["source"]["start"], 0)
            self.assertLessEqual(s["source"]["start"] + s["src_len"], dur)

    def test_remap_expression_matches_knots(self):
        knots = [(0.0, 0.0), (0.2, 0.08), (0.5, 0.5), (0.8, 1.2)]
        expr = render.remap_expr(knots)[1:-4]  # strip "(" ... ")/TB"
        for o, src in knots:
            val = eval(expr.replace("if(", "_if(").replace("lt(", "_lt("),
                       {"_if": lambda c, a, b: a if c else b, "_lt": lambda a, b: a < b, "T": src})
            self.assertAlmostEqual(val, o, places=4)


class RenderTest(unittest.TestCase):
    """End-to-end: tiny render, then check frame count, A/V length and loudness."""

    def test_end_to_end_draft_render(self):
        ws = Workspace(draft={"width": 270, "height": 480, "fps": 24, "x264_preset": "ultrafast"},
                       output={"parallel_jobs": 2}, edit={"duration_seconds": 6})
        self.addCleanup(ws.cleanup)
        clip = ws.dir / "input" / "clip with spaces – テスト.mp4"
        make_motion_clip(clip)
        song = ws.dir / "music" / "song ソング.wav"
        make_music(song, bpm=140, bars=12)

        from editlib import pipeline
        media = pipeline.scan(ws.cfg)
        data = pipeline.analyze(ws.cfg, media)
        plan = pipeline.make_plan(ws.cfg, data)
        prof = pipeline.profile(ws.cfg, draft=True)
        out = pipeline.output_path(ws.cfg, "draft")
        res = render.render_plan(plan, (0, len(plan["segments"])), data["videos"], prof, ws.cfg, out,
                                 ws.dir / "work")
        self.assertEqual(res["problems"], [])
        self.assertEqual(res["frames"], int(round(plan["length"] * 24)))
        self.assertAlmostEqual(res["audio_duration"], plan["length"], delta=0.05)
        self.assertAlmostEqual(res["loudness_lufs"], -14.0, delta=1.0)
        self.assertTrue(clip.exists() and song.exists())          # originals untouched
        self.assertFalse(any((ws.dir / "work").glob("render_*")))  # temp clips cleaned up


if __name__ == "__main__":
    unittest.main()
