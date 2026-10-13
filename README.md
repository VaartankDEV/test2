# Anime Edit Generator — Velocity + Shakes + Glow

Turn your anime clips and a song into a vertical **TikTok / YouTube Shorts / Instagram Reels**
edit, in the high-energy "velocity" style of Naruto and Dragon Ball fan edits:

- cuts locked to the **beats** of your song (detected automatically),
- **velocity ramps** (slow motion on the hit, speed-up into the next cut),
- **punch-in zooms**, **impact shakes** (with a little rotation), **white flashes** on the drop,
- soft **glow**, contrast and **color grading**,
- selective **transition accents** (whip blur, chromatic hit, zoom snap) — not on every cut,
- automatic picking of **action moments** from your clips (simple motion analysis),
- 1080×1920 at 60 fps, music loudness-normalised to −14 LUFS with no clipping.

Everything runs locally on your PC with Python + FFmpeg. Your footage is never uploaded and
your original files are never modified.

---

## 1. One-time setup (Windows)

1. **Python 3.9 or newer** — download from <https://www.python.org/downloads/>.
   In the installer tick **"Add python.exe to PATH"**.
2. **FFmpeg** — open *Command Prompt* and run:
   ```
   winget install Gyan.FFmpeg
   ```
   Then **close and reopen** the Command Prompt. Check it works with `ffmpeg -version`.
   *No winget?* Download the "full" build from <https://www.gyan.dev/ffmpeg/builds/>, unzip it,
   and copy `ffmpeg.exe` and `ffprobe.exe` into this project's `tools\ffmpeg\bin\` folder.
3. **numpy** (the only Python package needed). `run_edit.bat` asks you before installing it, or
   you can install it yourself:
   ```
   python -m pip install -r requirements.txt
   ```

## 2. Make an edit

1. Put your anime clips into the **`input`** folder (MP4, MKV, MOV, WebM, AVI, … — sub-folders,
   spaces and Japanese/Chinese/accented file names are all fine).
2. Put one song into the **`music`** folder (MP3, WAV, FLAC, M4A, OGG, …).
   No song yet? `python scripts\make_demo_media.py` creates a royalty-free 140 BPM demo track.
3. **Double-click `run_edit.bat`**. It will:
   1. check every clip and the song (and tell you exactly what is wrong with any bad file),
   2. render a ~10-second **preview** around the beat drop and open it,
   3. ask you whether to render the **full-quality** version.

**Your finished video appears in the `output` folder**, e.g.
`output\anime_edit_20261013_154210.mp4` (the preview is `..._preview.mp4`).
Every render gets a new timestamped name, so nothing is ever overwritten.

### Running the steps yourself (any OS)

```
python scripts/analyze_media.py      # inspect clips + music, detect beats and action moments
python scripts/preview_edit.py       # short low-res preview  (--seconds 15, --open)
python scripts/create_edit.py        # full-quality export
python scripts/create_edit.py --draft             # full length, fast low-res
python scripts/create_edit.py --duration 20 --music "my song.mp3"
```

## 3. Settings — `config.json`

Open `config.json` in Notepad. Each group has a `_help` line explaining it. The most useful ones:

| Setting | Default | What it does |
|---|---|---|
| `edit.duration_seconds` | `30` | Length of the edit (it ends on the nearest musical downbeat). |
| `edit.music_file` | `"auto"` | Which song to use if there are several (file name). |
| `edit.music_start_seconds` | `"auto"` | Where in the song to start. Auto = the most energetic part including the drop. |
| `edit.clip_order` | `"hybrid"` | `hybrid`: story order, but the drop and biggest hits get the best action; `chronological`; `best_moments`. |
| `edit.skip_start_seconds` / `skip_end_seconds` | `0` | Ignore the opening/ending of every clip (e.g. `90` to skip an anime opening song). |
| `edit.random_seed` | `7` | Change it to get a different variation of the same edit. |
| `output.width` × `output.height` | `1080` × `1920` | Final resolution (9:16 portrait). |
| `output.fps` | `60` | Frame rate. `30` renders about twice as fast. |
| `output.framing` | `"smart_crop"` | `smart_crop` (crop follows the action), `center_crop`, `blur_fill` (whole picture on a blurred background), `fit` (black bars). Never stretches. |
| `draft.*` | 540×960, 30 fps | Settings for previews and `--draft` renders. |
| `beats.cut_every_beats` | `2` | Cut frequency: 1 = every beat, 2 = every 2nd beat, 4 = every bar. |
| `beats.beat_sensitivity` | `0.6` | 0–1. Higher = more beats count as hard hits (more zoom/shake hits). |
| `beats.adaptive_cuts` | `true` | Faster cuts in loud parts, slower in quiet parts. |
| `beats.bpm_override` | `null` | Set the song's BPM if automatic detection gets it wrong. |
| `effects.velocity_intensity` | `0.6` | Speed ramps (0 = off, 1 = extreme). |
| `effects.shake_intensity` | `0.5` | Impact camera shake. |
| `effects.zoom_intensity` | `0.5` | Punch-in / push zooms. |
| `effects.glow_strength` | `0.4` | Bloom on bright areas. |
| `effects.flash_intensity` | `0.35` | White flash on the biggest hits (max 0.8). |
| `effects.max_flashes_per_second` | `0.5` | Flash rate limit (hard cap 3/s, see *Safety* below). |
| `effects.transition_rate` | `0.3` | Share of hits that get a whip-blur / chromatic / zoom-snap accent. |
| `effects.color_grade` | `"anime_punch"` | `anime_punch`, `super_saiyan_gold`, `ninja_orange`, `cold_night`, `natural`, `none`. |
| `effects.contrast` / `saturation` / `brightness` | `1` / `1` / `0` | Fine-tune the grade. |
| `effects.slowmo_smoothing` | `"none"` | `"blend"` = frame blending for softer slow motion. |
| `audio.target_lufs` | `-14` | Music loudness (−14 suits TikTok/YouTube/Instagram). |
| `audio.output_gain_db` | `0` | Extra volume; a limiter still prevents clipping. |

Out-of-range values are clamped automatically, and the log says so. A typo in a setting name gives
a warning instead of being silently ignored.

**Quick recipes**

- *Calmer / easier to watch:* `velocity_intensity 0.4`, `shake_intensity 0.25`, `flash_intensity 0.2`,
  `cut_every_beats 4`.
- *Maximum hype:* `velocity_intensity 0.85`, `shake_intensity 0.8`, `zoom_intensity 0.8`,
  `beat_sensitivity 0.8`, `transition_rate 0.5`.
- *Faster rendering:* `output.fps 30`, `output.x264_preset "fast"`, `output.parallel_jobs 3`.

## 4. How it works

1. **Inspect** (`analyze_media.py`): FFprobe reads duration, frame rate, resolution, codecs,
   rotation, interlacing and audio streams; FFmpeg test-decodes a few frames of every file, so
   unsupported or damaged files are reported with a fix instead of crashing later.
2. **Beats**: the song is decoded to mono, a log-frequency spectrogram gives an onset-strength
   curve, the tempo comes from its autocorrelation, and a dynamic-programming beat tracker
   (Ellis 2007) places the beats with sub-frame precision. Beat *strength* (kick/bass punch ×
   loudness), downbeats (bar starts) and the drop (biggest rise into a loud section) are derived from it.
3. **Action**: each clip is decoded to tiny grayscale frames (128 px, 10 fps). Frame differences
   after removing camera pans (phase correlation) measure on-screen action; brightness jumps
   catch impact flashes; hard cuts, fades, black and blank frames are recognised and not counted.
4. **Plan**: at every downbeat a bar pattern is chosen (e.g. 2+2 beats, 1+1+2 in loud parts,
   4 in quiet parts), so cuts never drift off the bar. The hardest hits of the edit are ranked;
   only those get shakes/punch zooms/flashes/accents. Each clip's speed curve is integrated to
   know how much source footage it needs, then the best-scoring source window is chosen.
5. **Render**: every clip is rendered by FFmpeg with its own filter chain (seek → deinterlace →
   9:16 framing → `setpts` time remap → grade → glow → `perspective` zoom/shake/rotation →
   accents → flash), with an exact frame count. The clips are joined and encoded once together
   with the music, which is faded, 2-pass loudness-normalised and peak-limited.
6. **Verify**: the output's resolution, frame count, audio length and loudness are checked and
   written to the log.

## 5. Logs and troubleshooting

Every run writes a detailed log to the `logs` folder (all FFmpeg commands, file details,
beat/tempo results, the full edit plan as JSON, timings and the final checks).

| Message | What to do |
|---|---|
| *Could not find 'ffmpeg'* | Install FFmpeg (step 1.2) and open a **new** terminal, or set `paths.ffmpeg` in `config.json`. |
| *No valid input media* | Put clips in `input` and a song in `music`. |
| *FFprobe cannot read this file* | The file is damaged, incomplete or DRM-protected. Try it in VLC and re-download/convert it. |
| *Your FFmpeg has no decoder for video codec ...* | Install the "full" FFmpeg build, or convert the clip to H.264 (the message shows the exact command). |
| *HDR video detected* | Use an SDR version of the clip; HDR→SDR tone-mapping is not applied. |
| *Beat detection is unsure* / cuts feel off-beat | Set `beats.bpm_override` to the song's BPM. |
| *Not enough unique footage* | Add clips, shorten the edit, or set `edit.allow_clip_reuse` to `true`. |
| Rendering is slow | Use `output.fps 30`, `x264_preset "fast"`, or a shorter edit. |

Run the automated tests with `python -m unittest discover -s tests -v` (they use generated test
media in a temporary folder, not your files).

## 6. What was tested

Tested end-to-end (Python 3.13, FFmpeg 6.1, numpy 2, Linux, 4 CPU cores) with two Big Buck Bunny
clips (1080p60 and 360p24, file names with spaces + Japanese characters) and the synthesized
140 BPM demo song. Example log: [`docs/example_create_edit.log`](docs/example_create_edit.log).

| Check | Result |
|---|---|
| Beat detection on synthetic songs (90–174 BPM, MP3) | tempo within 1 %; beats after the intro within ±15 ms of the truth |
| Cuts in the final 1080×1920/60 fps export vs. the song's true beat grid | 42/42 within 15 ms (median 2.8 ms) — less than one frame |
| Cuts visible in the picture vs. planned frames | 41/41 exactly on the planned frame |
| Frame count / audio length | 1749 of 1749 frames, audio = video length |
| Loudness | −13.9 LUFS (target −14), peak −3.2 dBFS (no clipping) |
| Color pipeline with neutral settings | ≤ 1.5/255 average difference (no color shift) |
| Damaged file, audio-only file in `input`, empty file, no media | clear message, file skipped / helpful error |
| Automated tests (`tests/test_pipeline.py`) | 16 passed |
| Render time (4-core cloud VM) | 10 s preview: ~20 s · 30 s final 1080×1920@60: ~3.6 min |

## 7. Limitations (honest list)

- **Action detection is simple motion analysis**, not AI. It reliably finds fast motion, impact
  flashes and cuts, and it compensates flat 2D camera pans. Camera moves with strong 3D
  parallax (CG crane shots) can still be mistaken for action, and it does not know *who* is on
  screen. Use `skip_start_seconds` / `skip_end_seconds` to exclude openings, credits and title cards.
- **Smart crop** is static per clip (follows where the motion is). It does not track faces.
- **Slow motion** repeats frames (or blends them with `slowmo_smoothing: "blend"`); there is no
  optical-flow interpolation — FFmpeg's `minterpolate` was too slow and produces artifacts on anime.
- **Transitions** are applied at the start of the incoming clip (flash, whip blur, chromatic hit,
  zoom snap). Cross-dissolves that overlap two clips are not used.
- **Original clip audio** (voices, sound effects) is not mixed in — only the music track is used.
- `run_edit.bat` has been written for Windows 10/11 but was developed and tested in a Linux
  environment; the Python scripts were tested end-to-end there.

## 8. Safety and rights

- **Photosensitivity:** flashes are limited to `max_flashes_per_second` (default 0.5, hard cap 3 per
  second following common accessibility guidance), intensity is capped, and shakes decay within
  half a second. Keep these low if your audience may be sensitive to flashing or motion.
- **Copyright:** anime footage and commercial music are usually copyrighted. Fan edits are posted
  at your own risk and are subject to each platform's rules.

## 9. Demo media and credits

The tested example uses footage from **Big Buck Bunny** — © 2008 Blender Foundation,
www.bigbuckbunny.org, licensed under
[Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/). Clips were
re-cut, cropped, color-graded and given effects by this tool. The demo music is synthesized by
`scripts/make_demo_media.py` and is original. Media files are not stored in this repository
(see `.gitignore`).

## 10. Project files

```
run_edit.bat              double-click launcher (Windows)
config.json               all settings
requirements.txt          Python packages (numpy)
scripts/analyze_media.py  step 1: inspect media, detect beats + action
scripts/preview_edit.py   step 2: short draft preview
scripts/create_edit.py    step 3: full export (or --draft)
scripts/make_demo_media.py  royalty-free demo song / motion test clip
scripts/editlib/          the engine (probe, audio, motion, planner, render, pipeline)
tests/test_pipeline.py    automated tests
docs/example_create_edit.log  example of a full diagnostic log
input/  music/            your media (never modified)
output/                   finished MP4s
logs/  work/              logs, edit plans, analysis cache (safe to delete)
```

---

## Other contents of this repository: AFRICABET reference asset pack

*(The text below is the original README of the image files in the repository root; it is kept
unchanged and is unrelated to the anime edit generator.)*

# AFRICABET reference asset pack

This pack was generated from the clean AFRICABET screenshot supplied by the user.

It contains:
- full desktop reference
- mobile reference
- separate crops of the marked sections
- separate team-logo crops: Arsenal, Chelsea, Barcelona, Real Madrid, Bayern Munich, Borussia Dortmund
- separate competition-logo crops: Champions League, Premier League, LaLiga, Serie A, Bundesliga
- separate UI icon crops

IMPORTANT:
These are raster crops from the supplied screenshot. They are useful as exact visual references for an AI/coding agent, but they are NOT official SVG/vector files and should not be represented as licensed official team/league artwork. For a production commercial site, replace them with properly licensed/official assets where required.

Recommended AI instruction:
"Use the supplied asset pack as visual reference. Preserve the exact appearance of the supplied logos and UI, but use properly licensed official assets for production. Do not invent substitute logos."
