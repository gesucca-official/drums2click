# drums2click

Turn a **drum stem** into a **click track that follows the drummer**: tempo drift, pushes and
pulls included. Use it to replace the drums in a multitrack and play your own drums over it, or
import the MIDI tempo map into your DAW so its grid lines up with a human performance.

Outputs:

- **MIDI** (`.mid`): a tempo map with one tempo change per beat, so every beat lands exactly on the
  DAW grid, plus a click track with the downbeat accented. Odd-length bars get time signature changes.
- **Click audio** (`.flac` / `.wav` / `.mp3`): normalized to -1 dBFS, same length and sample rate
  as the stem, so it drops into the multitrack in place of the drums. Clicks also fill the silence
  before the drums come in.
- **Check WAV** (optional): the stem mixed with the click rendered back from the MIDI file, to
  verify by ear.

## How it works

1. [Beat This!](https://github.com/CPJKU/beat_this) detects beats and downbeats.
2. Each beat is snapped to the actual drum hit (spectral flux at ~3 ms resolution; the model works
   at 20 ms frames).
3. Double-time detections are dropped and gaps where the drums stop are filled.
4. The tempo is smoothed over ±N beats (default 4). The click follows tempo drift but ignores the
   drummer's per-hit micro-timing. `--smooth 0` follows every hit.
5. Bar positions come from the detected downbeats. A shift of the downbeat is followed only once it
   is confirmed for several bars in a row. Half-bar shifts (beat 1 vs beat 3) need even more evidence,
   because a kick/snare groove often looks the same in both halves of the bar.

On the test songs the click lands within ~3 ms RMS of the drum hits.

## Setup (Windows)

Requires Python 3.10+ and, for MP3 output only, `ffmpeg` on the PATH.

```bat
setup.bat
```

This creates a `.venv`, installs PyTorch (CPU build), Beat This! and the other dependencies, and
creates the `Drums2Click` shortcut.
The model checkpoint is downloaded automatically on first run.

## Usage

**GUI:** double-click the **Drums2Click** shortcut (metronome icon, created by `setup.bat`, or
run `make_shortcut.ps1`), or `Drums2Click.bat`. Choose the stem and the output folder, then click
*Genera click*. The window follows the Windows light/dark theme, and the technical log is under
*Dettagli*. Icon and checkbox graphics are drawn by `tools/make_assets.py`.

**Command line:**

```bat
.venv\Scripts\python drums2click.py "drums.flac" --click-audio click.flac
.venv\Scripts\python drums2click.py "drums.flac" -o tempo.mid --check-wav check.wav
```

Useful options:

| Option | Default | |
|---|---|---|
| `--click-audio PATH` | | write the click as audio (`.flac`, `.wav`, `.mp3`) |
| `--check-wav PATH` | | write stem + click for checking by ear |
| `--smooth N` | 4 | average tempo over ±N beats, 0 = follow every hit |
| `--beats-per-bar N` | detected | force the meter |
| `--count-in N` | -1 (all) | clicks added in the silence before the drums |
| `--click-sample` / `--accent-sample` | built-in beep | use your own click sounds |
| `--dbn` | off | steadier tracking for songs with an almost constant tempo |
| `--device cuda` | cpu | run the model on an NVIDIA GPU |

Run `drums2click.py --help` for the full list.

## Testing

Ground truth is listening: a click is validated when it sounds right against the drums in the
check WAV. Two tools build on that.

**Regression** (`tools/regression.py`): re-runs the tool on stems validated by ear and compares
the result with their reference MIDI. A change passes only if every song gives the same clicks
and accents within 1 ms. References live in `reference/`, which is not in git because they are
derived from copyrighted stems.

```bat
.venv\Scripts\python tools\regression.py
.venv\Scripts\python tools\regression.py --add my_song "my song (Drums).flac"
```

Add a reference only after listening to its check WAV. The current set: Billie Jean, Livin' on a
Prayer, Don't Fear the Reaper, It's My Life, Thriller.

**Alignment check** (`tools/check_alignment.py stem.flac click.mid`): for a new song.
- **Timing** is reliable. The onset envelope is averaged around each click, and the peak must be
  near 0 ms (+1.5..+6 ms on the validated songs). Shifting the click by ±20 ms must move the peak
  accordingly, as a control.
- **Beat vs off-beat cannot be measured** automatically. A click moved by half a beat still lands
  on hi-hat eighths, and the kick/snare ratio it prints ranges from 1.0x to 5.3x on correct
  clicks. Listen to settle it.

A "percentage of beats on a hit" metric is deliberately not used. With dense drums it scores ~90%
even for a click shifted by half a beat, and it reported Battle Hymn as fine when it was not.

**Known limit:** dense, complex metal (tested: Manowar's Battle Hymn, with 3/4 vs 6/8 feel
changes, rolls and a free intro) does not give a usable click. An attempt to handle it with extra
heuristics was reverted because it changed songs that already worked.

## Notes

- Prefer FLAC or WAV for the click. MP3 adds encoder delay at the start, which some DAWs don't
  compensate.
- In MIDI, beat 1 must be at time 0. If the first hit is a few ms into the file, it is moved to 0;
  otherwise a count-in is generated. The audio click always uses the real hit times.
- Some DAWs (e.g. older FL Studio versions) ignore time signature changes in imported MIDI. The
  beats stay aligned; only the bar lines shift after an odd bar.
- Beat This! and its model checkpoints are distributed under their own license. See the
  [Beat This! repository](https://github.com/CPJKU/beat_this).
