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

This creates a `.venv` and installs PyTorch (CPU build), Beat This! and the other dependencies.
The model checkpoint is downloaded automatically on first run.

## Usage

**GUI:** double-click `Drums2Click.bat`, choose the stem and the output folder, then click *Genera*.

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

## Notes

- Prefer FLAC or WAV for the click. MP3 adds encoder delay at the start, which some DAWs don't
  compensate.
- In MIDI, beat 1 must be at time 0. If the first hit is a few ms into the file, it is moved to 0;
  otherwise a count-in is generated. The audio click always uses the real hit times.
- Some DAWs (e.g. older FL Studio versions) ignore time signature changes in imported MIDI. The
  beats stay aligned; only the bar lines shift after an odd bar.
- Beat This! and its model checkpoints are distributed under their own license. See the
  [Beat This! repository](https://github.com/CPJKU/beat_this).
