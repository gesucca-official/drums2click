"""Check how well a click (MIDI from drums2click) sits on the drum hits of a stem.

    python tools/check_alignment.py stem.flac click.mid

What it measures, and what it can't:
- Timing. The onset envelope is averaged around every click. If the click is on the hits, the
  peak of that average is near 0 ms; the same profile is computed with the click moved +-20 ms
  as a control, and its peak must move by about -+20 ms. On the five songs validated by ear the
  peak is at +1.5..+6 ms.
- Beat vs off-beat: NOT reliable. Drums have hits on the off-beats too (hi-hat eighths), so a
  click shifted by half a beat also looks "on a hit". The kick/snare ratio below is printed as a
  hint only: on the validated songs it ranges from 1.0x (Don't Fear the Reaper, It's My Life,
  both correct) to 5.3x. Only listening settles it.
- "Percentage of beats on a hit" is deliberately not reported: with a threshold-based onset
  detector it scores ~90% even for a click shifted by half a beat.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import drums2click as d  # noqa: E402

PEAK_OK_MS = (-5.0, 10.0)  # validated songs: +1.5..+6 ms


def low_band_flux(mono, sr, hop=64, n_fft=1024, fmax=250):
    """Onset envelope of the kick/snare range only."""
    x = np.pad(mono, (n_fft // 2, n_fft // 2))
    n = 1 + (len(x) - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    kmax = int(fmax * n_fft / sr) + 1
    env = np.zeros(n, dtype=np.float32)
    prev = None
    for start in range(0, n, 4096):
        idx = np.arange(start, min(n, start + 4096))
        frames = x[idx[:, None] * hop + np.arange(n_fft)] * win
        mag = np.log1p(100.0 * np.abs(np.fft.rfft(frames, axis=1)[:, :kmax]))
        stacked = mag if prev is None else np.vstack([prev, mag])
        flux = np.maximum(np.diff(stacked, axis=0), 0).sum(axis=1)
        if prev is None:
            env[idx[1:]] = flux
        else:
            env[idx] = flux
        prev = mag[-1:]
    return env, sr / hop


def profile_peak(env, fps, times, span=0.08):
    """Lag (ms) of the peak of the onset envelope averaged around the given times."""
    w = int(span * fps)
    acc = np.zeros(2 * w + 1)
    used = 0
    for t in times:
        c = int(round(t * fps))
        if w <= c < len(env) - w:
            acc += env[c - w:c + w + 1]
            used += 1
    return (int(np.argmax(acc)) - w) / fps * 1000 if used else float("nan")


def strongest_near(env, fps, times, win=0.025):
    w = int(win * fps)
    out = [env[c - w:c + w + 1].max() for c in (int(round(t * fps)) for t in times)
           if w <= c < len(env) - w]
    return np.array(out)


def check(stem, midi):
    """Returns a dict of measurements for a stem and a drums2click MIDI file."""
    _, mono, sr = d.load_audio(stem)
    clicks = np.array([t for t, _ in d.read_midi_clicks(midi)])
    env, fps = d.onset_envelope(mono, sr, hop=64, n_fft=512)
    peak = profile_peak(env, fps, clicks)
    plus = profile_peak(env, fps, clicks + 0.02)
    minus = profile_peak(env, fps, clicks - 0.02)
    low, lfps = low_band_flux(mono, sr)
    half = (clicks[:-1] + clicks[1:]) / 2
    ratio = np.median(strongest_near(low, lfps, clicks)) / np.median(strongest_near(low, lfps, half))
    timing_ok = (PEAK_OK_MS[0] <= peak <= PEAK_OK_MS[1]
                 and abs(plus - (peak - 20)) < 8 and abs(minus - (peak + 20)) < 8)
    return {"clicks": len(clicks), "peak_ms": peak, "peak_plus20_ms": plus, "peak_minus20_ms": minus,
            "kick_snare_ratio": ratio, "timing_ok": timing_ok}


def report(m):
    print(f"  clicks: {m['clicks']}")
    print(f"  timing: hit profile peak {m['peak_ms']:+.1f} ms "
          f"(controls: click +20 ms -> {m['peak_plus20_ms']:+.1f}, -20 ms -> {m['peak_minus20_ms']:+.1f}) "
          f"-> {'OK' if m['timing_ok'] else 'CHECK'}")
    print(f"  beat vs off-beat: kick/snare {m['kick_snare_ratio']:.1f}x on the click vs half a beat later "
          f"(hint only, listen to confirm)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    report(check(sys.argv[1], sys.argv[2]))
