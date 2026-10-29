#!/usr/bin/env python
"""drums2click: drum stem WAV -> MIDI with a tempo map and a click track that follows the drummer.

Every detected beat becomes exactly one quarter note on the MIDI grid: the tempo map changes
at each beat so that the DAW grid (bars, beats, quantize, metronome) sits on top of the audio.

Usage:
    python drums2click.py stem.wav                      # writes stem.mid
    python drums2click.py stem.wav -o click.mid --check-wav check.wav
    python drums2click.py stem.flac --click-audio click.flac   # click to replace the drum stem
"""
import argparse
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

FL_MIN_BPM, FL_MAX_BPM = 20.0, 480.0  # safe range inside FL Studio's 10-522 BPM limits


# ---------------------------------------------------------------- analysis

def load_audio(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x, x.mean(axis=1), sr


def track_beats(mono, sr, device, dbn):
    from beat_this.inference import Audio2Beats
    a2b = Audio2Beats(checkpoint_path="final0", device=device, dbn=dbn)
    beats, downbeats = a2b(mono, sr)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def onset_envelope(mono, sr, hop=128, n_fft=1024):
    """Spectral flux at ~3 ms resolution (Beat This works at 20 ms frames)."""
    x = np.pad(mono, (n_fft // 2, n_fft // 2))
    n = 1 + (len(x) - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    env = np.zeros(n, dtype=np.float32)
    prev = None
    for start in range(0, n, 4096):
        idx = np.arange(start, min(n, start + 4096))
        frames = x[idx[:, None] * hop + np.arange(n_fft)] * win
        mag = np.log1p(100.0 * np.abs(np.fft.rfft(frames, axis=1)))
        stacked = mag if prev is None else np.vstack([prev, mag])
        flux = np.maximum(np.diff(stacked, axis=0), 0).sum(axis=1)
        if prev is None:
            env[idx[1:]] = flux
        else:
            env[idx] = flux
        prev = mag[-1:]
    return env, sr / hop


def refine_to_onsets(beats, env, fps, window_s):
    """Move each beat to the strongest drum hit within +-window_s, if there is a real hit there."""
    out = beats.copy()
    w = max(1, int(round(window_s * fps)))
    thr = 0.2 * np.percentile(env, 99)
    moved = 0
    for i, b in enumerate(beats):
        c = int(round(b * fps))
        lo, hi = max(0, c - w), min(len(env), c + w + 1)
        if hi <= lo:
            continue
        j = int(np.argmax(env[lo:hi]))
        if env[lo + j] >= thr:
            out[i] = (lo + j) / fps
            moved += 1
    return out, moved


def drop_extra(beats, min_island=4):
    """Drop isolated hits (a lone roll in a free intro) and doubled detections (eighth-note fills).

    Returns (beats, isolated dropped, doubles dropped).
    """
    med = float(np.median(np.diff(beats)))
    runs = np.split(beats, np.where(np.diff(beats) > 4 * med)[0] + 1)
    kept = [r for r in runs if len(r) >= min_island]
    isolated = len(beats) - sum(len(r) for r in kept)
    beats = np.concatenate(kept) if kept else beats

    # compare against the tempo of the beats already kept, so a run of double-time detections
    # can't become the reference
    ref = float(np.median(np.diff(beats)))
    keep = [beats[0]]
    for b in beats[1:]:
        if b - keep[-1] >= 0.6 * ref:  # 0.6 < 2/3, so a 3:2 switch (e.g. 100 -> 150) is kept
            keep.append(b)
            if len(keep) >= 17:
                ref = float(np.median(np.diff(keep[-17:])))
    return np.array(keep), isolated, len(beats) - len(keep)


# (ref beats, detected intervals): e.g. (3, 2) = two detected beats span three reference beats
LEVEL_RATIOS = [(1, 3), (1, 2), (2, 3), (1, 1), (3, 2), (2, 1), (3, 1)]


def unify_levels(beats, bpm_hint=None, tol=0.06):
    """Keep one metrical level for the whole song.

    Trackers jump between pulse levels when the feel changes: 3/4 vs 6/8 (3:2), half-time
    (2:1). Sections whose local tempo is a simple ratio of the reference are re-divided to the
    reference pulse; real tempo changes (non-simple ratios) are left alone. The reference is the
    level that covers most of the song, or the one closest to bpm_hint.
    Returns (beats, number of re-divided sections, reference BPM).
    """
    ibi = np.diff(beats)
    n = len(ibi)
    loc = np.array([np.median(ibi[max(0, i - 4):i + 5]) for i in range(n)])
    logs = np.log(loc)
    weight = np.array([ibi[np.abs(logs - l) < tol].sum() for l in logs])
    ref = float(loc[np.argmax(weight)])
    if bpm_hint:
        levels = [ref * den / num for num, den in LEVEL_RATIOS]
        ref = min(levels, key=lambda p: abs(np.log(p * bpm_hint / 60.0)))

    def ratio(period):
        for num, den in LEVEL_RATIOS:
            if abs(np.log(period / ref) - np.log(num / den)) < tol:
                return num, den
        return 1, 1

    k = [ratio(p) for p in loc]
    k = [Counter(k[max(0, i - 4):i + 5]).most_common(1)[0][0] for i in range(n)]

    out, sections, a = [beats[0]], 0, 0
    pos = np.arange(len(beats))
    while a < n:
        b = a
        while b < n and k[b] == k[a]:
            b += 1
        num, den = k[a]
        use = (b - a) - (b - a) % den
        if (num, den) != (1, 1) and use:
            m = use * num // den
            frac = a + np.arange(1, m + 1) * (den / num)
            out.extend(np.interp(frac, pos, beats))
            sections += 1
        else:
            use = 0
        out.extend(beats[a + use + 1:b + 1])
        a = b
    return np.array(out), sections, 60.0 / ref


def roll_ranges(raw, ref_period, min_fast=3):
    """Time ranges where the tracker lost the beat in a roll or blast: several raw intervals in a
    row much faster than the song's pulse (> 2.5x)."""
    fast = np.diff(raw) < 0.4 * ref_period
    ranges = []
    for i in range(len(fast)):
        if fast[max(0, i - 4):i + 5].sum() >= min_fast:
            a, b = raw[i], raw[i + 1]
            if ranges and a <= ranges[-1][1] + 2 * ref_period:
                ranges[-1] = (ranges[-1][0], b)
            else:
                ranges.append((a, b))
    return [(a, b) for a, b in ranges if b - a > 4 * ref_period]


def fill_gaps(beats, is_quiet, max_click_fill=16):
    """Fill gaps (drums stopping, or rolls removed from the guide) using the tempo on both sides.

    Short gaps (breaks, stops) get normal clicks. In long gaps where the drums are silent the
    tempo is only a guess (the band may play freely), so those beats stay in the MIDI grid but
    are marked silent for the click. Long gaps with drums playing (rolls) keep their clicks.
    Returns (beats, quiet gap ranges, silent ranges, filled, silent count).
    """
    ibi = np.diff(beats)
    out, quiet, ranges, filled, muted = [beats[0]], [], [], 0, 0
    for i in range(len(ibi)):
        around = np.r_[ibi[max(0, i - 8):i], ibi[i + 1:i + 9]]
        n = int(round(ibi[i] / np.median(around))) if len(around) else 1
        if n >= 2:
            new = np.linspace(beats[i], beats[i + 1], n + 1)[1:-1]
            out.extend(new)
            filled += len(new)
            if is_quiet(beats[i], beats[i + 1]):
                quiet.append((beats[i], beats[i + 1]))
                if len(new) > max_click_fill:
                    ranges.append((beats[i], beats[i + 1]))
                    muted += len(new)
        out.append(beats[i + 1])
    return np.array(out), quiet, ranges, filled, muted


def in_ranges(beats, ranges, margin=0.05):
    """Mask of beats strictly inside any (start, end) range."""
    mask = np.zeros(len(beats), dtype=bool)
    for a, b in ranges:
        mask |= (beats > a + margin) & (beats < b - margin)
    return mask


def dp_track(guide, mono, sr, tightness=100.0):
    """Re-track the beats with dynamic programming (Ellis 2007) on the onset envelope.

    The cleaned-up beats are used only as a tempo guide; the result is the single most
    consistent beat sequence that sits on the strongest hits, so section changes, fills and
    rolls can't add or lose a beat.
    """
    env, fps = onset_envelope(mono, sr, hop=256, n_fft=1024)
    env = env / (env.std() + 1e-9)
    ibi = np.diff(guide)
    per = np.array([np.median(ibi[max(0, i - 8):i + 9]) for i in range(len(ibi))])
    mids = (guide[:-1] + guide[1:]) / 2
    lo = int(max(0, (guide[0] - 0.05) * fps))  # start on the first guide beat, not a pickup
    hi = int(min(len(env), (guide[-1] + 0.3 * per[-1]) * fps) + 1)
    o = env[lo:hi]
    P = np.interp((np.arange(lo, hi)) / fps, mids, per) * fps  # period in frames

    score = o.copy()
    back = np.full(len(o), -1)
    for t in range(len(o)):
        a, b = int(round(t - 2 * P[t])), int(round(t - P[t] / 2))
        if b < 0:
            continue
        a = max(a, 0)
        cand = score[a:b + 1] - tightness * np.log((t - np.arange(a, b + 1)) / P[t]) ** 2
        k = int(np.argmax(cand))
        if cand[k] > 0:
            score[t] += cand[k]
            back[t] = a + k

    # last beat: best score near the end of the guide
    tail = int(1.5 * P[-1])
    t = len(o) - tail + int(np.argmax(score[-tail:]))
    path = []
    while t >= 0:
        path.append(t)
        t = back[t]
    return (np.array(path[::-1]) + lo) / fps


def smooth_beats(beats, n):
    """Local linear fit over 2n+1 beats: keeps tempo drift, removes per-beat jitter."""
    if n <= 0 or len(beats) < 3:
        return beats
    idx = np.arange(len(beats))
    out = beats.copy()
    for i in range(len(beats)):
        lo, hi = max(0, i - n), min(len(beats), i + n + 1)
        out[i] = np.polyval(np.polyfit(idx[lo:hi], beats[lo:hi], 1), i)
    return out


def bar_starts(beats, downbeats, beats_per_bar, win_bars=8):
    """Beat indices where bars start.

    The meter is the most common downbeat spacing. The bar position is decided by the downbeats
    voting over a window of win_bars bars, so scattered wrong detections don't move it. A shift
    (an odd bar, or a beat lost in a break) is followed when the new position clearly wins and
    keeps winning for two bars. Half-bar shifts (beat 1 vs 3, which share the same kick/snare
    pattern) need a much larger margin.
    Returns (beats_per_bar, phase of the first bar, sorted bar-start indices, odd bars).
    """
    n = len(beats)
    idx = sorted({int(np.argmin(np.abs(beats - d))) for d in downbeats})
    if beats_per_bar is None:
        gaps = Counter(g for g in np.diff(idx) if g > 1)
        long_gaps = Counter({g: c for g, c in gaps.items() if g >= 3})
        if long_gaps and sum(long_gaps.values()) >= 0.2 * sum(gaps.values()):
            gaps = long_gaps
        beats_per_bar = gaps.most_common(1)[0][0] if gaps else 4
    B = beats_per_bar
    if not idx:
        return B, 0, list(range(0, n, B)), []

    hit = np.zeros(n)
    hit[idx] = 1
    kernel = np.ones(win_bars * B + 1)
    pos = np.arange(n)
    votes = np.stack([np.convolve(hit * (pos % B == p), kernel, "same") for p in range(B)], axis=1)

    def is_half(new, cur):
        return B % 2 == 0 and (new - cur) % B == B // 2

    def wins(i, new, cur):
        return votes[i, new] >= (2.0 if is_half(new, cur) else 1.3) * max(votes[i, cur], 1.0)

    def confirmed(s, new, need=8):
        """need downbeats in a row, one bar apart, all on the new phase, starting at s"""
        run = [j for j in idx if j >= s][:need]
        return (len(run) == need and all(j % B == new for j in run)
                and all(b - a == B for a, b in zip(run, run[1:])))

    first_phase = current = int(np.argmax(votes[idx[0]]))
    switches, i = [], 0  # (beat index where the new phase starts, phase)
    while i < n:
        new = int(np.argmax(votes[i]))
        if new != current and all(wins(j, new, current) for j in range(i, min(n, i + 2 * B))):
            # switch at the first new-phase downbeat after the last old-phase one nearby
            last_old = max([j for j in idx if j % B == current and j < i] or [i - 1])
            s = min([j for j in idx if j % B == new and j > last_old] or [i])
            if not confirmed(s, new, 8 if is_half(new, current) else 4):
                i += 1
                continue
            if not switches or s > switches[-1][0]:
                switches.append((s, new))
            current = new
            i = max(i, s) + 1
            continue
        i += 1

    starts, seg_phase = [], first_phase
    for i in range(len(beats)):
        if switches and i == switches[0][0]:
            seg_phase = switches.pop(0)[1]
            if starts and i - starts[-1] < 2:
                starts.pop()  # merge a 1-beat bar into the previous one
            starts.append(i)
        elif i % B == seg_phase and (not starts or i - starts[-1] >= 2):
            starts.append(i)
    odd = [(beats[s], b - s) for s, b in zip(starts, starts[1:]) if b - s != B]
    return B, first_phase, starts, odd


def build_grid(beats, beats_per_bar, phase, max_snap):
    """Prepend lead-in beats so that time 0 = bar 1 beat 1 and downbeats land on bar lines.

    Returns (grid times, number of virtual lead-in beats, snapped flag).
    """
    t0 = beats[0]
    ibi = float(np.median(np.diff(beats[:9])))
    if t0 <= max_snap and phase == 0:
        grid = beats.copy()
        grid[0] = 0.0
        return grid, 0, t0 > 0

    best, fallback = None, None
    for r in range(1, 8 * beats_per_bar + 1):
        if (r + phase) % beats_per_bar:
            continue
        bpm = 60.0 * r / t0 if t0 > 0 else float("inf")
        cost = abs(np.log((r * ibi) / t0)) if t0 > 0 else float("inf")
        if fallback is None or cost < fallback[1]:
            fallback = (r, cost)
        if FL_MIN_BPM <= bpm <= FL_MAX_BPM and (best is None or cost < best[1]):
            best = (r, cost)
    if best is None:
        if phase == 0:
            print(f"  warning: first beat at {t0 * 1000:.0f} ms is too close to the start for a "
                  f"count-in; snapping it to 0 (only the first hit will be off)")
            grid = beats.copy()
            grid[0] = 0.0
            return grid, 0, True
        best = fallback
        print("  warning: count-in tempo is outside FL Studio's range; consider adding silence "
              "before the stems")
    r = best[0]
    return np.concatenate([np.linspace(0.0, t0, r, endpoint=False), beats]), r, False


# ---------------------------------------------------------------- MIDI

def vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def meta(kind, data):
    return bytes([0xFF, kind]) + vlq(len(data)) + data


def track_chunk(events):
    data = bytearray()
    last = 0
    for tick, ev in sorted(events, key=lambda e: e[0]):
        data += vlq(tick - last) + ev
        last = tick
    data += vlq(0) + b"\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(data)) + data


def write_midi(path, grid, first_real, starts, beats_per_bar, ppq, accent_note, beat_note,
               silent=()):
    """starts: bar-start indices into grid (lead-in bars included); silent: grid indices that
    stay in the tempo map but get no click note."""
    us = np.round(grid * 1e6).astype(np.int64)
    tempo_events = [(0, meta(0x03, b"Tempo Map"))]
    last = None
    for i in range(len(grid) - 1):
        per_beat = int(min(us[i + 1] - us[i], 0xFFFFFF))
        if per_beat != last:
            tempo_events.append((i * ppq, meta(0x51, per_beat.to_bytes(3, "big"))))
            last = per_beat

    # time signature changes for odd bars (and back), so bar lines follow the music
    bounds = list(starts) + [starts[-1] + beats_per_bar]
    last = None
    for s, e in zip(bounds, bounds[1:]):
        n = e - s
        if n != last:
            tempo_events.append((s * ppq, meta(0x58, bytes([n, 2, 24, 8]))))
            last = n

    accents, silent = set(starts), set(silent)
    click_events = [(0, meta(0x03, b"Click"))]
    length = ppq // 4
    for i in range(first_real, len(grid)):
        if i in silent:
            continue
        note = accent_note if i in accents else beat_note
        vel = 127 if i in accents else 90
        click_events.append((i * ppq, bytes([0x99, note, vel])))
        click_events.append((i * ppq + length, bytes([0x89, note, 0])))

    header = b"MThd" + struct.pack(">IHHH", 6, 1, 2, ppq)
    Path(path).write_bytes(header + track_chunk(tempo_events) + track_chunk(click_events))


def read_midi_clicks(path):
    """Read back the file we wrote: returns (click times in seconds, is_accent) via its tempo map."""
    data = Path(path).read_bytes()
    ppq = struct.unpack(">H", data[12:14])[0]
    pos, tracks = 14, []
    while pos < len(data):
        size = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        tracks.append(data[pos + 8:pos + 8 + size])
        pos += 8 + size

    def events(trk):
        p, tick, status = 0, 0, 0
        while p < len(trk):
            delta = 0
            while True:
                b = trk[p]; p += 1
                delta = (delta << 7) | (b & 0x7F)
                if b < 0x80:
                    break
            tick += delta
            if trk[p] == 0xFF:
                kind = trk[p + 1]; p += 2
                ln = 0
                while True:
                    b = trk[p]; p += 1
                    ln = (ln << 7) | (b & 0x7F)
                    if b < 0x80:
                        break
                yield tick, ("meta", kind, trk[p:p + ln]); p += ln
            else:
                if trk[p] & 0x80:
                    status = trk[p]; p += 1
                n = 1 if status & 0xF0 in (0xC0, 0xD0) else 2
                yield tick, ("midi", status, trk[p:p + n]); p += n

    tempos = [(t, int.from_bytes(d, "big")) for trk in tracks for t, (k, kind, d) in events(trk)
              if k == "meta" and kind == 0x51]
    tempos.sort()

    def tick_to_sec(tick):
        sec, prev_tick, prev_us = 0.0, 0, 500000
        for t, u in tempos:
            if t >= tick:
                break
            sec += (t - prev_tick) * prev_us / ppq / 1e6
            prev_tick, prev_us = t, u
        return sec + (tick - prev_tick) * prev_us / ppq / 1e6

    clicks = [(tick_to_sec(t), d[0]) for trk in tracks for t, (k, st, d) in events(trk)
              if k == "midi" and st & 0xF0 == 0x90 and d[1] > 0]
    return clicks


# ---------------------------------------------------------------- audio click

def synth_click(sr, freq):
    """Classic digital-metronome beep: fast attack, short exponential decay."""
    n = int(0.05 * sr)
    t = np.arange(n) / sr
    env = np.minimum(t / 0.0005, 1.0) * np.exp(-t / 0.012)
    tone = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    return (tone * env / 1.3).astype(np.float32)


def load_click_sample(path, sr):
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        import soxr
        x = soxr.resample(x, file_sr, sr).astype(np.float32)
    return x / max(np.abs(x).max(), 1e-9)


def render_clicks(n_samples, sr, clicks, accent, normal):
    """clicks: list of (seconds, is_accent). Mono track of exactly n_samples."""
    out = np.zeros(n_samples, dtype=np.float32)
    for sec, is_accent in clicks:
        s = int(round(sec * sr))
        if not 0 <= s < n_samples:
            continue
        blip = (accent if is_accent else normal)[: n_samples - s]
        out[s:s + len(blip)] += blip * (1.0 if is_accent else 0.7)
    return out


def normalize(x, peak_db=-1.0):
    peak = np.abs(x).max()
    return x * (10 ** (peak_db / 20) / peak) if peak > 0 else x


def write_audio(path, x, sr):
    path = Path(path)
    if path.suffix.lower() == ".mp3":
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "click.wav"
            sf.write(str(wav), x, sr, subtype="PCM_24")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                            "-codec:a", "libmp3lame", "-b:a", "320k", str(path)], check=True)
        print("  note: MP3 adds encoder delay that some DAWs don't compensate; "
              "use FLAC/WAV for sample-accurate alignment")
    else:
        sf.write(str(path), x, sr, subtype="PCM_24")


def count_in(beats, n_before):
    """Extend the beat grid backwards into the silence before the first hit."""
    ibi = float(np.median(np.diff(beats[:9])))
    if n_before < 0:
        n_before = int(beats[0] / ibi) + 1
    extra = beats[0] - ibi * np.arange(1, n_before + 1)
    return extra[extra >= 0][::-1]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Drum stem WAV -> MIDI tempo map + click track")
    ap.add_argument("input", type=Path, help="drum stem (wav/flac/aiff)")
    ap.add_argument("-o", "--output", type=Path, help="output .mid (default: next to input)")
    ap.add_argument("--check-wav", type=Path, help="also write the stem mixed with the click, "
                                                   "rendered from the MIDI file, to verify by ear")
    ap.add_argument("--beats-per-bar", type=int, help="force the meter (default: detected)")
    ap.add_argument("--smooth", type=int, default=4,
                    help="follow tempo drift averaged over +-N beats, ignoring the drummer's "
                         "per-hit micro-timing (default 4; 0 = follow every hit exactly)")
    ap.add_argument("--no-refine", action="store_true",
                    help="don't snap beats to the drum transients")
    ap.add_argument("--refine-window", type=float, default=0.035,
                    help="max distance in seconds for snapping a beat to a hit (default 0.035)")
    ap.add_argument("--max-snap", type=float, default=0.03,
                    help="if the first beat is within this many seconds of the start, "
                         "put it at 0 instead of adding a count-in (default 0.03)")
    ap.add_argument("--dbn", action="store_true",
                    help="use the DBN postprocessor (steadier, assumes a fairly constant tempo)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--ppq", type=int, default=960)
    ap.add_argument("--accent-note", type=int, default=76, help="downbeat note (76 hi wood block)")
    ap.add_argument("--beat-note", type=int, default=77, help="other beats (77 low wood block)")
    ap.add_argument("--dump-beats", type=Path, help="write the final beat times to a text file")
    ap.add_argument("--click-audio", type=Path,
                    help="also write the click as audio (.flac/.wav/.mp3), normalized, same length "
                         "and sample rate as the stem, to replace the drums in a multitrack")
    ap.add_argument("--count-in", type=int, default=8,
                    help="max clicks added in the silence before the first hit "
                         "(default 8, -1 = fill all of it, 0 = none)")
    ap.add_argument("--bpm", type=float,
                    help="approximate tempo you count the song in: picks the pulse level when the "
                         "drums are ambiguous (e.g. 100 vs 150 in a 6/8 song)")
    ap.add_argument("--keep-levels", action="store_true",
                    help="don't unify the pulse level across sections")
    ap.add_argument("--no-dp", action="store_true",
                    help="skip the final dynamic-programming re-tracking")
    ap.add_argument("--click-sample", type=Path, help="custom sample for the normal clicks")
    ap.add_argument("--accent-sample", type=Path, help="custom sample for the downbeat clicks")
    args = ap.parse_args()

    out_mid = args.output or args.input.with_suffix(".mid")
    print(f"Loading {args.input}")
    audio, mono, sr = load_audio(args.input)

    print("Tracking beats (Beat This!)...")
    beats, downbeats = track_beats(mono, sr, args.device, args.dbn)
    if len(beats) < 4:
        sys.exit("Too few beats detected.")
    print(f"  {len(beats)} beats, {len(downbeats)} downbeats")

    env, fps = onset_envelope(mono, sr)

    def level_db(a, b):
        seg = mono[int(a * sr):int(b * sr)]
        return 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else -120.0

    playing = [x for x in (level_db(t, t + 1) for t in range(int(len(mono) / sr))) if x > -60]
    song_db = float(np.median(playing)) if playing else -20.0

    def is_quiet(a, b):
        """drums absent: much softer than the song's typical level (bleed only)"""
        return level_db(a, b) < song_db - 15

    raw = beats
    if not args.no_refine:
        beats, moved = refine_to_onsets(beats, env, fps, args.refine_window)
        print(f"  refined {moved}/{len(beats)} beats onto drum hits")
    beats = np.unique(np.round(beats, 6))
    beats, isolated, doubles = drop_extra(beats)
    if isolated or doubles:
        print(f"  removed {isolated} isolated hits, {doubles} doubled beats")
    ref_bpm = 60.0 / np.median(np.diff(beats))
    if not args.keep_levels:
        beats, sections, ref_bpm = unify_levels(beats, args.bpm)
        if sections:
            print(f"  {sections} sections re-divided to a single pulse of ~{ref_bpm:.0f} BPM "
                  f"(--bpm to pick another level, --keep-levels to disable)")
            if not args.no_refine:
                beats, _ = refine_to_onsets(beats, env, fps, args.refine_window)
                beats = np.unique(np.round(beats, 6))
    rolls = roll_ranges(raw, 60.0 / ref_bpm)
    if rolls:
        inside = in_ranges(beats, rolls, margin=0.0)
        beats = beats[~inside]
        print(f"  {len(rolls)} rolls/blasts where the tracker lost the beat, tempo bridged: "
              + ", ".join(f"{a:.0f}-{b:.0f}s" for a, b in rolls))
    beats, quiet, ranges, filled, muted = fill_gaps(beats, is_quiet)
    if filled:
        print(f"  filled {filled} beats in gaps" + (f" ({muted} silent in long drumless parts)" if muted else ""))
    if not args.no_dp:
        guide = beats
        beats = dp_track(guide, mono, sr)
        if not args.no_refine:
            beats, _ = refine_to_onsets(beats, env, fps, args.refine_window)
            beats = np.unique(np.round(beats, 6))
        # where the drums are silent there is nothing to track: keep the interpolated beats
        beats = np.sort(np.r_[beats[~in_ranges(beats, quiet, margin=0.1)],
                              guide[in_ranges(guide, quiet, margin=0.1)]])
        print(f"  re-tracked with dynamic programming: {len(beats)} beats (guide {len(guide)})")
    silent = in_ranges(beats, ranges)
    beats = smooth_beats(beats, args.smooth)

    beats_per_bar, phase, starts, odd = bar_starts(beats, downbeats, args.beats_per_bar)
    grid, lead_in, snapped = build_grid(beats, beats_per_bar, phase, args.max_snap)
    grid_starts = list(range(0, lead_in + phase, beats_per_bar)) + [s + lead_in for s in starts]

    bpm = 60.0 / np.diff(beats)
    print(f"  meter {beats_per_bar}/4, tempo median {np.median(bpm):.2f} BPM "
          f"(range {bpm.min():.1f}-{bpm.max():.1f})")
    if odd:
        print(f"  {len(odd)} odd-length bars (MIDI gets time signature changes): "
              + ", ".join(f"{t:.1f}s {n}/4" for t, n in odd[:10]) + (" ..." if len(odd) > 10 else ""))
    if lead_in:
        print(f"  first beat at {beats[0]:.3f}s -> {lead_in}-beat count-in")
    elif snapped:
        print(f"  first beat at {beats[0] * 1000:.0f} ms snapped to 0")

    write_midi(out_mid, grid, lead_in, grid_starts, beats_per_bar, args.ppq,
               args.accent_note, args.beat_note, silent=np.where(silent)[0] + lead_in)
    print(f"Wrote {out_mid}")

    if args.dump_beats:
        np.savetxt(args.dump_beats, beats, fmt="%.6f")
    accent = load_click_sample(args.accent_sample, sr) if args.accent_sample else synth_click(sr, 1760)
    normal = load_click_sample(args.click_sample, sr) if args.click_sample else synth_click(sr, 1320)

    if args.check_wav:
        clicks = read_midi_clicks(out_mid)
        err = np.array([c for c, _ in clicks]) - grid[lead_in:][~silent]
        print(f"  MIDI round-trip error: max {np.abs(err).max() * 1000:.3f} ms")
        click = render_clicks(len(audio), sr, [(c, n == args.accent_note) for c, n in clicks],
                              accent, normal)
        mix = audio * 0.7 + 0.5 * click[:, None]
        sf.write(str(args.check_wav), mix / max(1.0, np.abs(mix).max()), sr)
        print(f"Wrote {args.check_wav}")

    if args.click_audio:
        # real beat times (not the MIDI snap/count-in), plus clicks in the silence before the drums
        pre = count_in(beats, args.count_in)
        accents = set(starts)
        clicks = [(t, (j - len(pre) - phase) % beats_per_bar == 0) for j, t in enumerate(pre)]
        clicks += [(t, i in accents) for i, t in enumerate(beats) if not silent[i]]
        write_audio(args.click_audio, normalize(render_clicks(len(audio), sr, clicks, accent, normal)), sr)
        print(f"Wrote {args.click_audio} ({len(pre)} count-in clicks, {len(audio) / sr:.2f}s)")


if __name__ == "__main__":
    main()
