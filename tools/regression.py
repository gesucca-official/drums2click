"""Regression test: re-run drums2click on stems validated by ear and compare with their reference MIDI.

The references live in reference/ (not in git, they come from copyrighted stems):
    reference/manifest.json   {"billie_jean": {"stem": "1_06. Billie Jean_(Drums).flac", "args": []}, ...}
    reference/<name>.mid      the MIDI that was validated by listening

    python tools/regression.py                 # check every reference
    python tools/regression.py billie_jean     # check some
    python tools/regression.py --add NAME STEM # store the current output as a new reference
                                               # (only after listening to its check WAV!)

A change passes when every song gives the same clicks (same count, same accents) within 1 ms.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"
sys.path.insert(0, str(ROOT))
import drums2click as d  # noqa: E402
from tools.check_alignment import check, report  # noqa: E402

TOLERANCE_MS = 1.0


def run(stem, args, out_mid):
    cmd = [sys.executable, str(ROOT / "drums2click.py"), str(ROOT / stem), "-o", str(out_mid), *args]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode:
        raise RuntimeError(res.stdout[-2000:] + res.stderr[-2000:])


def compare(ref_mid, new_mid):
    ref, new = d.read_midi_clicks(ref_mid), d.read_midi_clicks(new_mid)
    problems = []
    if len(ref) != len(new):
        problems.append(f"click count {len(ref)} -> {len(new)}")
    n = min(len(ref), len(new))
    dt = np.abs(np.array([a for a, _ in ref[:n]]) - np.array([b for b, _ in new[:n]])) * 1000
    if n and dt.max() > TOLERANCE_MS:
        first = int(np.argmax(dt > TOLERANCE_MS))
        problems.append(f"timing: {int((dt > TOLERANCE_MS).sum())} clicks moved > {TOLERANCE_MS} ms "
                        f"(max {dt.max():.1f} ms, first at {ref[first][0]:.2f}s)")
    acc = [i for i in range(n) if ref[i][1] != new[i][1]]
    if acc:
        problems.append(f"accents: {len(acc)} differ (first at {ref[acc[0]][0]:.2f}s)")
    return problems, (dt.max() if n else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--add", nargs=2, metavar=("NAME", "STEM"))
    args = ap.parse_args()

    manifest_path = REF / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    if args.add:
        name, stem = args.add
        REF.mkdir(exist_ok=True)
        run(stem, [], REF / f"{name}.mid")
        manifest[name] = {"stem": stem, "args": []}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Added reference {name}")
        return

    names = args.names or sorted(manifest)
    if not names:
        sys.exit("No references. Add one with --add NAME STEM after validating it by ear.")
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            entry = manifest[name]
            print(f"=== {name}")
            new_mid = Path(tmp) / f"{name}.mid"
            run(entry["stem"], entry.get("args", []), new_mid)
            problems, worst = compare(REF / f"{name}.mid", new_mid)
            if problems:
                failed += 1
                for p in problems:
                    print(f"  FAIL {p}")
            else:
                print(f"  PASS same clicks and accents (max difference {worst:.3f} ms)")
            report(check(ROOT / entry["stem"], new_mid))
    print(f"\n{len(names) - failed}/{len(names)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
