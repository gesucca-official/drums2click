"""Small window for drums2click: pick a drum stem and an output folder, then generate."""
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "drums2click.py"
PYTHON = Path(sys.executable).with_name("python.exe")  # launched with pythonw; run the script with python
AUDIO_TYPES = [("Audio", "*.wav *.flac *.aif *.aiff *.mp3 *.ogg"), ("Tutti i file", "*.*")]
PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Drums → Click")
        self.resizable(True, False)
        self.minsize(560, 0)
        self.proc = None
        self.lines = queue.Queue()

        self.input = tk.StringVar()
        self.outdir = tk.StringVar()
        self.click = tk.BooleanVar(value=True)
        self.fmt = tk.StringVar(value="flac")
        self.check = tk.BooleanVar(value=False)
        self.smooth = tk.IntVar(value=4)
        self.bpm = tk.StringVar()

        pad = {"padx": 8, "pady": 4}
        f = ttk.Frame(self, padding=8)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Stem batteria").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(f, textvariable=self.input).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(f, text="Sfoglia…", command=self.pick_input).grid(row=0, column=2, **pad)

        ttk.Label(f, text="Cartella output").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(f, textvariable=self.outdir).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(f, text="Sfoglia…", command=self.pick_outdir).grid(row=1, column=2, **pad)

        opts = ttk.LabelFrame(f, text="Output", padding=6)
        opts.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(opts, text="MIDI con tempo map (sempre)").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(opts, text="Traccia click audio", variable=self.click).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(opts, text="FLAC", value="flac", variable=self.fmt).grid(row=1, column=1, padx=6)
        ttk.Radiobutton(opts, text="MP3", value="mp3", variable=self.fmt).grid(row=1, column=2, padx=6)
        ttk.Checkbutton(opts, text="WAV di controllo (stem + click)", variable=self.check).grid(
            row=2, column=0, columnspan=3, sticky="w")
        ttk.Label(opts, text="Smoothing tempo (±beat, 0 = segue ogni colpo)").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(opts, from_=0, to=32, width=5, textvariable=self.smooth).grid(row=3, column=1, padx=6)
        ttk.Label(opts, text="BPM indicativo (opzionale, sceglie la pulsazione)").grid(row=4, column=0, sticky="w")
        ttk.Entry(opts, width=7, textvariable=self.bpm).grid(row=4, column=1, padx=6)

        self.run_btn = ttk.Button(f, text="Genera", command=self.run)
        self.run_btn.grid(row=3, column=0, sticky="w", **pad)
        self.open_btn = ttk.Button(f, text="Apri cartella", command=self.open_outdir, state="disabled")
        self.open_btn.grid(row=3, column=2, sticky="e", **pad)
        self.progress = ttk.Progressbar(f, maximum=100)
        self.progress.grid(row=3, column=1, sticky="ew", **pad)

        self.log = tk.Text(f, height=12, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.grid(row=4, column=0, columnspan=3, sticky="nsew", **pad)

        self.after(100, self.poll)

    def pick_input(self):
        path = filedialog.askopenfilename(title="Scegli lo stem di batteria", filetypes=AUDIO_TYPES)
        if path:
            self.input.set(path)
            if not self.outdir.get():
                self.outdir.set(str(Path(path).parent / "out"))

    def pick_outdir(self):
        path = filedialog.askdirectory(title="Scegli la cartella di output")
        if path:
            self.outdir.set(path)

    def open_outdir(self):
        if Path(self.outdir.get()).is_dir():
            os.startfile(self.outdir.get())

    def write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def run(self):
        src = Path(self.input.get())
        if not src.is_file():
            messagebox.showerror("Drums → Click", "Scegli uno stem di batteria esistente.")
            return
        out = Path(self.outdir.get() or src.parent)
        out.mkdir(parents=True, exist_ok=True)
        base = out / src.stem

        cmd = [str(PYTHON), "-u", str(SCRIPT), str(src), "-o", f"{base}.mid",
               "--smooth", str(self.smooth.get())]
        if self.click.get():
            cmd += ["--click-audio", f"{base}_click.{self.fmt.get()}"]
        if self.check.get():
            cmd += ["--check-wav", f"{base}_check.wav"]
        bpm = self.bpm.get().strip().replace(",", ".")
        if bpm:
            try:
                cmd += ["--bpm", str(float(bpm))]
            except ValueError:
                messagebox.showerror("Drums → Click", "Il BPM indicativo deve essere un numero.")
                return

        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.progress["value"] = 0
        self.run_btn.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        threading.Thread(target=self.worker, args=(cmd,), daemon=True).start()

    def worker(self, cmd):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
        self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        buf = b""
        while chunk := self.proc.stdout.read(256):
            buf += chunk
            *parts, buf = re.split(rb"[\r\n]", buf)
            for p in parts:
                self.lines.put(p.decode("utf-8", "replace"))
        if buf:
            self.lines.put(buf.decode("utf-8", "replace"))
        self.lines.put(("done", self.proc.wait()))

    def poll(self):
        try:
            while True:
                item = self.lines.get_nowait()
                if isinstance(item, tuple):
                    self.finish(item[1])
                    continue
                line = item.strip()
                m = PERCENT.fullmatch(line) or (PERCENT.search(line) if line.endswith("%") else None)
                if m:
                    self.progress["value"] = float(m.group(1))
                elif line:
                    self.write(line)
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def finish(self, code):
        self.run_btn.configure(state="normal")
        if code == 0:
            self.progress["value"] = 100
            self.open_btn.configure(state="normal")
            self.write("\nFatto!")
        else:
            self.write(f"\nErrore (codice {code}).")


if __name__ == "__main__":
    App().mainloop()
