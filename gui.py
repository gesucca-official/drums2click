"""Small window for drums2click: pick a drum stem and an output folder, then generate."""
import ctypes
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "drums2click.py"
ASSETS = HERE / "assets"
PYTHON = Path(sys.executable).with_name("python.exe")  # launched with pythonw; run the script with python
AUDIO_TYPES = [("Audio", "*.wav *.flac *.aif *.aiff *.mp3 *.ogg"), ("Tutti i file", "*.*")]
PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")
SUMMARY = re.compile(r"meter (\d+)/4, tempo median ([\d.]+) BPM")
APP = "Drums → Click"

LIGHT = dict(bg="#f3f3f7", card="#ffffff", border="#e2e3ea", text="#1d1e24", muted="#6b6e7d",
             field="#f6f6f9", accent="#f0602c", accent_hover="#ff7743", accent_text="#ffffff",
             button="#ebebf0", button_hover="#e0e0e7", log="#fafafc", error="#c62828")
DARK = dict(bg="#15161c", card="#1f2129", border="#2e313c", text="#e9eaef", muted="#9a9db0",
            field="#2a2d38", accent="#ff7a3d", accent_hover="#ff925e", accent_text="#1a1206",
            button="#2c2f3a", button_hover="#373a47", log="#111217", error="#ff6b6b")


def windows_dark_mode():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


class ProgressLine(tk.Canvas):
    """Thin progress bar; set it with bar["value"] = 0..100 like a ttk.Progressbar."""

    def __init__(self, master, track, fill, height=4):
        super().__init__(master, height=height, bg=track, highlightthickness=0, bd=0)
        self._value = 0.0
        self._height = height
        self._bar = self.create_rectangle(0, 0, 0, height, fill=fill, width=0)
        self.bind("<Configure>", lambda _e: self._draw())

    def __setitem__(self, key, value):
        if key == "value":
            self._value = float(value)
            self._draw()
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        return self._value if key == "value" else super().__getitem__(key)

    def _draw(self):
        self.coords(self._bar, 0, 0, self.winfo_width() * self._value / 100, self._height)


class App(tk.Tk):
    def __init__(self):
        try:  # own taskbar entry and icon instead of python's
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("drums2click.gui")
        except (AttributeError, OSError):
            pass
        super().__init__()
        self.dark = windows_dark_mode()
        self.c = DARK if self.dark else LIGHT
        self.title(APP)
        self.configure(bg=self.c["bg"])
        self.resizable(False, False)
        self.set_icon()
        self.setup_style()

        self.proc = None
        self.lines = queue.Queue()
        self.started = 0.0
        self.summary = ""

        self.input = tk.StringVar()
        self.outdir = tk.StringVar()
        self.click = tk.BooleanVar(value=True)
        self.fmt = tk.StringVar(value="flac")
        self.check = tk.BooleanVar(value=False)
        self.smooth = tk.IntVar(value=4)
        self.status = tk.StringVar(value="Scegli uno stem di batteria per iniziare.")

        self.build()
        self.after(10, self.dark_title_bar)
        self.after(100, self.poll)

    # ------------------------------------------------------------ look

    def set_icon(self):
        try:
            self.iconbitmap(default=str(ASSETS / "icon.ico"))
            self.header_icon = tk.PhotoImage(file=str(ASSETS / "icon_48.png"))
        except tk.TclError:
            self.header_icon = None

    def dark_title_bar(self):
        if not self.dark:
            return
        try:
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
        except (AttributeError, OSError):
            pass

    def setup_style(self):
        c = self.c
        s = ttk.Style(self)
        s.theme_use("clam")
        base = ("Segoe UI", 10)
        s.configure(".", background=c["card"], foreground=c["text"], font=base,
                    bordercolor=c["border"], lightcolor=c["card"], darkcolor=c["card"],
                    troughcolor=c["field"], focuscolor=c["accent"])
        s.configure("TFrame", background=c["card"])
        s.configure("Bg.TFrame", background=c["bg"])
        s.configure("TLabel", background=c["card"], foreground=c["text"])
        s.configure("Muted.TLabel", foreground=c["muted"], font=("Segoe UI", 9))
        s.configure("Section.TLabel", font=("Segoe UI Semibold", 10))
        s.configure("Title.TLabel", background=c["bg"], font=("Segoe UI Semibold", 17))
        s.configure("Subtitle.TLabel", background=c["bg"], foreground=c["muted"])
        s.configure("Status.TLabel", background=c["bg"], foreground=c["muted"])
        s.configure("Error.TLabel", background=c["bg"], foreground=c["error"])

        s.configure("TEntry", fieldbackground=c["field"], foreground=c["text"], insertcolor=c["text"],
                    bordercolor=c["border"], lightcolor=c["field"], darkcolor=c["field"], padding=(8, 6))
        s.map("TEntry", bordercolor=[("focus", c["accent"])], lightcolor=[("focus", c["accent"])])
        s.configure("TSpinbox", fieldbackground=c["field"], foreground=c["text"], insertcolor=c["text"],
                    bordercolor=c["border"], lightcolor=c["field"], darkcolor=c["field"],
                    arrowcolor=c["muted"], background=c["field"], padding=(6, 4))
        s.map("TSpinbox", bordercolor=[("focus", c["accent"])])

        # checkbox and radio indicators drawn by tools/make_assets.py, instead of clam's "X"
        theme = "dark" if self.dark else "light"
        self.indicators = {f"{k}_{st}": tk.PhotoImage(file=str(ASSETS / f"{k}_{st}_{theme}.png"))
                           for k in ("check", "radio") for st in ("on", "off")}
        for kind, img in (("Checkbutton", "check"), ("Radiobutton", "radio")):
            s.element_create(f"Pretty.{kind}.indicator", "image", self.indicators[f"{img}_off"],
                             ("selected", self.indicators[f"{img}_on"]), sticky="")
            s.layout(f"T{kind}", [(f"{kind}.padding", {"sticky": "nswe", "children": [
                (f"Pretty.{kind}.indicator", {"side": "left", "sticky": ""}),
                (f"{kind}.focus", {"side": "left", "sticky": "w",
                                   "children": [(f"{kind}.label", {"sticky": "nswe"})]})]})])
            s.configure(f"T{kind}", background=c["card"], foreground=c["text"], padding=(0, 3))
            s.map(f"T{kind}", background=[("active", c["card"])], foreground=[("disabled", c["muted"])])

        s.configure("TButton", background=c["button"], foreground=c["text"], bordercolor=c["button"],
                    lightcolor=c["button"], darkcolor=c["button"], padding=(12, 6))
        s.map("TButton", background=[("disabled", c["button"]), ("active", c["button_hover"])],
              foreground=[("disabled", c["muted"])],
              bordercolor=[("active", c["button_hover"])],
              lightcolor=[("active", c["button_hover"])], darkcolor=[("active", c["button_hover"])])
        s.configure("Accent.TButton", background=c["accent"], foreground=c["accent_text"],
                    bordercolor=c["accent"], lightcolor=c["accent"], darkcolor=c["accent"],
                    font=("Segoe UI Semibold", 11), padding=(22, 9))
        s.map("Accent.TButton", background=[("disabled", c["button"]), ("active", c["accent_hover"])],
              foreground=[("disabled", c["muted"])],
              bordercolor=[("disabled", c["button"]), ("active", c["accent_hover"])],
              lightcolor=[("disabled", c["button"]), ("active", c["accent_hover"])],
              darkcolor=[("disabled", c["button"]), ("active", c["accent_hover"])])
        s.configure("Link.TButton", background=c["bg"], foreground=c["muted"], bordercolor=c["bg"],
                    lightcolor=c["bg"], darkcolor=c["bg"], padding=(0, 2), font=("Segoe UI", 9))
        s.map("Link.TButton", background=[("active", c["bg"])], foreground=[("active", c["text"])],
              bordercolor=[("active", c["bg"])], lightcolor=[("active", c["bg"])],
              darkcolor=[("active", c["bg"])])
        s.configure("TSeparator", background=c["border"])

    def card(self, parent):
        return tk.Frame(parent, bg=self.c["card"], highlightthickness=1,
                        highlightbackground=self.c["border"], padx=18, pady=14)

    # ------------------------------------------------------------ layout

    def build(self):
        c = self.c
        root = ttk.Frame(self, style="Bg.TFrame", padding=(22, 18, 22, 16))
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)

        header = ttk.Frame(root, style="Bg.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        if self.header_icon:
            tk.Label(header, image=self.header_icon, bg=c["bg"]).grid(row=0, column=0, rowspan=2, padx=(0, 12))
        ttk.Label(header, text=APP, style="Title.TLabel").grid(row=0, column=1, sticky="sw")
        ttk.Label(header, text="Dallo stem di batteria a un click che segue il batterista",
                  style="Subtitle.TLabel").grid(row=1, column=1, sticky="nw")

        files = self.card(root)
        files.grid(row=1, column=0, sticky="ew")
        files.columnconfigure(0, weight=1)
        ttk.Label(files, text="Stem di batteria", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(files, textvariable=self.input, width=52).grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(files, text="Sfoglia…", command=self.pick_input).grid(row=1, column=1, padx=(8, 0), pady=(4, 0))
        ttk.Label(files, text="Cartella di output", style="Section.TLabel").grid(row=2, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(files, textvariable=self.outdir).grid(row=3, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(files, text="Sfoglia…", command=self.pick_outdir).grid(row=3, column=1, padx=(8, 0), pady=(4, 0))
        ttk.Label(files, text="Se la lasci vuota, i file vanno in una cartella “out” accanto allo stem.",
                  style="Muted.TLabel").grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

        opts = self.card(root)
        opts.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        opts.columnconfigure(0, weight=1)
        ttk.Label(opts, text="Cosa generare", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(opts, text="Traccia click audio, per sostituire la batteria nel multitraccia",
                        variable=self.click, command=self.sync_format).grid(row=1, column=0, sticky="w", pady=(6, 0))
        fmt = ttk.Frame(opts)
        fmt.grid(row=1, column=1, sticky="e", pady=(6, 0))
        self.fmt_buttons = [ttk.Radiobutton(fmt, text=t, value=v, variable=self.fmt)
                            for t, v in (("FLAC", "flac"), ("MP3", "mp3"))]
        for i, b in enumerate(self.fmt_buttons):
            b.grid(row=0, column=i, padx=(10, 0))
        ttk.Checkbutton(opts, text="WAV di controllo (stem + click mixati, per verificare a orecchio)",
                        variable=self.check).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(opts, text="Il MIDI con la tempo map viene sempre generato.",
                  style="Muted.TLabel").grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))
        ttk.Separator(opts).grid(row=4, column=0, columnspan=2, sticky="ew", pady=12)
        smooth = ttk.Frame(opts)
        smooth.grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Label(smooth, text="Smoothing tempo", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(smooth, from_=0, to=32, width=4, textvariable=self.smooth).grid(row=0, column=1, padx=10)
        ttk.Label(smooth, text="± beat su cui fare la media · 0 = segue ogni singolo colpo",
                  style="Muted.TLabel").grid(row=0, column=2, sticky="w")

        actions = ttk.Frame(root, style="Bg.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(16, 0))
        actions.columnconfigure(0, weight=1)
        self.run_btn = ttk.Button(actions, text="Genera click", style="Accent.TButton", command=self.run)
        self.run_btn.grid(row=0, column=0, sticky="ew")
        self.open_btn = ttk.Button(actions, text="Apri cartella", command=self.open_outdir, state="disabled")
        self.open_btn.grid(row=0, column=1, sticky="ns", padx=(10, 0))

        self.progress = ProgressLine(root, track=c["border"], fill=c["accent"])
        self.progress.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        status = ttk.Frame(root, style="Bg.TFrame")
        status.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        status.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(status, textvariable=self.status, style="Status.TLabel")
        self.status_label.grid(row=0, column=0, sticky="w")
        self.details_btn = ttk.Button(status, text="Dettagli ▸", style="Link.TButton", command=self.toggle_details)
        self.details_btn.grid(row=0, column=1, sticky="e")

        self.log = tk.Text(root, width=10, height=10, wrap="word", state="disabled", font=("Consolas", 9),
                           bg=c["log"], fg=c["muted"], insertbackground=c["text"], relief="flat",
                           highlightthickness=1, highlightbackground=c["border"], padx=10, pady=8)
        self.log_row = 6

    def sync_format(self):
        for b in self.fmt_buttons:
            b.configure(state="normal" if self.click.get() else "disabled")

    def toggle_details(self, show=None):
        visible = bool(self.log.winfo_ismapped())
        show = not visible if show is None else show
        if show and not visible:
            self.log.grid(row=self.log_row, column=0, sticky="nsew", pady=(8, 0), in_=self.progress.master)
            self.details_btn.configure(text="Dettagli ▾")
        elif not show and visible:
            self.log.grid_remove()
            self.details_btn.configure(text="Dettagli ▸")

    # ------------------------------------------------------------ actions

    def pick_input(self):
        path = filedialog.askopenfilename(title="Scegli lo stem di batteria", filetypes=AUDIO_TYPES)
        if path:
            self.input.set(path)
            if not self.outdir.get():
                self.outdir.set(str(Path(path).parent / "out"))
            self.set_status(f"Pronto: {Path(path).name}")

    def pick_outdir(self):
        path = filedialog.askdirectory(title="Scegli la cartella di output")
        if path:
            self.outdir.set(path)

    def open_outdir(self):
        if Path(self.outdir.get()).is_dir():
            os.startfile(self.outdir.get())

    def set_status(self, text, error=False):
        self.status.set(text)
        self.status_label.configure(style="Error.TLabel" if error else "Status.TLabel")

    def write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def run(self):
        src = Path(self.input.get())
        if not src.is_file():
            messagebox.showerror(APP, "Scegli uno stem di batteria esistente.")
            return
        out = Path(self.outdir.get() or src.parent / "out")
        self.outdir.set(str(out))
        out.mkdir(parents=True, exist_ok=True)
        base = out / src.stem

        cmd = [str(PYTHON), "-u", str(SCRIPT), str(src), "-o", f"{base}.mid",
               "--smooth", str(self.smooth.get())]
        if self.click.get():
            cmd += ["--click-audio", f"{base}_click.{self.fmt.get()}"]
        if self.check.get():
            cmd += ["--check-wav", f"{base}_check.wav"]

        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.progress["value"] = 0
        self.summary = ""
        self.started = time.monotonic()
        self.set_status("Avvio…")
        self.run_btn.configure(state="disabled", text="Elaborazione…")
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
                self.handle(item.strip())
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def handle(self, line):
        m = PERCENT.fullmatch(line) or (PERCENT.search(line) if line.endswith("%") else None)
        if m:
            p = float(m.group(1))
            self.progress["value"] = 5 + 0.85 * p
            self.set_status(f"Analisi dei beat… {p:.0f}%")
            return
        if not line:
            return
        self.write(line)
        if line.startswith("Loading"):
            self.progress["value"] = 2
            self.set_status("Caricamento audio…")
        elif line.startswith("Downloading"):
            self.set_status("Download del modello (solo la prima volta)…")
        elif line.startswith("Tracking"):
            self.progress["value"] = 5
            self.set_status("Analisi dei beat…")
        elif line.startswith("Wrote"):
            self.progress["value"] = max(self.progress["value"], 95)
            self.set_status("Scrittura dei file…")
        s = SUMMARY.search(line)
        if s:
            self.summary = f"{float(s.group(2)):.1f} BPM · {s.group(1)}/4"

    def finish(self, code):
        self.run_btn.configure(state="normal", text="Genera click")
        elapsed = time.monotonic() - self.started
        if code == 0:
            self.progress["value"] = 100
            self.open_btn.configure(state="normal")
            parts = ["Fatto", self.summary, f"{elapsed:.0f} s"]
            self.set_status(" · ".join(p for p in parts if p))
            self.write("\nFatto!")
        else:
            self.set_status("Qualcosa è andato storto: guarda i dettagli.", error=True)
            self.write(f"\nErrore (codice {code}).")
            self.toggle_details(True)


if __name__ == "__main__":
    App().mainloop()
