"""Graphical (non-terminal) front end for gputaskman, built on tkinter.

Same data and the same provenance rules as the Textual UI (app.py): it only renders a
collector.Snapshot. Nothing here measures, sums or infers anything the TUI doesn't.

Run:  gputaskman-gui            (installed entry point, no console window on Windows)
      python -m gputaskman.gui  [--gpu N] [--interval S] [--no-host]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import filedialog, font as tkfont, messagebox, ttk

from .collector import BACKEND, Collector, ProcRow, Snapshot, enumerate_adapters, ordered_adapters
from .llm import gb
from .nvml import format_metric
from .report import (METAL_NOTE, WSL_EMPTY, has_engine_counters, has_wddm_counters, inferred_note, to_dict,
                     to_text, vram_label, wsl_note)

try:  # reuse the TUI's explanations so both front ends say exactly the same thing
    from .app import EXPLAIN, HELP_MD
except Exception:  # pragma: no cover - textual missing in a trimmed build
    EXPLAIN, HELP_MD = {}, "See README.md"

# ---------------------------------------------------------------------------------------------- theme

BG = "#1b1d21"
PANEL = "#23262b"
PANEL_2 = "#2b2f35"
FG = "#e6e6e6"
MUTED = "#9aa0a6"
DIM = "#6b7078"
GRID = "#353a41"
ACCENT = "#4ade80"
RED = "#f87171"
YELLOW = "#facc15"
GREEN = "#4ade80"
CYAN = "#22d3ee"
MAGENTA = "#e879f9"

KIND_COLOR = {
    "LLM": "#e879f9", "decision": "#da70d6", "decision?": "#c79bd6", "embedding": "#e0b0e0",
    "speech": "#afd7ff", "diffusion": "#da70d6", "AI": "#b197fc", "compute": "#f59e0b",
    "3D/game": "#22c55e", "video": "#eab308", "browser": "#3b82f6", "app": "#06b6d4",
    "compositor": "#7c8088", "kernel": "#7c8088", "system": "#7c8088", "wsl-vm": "#34d399",
    "external": "#7c8088",
}
ROW_TINT = {k: KIND_COLOR[k] for k in ("LLM", "decision", "decision?", "embedding", "speech", "diffusion", "AI",
                                       "wsl-vm")}
SEG_COLOR = {"weights": "#e879f9", "kv": "#fb923c", "compute": "#facc15", "runtime": "#94a3b8",
             "unknown": "#64748b"}
UNATTRIBUTED = "#4b5058"
FREE = "#2f3338"

CHARTS = (  # name, colour, unit
    ("GPU", "#4ade80", "%"), ("SM clock", "#fb923c", " MHz"), ("VRAM", "#60a5fa", "%"),
    ("Mem activity", "#a78bfa", "%"), ("Temperature", "#ef4444", "°C"), ("Power", "#facc15", " W"),
    ("Encoder", "#e2e8f0", "%"), ("Decoder", "#22d3ee", "%"),
)
METAL_CHARTS = ("GPU", "VRAM")
HISTORY = 120  # samples kept per chart

COLUMNS = (  # id, heading, width, anchor, stretch
    ("pid", "PID", 70, "e", False), ("name", "Process", 170, "w", False), ("kind", "Kind", 90, "w", False),
    ("runtime", "Runtime", 150, "w", False), ("vram", "VRAM", 85, "e", False), ("spill", "Spill", 75, "e", False),
    ("paged", "Paged", 75, "e", False), ("gpu", "GPU%", 55, "e", False), ("3d", "3D", 45, "e", False),
    ("copy", "Copy", 52, "e", False), ("video", "Video", 56, "e", False), ("inside", "What's inside", 320, "w", True),
)


def _blend(color: str, other: str, t: float) -> str:
    a = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x * t + y * (1 - t)):02x}" for x, y in zip(a, b))


# ---------------------------------------------------------------------------------------------- sampling

class Poller(threading.Thread):
    """Owns one Collector on a background thread and posts snapshots to the UI queue."""

    def __init__(self, gpu_index: int, host: bool, interval: float, out: queue.Queue, generation: int) -> None:
        super().__init__(daemon=True, name=f"gputaskman-poll-{generation}")
        self.gpu_index, self.host, self.interval = gpu_index, host, interval
        self.out, self.generation = out, generation
        self.stop_event = threading.Event()
        self.paused = False

    def run(self) -> None:
        try:
            collector = Collector(self.gpu_index, host=self.host)
        except Exception as error:  # degrade with a reason, never crash the window
            self.out.put((self.generation, "error", f"Cannot start the collector: {error}"))
            return
        try:
            self.stop_event.wait(0.5)  # engine utilisation is a rate: give PDH two samples
            while not self.stop_event.is_set():
                if not self.paused:
                    started = time.perf_counter()
                    try:
                        snap = collector.sample()
                    except Exception as error:
                        self.out.put((self.generation, "error", f"Sampling failed: {error}"))
                    else:
                        self.out.put((self.generation, "snap", (snap, time.perf_counter() - started)))
                self.stop_event.wait(self.interval)
        finally:
            collector.close()

    def stop(self) -> None:
        self.stop_event.set()


# ---------------------------------------------------------------------------------------------- widgets

class Chart(tk.Canvas):
    """A small trend chart. None values are gaps (no reading); zero is drawn on the baseline."""

    def __init__(self, master, title: str, color: str, unit: str) -> None:
        super().__init__(master, height=74, bg=PANEL, highlightthickness=0, bd=0)
        self.title, self.color, self.unit = title, color, unit
        self.values: list[float | None] = []
        self.top = 100.0
        self.bind("<Configure>", lambda _e: self.redraw())

    def set(self, values, top: float, title: str | None = None) -> None:
        self.values, self.top = list(values), max(top, 1e-9)
        if title:
            self.title = title
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10:
            return
        pad_t, pad_b, pad_x = 22, 6, 8
        x0, x1, y0, y1 = pad_x, w - pad_x, pad_t, h - pad_b
        for frac in (0.0, 0.5, 1.0):
            y = y1 - (y1 - y0) * frac
            self.create_line(x0, y, x1, y, fill=GRID, dash=(2, 3) if frac else ())
        current = self.values[-1] if self.values else None
        self.create_text(x0, 4, anchor="nw", text=self.title, fill=self.color, font=("TkDefaultFont", 9, "bold"))
        self.create_text(x1, 4, anchor="ne", text=format_metric(current, ".0f", self.unit), fill=FG,
                         font=("TkDefaultFont", 9, "bold"))
        n = HISTORY
        step = (x1 - x0) / (n - 1)
        offset = n - len(self.values)
        fill = _blend(self.color, PANEL, 0.22)
        run: list[tuple[float, float]] = []

        def flush() -> None:
            if not run:
                return
            if len(run) > 1:
                poly = [run[0][0], y1, *[c for p in run for c in p], run[-1][0], y1]
                self.create_polygon(*poly, fill=fill, outline="")
                self.create_line(*[c for p in run for c in p], fill=self.color, width=1.6)
            else:
                self.create_oval(run[0][0] - 1.5, run[0][1] - 1.5, run[0][0] + 1.5, run[0][1] + 1.5,
                                 fill=self.color, outline="")
            run.clear()

        for i, v in enumerate(self.values):
            if v is None:
                flush()
                continue
            frac = min(max(v / self.top, 0.0), 1.0)
            run.append((x0 + (offset + i) * step, y1 - (y1 - y0) * frac))
        flush()


class VramBar(tk.Canvas):
    """Adapter VRAM as a stacked bar by process kind, plus a wrapped legend."""

    def __init__(self, master) -> None:
        super().__init__(master, height=60, bg=PANEL, highlightthickness=0, bd=0)
        self.parts: list[tuple[float, str]] = []
        self.total = 0.0
        self.caption = ""
        self.legend: list[tuple[str, str]] = []
        self.bind("<Configure>", lambda _e: self.redraw())

    def set(self, parts, total, caption, legend) -> None:
        self.parts, self.total, self.caption, self.legend = parts, total, caption, legend
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w = self.winfo_width()
        if w < 20:
            return
        cap = self.create_text(w, 13, anchor="e", text=self.caption, fill=FG, font=("TkDefaultFont", 10, "bold"))
        x0, x1, y0, y1 = 0, max(self.bbox(cap)[0] - 12, 50), 2, 24
        self.create_rectangle(x0, y0, x1, y1, fill=FREE, outline="")
        x = float(x0)
        for value, color in self.parts:
            seg = (value / self.total) * (x1 - x0) if self.total else 0
            seg = min(seg, x1 - x)
            if seg >= 0.5:
                self.create_rectangle(x, y0, x + seg, y1, fill=color, outline="")
                x += seg
        lx, ly = 0, y1 + 8
        for text, color in self.legend:
            item = self.create_text(lx + 14, ly, anchor="nw", text=text, fill=MUTED if color is None else FG)
            bx0, _, bx1, by1 = self.bbox(item)
            if bx1 > w and lx > 0:  # wrap
                lx, ly = 0, by1 + 4
                self.coords(item, lx + 14, ly)
                bx0, _, bx1, by1 = self.bbox(item)
            if color:
                self.create_rectangle(lx, ly + 3, lx + 10, ly + 13, fill=color, outline="")
            lx = bx1 + 16
        needed = ly + 22
        if abs(int(self["height"]) - needed) > 2:
            self.configure(height=needed)


# ---------------------------------------------------------------------------------------------- the app

class GpuTaskManagerGui(tk.Tk):
    def __init__(self, gpu_index: int = 0, interval: float = 1.0, host: bool = True) -> None:
        super().__init__()
        self.title("gputaskman - GPU Task Manager")
        self.geometry("1320x860")
        self.minsize(900, 600)
        self.configure(bg=BG)
        self.host = host
        self.gpu_index = gpu_index
        self.interval = tk.DoubleVar(value=interval)
        self.show_all = tk.BooleanVar(value=False)
        self.paused = tk.BooleanVar(value=False)
        self.sort_col, self.sort_desc = "vram", True
        self.snap: Snapshot | None = None
        self.selected_key: str | None = None
        self._auto_selected = False
        self._rows: dict[str, ProcRow] = {}
        self.hist = {name: deque(maxlen=HISTORY) for name, _, _ in CHARTS}
        self.q: queue.Queue = queue.Queue()
        self.generation = 0
        self.poller: Poller | None = None

        self._fonts()
        self._style()
        self._build()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.start_poller()
        self.after(100, self.drain)

    # ---------------------------------------------------------------- look

    def _fonts(self) -> None:
        win = sys.platform == "win32"
        family = "Segoe UI" if win else ("SF Pro Text" if sys.platform == "darwin" else "DejaVu Sans")
        mono = "Consolas" if win else ("Menlo" if sys.platform == "darwin" else "DejaVu Sans Mono")
        for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont"):
            try:
                tkfont.nametofont(name).configure(family=family, size=9)
            except tk.TclError:
                pass
        self.mono = (mono, 9)
        self.big = (family, 12, "bold")

    def _style(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=PANEL_2, bordercolor=GRID,
                    lightcolor=PANEL, darkcolor=PANEL, troughcolor=PANEL, focuscolor=ACCENT)
        s.configure("TFrame", background=BG)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("Panel.TLabel", background=PANEL, foreground=FG)
        s.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        s.configure("Status.TLabel", background=PANEL_2, foreground=MUTED, padding=(8, 3))
        s.configure("Title.TLabel", background=PANEL, foreground=FG, font=self.big)
        s.configure("TButton", background=PANEL_2, foreground=FG, padding=(10, 3), borderwidth=1)
        s.map("TButton", background=[("active", "#3a3f46")])
        s.configure("TCheckbutton", background=BG, foreground=FG)
        s.map("TCheckbutton", background=[("active", BG)], indicatorcolor=[("selected", ACCENT), ("!selected", PANEL_2)])
        s.configure("TCombobox", foreground=FG, background=PANEL_2, arrowcolor=FG)
        s.map("TCombobox", fieldbackground=[("readonly", PANEL_2)], foreground=[("readonly", FG)])
        self.option_add("*TCombobox*Listbox.background", PANEL_2)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", "#3b4252")
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG, rowheight=22,
                    borderwidth=0)
        s.map("Treeview", background=[("selected", "#2f4a6b")], foreground=[("selected", "#ffffff")])
        s.configure("Treeview.Heading", background=PANEL_2, foreground=MUTED, relief="flat", padding=(4, 4))
        s.map("Treeview.Heading", background=[("active", "#353a41")])
        s.configure("TPanedwindow", background=BG)
        s.configure("Vertical.TScrollbar", background=PANEL_2, troughcolor=PANEL, arrowcolor=MUTED)

    # ---------------------------------------------------------------- layout

    def _build(self) -> None:
        # toolbar
        bar = ttk.Frame(self, padding=(10, 8, 10, 4))
        bar.pack(fill="x")
        ttk.Label(bar, text="GPU").pack(side="left")
        self.gpu_box = ttk.Combobox(bar, state="readonly", width=46)
        self.gpu_box.pack(side="left", padx=(6, 14))
        self.gpu_box.bind("<<ComboboxSelected>>", self.on_gpu_change)
        self._load_adapters()
        ttk.Label(bar, text="Refresh").pack(side="left")
        self.int_box = ttk.Combobox(bar, state="readonly", width=6, values=("0.5 s", "1 s", "2 s", "5 s"))
        self.int_box.set(f"{self.interval.get():g} s")
        self.int_box.pack(side="left", padx=(6, 14))
        self.int_box.bind("<<ComboboxSelected>>", self.on_interval_change)
        ttk.Checkbutton(bar, text="All processes", variable=self.show_all, command=self.refresh_table).pack(side="left")
        ttk.Checkbutton(bar, text="Pause", variable=self.paused, command=self.on_pause).pack(side="left", padx=10)
        ttk.Button(bar, text="Help", command=self.show_help).pack(side="right")
        ttk.Button(bar, text="Copy report", command=self.copy_report).pack(side="right", padx=6)
        ttk.Button(bar, text="Export JSON…", command=self.export_json).pack(side="right")
        if sys.platform == "win32" and not _is_admin():
            ttk.Button(bar, text="Run as admin", command=self.relaunch_admin).pack(side="right", padx=6)

        # device header
        head = ttk.Frame(self, style="Panel.TFrame", padding=(12, 10))
        head.pack(fill="x", padx=10, pady=(4, 6))
        self.title_lbl = ttk.Label(head, text="Sampling GPU counters…", style="Title.TLabel")
        self.title_lbl.pack(anchor="w")
        self.vram_bar = VramBar(head)
        self.vram_bar.pack(fill="x", pady=(8, 2))
        self.metrics_lbl = ttk.Label(head, text="", style="Panel.TLabel")
        self.metrics_lbl.pack(anchor="w")
        self.detail_lbl = ttk.Label(head, text="", style="Muted.TLabel")
        self.detail_lbl.pack(anchor="w")
        self.note_lbl = ttk.Label(head, text="", style="Muted.TLabel", wraplength=1200, justify="left")
        self.note_lbl.pack(anchor="w")
        head.bind("<Configure>", lambda e: self.note_lbl.configure(wraplength=max(e.width - 30, 300)))

        # charts
        charts = ttk.Frame(self, padding=(10, 0))
        charts.pack(fill="x")
        self.charts: dict[str, Chart] = {}
        names = [c for c in CHARTS if BACKEND != "metal" or c[0] in METAL_CHARTS]
        cols = 4
        for i, (name, color, unit) in enumerate(names):
            ch = Chart(charts, name, color, unit)
            ch.grid(row=i // cols, column=i % cols, sticky="nsew", padx=3, pady=3)
            self.charts[name] = ch
        for c in range(cols):
            charts.columnconfigure(c, weight=1, uniform="charts")

        self.status = ttk.Label(self, text="", style="Status.TLabel", anchor="w")
        self.status.pack(fill="x", side="bottom")

        # processes + inspector
        panes = ttk.PanedWindow(self, orient="vertical")
        panes.pack(fill="both", expand=True, padx=10, pady=(6, 0))
        top = ttk.Frame(panes, style="Panel.TFrame")
        self.tree = ttk.Treeview(top, columns=[c[0] for c in COLUMNS], show="headings", selectmode="browse")
        for cid, heading, width, anchor, stretch in COLUMNS:
            self.tree.heading(cid, text=heading, anchor=anchor, command=lambda c=cid: self.on_sort(c))
            self.tree.column(cid, width=width, minwidth=40, anchor=anchor, stretch=stretch)
        ysb = ttk.Scrollbar(top, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ysb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        for kind, color in ROW_TINT.items():
            self.tree.tag_configure(f"k:{kind}", foreground=color)
        self.tree.tag_configure("muted", foreground=MUTED)
        self.tree.tag_configure("odd", background="#262a30")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        panes.add(top, weight=3)

        bottom = ttk.Frame(panes, style="Panel.TFrame", padding=(10, 8))
        self.insp_title = ttk.Label(bottom, text="", style="Panel.TLabel", font=(self.big[0], 10, "bold"))
        self.insp_title.pack(anchor="w")
        self.insp = tk.Text(bottom, height=10, bg=PANEL, fg=FG, bd=0, highlightthickness=0, wrap="word",
                            padx=2, pady=4, cursor="arrow", insertwidth=0)
        isb = ttk.Scrollbar(bottom, orient="vertical", command=self.insp.yview)
        self.insp.configure(yscrollcommand=isb.set)
        isb.pack(side="right", fill="y")
        self.insp.pack(fill="both", expand=True)
        for tag, opts in {
            "b": {"font": (self.big[0], 9, "bold")}, "muted": {"foreground": MUTED}, "dim": {"foreground": DIM},
            "red": {"foreground": RED, "font": (self.big[0], 9, "bold")}, "yellow": {"foreground": YELLOW},
            "green": {"foreground": GREEN}, "cyan": {"foreground": CYAN}, "magenta": {"foreground": MAGENTA},
            "mono": {"font": self.mono}, "monohead": {"font": (self.mono[0], 9, "bold"), "foreground": MUTED},
        }.items():
            self.insp.tag_configure(tag, **opts)
        self.insp.bind("<Key>", lambda e: None if (e.state & 0x4 and e.keysym.lower() in ("c", "a")) else "break")
        panes.add(bottom, weight=2)

    def _bind_keys(self) -> None:
        self.bind("<F1>", lambda _e: self.show_help())
        self.bind("<Control-s>", lambda _e: self.export_json())
        self.bind("<Control-p>", lambda _e: (self.paused.set(not self.paused.get()), self.on_pause()))
        self.bind("<Control-a>", lambda _e: (self.show_all.set(not self.show_all.get()), self.refresh_table()))

    def _load_adapters(self) -> None:
        try:
            self.adapters = ordered_adapters(enumerate_adapters())
        except Exception:
            self.adapters = []
        labels = [f"{i}: {a.name} ({a.vendor}, {gb(a.dedicated_vram)})" for i, a in enumerate(self.adapters)]
        self.gpu_box.configure(values=labels or ["0: default"])
        self.gpu_box.current(min(self.gpu_index, max(len(labels) - 1, 0)))

    # ---------------------------------------------------------------- polling

    def start_poller(self) -> None:
        if self.poller is not None:
            self.poller.stop()
        self.generation += 1
        self.poller = Poller(self.gpu_index, self.host, self.interval.get(), self.q, self.generation)
        self.poller.paused = self.paused.get()
        self.poller.start()

    def drain(self) -> None:
        latest = None
        try:
            while True:
                gen, kind, payload = self.q.get_nowait()
                if gen != self.generation:
                    continue  # from a GPU we switched away from
                if kind == "error":
                    self.status.configure(text="⚠ " + payload, foreground=RED)
                else:
                    latest = payload
        except queue.Empty:
            pass
        if latest is not None:
            snap, took = latest
            try:
                self.apply(snap, took)
            except Exception as error:  # a rendering bug must not kill the window
                self.status.configure(text=f"⚠ UI error: {error}", foreground=RED)
        self.after(100, self.drain)

    def on_gpu_change(self, _e=None) -> None:
        self.gpu_index = self.gpu_box.current()
        for h in self.hist.values():
            h.clear()
        self.snap, self.selected_key, self._auto_selected = None, None, False
        self.tree.delete(*self.tree.get_children())
        self.title_lbl.configure(text="Sampling GPU counters…")
        self.start_poller()

    def on_interval_change(self, _e=None) -> None:
        self.interval.set(float(self.int_box.get().split()[0]))
        if self.poller:
            self.poller.interval = self.interval.get()

    def on_pause(self) -> None:
        if self.poller:
            self.poller.paused = self.paused.get()
        if self.snap:
            self.render_header(self.snap)

    def on_close(self) -> None:
        if self.poller:
            self.poller.stop()
            self.poller.join(timeout=2.0)
        self.destroy()

    # ---------------------------------------------------------------- rendering

    def apply(self, snap: Snapshot, took: float) -> None:
        self.snap = snap
        nv = snap.nv
        amd = bool(snap.adapter and snap.adapter.vendor_id == 0x1002)
        self.hist["GPU"].append(nv.util_gpu if nv else None)
        self.hist["SM clock"].append((nv.gfx_clock if amd else nv.sm_clock) if nv else None)
        self.hist["Mem activity"].append(nv.util_mem if nv else None)
        self.hist["Temperature"].append(nv.temp if nv else None)
        self.hist["Power"].append(nv.power_w if nv else None)
        self.hist["Encoder"].append(nv.enc if nv else None)
        self.hist["Decoder"].append(nv.dec if nv else None)
        self.hist["VRAM"].append(100 * snap.vram_used / snap.vram_total
                                 if snap.vram_total and snap.vram_used is not None else None)
        if not self._auto_selected:
            llm = next((r for r in sorted(snap.procs, key=lambda r: -(r.gpu.shown_vram or 0)) if r.llm), None)
            if llm:
                self.selected_key = llm.key
            self._auto_selected = True
        self.render_header(snap)
        self.render_charts(snap, amd)
        self.refresh_table()
        self.render_inspector()
        shown = len(self.tree.get_children())
        host = f"  ·  Windows host: {snap.host_status}" if snap.host_status else ""
        self.status.configure(
            foreground=MUTED,
            text=f"Updated {time.strftime('%H:%M:%S', time.localtime(snap.taken))}  ·  sample {took * 1000:.0f} ms  ·  "
                 f"backend {snap.backend}{host}  ·  {shown} of {len(snap.procs)} processes shown  ·  "
                 f"click a column to sort  ·  F1 help")

    def render_header(self, s: Snapshot) -> None:
        nv = s.nv
        name = s.adapter.name if s.adapter else (nv.name if nv else "GPU")
        if s.backend == "metal":
            title = f"{name}    {s.driver}  ·  Metal  ·  unified memory"
        else:
            title = name + (f"    driver {s.driver}" if s.driver else "") + (f"  ·  CUDA {s.cuda}" if s.cuda else "")
            if nv and nv.driver_model:
                title += f"  ·  {nv.driver_model}"
        if self.paused.get():
            title += "    [PAUSED]"
        self.title_lbl.configure(text=title)

        # Same rule as the TUI: only measured bytes form the bar; scale rows to the adapter total.
        by_kind: dict[str, float] = {}
        for r in s.procs:
            if r.gpu.attributed:
                by_kind[r.kind] = by_kind.get(r.kind, 0) + r.gpu.dedicated
        attributed = sum(by_kind.values())
        scale = min(1.0, s.vram_used / attributed) if attributed and s.vram_used is not None else 1.0
        order = sorted(by_kind.items(), key=lambda kv: -kv[1])
        parts = [(v * scale, KIND_COLOR.get(k, "#cbd5e1")) for k, v in order] if s.vram_used is not None else []
        other = max(s.vram_used - attributed * scale, 0) if s.vram_used is not None else 0
        parts.append((other, UNATTRIBUTED))
        used = gb(s.vram_used) if s.vram_used is not None else "N/A"
        caption = f"{'GPU mem' if s.backend == 'metal' else 'VRAM'}  {used} / {gb(s.vram_total)}"
        legend = [(f"{k} {gb(v)}", KIND_COLOR.get(k, "#cbd5e1")) for k, v in order[:8]]
        if other > 32 << 20:
            legend.append((f"unattributed {gb(other)}", UNATTRIBUTED))
        if s.vram_used is not None:
            legend.append((f"free {gb(max(s.vram_total - s.vram_used, 0))}", FREE))
        if s.shared_used is not None and s.shared_used > 64 << 20:
            legend.append((f"shared (system RAM) {gb(s.shared_used)}", YELLOW))
        if scale < 1.0:
            legend.append(("bar normalized; labels are raw", None))
        self.vram_bar.set(parts, s.vram_total or 1, caption, legend)

        notes = []
        if s.backend == "wsl":
            notes.append(wsl_note(s))
        if s.backend == "metal":
            notes.append(METAL_NOTE)
        if s.backend == "wsl" and not any(not r.host for r in s.procs):
            notes.append(WSL_EMPTY)
        self.note_lbl.configure(text="\n".join(notes))

        if nv and s.backend != "metal":
            rx = nv.pcie_rx_kbs / 1024 if nv.pcie_rx_kbs is not None else None
            tx = nv.pcie_tx_kbs / 1024 if nv.pcie_tx_kbs is not None else None
            label = "AMD gfx" if nv.source.startswith("AMD ") else "GPU"
            line = (f"{label} {format_metric(nv.util_gpu, '.0f', '%')}    Mem activity {format_metric(nv.util_mem, '.0f', '%')}"
                    f"    {format_metric(nv.temp, '.0f', ' °C')}    {format_metric(nv.power_w, '.0f')} / "
                    f"{format_metric(nv.power_limit_w, '.0f')} W{(' ' + nv.power_kind) if nv.power_kind else ''}"
                    f"    PCIe ↓ {format_metric(rx, '.0f')} ↑ {format_metric(tx, '.0f')} MB/s")
            if nv.enc or nv.dec:
                line += f"    enc {format_metric(nv.enc, '.0f', '%')} dec {format_metric(nv.dec, '.0f', '%')}"
            clock = (f"gfx {format_metric(nv.gfx_clock)} MHz" if nv.source.startswith("AMD ") else
                     f"SM {format_metric(nv.sm_clock)}/{format_metric(nv.sm_clock_max)} MHz")
            pstate = f"P{nv.pstate}" if nv.pstate is not None else "P N/A"
            detail = (f"{pstate}  ·  {clock}  ·  mem {format_metric(nv.mem_clock)} MHz  ·  "
                      f"video {format_metric(nv.video_clock)} MHz  ·  PCIe gen{format_metric(nv.pcie_gen)} "
                      f"x{format_metric(nv.pcie_width)}  ·  "
                      + (f"CC {nv.compute_cap or 'N/A'}" if nv.source == "NVML" else nv.source))
            if nv.throttle:
                detail += "  ·  throttle: " + ", ".join(nv.throttle)
            self.metrics_lbl.configure(text=line)
            self.detail_lbl.configure(text=detail)
        elif nv:
            self.metrics_lbl.configure(text=f"GPU {format_metric(nv.util_gpu, '.0f', '%')}")
            self.detail_lbl.configure(text="")
        else:
            self.metrics_lbl.configure(text="Device telemetry unavailable (no NVML / AMD ADL reading for this adapter).")
            self.detail_lbl.configure(text="")

    def render_charts(self, s: Snapshot, amd: bool) -> None:
        nv = s.nv
        for name, ch in self.charts.items():
            values = self.hist[name]
            top, title = 100.0, None
            if name == "SM clock":
                top = max((nv.sm_clock_max if nv else None) or 0, max((v for v in values if v is not None), default=0), 1)
                title = "GFX clock" if amd else "SM clock"
            elif name == "Power":
                top = max((nv.power_limit_w if nv else None) or 0, max((v for v in values if v is not None), default=0), 1)
            elif name == "Temperature":
                top = 120.0
            ch.set(values, top, title)

    # ---------------------------------------------------------------- table

    def _sort_key(self, col: str):
        def pct(r: ProcRow, val: float) -> float:
            return val if has_engine_counters(self.snap, r) else -1.0
        return {
            "pid": lambda r: (r.host, r.pid),
            "name": lambda r: r.name.lower(),
            "kind": lambda r: r.kind,
            "runtime": lambda r: r.runtime.lower(),
            "vram": lambda r: r.gpu.shown_vram if r.gpu.shown_vram is not None else -1.0,
            "spill": lambda r: r.gpu.shared,
            "paged": lambda r: r.gpu.paged_out,
            "gpu": lambda r: pct(r, r.gpu.gpu_pct),
            "3d": lambda r: pct(r, r.gpu.engines.get("3d", 0) + r.gpu.engines.get("compute", 0)),
            "copy": lambda r: pct(r, r.gpu.engines.get("copy", 0)),
            "video": lambda r: pct(r, r.gpu.engines.get("decode", 0) + r.gpu.engines.get("encode", 0)),
            "inside": lambda r: r.inside.lower(),
        }[col]

    def visible_rows(self, s: Snapshot) -> list[ProcRow]:
        rows = s.procs
        if not self.show_all.get():
            rows = [r for r in rows if not r.gpu.attributed or r.gpu.dedicated >= 32 << 20 or r.gpu.gpu_pct >= 1
                    or r.gpu.shared >= 64 << 20 or r.gpu.paged_out >= 128 << 20 or r.llm]
        rows = sorted(rows, key=self._sort_key(self.sort_col), reverse=self.sort_desc)
        vm = next((i for i, r in enumerate(rows) if r.kind == "wsl-vm"), None)
        if vm is not None and self.sort_col == "vram" and self.sort_desc:
            # Linux rows share the WSL VM row's total: keep them right under it.
            linux = [r for r in rows if not r.gpu.attributed and not r.host]
            rest = [r for r in rows if r not in linux]
            at = rest.index(rows[vm])
            rows = rest[:at + 1] + linux + rest[at + 1:]
        return rows

    def _row_values(self, s: Snapshot, r: ProcRow) -> tuple:
        e = r.gpu.engines
        wddm = has_wddm_counters(s, r)
        engines = has_engine_counters(s, r)
        split = engines and s.backend != "metal"

        def pct(v: float | None) -> str:
            return "n/a" if v is None else (f"{v:.0f}" if v >= 0.5 else "")

        spill = gb(r.gpu.shared) if wddm and r.gpu.shared >= 1 << 20 else ""
        paged = gb(r.gpu.paged_out) if wddm and r.gpu.paged_out >= 1 << 20 else ""
        if wddm and "shared" not in r.gpu.available:
            spill = "n/a"
        if wddm and not r.gpu.paged_available:
            paged = "n/a"
        return (r.pid_label, r.name, r.kind, " ".join(r.detail.runtimes[:3]), vram_label(r.gpu), spill, paged,
                pct(r.gpu.gpu_pct if engines else None),
                pct(e.get("3d", 0) + e.get("compute", 0) if split else None),
                pct(e.get("copy", 0) if split else None),
                pct(e.get("decode", 0) + e.get("encode", 0) if split else None),
                r.inside)

    def refresh_table(self) -> None:
        s = self.snap
        if s is None:
            return
        rows = self.visible_rows(s)
        self._rows = {r.key: r for r in s.procs}
        wanted = [r.key for r in rows]
        existing = set(self.tree.get_children())
        for iid in existing - set(wanted):
            self.tree.delete(iid)
        for index, r in enumerate(rows):
            tags = []
            if r.kind in ROW_TINT:
                tags.append(f"k:{r.kind}")
            elif r.host or r.kind in ("compositor", "kernel", "system", "external"):
                tags.append("muted")
            if index % 2:
                tags.append("odd")
            values = self._row_values(s, r)
            if r.key in existing:
                self.tree.item(r.key, values=values, tags=tags)
            else:
                self.tree.insert("", "end", iid=r.key, values=values, tags=tags)
            self.tree.move(r.key, "", index)
        for cid, heading, *_ in COLUMNS:
            arrow = (" ▼" if self.sort_desc else " ▲") if cid == self.sort_col else ""
            self.tree.heading(cid, text=heading + arrow)
        if self.selected_key in wanted and self.tree.selection() != (self.selected_key,):
            self.tree.selection_set(self.selected_key)

    def on_sort(self, col: str) -> None:
        if col == self.sort_col:
            self.sort_desc = not self.sort_desc
        else:
            self.sort_col = col
            self.sort_desc = col not in ("pid", "name", "kind", "runtime", "inside")
        self.refresh_table()

    def on_select(self, _e=None) -> None:
        sel = self.tree.selection()
        if sel and sel[0] != self.selected_key:
            self.selected_key = sel[0]
            self.render_inspector(keep_scroll=False)

    # ---------------------------------------------------------------- inspector

    def render_inspector(self, keep_scroll: bool = True) -> None:
        s = self.snap
        t = self.insp
        r = self._rows.get(self.selected_key) if s and self.selected_key else None
        pos = t.yview()[0] if keep_scroll else 0.0
        t.configure(state="normal")
        t.delete("1.0", "end")
        if r is None:
            self.insp_title.configure(text="Inspector")
            t.insert("end", "Select a process to see what's inside its GPU memory. F1 explains what can be known.",
                     "muted")
            return
        where = {"windows": "Windows PID ", "outside": "host PID "}.get(r.origin, "") if r.host else ""
        self.insp_title.configure(text=f"{r.name}  ({where}{r.pid})  ·  {r.kind}",
                                  foreground=KIND_COLOR.get(r.kind, FG))
        e = r.gpu.engines
        wddm = has_wddm_counters(s, r)
        engines = " ".join(f"{k} {v:.0f}%" for k, v in e.items() if v >= 0.5) or "idle"
        put = t.insert
        put("end", "VRAM ", "b")
        put("end", vram_label(r.gpu), "b")
        put("end", "    spill (RAM) ", "muted")
        put("end", gb(r.gpu.shared) if wddm and "shared" in r.gpu.available else "n/a",
            "red" if r.gpu.shared > 256 << 20 else ())
        put("end", "    paged (est) ", "muted")
        put("end", gb(r.gpu.paged_out) if wddm and r.gpu.paged_available else "n/a")
        put("end", "    engines ", "muted")
        put("end", engines if has_engine_counters(s, r) else "n/a")
        if r.detail.runtimes:
            put("end", "    runtime ", "muted")
            put("end", " ".join(r.detail.runtimes), "cyan")
        put("end", "\n")
        if not r.host and r.gpu.attributed and r.gpu.inferred_by:
            put("end", inferred_note(r.gpu) + "\n", "muted")
        if not r.host and not r.gpu.attributed:
            if r.gpu.inferred is not None:
                put("end", inferred_note(r.gpu) + "\n", "muted")
            elif s.wsl_split:
                put("end", f"No per-process VRAM: {s.wsl_split}.\n", "muted")
            elif s.backend == "metal":
                put("end", "GPU memory n/a: footprint can't read other users' processes without root "
                           "(or hasn't measured this one yet). GPU% comes from Metal's per-process GPU time.\n", "muted")
        if wddm and r.gpu.paged_out >= 64 << 20:
            put("end", f"{gb(r.gpu.paged_out)} of {gb(r.gpu.committed)} committed is not accounted for by "
                       "local/non-local usage. Estimated non-residency, not proof of eviction or disk paging.\n",
                "yellow")
        if r.handy:
            if r.handy.selected_model:
                put("end", f"Handy selected (settings; not load proof): {r.handy.selected_model}\n", "muted")
            if r.handy.last_event:
                put("end", f"Handy last event: {r.handy.last_event} {r.handy.last_model or ''} at "
                           f"{r.handy.last_event_local} (runtime-reported history, not current GPU residency)\n",
                    "green")

        v = r.llm
        if v:
            put("end", "\n")
            total = max(r.gpu.shown_vram or 0, sum(x.bytes for x in v.segments), 1)
            bar = tk.Canvas(t, height=18, width=max(t.winfo_width() - 30, 300), bg=PANEL, highlightthickness=0)
            bw = int(bar["width"])
            bar.create_rectangle(0, 2, bw, 16, fill=FREE, outline="")
            x = 0.0
            for seg in v.segments:
                w = seg.bytes / total * bw
                bar.create_rectangle(x, 2, x + w, 16, fill=SEG_COLOR.get(seg.kind, "#888888"), outline="")
                x += w
            t.window_create("end", window=bar)
            put("end", "\n")
            for seg in v.segments:
                tag = f"seg:{seg.kind}"
                t.tag_configure(tag, foreground=SEG_COLOR.get(seg.kind, FG))
                put("end", "■ ", tag)
                put("end", f"{seg.label} {gb(seg.bytes)}", "b" if seg.exact else ())
                put("end", "    ")
            if v.log_source:
                put("end", f"(allocated sizes from {os.path.basename(v.log_source)})", "green")
            put("end", "\n")
            if v.models:
                heads = ("kind", "model", "arch", "params", "quant", "layers", "train ctx", "on GPU (est)", "evidence", "file")
                rows = []
                for i, g in enumerate(v.models):
                    f = v.files[i] if i < len(v.files) else None
                    rows.append((v.kind_label(i), v.label(i), g.arch, g.params_label, g.quant, str(g.n_layers or ""),
                                 f"{g.ctx_train:,}" if g.ctx_train else "", gb(g.gpu_weight_bytes()),
                                 f.evidence if f else "", os.path.basename(g.path)))
                widths = [max(len(str(x)) for x in col) for col in zip(heads, *rows)]
                fmt = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()
                put("end", "\n" + fmt(heads) + "\n", "monohead")
                for row in rows:
                    put("end", fmt(row) + "\n", "mono")
            for f in v.other_files:
                kind = f", {f.kind_label}" if f.kind_label else ""
                put("end", f"file: {f.store_name or f.path}  ({gb(f.size)}, {f.evidence}{kind})\n", "muted")
            if v.server:
                sv = v.server
                put("end", "\nserver ", "b")
                put("end", f"{sv.kind} {sv.url}", "cyan")
                if sv.n_ctx:
                    put("end", f"  ctx {sv.n_ctx:,}")
                if sv.slots_total is not None:
                    put("end", f"  slots busy {sv.slots_busy}/{sv.slots_total}")
                if sv.gen_tok_s:
                    put("end", f"  gen {sv.gen_tok_s:.1f} tok/s", ("green", "b"))
                if sv.prompt_tok_s:
                    put("end", f"  prompt {sv.prompt_tok_s:.0f} tok/s")
                for m in sv.models[:4]:
                    if sv.kind == "lmstudio":
                        put("end", f"\n  loaded: {m['name']} {m.get('type') or ''} {m.get('quant') or ''} "
                                   f"ctx {m.get('ctx') or '?'}")
                put("end", "\n")
            for n in v.notes:
                put("end", "• " + n + "\n", "red" if n.startswith("!") else "muted")
        else:
            if EXPLAIN.get(r.kind):
                put("end", EXPLAIN[r.kind] + "\n", "muted")
            for m in r.detail.models:
                kind = f", {m.kind_label}" if m.kind_label else ""
                put("end", f"file: {m.store_name or m.path}  ({gb(m.size)}, {m.evidence}{kind})\n", "magenta")
            if r.detail.candidates:
                put("end", "No mapped model file was observed. Some runtimes read then close model files. "
                           "Installed candidates (not proven loaded):\n", "muted")
                for m in r.detail.candidates:
                    kind = f", {m.kind_label}" if m.kind_label else ""
                    put("end", f"  {m.store_name or m.path}  ({gb(m.size)}{kind})\n", "dim")
        if r.detail.ports:
            put("end", "listening: " + ", ".join(map(str, r.detail.ports[:8])) + "\n", "dim")
        if r.detail.cmdline:
            put("end", " ".join(r.detail.cmdline)[:600] + "\n", ("dim", "mono"))
        elif r.detail.access_denied:
            put("end", "(protected process: use 'Run as admin' for runtime and file details)\n", "dim")
        t.yview_moveto(pos)

    # ---------------------------------------------------------------- actions

    def export_json(self) -> None:
        if self.snap is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Save snapshot", defaultextension=".json",
            initialfile=time.strftime("gputaskman-%Y%m%d-%H%M%S.json"), filetypes=[("JSON", "*.json")])
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(to_dict(self.snap), fh, indent=2, default=str)
            self.status.configure(text=f"Snapshot saved to {path} (compare two with: gputaskman diff A.json B.json)",
                                  foreground=GREEN)

    def copy_report(self) -> None:
        if self.snap is None:
            return
        self.clipboard_clear()
        self.clipboard_append(to_text(self.snap))
        self.status.configure(text="Text report copied to the clipboard.", foreground=GREEN)

    def relaunch_admin(self) -> None:
        import ctypes

        if getattr(sys, "frozen", False):
            exe, params = sys.executable, ""
        else:
            exe = sys.executable.replace("python.exe", "pythonw.exe")
            params = "-m gputaskman.gui"
        params += f" --gpu {self.gpu_index} --interval {self.interval.get():g}" + ("" if self.host else " --no-host")
        rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", exe, params, None, 1)
        if rc > 32:
            self.on_close()
        else:
            messagebox.showwarning("gputaskman", "Elevation was cancelled or failed.", parent=self)

    def show_help(self) -> None:
        win = tk.Toplevel(self)
        win.title("What can be known about GPU memory?")
        win.geometry("900x700")
        win.configure(bg=BG)
        txt = tk.Text(win, bg=PANEL, fg=FG, bd=0, wrap="word", padx=16, pady=12, highlightthickness=0,
                      font=("TkDefaultFont", 10), spacing1=1, spacing3=1)
        sb = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.tag_configure("h", font=(self.big[0], 13, "bold"), spacing3=8)
        txt.tag_configure("b", font=(self.big[0], 9, "bold"))
        txt.tag_configure("code", font=self.mono, foreground=CYAN)
        txt.tag_configure("table", font=self.mono, foreground=MUTED)
        help_md = HELP_MD.replace("Keys: `↑/↓` select · `s` sort · `c` chart · `a` show all · `p` pause · "
                                  "`?` this help · `q` quit",
                                  "GUI: click a column heading to sort · Ctrl+A all processes · Ctrl+P pause · "
                                  "Ctrl+S export JSON · F1 this help")
        blocks: list[str] = []
        for line in help_md.splitlines():  # re-flow hard-wrapped markdown paragraphs
            if line.startswith(("#", "|")) or not line.strip() or not blocks or not blocks[-1] \
                    or blocks[-1].startswith(("#", "|")):
                blocks.append(line)
            else:
                blocks[-1] += " " + line.strip()
        header_seen = False
        for line in blocks:
            if line.startswith("# "):
                txt.insert("end", line[2:] + "\n", "h")
                continue
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if set(line) <= set("|- ") or not header_seen:
                    header_seen = True
                    continue
                line = f"• {cells[0]}: {cells[1]}" + (f" (works for {cells[2]})" if len(cells) > 2 else "")
            for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", line):
                if part.startswith("**"):
                    txt.insert("end", part[2:-2], "b")
                elif part.startswith("`"):
                    txt.insert("end", part[1:-1], "code")
                else:
                    txt.insert("end", part.replace("*", ""))
            txt.insert("end", "\n")
        txt.configure(state="disabled")
        win.bind("<Escape>", lambda _e: win.destroy())


# ---------------------------------------------------------------------------------------------- entry

def _is_admin() -> bool:
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _dpi_aware() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on scaled displays
    except Exception:
        pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gputaskman-gui", description="Graphical GPU task manager (gputaskman).")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (default 0)")
    ap.add_argument("-i", "--interval", type=float, default=1.0, help="refresh seconds (default 1)")
    ap.add_argument("--no-host", action="store_true", help="WSL: don't read Windows host GPU counters")
    args = ap.parse_args(argv)
    if args.gpu < 0 or not math.isfinite(args.interval) or args.interval <= 0:
        ap.error("--gpu must be >= 0 and --interval > 0")
    _dpi_aware()
    GpuTaskManagerGui(gpu_index=args.gpu, interval=args.interval, host=not args.no_host).mainloop()


if __name__ == "__main__":
    main()
