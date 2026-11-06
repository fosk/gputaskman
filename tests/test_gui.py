"""GUI (tkinter) smoke test: renders a realistic snapshot built from the real data classes.

It guards the GUI against upstream renames in Snapshot / ProcRow / ProcGpu / LlmView / DeviceTelemetry.
Skipped where no display is available (CI runs it under xvfb-run on Linux).
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from collections import defaultdict
from unittest.mock import patch

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - Python built without Tk
    tk = None

from gputaskman.collector import ProcRow, Snapshot
from gputaskman.dxgi import Adapter
from gputaskman.gguf import GgufInfo
from gputaskman.llm import LlmView, Segment, ServerInfo
from gputaskman.pdh import ProcGpu
from gputaskman.procinfo import MappedModel, ProcDetail
from gputaskman.report import to_dict
from gputaskman.telemetry import DeviceTelemetry

G, M = 1 << 30, 1 << 20
ADAPTER = Adapter(luid=1, name="NVIDIA GeForce RTX 4080 SUPER", vendor_id=0x10DE, dedicated_vram=16 * G,
                  shared_system=32 * G, software=False)


def _gpu(dedicated: float, shared: float = 0, committed: float = 0, engines: dict | None = None) -> ProcGpu:
    g = ProcGpu(dedicated=dedicated, shared=shared, committed=committed or dedicated + shared)
    g.engines = defaultdict(float, engines or {})
    return g


def make_snapshot() -> Snapshot:
    gguf = GgufInfo(path="C:/models/qwen3-14b-q4_k_m.gguf", file_size=int(8.4 * G), version=3,
                    metadata={"general.architecture": "qwen3", "general.basename": "Qwen3",
                              "qwen3.block_count": 40, "qwen3.context_length": 40960},
                    n_params=14_800_000_000, tensor_bytes=int(8.4 * G))
    mapped = MappedModel(path=gguf.path, size=int(8.4 * G), resident=int(2 * G), gguf=True,
                         sources=["mapped", "api"], store_name="qwen3:14b", kind="LLM")
    llm = LlmView(models=[gguf], files=[mapped],
                  segments=[Segment("weights", int(8.1 * G), "weights", True), Segment("KV cache", int(1.25 * G), "kv"),
                            Segment("compute", int(0.4 * G), "compute", True)],
                  server=ServerInfo(kind="ollama", url="http://127.0.0.1:11434"),
                  notes=["KV cache estimated from n_ctx 8192."])
    rows = [
        ProcRow(4120, "llama-server.exe", "LLM", _gpu(9.95 * G, 120 * M, engines={"3d": 60, "copy": 4}),
                ProcDetail(4120, "llama-server.exe", runtimes=["CUDA", "ggml"], cmdline=["llama-server.exe", "-m", gguf.path],
                           ports=[11434]), llm=llm),
        ProcRow(880, "dwm.exe", "compositor", _gpu(410 * M, engines={"3d": 3}), ProcDetail(880, "dwm.exe")),
        ProcRow(15100, "python.exe", "AI", _gpu(2.1 * G, 300 * M, 3 * G, {"compute": 12}),
                ProcDetail(15100, "python.exe", runtimes=["CUDA", "PyTorch"])),
        ProcRow(4, "System", "kernel", _gpu(180 * M), ProcDetail(4, "System", access_denied=True)),
    ]
    nv = DeviceTelemetry(index=0, name=ADAPTER.name, driver_model="WDDM", mem_total=16 * G, util_gpu=60, util_mem=30,
                         temp=62, power_w=180, power_limit_w=320, pstate=2, sm_clock=2400, sm_clock_max=2820,
                         mem_clock=11501, video_clock=1650, pcie_rx_kbs=100_000, pcie_tx_kbs=50_000, pcie_gen=4,
                         pcie_width=16, enc=0, dec=7, compute_cap="8.9")
    return Snapshot(taken=time.time(), adapter=ADAPTER, adapters=[ADAPTER], nv=nv, driver="581.57", cuda="13.0",
                    vram_total=16 * G, vram_used=int(13 * G), shared_used=int(1.2 * G), procs=rows, backend="wddm")


def _display_available() -> bool:
    if tk is None:
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    return True


@unittest.skipUnless(_display_available(), "no display for tkinter")
class GuiRenderTest(unittest.TestCase):
    def setUp(self) -> None:
        from gputaskman import gui

        self.gui = gui
        patches = [patch.object(gui.GpuTaskManagerGui, "start_poller", lambda self: None),
                   patch.object(gui.GpuTaskManagerGui, "drain", lambda self: None),
                   patch.object(gui, "enumerate_adapters", lambda: [ADAPTER])]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        try:
            self.app = gui.GpuTaskManagerGui()
        except tk.TclError as error:
            self.skipTest(f"tkinter cannot open a window here: {error}")
        self.addCleanup(self.app.destroy)
        self.app.withdraw()

    def test_renders_snapshot(self) -> None:
        snap = make_snapshot()
        for _ in range(3):  # several samples, so the charts have history
            self.app.apply(snap, 0.01)
        self.app.update_idletasks()
        rows = self.app.tree.get_children()
        self.assertEqual(rows[0], "4120")  # sorted by VRAM, largest first
        self.assertEqual(self.app.selected_key, "4120")  # the LLM is auto-selected
        self.assertIn("qwen3:14b", self.app.insp.get("1.0", "end"))
        self.assertIn("weights", self.app.insp.get("1.0", "end"))
        self.assertIn("RTX 4080 SUPER", self.app.title_lbl.cget("text"))

    def test_sort_filter_and_inspect_every_row(self) -> None:
        snap = make_snapshot()
        self.app.apply(snap, 0.01)
        for column, *_ in self.gui.COLUMNS:
            self.app.on_sort(column)
            self.app.on_sort(column)
        self.app.show_all.set(True)
        self.app.refresh_table()
        for key in self.app.tree.get_children():
            self.app.selected_key = key
            self.app.render_inspector()
        self.assertEqual(len(self.app.tree.get_children()), len(snap.procs))



class GuiDataTest(unittest.TestCase):
    def test_gui_module_imports_and_export_is_serialisable(self) -> None:
        if tk is not None:
            from gputaskman import gui  # noqa: F401  (catches renamed imports even without a display)
        json.dumps(to_dict(make_snapshot()), default=str)


if __name__ == "__main__":
    unittest.main()
