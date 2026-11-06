import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

root = Path(SPECPATH).parent
analysis = Analysis(
    [str(root / "tools" / "portable_gui_entry.py")],
    pathex=[str(root / "src")],
    datas=collect_data_files("textual") + copy_metadata("gputaskman", recursive=True),
)

# These belong to the operating system or the installed NVIDIA driver, not the portable app.
driver_libs = ("nvml.dll", "nvcuda.dll", "atiadlxx.dll", "amd_smi.dll", "libamd_smi.so", "dxgi.dll", "pdh.dll", "libnvidia-ml.so", "libcuda.so", "libdxcore.so")


def is_driver_lib(name):
    name = name.lower()
    return any(name == lib or name.startswith(lib + ".") for lib in driver_libs)


analysis.binaries = [entry for entry in analysis.binaries if not is_driver_lib(Path(entry[0]).name)]

pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="gputaskman-gui",
    console=False,
)
bundle = COLLECT(exe, analysis.binaries, analysis.datas, name="gputaskman-gui")

if sys.platform == "darwin":  # a double-clickable app bundle on macOS
    app = BUNDLE(bundle, name="gputaskman-gui.app", bundle_identifier="com.github.gputaskman.gui",
                 info_plist={"NSHighResolutionCapable": True, "LSMinimumSystemVersion": "12.0"})
