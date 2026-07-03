"""
=============================================================================
verify_gpu.py — Comprehensive GPU/ROCm/PyTorch Verification
GPU Optimization / Tool / Audio / tests
=============================================================================
Run this FIRST to confirm everything is working before generating audio.
Usage: python3 tests/verify_gpu.py
=============================================================================
"""

import os
import sys
import subprocess
from pathlib import Path

# AMD ROCm flags
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def section(title: str):
    print(f"\n{'━'*55}")
    print(f"  {title}")
    print(f"{'━'*55}")


def check(label: str, condition: bool, detail: str = "", warn_only: bool = False):
    icon = PASS if condition else (WARN if warn_only else FAIL)
    status = "PASS" if condition else ("WARN" if warn_only else "FAIL")
    print(f"  {icon} {label:<30} [{status}]")
    if detail:
        print(f"       {detail}")
    return condition


# ── 1. System checks ────────────────────────────────────────────────────────
section("1. System Environment")

# ROCm env vars
check("HSA_OVERRIDE_GFX_VERSION",
      os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0",
      detail=f"Value: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'NOT SET')}")

check("HIP_VISIBLE_DEVICES",
      os.environ.get("HIP_VISIBLE_DEVICES") == "0",
      detail=f"Value: {os.environ.get('HIP_VISIBLE_DEVICES', 'NOT SET')}")

# ROCm tools
try:
    result = subprocess.run(["rocm-smi", "--version"], capture_output=True, text=True)
    rocm_ok = result.returncode == 0
    rocm_ver = result.stdout.strip().split("\n")[0] if rocm_ok else "N/A"
    check("rocm-smi available", rocm_ok, detail=rocm_ver)
except FileNotFoundError:
    check("rocm-smi available", False, detail="rocm-smi not found in PATH")

# Python version
import platform
py_ver = platform.python_version()
py_ok = int(py_ver.split(".")[1]) >= 10
check(f"Python version", py_ok, detail=f"Python {py_ver} (need 3.10+)")


# ── 2. PyTorch checks ───────────────────────────────────────────────────────
section("2. PyTorch & ROCm")

import torch

torch_ver = torch.__version__
rocm_in_torch = "rocm" in torch_ver
check("PyTorch installed", True, detail=f"Version: {torch_ver}")
check("PyTorch has ROCm build", rocm_in_torch,
      detail=f"Expected 'rocmX.Y' in version string")

cuda_avail = torch.cuda.is_available()
check("torch.cuda.is_available()", cuda_avail,
      detail="NOTE: PyTorch maps AMD GPUs to cuda namespace")

if cuda_avail:
    gpu_count = torch.cuda.device_count()
    check("GPU count ≥ 1", gpu_count >= 1, detail=f"Found {gpu_count} GPU(s)")

    gpu_name = torch.cuda.get_device_name(0)
    is_7900xtx = "7900" in gpu_name
    check("RX 7900 XTX detected", is_7900xtx, detail=f"Name: {gpu_name}")

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    enough_vram = vram_gb >= 20.0
    check("VRAM ≥ 20 GB", enough_vram, detail=f"VRAM: {vram_gb:.1f} GB")


# ── 3. Basic VRAM allocation test ───────────────────────────────────────────
section("3. GPU Memory Allocation Test")

if cuda_avail:
    try:
        # Allocate a 1GB tensor on GPU
        test_tensor = torch.ones(1024, 1024, 256, device="cuda", dtype=torch.float32)
        alloc_gb = test_tensor.element_size() * test_tensor.nelement() / 1024**3
        del test_tensor
        torch.cuda.empty_cache()
        check(f"Allocate {alloc_gb:.1f}GB on GPU", True,
              detail="Tensor created and freed successfully")

        # Check free VRAM
        free_vram = torch.cuda.mem_get_info()[0] / 1024**3
        total_vram = torch.cuda.mem_get_info()[1] / 1024**3
        check("Free VRAM > 10 GB", free_vram > 10,
              detail=f"Free: {free_vram:.1f} GB / Total: {total_vram:.1f} GB")

    except Exception as e:
        check("GPU memory allocation", False, detail=str(e))
else:
    print(f"  {WARN} Skipping GPU tests (no CUDA device)")


# ── 4. Dependency checks ─────────────────────────────────────────────────────
section("4. Python Dependencies")

def check_import(name: str, package: str = None, warn_only: bool = False):
    pkg = package or name
    try:
        __import__(pkg)
        check(f"import {name}", True)
        return True
    except ImportError:
        check(f"import {name}", False, detail=f"pip install {pkg}", warn_only=warn_only)
        return False

check_import("torch")
check_import("torchaudio")
check_import("soundfile")
check_import("omnivoice", warn_only=True)   # May need installation
check_import("pydub", warn_only=True)       # For audio merging

# FFmpeg check (needed by pydub)
try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    check("ffmpeg", result.returncode == 0,
          detail="Required by pydub for audio format conversion")
except FileNotFoundError:
    check("ffmpeg", False, detail="sudo apt install ffmpeg", warn_only=True)


# ── 5. torchaudio GPU test ───────────────────────────────────────────────────
section("5. torchaudio GPU Test")

try:
    import torchaudio
    check("torchaudio imported", True, detail=f"Version: {torchaudio.__version__}")

    if cuda_avail:
        # Create a test waveform on GPU
        waveform = torch.randn(1, 24000, device="cuda")  # 1 sec at 24kHz
        check("torchaudio GPU tensor", waveform.is_cuda,
              detail=f"Shape: {waveform.shape}, Device: {waveform.device}")
        del waveform
        torch.cuda.empty_cache()
except Exception as e:
    check("torchaudio GPU test", False, detail=str(e))


# ── Final summary ─────────────────────────────────────────────────────────────
section("Summary")

if cuda_avail and rocm_in_torch:
    print(f"\n  {PASS} SYSTEM IS READY FOR OMNIVOICE AUDIO GENERATION!")
    print(f"\n  Next steps:")
    print(f"    1. pip install omnivoice soundfile pydub")
    print(f"    2. source rocm_env.sh")
    print(f"    3. python3 scripts/generate_audio.py --text 'Hello world'")
elif not rocm_in_torch:
    print(f"\n  {FAIL} PyTorch ROCm build not detected!")
    print(f"     Fix: pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.1")
else:
    print(f"\n  {FAIL} GPU not detected by PyTorch!")
    print(f"     Fix: source rocm_env.sh  (sets HSA_OVERRIDE_GFX_VERSION=11.0.0)")
    print(f"     Then re-run: python3 tests/verify_gpu.py")

print()
