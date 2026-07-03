#!/usr/bin/env bash
# =============================================================================
# rocm_env.sh — AMD ROCm Environment for RX 7900 XTX (RDNA3 / gfx1100)
# Usage: source rocm_env.sh
# =============================================================================

# Force ROCm to correctly target RDNA3 architecture (gfx1100)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Target the RX 7900 XTX (GPU 0) — skip integrated Raphael APU (GPU 1)
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

# Allow PyTorch to dynamically expand GPU memory allocations (prevents OOM)
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Maximize hardware compute queues for RDNA3 throughput
export GPU_MAX_HW_QUEUES=8

# Suppress non-critical HIP warnings
export AMD_SERIALIZE_KERNEL=0

echo "✅ AMD ROCm environment loaded"
echo "   GPU         : RX 7900 XTX (gfx1100 / RDNA3)"
echo "   GFX Version : $HSA_OVERRIDE_GFX_VERSION"
echo "   Visible GPUs: $HIP_VISIBLE_DEVICES (7900 XTX only)"
echo "   VRAM Budget : 24 GB"
