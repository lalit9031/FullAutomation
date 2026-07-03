#!/usr/bin/env bash
# =============================================================================
# setup_rocm_env.sh — OmniVoice + ROCm Setup for AMD RX 7900 XTX
# GPU optimization/Tool/Audio
# =============================================================================

set -e  # Exit on error

AUDIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$AUDIO_DIR/logs/setup_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$AUDIO_DIR/logs"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  OmniVoice + ROCm Environment Setup"
log "  GPU: AMD RX 7900 XTX (gfx1100 / RDNA3)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 1: Detect ROCm version ────────────────────────────────────────────
log "Step 1: Detecting ROCm version..."
ROCM_VER=$(rocm-smi --version 2>/dev/null | grep "ROCM-SMI-LIB" | awk '{print $NF}' | cut -d. -f1-2)
log "ROCm SMI Lib: $ROCM_VER"

# Detect PyTorch ROCm version already installed
TORCH_ROCM=$(pip3 show torch 2>/dev/null | grep Version | grep -o "rocm[0-9.]*" || echo "none")
log "Current PyTorch: $(pip3 show torch 2>/dev/null | grep Version)"

# ── Step 2: Install / Upgrade PyTorch for ROCm ────────────────────────────
log ""
log "Step 2: Installing PyTorch with ROCm support..."

# Check if ROCm 6.x torch is installed; upgrade if needed
CURRENT_TORCH=$(pip3 show torch 2>/dev/null | grep Version | awk '{print $2}')
log "Currently installed: torch $CURRENT_TORCH"

if echo "$CURRENT_TORCH" | grep -q "rocm6"; then
    log "✅ ROCm PyTorch already installed: $CURRENT_TORCH"
else
    log "⚠️  Installing ROCm-optimized PyTorch (rocm6.1)..."
    pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.1 \
        2>&1 | tee -a "$LOG_FILE"
fi

# ── Step 3: Set AMD environment flags ─────────────────────────────────────
log ""
log "Step 3: Configuring AMD environment flags..."

# Write persistent env config
ENV_FILE="$AUDIO_DIR/rocm_env.sh"
cat > "$ENV_FILE" << 'ENVEOF'
#!/usr/bin/env bash
# OmniVoice ROCm Environment Variables for RX 7900 XTX
# Source this before running any scripts: source rocm_env.sh

# Force ROCm to use correct GFX version for RDNA3 (gfx1100)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Target the RX 7900 XTX (GPU 0) not the integrated Raphael APU (GPU 1)
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

# Disable memory fragmentation for large model loading
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Better performance on RDNA3
export GPU_MAX_HW_QUEUES=8

echo "✅ AMD ROCm environment loaded for RX 7900 XTX"
echo "   HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "   HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
ENVEOF

chmod +x "$ENV_FILE"
log "✅ ROCm env file written: $ENV_FILE"

# Source it now
source "$ENV_FILE"

# ── Step 4: Install OmniVoice and dependencies ─────────────────────────────
log ""
log "Step 4: Installing OmniVoice and audio dependencies..."

pip3 install omnivoice soundfile pydub 2>&1 | tee -a "$LOG_FILE" || {
    log "⚠️  omnivoice pip failed, trying with --break-system-packages..."
    pip3 install omnivoice soundfile pydub --break-system-packages 2>&1 | tee -a "$LOG_FILE"
}

# Install ffmpeg for pydub audio merging
if ! command -v ffmpeg &>/dev/null; then
    log "Installing ffmpeg..."
    sudo apt-get install -y ffmpeg 2>&1 | tee -a "$LOG_FILE"
else
    log "✅ ffmpeg already installed: $(ffmpeg -version 2>&1 | head -1)"
fi

# ── Step 5: Verify GPU detection ────────────────────────────────────────────
log ""
log "Step 5: Verifying GPU detection..."

python3 << 'PYEOF'
import torch
import os

# Apply env flags in Python too
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count       : {torch.cuda.device_count()}")
    print(f"GPU name        : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    print(f"VRAM            : {vram_gb:.1f} GB")
    print(f"✅ AMD GPU is READY for OmniVoice!")
else:
    print("❌ GPU not detected — check ROCm drivers and HSA_OVERRIDE_GFX_VERSION")
PYEOF

log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "✅ Setup complete! Log saved to: $LOG_FILE"
log ""
log "Next step — run the main generator:"
log "  source rocm_env.sh && python3 scripts/generate_audio.py"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
