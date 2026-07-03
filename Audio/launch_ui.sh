#!/usr/bin/env bash
# =============================================================================
# launch_ui.sh — Start the OmniVoice Web Studio
# GPU Optimization / Tool / Audio
# =============================================================================

AUDIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$AUDIO_DIR"

# 1. Source ROCm env flags
if [ -f "rocm_env.sh" ]; then
    source rocm_env.sh
else
    echo "⚠️  rocm_env.sh not found. Using system defaults."
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export HIP_VISIBLE_DEVICES=0
fi

# 2. Check if server dependencies are available
python3 -c "import fastapi, uvicorn, omnivoice, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing Python packages. Running setup..."
    bash setup_rocm_env.sh
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting OmniVoice Web Studio..."
echo "   GPU   : AMD RX 7900 XTX (gfx1100)"
echo "   URL   : http://localhost:8005"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 3. Launch FastAPI server
python3 ui/server.py
