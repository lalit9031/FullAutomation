#!/bin/bash
# Music Studio — Launch Script (Port 8006)
# Separate from Audio Studio (Port 8005)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="$(dirname "$SCRIPT_DIR")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎵 Starting Music Studio (Singing Pipeline)..."
echo "   URL   : http://localhost:8006"
echo "   Model : ModelsLab/omnivoice-singing"
echo "   Note  : English only — [singing] tags applied automatically"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# AMD ROCm environment
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
echo "✅ AMD ROCm environment loaded (RX 7900 XTX / gfx1100)"

# Activate virtualenv if present
VENV="$TOOL_DIR/Audio/venv"
if [ -d "$VENV" ]; then
  source "$VENV/bin/activate"
  echo "✅ Virtualenv activated"
fi

cd "$SCRIPT_DIR/ui"
exec python3 -m uvicorn server:app --host 0.0.0.0 --port 8006 --reload
