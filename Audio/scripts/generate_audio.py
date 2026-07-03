"""
=============================================================================
generate_audio.py — OmniVoice AMD ROCm Audio Generator
GPU Optimization / Tool / Audio / scripts
=============================================================================
Features:
  - Full ROCm/AMD RX 7900 XTX support
  - Float32 / BFloat16 precision (AMD optimal)
  - Configurable generation parameters
  - Single text → WAV generation
  - Voice design via natural language instructions
  - Voice cloning via reference audio

Usage:
  source ../rocm_env.sh
  python3 generate_audio.py
  python3 generate_audio.py --text "Hello world" --instruct "female, british"
  python3 generate_audio.py --ref_audio ../reference_audio/voice.wav --text "Hello"
=============================================================================
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime

# ── AMD ROCm environment flags ─────────────────────────────────────────────
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"   # RDNA3 / gfx1100
os.environ["HIP_VISIBLE_DEVICES"] = "0"              # Only RX 7900 XTX
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["GPU_MAX_HW_QUEUES"] = "8"

import torch

# ── Logging setup ──────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"generate_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Output directory ───────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def verify_gpu() -> str:
    """Verify AMD GPU is detected and return device string."""
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  OmniVoice Audio Generator — AMD ROCm Edition")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info(f"PyTorch version : {torch.__version__}")
    log.info(f"CUDA/ROCm avail : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info(f"GPU             : {gpu_name}")
        log.info(f"VRAM            : {vram_gb:.1f} GB")
        log.info("✅ AMD GPU ready — running on hardware acceleration")
        return "cuda:0"
    else:
        log.warning("⚠️  GPU not detected — falling back to CPU (slow!)")
        log.warning("   Fix: source rocm_env.sh && check ROCm drivers")
        return "cpu"


def load_model(device: str, dtype: torch.dtype = torch.float32):
    """Load OmniVoice model onto AMD GPU."""
    try:
        from omnivoice import OmniVoice
    except ImportError:
        log.error("❌ omnivoice not installed. Run: pip install omnivoice")
        sys.exit(1)

    log.info("")
    log.info("Loading OmniVoice model...")
    log.info(f"  Model   : k2-fsa/OmniVoice")
    log.info(f"  Device  : {device}")
    log.info(f"  Dtype   : {dtype} (AMD FP32/BF16 optimized)")

    t0 = time.time()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype,
    )
    load_time = time.time() - t0
    log.info(f"✅ Model loaded in {load_time:.1f}s")
    return model


def build_generation_config(
    num_step: int = 50,
    guidance_scale: float = 2.5,
    denoise: bool = True,
    postprocess: bool = True,
):
    """Build OmniVoiceGenerationConfig with quality-optimized settings."""
    try:
        from omnivoice import OmniVoiceGenerationConfig
        config = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=denoise,
            postprocess_output=postprocess,
        )
        log.info(f"Generation config: steps={num_step}, guidance={guidance_scale}, "
                 f"denoise={denoise}, postprocess={postprocess}")
        return config
    except Exception as e:
        log.warning(f"Could not create OmniVoiceGenerationConfig: {e}")
        log.warning("Using default generation settings")
        return None


def generate(
    model,
    text: str,
    instruct: str = None,
    ref_audio: str = None,
    config=None,
    output_name: str = "output",
    sample_rate: int = 24000,
) -> Path:
    """
    Generate audio from text.

    Args:
        model       : Loaded OmniVoice model
        text        : Text to synthesize
        instruct    : Voice design prompt (e.g. "male, british, low pitch")
        ref_audio   : Path to reference audio for voice cloning
        config      : OmniVoiceGenerationConfig instance
        output_name : Output filename (without extension)
        sample_rate : Audio sample rate (OmniVoice default: 24000 Hz)

    Returns:
        Path to saved .wav file
    """
    import soundfile as sf

    log.info("")
    log.info(f"Generating audio...")
    log.info(f"  Text    : {text[:80]}{'...' if len(text) > 80 else ''}")
    if instruct:
        log.info(f"  Voice   : {instruct}")
    if ref_audio:
        log.info(f"  Ref     : {ref_audio}")

    # Build generate() kwargs
    kwargs = {"text": text}
    if config:
        kwargs["config"] = config
    if instruct and not ref_audio:
        kwargs["instruct"] = instruct
    if ref_audio:
        kwargs["ref_audio"] = ref_audio

    t0 = time.time()
    audio = model.generate(**kwargs)
    gen_time = time.time() - t0

    # Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{output_name}_{timestamp}.wav"

    # audio is typically [samples] or [1, samples]
    audio_data = audio[0] if hasattr(audio, '__len__') and len(audio.shape) > 1 else audio
    if hasattr(audio_data, 'cpu'):
        audio_data = audio_data.cpu().numpy()

    sf.write(str(out_path), audio_data, sample_rate)

    duration = len(audio_data) / sample_rate
    rtf = gen_time / duration  # Real-Time Factor (lower = faster)

    log.info(f"✅ Audio saved: {out_path}")
    log.info(f"   Duration  : {duration:.2f}s")
    log.info(f"   Gen time  : {gen_time:.2f}s")
    log.info(f"   RTF       : {rtf:.3f}x (< 1.0 = faster than real-time ✅)")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="OmniVoice Audio Generator — AMD ROCm Edition"
    )
    parser.add_argument("--text", type=str,
        default="The old clock struck midnight. Suddenly, the door creaked open...",
        help="Text to synthesize")
    parser.add_argument("--instruct", type=str,
        default="male, middle-aged, low pitch, british accent",
        help=(
            "Voice design keywords (comma + space separated). "
            "Valid: male, female, young adult, middle-aged, elderly, child, teenager, "
            "low pitch, moderate pitch, high pitch, very low pitch, very high pitch, whisper, "
            "american accent, british accent, australian accent, canadian accent, "
            "indian accent, russian accent, portuguese accent, chinese accent, "
            "japanese accent, korean accent"
        ))
    parser.add_argument("--ref_audio", type=str, default=None,
        help="Path to reference audio for voice cloning (3-10 seconds)")
    parser.add_argument("--num_step", type=int, default=50,
        help="Diffusion steps (default: 50, more = better quality)")
    parser.add_argument("--guidance_scale", type=float, default=2.5,
        help="Guidance scale (default: 2.5)")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
        help="Precision (float32 recommended for AMD, avoids underflow)")
    parser.add_argument("--output", type=str, default="output",
        help="Output filename prefix")
    args = parser.parse_args()

    # Map dtype string to torch dtype
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Run pipeline
    device  = verify_gpu()
    model   = load_model(device, dtype)
    config  = build_generation_config(
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
    )
    out = generate(
        model=model,
        text=args.text,
        instruct=args.instruct,
        ref_audio=args.ref_audio,
        config=config,
        output_name=args.output,
    )

    log.info("")
    log.info(f"🎙️  Output file: {out}")
    log.info("Done!")


if __name__ == "__main__":
    main()
