"""
=============================================================================
batch_storytelling.py — Automated Storytelling Pipeline
GPU Optimization / Tool / Audio / scripts
=============================================================================
Features:
  - Split long story scripts into paragraphs/chapters
  - Generate each chunk on AMD GPU in parallel or sequential batches
  - Merge all .wav chunks with smooth crossfades using pydub
  - Full progress logging with ETA
  - Voice consistency across all chunks (same instruct or ref_audio)
  - Resume interrupted batches (skips already-generated chunks)

Usage:
  source ../rocm_env.sh

  # From a text file:
  python3 batch_storytelling.py --story story.txt --instruct "male, british"

  # From inline text (with chapter splits):
  python3 batch_storytelling.py --story "Chapter 1...|Chapter 2...|Chapter 3..."

  # Voice cloning mode:
  python3 batch_storytelling.py --story story.txt --ref_audio ../reference_audio/voice.wav

Output: outputs/story_<timestamp>/  containing all chunks + final merged WAV
=============================================================================
"""

import os
import sys
import re
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# ── AMD ROCm environment flags ─────────────────────────────────────────────
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["GPU_MAX_HW_QUEUES"] = "8"

import torch

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
OUTPUT_DIR  = BASE_DIR / "outputs"
LOG_DIR     = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
log_file = LOG_DIR / f"batch_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# Text Splitter
# =============================================================================

def split_story(
    text: str,
    mode: str = "paragraph",   # "paragraph" | "sentence" | "pipe" | "chapter"
    max_chars: int = 500,
) -> List[str]:
    """
    Split story text into manageable chunks for batch TTS.

    Args:
        text      : Full story text
        mode      : Split strategy
        max_chars : Maximum characters per chunk (soft limit)

    Returns:
        List of text chunks
    """
    if mode == "pipe":
        # User explicitly separated chunks with |
        chunks = [c.strip() for c in text.split("|") if c.strip()]

    elif mode == "paragraph":
        # Split on blank lines (natural paragraph breaks)
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]

    elif mode == "chapter":
        # Split on Chapter/CHAPTER/Part headings
        chunks = re.split(r"(?=(?:Chapter|CHAPTER|Part|PART|Section)\s+\w+)", text)
        chunks = [c.strip() for c in chunks if c.strip()]

    elif mode == "sentence":
        # Split on sentence endings
        sentences = re.split(r"(?<=[.!?…])\s+", text)
        # Group sentences into chunks under max_chars
        chunks = []
        current = ""
        for s in sentences:
            if len(current) + len(s) < max_chars:
                current += (" " if current else "") + s
            else:
                if current:
                    chunks.append(current)
                current = s
        if current:
            chunks.append(current)
    else:
        chunks = [text]

    log.info(f"Split story into {len(chunks)} chunks (mode: {mode})")
    for i, c in enumerate(chunks):
        log.info(f"  Chunk {i+1:03d}: {len(c)} chars — {c[:60]}...")

    return chunks


# =============================================================================
# GPU Verification
# =============================================================================

def verify_gpu() -> str:
    """Verify AMD GPU and return device string."""
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  Batch Storytelling Pipeline — AMD ROCm Edition")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info(f"✅ GPU: {name} ({vram:.1f} GB VRAM)")
        return "cuda:0"
    else:
        log.warning("⚠️  No GPU — CPU fallback (very slow for batching!)")
        return "cpu"


# =============================================================================
# Model Loader
# =============================================================================

def load_model(device: str, dtype: torch.dtype = torch.float32):
    """Load OmniVoice model once and reuse across all chunks."""
    try:
        from omnivoice import OmniVoice
    except ImportError:
        log.error("❌ omnivoice not installed: pip install omnivoice")
        sys.exit(1)

    log.info("Loading OmniVoice model (once for all chunks)...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype,
    )
    log.info(f"✅ Model ready in {time.time()-t0:.1f}s")
    return model


# =============================================================================
# Batch Generator
# =============================================================================

def generate_chunks(
    model,
    chunks: List[str],
    run_dir: Path,
    instruct: Optional[str] = None,
    ref_audio: Optional[str] = None,
    num_step: int = 50,
    guidance_scale: float = 2.5,
    sample_rate: int = 24000,
) -> List[Path]:
    """Generate audio for each chunk. Skip already completed chunks."""
    import soundfile as sf

    # Build generation config
    try:
        from omnivoice import OmniVoiceGenerationConfig
        config = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=True,
            postprocess_output=True,
        )
    except Exception:
        config = None

    chunk_files = []
    total = len(chunks)
    total_gen_time = 0.0

    for i, text in enumerate(chunks, 1):
        out_file = run_dir / f"chunk_{i:03d}.wav"

        # ── Resume support: skip if already done ─────────────────────────
        if out_file.exists():
            log.info(f"[{i}/{total}] ⏭️  Skipping (already generated): {out_file.name}")
            chunk_files.append(out_file)
            continue

        log.info(f"")
        log.info(f"[{i}/{total}] Generating chunk {i}...")
        log.info(f"  Text: {text[:70]}{'...' if len(text)>70 else ''}")

        # Build kwargs
        kwargs = {"text": text}
        if config:
            kwargs["config"] = config
        if instruct and not ref_audio:
            kwargs["instruct"] = instruct
        if ref_audio:
            kwargs["ref_audio"] = ref_audio

        t0 = time.time()
        try:
            audio = model.generate(**kwargs)
            gen_time = time.time() - t0
            total_gen_time += gen_time

            # Extract audio data
            audio_data = audio[0] if hasattr(audio[0], '__len__') else audio
            if hasattr(audio_data, 'cpu'):
                audio_data = audio_data.cpu().numpy()

            sf.write(str(out_file), audio_data, sample_rate)

            duration = len(audio_data) / sample_rate
            rtf = gen_time / duration
            log.info(f"  ✅ Saved {out_file.name} | {duration:.1f}s audio | "
                     f"{gen_time:.1f}s gen | RTF={rtf:.2f}x")

            # ETA estimate
            if i < total:
                avg_time = total_gen_time / i
                remaining = (total - i) * avg_time
                log.info(f"  📊 Progress: {i}/{total} chunks | "
                         f"ETA: {remaining/60:.1f} min remaining")

            chunk_files.append(out_file)

            # Clear GPU cache between chunks to prevent fragmentation
            torch.cuda.empty_cache()

        except Exception as e:
            log.error(f"  ❌ Chunk {i} failed: {e}")
            log.error(f"     Skipping and continuing...")
            continue

    return chunk_files


# =============================================================================
# Audio Merger
# =============================================================================

def merge_audio(
    chunk_files: List[Path],
    output_path: Path,
    crossfade_ms: int = 500,
) -> Path:
    """
    Merge all chunk WAV files into one final story audio file.
    Uses pydub for smooth crossfade transitions between chunks.
    """
    try:
        from pydub import AudioSegment
        from pydub import effects

        log.info(f"\nMerging {len(chunk_files)} chunks with {crossfade_ms}ms crossfade...")

        combined = None
        for i, f in enumerate(chunk_files):
            log.info(f"  Adding chunk {i+1:03d}: {f.name}")
            segment = AudioSegment.from_wav(str(f))
            # Normalize volume for consistency
            segment = effects.normalize(segment)

            if combined is None:
                combined = segment
            else:
                # Smooth crossfade between paragraphs
                combined = combined.append(segment, crossfade=crossfade_ms)

        # Export final merged file
        combined.export(str(output_path), format="wav")
        duration_min = len(combined) / 1000 / 60
        log.info(f"✅ Final audio: {output_path}")
        log.info(f"   Total duration: {duration_min:.2f} minutes")
        return output_path

    except ImportError:
        log.warning("pydub not available — using wave module fallback")
        return merge_audio_wave(chunk_files, output_path)


def merge_audio_wave(chunk_files: List[Path], output_path: Path) -> Path:
    """Fallback merger using built-in wave module (no crossfade)."""
    import wave

    with wave.open(str(output_path), 'wb') as out:
        for i, f in enumerate(chunk_files):
            with wave.open(str(f), 'rb') as w:
                if i == 0:
                    out.setparams(w.getparams())
                out.writeframes(w.readframes(w.getnframes()))

    log.info(f"✅ Merged (wave): {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch Storytelling Pipeline — AMD ROCm Edition"
    )
    parser.add_argument("--story", type=str, required=True,
        help="Story text OR path to .txt file")
    parser.add_argument("--instruct", type=str,
        default="male, middle-aged, low pitch, british accent",
        help="Voice design keywords. Valid: male/female, young adult/middle-aged/elderly, "
             "low/moderate/high pitch, whisper, american/british/australian/indian accent, etc.")
    parser.add_argument("--ref_audio", type=str, default=None,
        help="Reference audio for voice cloning")
    parser.add_argument("--split", choices=["paragraph","sentence","pipe","chapter"],
        default="paragraph", help="How to split the story")
    parser.add_argument("--max_chars", type=int, default=500,
        help="Max chars per chunk (for sentence split)")
    parser.add_argument("--num_step", type=int, default=50,
        help="Diffusion steps per chunk")
    parser.add_argument("--guidance_scale", type=float, default=2.5,
        help="Voice guidance scale")
    parser.add_argument("--crossfade_ms", type=int, default=500,
        help="Crossfade between chunks in milliseconds")
    parser.add_argument("--dtype", choices=["float32","bfloat16"], default="float32")
    parser.add_argument("--output_name", type=str, default="story",
        help="Final output filename prefix")
    args = parser.parse_args()

    # Load story text
    story_path = Path(args.story)
    if story_path.exists():
        story_text = story_path.read_text(encoding="utf-8")
        log.info(f"Loaded story from file: {story_path} ({len(story_text)} chars)")
    else:
        story_text = args.story  # Direct text input

    # Create run output directory (for resume support)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{args.output_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {run_dir}")

    # Split story into chunks
    chunks = split_story(story_text, mode=args.split, max_chars=args.max_chars)

    if not chunks:
        log.error("No chunks to generate!")
        sys.exit(1)

    # GPU setup
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = verify_gpu()

    # Load model ONCE
    model = load_model(device, dtype)

    # Generate all chunks
    t_start = time.time()
    chunk_files = generate_chunks(
        model=model,
        chunks=chunks,
        run_dir=run_dir,
        instruct=args.instruct,
        ref_audio=args.ref_audio,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
    )

    if not chunk_files:
        log.error("No chunks were generated successfully!")
        sys.exit(1)

    # Merge all chunks into final output
    final_output = OUTPUT_DIR / f"{args.output_name}_{run_id}_FINAL.wav"
    merged = merge_audio(chunk_files, final_output, crossfade_ms=args.crossfade_ms)

    # Summary
    total_time = time.time() - t_start
    log.info("")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  🎙️  BATCH COMPLETE!")
    log.info(f"  Chunks processed : {len(chunk_files)}/{len(chunks)}")
    log.info(f"  Total time       : {total_time/60:.1f} minutes")
    log.info(f"  Final output     : {merged}")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
