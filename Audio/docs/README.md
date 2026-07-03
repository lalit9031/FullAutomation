# OmniVoice Audio Module — AMD ROCm Edition

> **GPU**: AMD RX 7900 XTX (24 GB VRAM) | **Arch**: RDNA3 / gfx1100
> **Status**: ROCm-optimized, fully AMD native

---

## Folder Structure

```
Audio/
├── setup_rocm_env.sh          ← Run once to install everything
├── rocm_env.sh                ← Source this before every session
│
├── scripts/
│   ├── generate_audio.py      ← Single text → WAV generator
│   ├── batch_storytelling.py  ← Full story → merged WAV pipeline
│   └── example_story.txt      ← Demo thriller story
│
├── tests/
│   ├── verify_gpu.py          ← Verify GPU + all deps (run first!)
│   └── test_audio_pipeline.py ← Full unit test suite
│
├── outputs/                   ← All generated WAV files land here
├── reference_audio/           ← Drop voice cloning reference clips here
├── logs/                      ← Generation logs with timing/RTF metrics
└── docs/
    └── README.md              ← This file
```

---

## Quick Start (3 Steps)

### Step 1: Setup (One Time Only)
```bash
cd "/home/lalit/Desktop/GPU optimization/Tool/Audio"
chmod +x setup_rocm_env.sh
bash setup_rocm_env.sh
```

### Step 2: Verify Everything Works
```bash
source rocm_env.sh
python3 tests/verify_gpu.py
```

Expected output:
```
✅ AMD GPU ready — RX 7900 XTX (24.0 GB VRAM)
✅ SYSTEM IS READY FOR OMNIVOICE AUDIO GENERATION!
```

### Step 3: Generate Audio
```bash
source rocm_env.sh

# Single line voice design
python3 scripts/generate_audio.py \
  --text "The storm was coming." \
  --instruct "male, deep voice, dramatic, british accent"

# Full story from file
python3 scripts/batch_storytelling.py \
  --story scripts/example_story.txt \
  --instruct "male, middle-aged, low pitch, whisper, british accent"
```

---

## ROCm Configuration (Why These Settings)

| Variable | Value | Reason |
|----------|-------|--------|
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | Forces RDNA3 (gfx1100) code path |
| `HIP_VISIBLE_DEVICES` | `0` | Targets RX 7900 XTX, not integrated GPU |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | Prevents OOM on large models |
| `GPU_MAX_HW_QUEUES` | `8` | Maximizes RDNA3 compute throughput |

> The RX 7900 XTX (gfx1100) IS officially supported by ROCm 6.x.
> `HSA_OVERRIDE_GFX_VERSION` is set as a safety measure for apps that
> haven't updated their GFX detection tables yet.

---

## OmniVoice Generation Modes

### Mode 1: Voice Design (Natural Language)
```python
model.generate(
    text="Hello, this is a test.",
    instruct="female, young, high pitch, cheerful, american accent"
)
```

### Mode 2: Voice Cloning (Reference Audio)
```bash
# Put a 3-10 second WAV of the target voice in reference_audio/
python3 scripts/generate_audio.py \
  --ref_audio reference_audio/my_voice.wav \
  --text "This will sound like me."
```

### Mode 3: Auto Voice (No Instructions)
```python
model.generate(text="The model picks a voice automatically.")
```

---

## Generation Config Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_step` | `50` | Diffusion steps — more = clearer voice, slower |
| `guidance_scale` | `2.5` | Voice adherence — higher = stricter |
| `denoise` | `True` | Removes background hiss |
| `postprocess_output` | `True` | Cleans awkward silences |
| `dtype` | `float32` | AMD handles FP32/BF16 well; avoids half-precision artifacts |

---

## Batch Storytelling Options

```bash
python3 scripts/batch_storytelling.py \
  --story story.txt          # or inline text with | separators
  --split paragraph          # paragraph | sentence | pipe | chapter
  --num_step 50              # quality setting
  --guidance_scale 2.5
  --crossfade_ms 500         # smooth transitions between chunks
  --output_name my_story     # output file prefix
```

**Resume support**: If generation is interrupted, re-running the same command
will skip already-generated chunks and continue from where it stopped.

---

## Expected Performance (RX 7900 XTX)

| Text Length | Chunks | Est. Gen Time | RTF |
|-------------|--------|---------------|-----|
| 1 sentence | 1 | ~5-10s | 0.2x |
| 1 paragraph | 1 | ~10-20s | 0.3x |
| Short story (1000 words) | ~8 | ~2-4 min | 0.3x |
| Full chapter (5000 words) | ~40 | ~10-20 min | 0.3x |

RTF < 1.0 = faster than real-time ✅

---

## CLI Reference

```bash
# Run all unit tests
python3 tests/test_audio_pipeline.py

# Verify GPU setup
python3 tests/verify_gpu.py

# Single generation with all options
python3 scripts/generate_audio.py \
  --text "..." \
  --instruct "voice style" \
  --num_step 50 \
  --guidance_scale 2.5 \
  --dtype float32 \
  --output my_clip

# Batch story generation
python3 scripts/batch_storytelling.py \
  --story story.txt \
  --instruct "voice style" \
  --split paragraph \
  --crossfade_ms 500 \
  --output_name chapter_01
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `torch.cuda.is_available()` returns `False` | `source rocm_env.sh` first |
| Static / robotic output | Increase `--num_step` to 60-80 |
| OOM on large models | Model already loaded in FP32; try `--dtype bfloat16` |
| Wrong GPU targeted | Check `rocm-smi` — ensure GPU 0 is 7900 XTX |
| Slow generation | Normal for CPU fallback; fix ROCm setup |
| pydub merge fails | `sudo apt install ffmpeg` |

---

## Part of: GPU Optimization / Tool
This module will be **combined** with other tool modules (Video, Image, etc.)
when all are complete.

**Next modules planned**: Video, Image (follow same folder pattern)
