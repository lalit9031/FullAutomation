"""
=============================================================================
generate_all_samples.py — Generate All Voice Samples in One Shot
GPU Optimization / Tool / Audio / scripts
=============================================================================
Generates:
  - Whisper (improved, num_step=80)
  - Kids voices (English + Hindi)
  - Hindi voices — male, female, child in multiple styles

Run: source ../rocm_env.sh && python3 scripts/generate_all_samples.py
=============================================================================
"""

import os, sys, time, torch, soundfile as sf
from pathlib import Path

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

BASE    = Path(__file__).parent.parent
OUT     = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ── All samples to generate ────────────────────────────────────────────────
# (filename, text, instruct, num_step, description)
SAMPLES = [

    # ── WHISPER — Improved clarity (num_step=80) ──────────────────────────
    ("whisper_hq_01",
     "The old clock struck midnight. Suddenly, the door creaked open...",
     "male, middle-aged, very low pitch, whisper", 80,
     "Thriller whisper — male, HQ"),

    ("whisper_hq_02",
     "Listen carefully. I have a secret to tell you. Come closer...",
     "female, young adult, low pitch, whisper", 80,
     "Mystery whisper — female, HQ"),

    # ── KIDS — English ────────────────────────────────────────────────────
    ("kids_en_01",
     "Hello! My name is Lily. I love playing in the park with my puppy!",
     "female, child, high pitch", 70,
     "Kids — English girl"),

    ("kids_en_02",
     "Wow, look at that! A rainbow! I want to catch it!",
     "male, child, high pitch", 70,
     "Kids — English boy"),

    ("kids_en_03",
     "Once upon a time, there was a tiny dragon who was afraid of fire.",
     "female, teenager, moderate pitch", 70,
     "Kids — Teen girl storytelling"),

    # ── HINDI — Indian accent, multiple styles ────────────────────────────
    ("hindi_male_01",
     "नमस्ते! मेरा नाम राज है। आज का दिन बहुत सुंदर है।",
     "male, middle-aged, moderate pitch, indian accent", 70,
     "Hindi — Adult male, neutral"),

    ("hindi_male_02",
     "सुनो ध्यान से। यह कहानी बहुत पुरानी है, पहाड़ों की गहराई से आती है।",
     "male, elderly, low pitch, indian accent", 70,
     "Hindi — Elderly male, storytelling"),

    ("hindi_female_01",
     "नमस्ते! मैं आपकी मदद कैसे कर सकती हूँ? आज मौसम बहुत अच्छा है।",
     "female, young adult, moderate pitch, indian accent", 70,
     "Hindi — Young adult female"),

    ("hindi_female_02",
     "आओ बच्चों, एक कहानी सुनाती हूँ। एक बार की बात है, एक छोटा सा गाँव था।",
     "female, middle-aged, moderate pitch, indian accent", 70,
     "Hindi — Middle-aged female, storytelling"),

    ("hindi_kids_01",
     "अरे वाह! देखो कितनी बड़ी तितली है! मुझे तितलियाँ बहुत पसंद हैं।",
     "female, child, high pitch, indian accent", 70,
     "Hindi — Girl child"),

    ("hindi_kids_02",
     "मम्मी! मम्मी! मुझे आइसक्रीम चाहिए! प्लीज़ प्लीज़ प्लीज़!",
     "male, child, high pitch, indian accent", 70,
     "Hindi — Boy child"),

    ("hindi_teen",
     "यार, आज स्कूल में क्या हुआ पता है? बड़ा मज़ेदार था।",
     "male, teenager, moderate pitch, indian accent", 70,
     "Hindi — Teen male, casual"),

    ("hindi_whisper",
     "धीरे बोलो। कोई सुन लेगा। यह हमारा राज़ है।",
     "male, middle-aged, low pitch, whisper, indian accent", 80,
     "Hindi — Whisper male"),
]


def main():
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

    print("━"*55)
    print("  Loading OmniVoice (from cache)...")
    print("━"*55)
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float32
    )
    print(f"  Model ready in {time.time()-t0:.1f}s\n")

    total = len(SAMPLES)
    results = []

    for i, (fname, text, instruct, steps, desc) in enumerate(SAMPLES, 1):
        out_path = OUT / f"{fname}.wav"
        print(f"[{i:02d}/{total}] {desc}")
        print(f"        instruct : {instruct}")
        print(f"        steps    : {steps}")
        print(f"        text     : {text[:55]}...")

        config = OmniVoiceGenerationConfig(
            num_step=steps, guidance_scale=2.5,
            denoise=True, postprocess_output=True
        )

        t_gen = time.time()
        try:
            audio = model.generate(text=text, instruct=instruct, config=config)
            data  = audio[0].cpu().numpy() if hasattr(audio[0], "cpu") else audio[0]
            sf.write(str(out_path), data, 24000)
            duration = len(data) / 24000
            elapsed  = time.time() - t_gen
            print(f"        ✅ {out_path.name} | {duration:.1f}s audio | {elapsed:.1f}s gen\n")
            results.append((fname, True, duration, elapsed))
        except Exception as e:
            print(f"        ❌ FAILED: {e}\n")
            results.append((fname, False, 0, 0))

        torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────
    print("━"*55)
    print("  GENERATION COMPLETE")
    print("━"*55)
    ok = sum(1 for _,s,_,_ in results if s)
    print(f"  Generated : {ok}/{total} clips")
    print(f"  Location  : {OUT}\n")
    print("  🎧 To listen:")
    for fname, success, dur, _ in results:
        if success:
            print(f"     aplay outputs/{fname}.wav   ({dur:.1f}s)")
    print()


if __name__ == "__main__":
    main()
