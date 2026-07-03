"""
=============================================================================
generate_5min_audio.py — 5-Minute Audio Generation & Stress Test
GPU Optimization / Tool / Audio / tests
=============================================================================
Generates a long 5-minute children's story poem to stress-test sequential 
synthesis stability, memory footprint, and phrase coherence on ROCm.

Saves output to: outputs/five_minute_test.wav
=============================================================================
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# Force AMD ROCm visibility
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

# Check GPU availability
if not torch.cuda.is_available():
    print("❌ Error: AMD ROCm GPU is not available.")
    sys.exit(1)

print("✅ AMD ROCm Initialized Successfully!")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Import OmniVoice modules
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig

# Create a long 5-minute (130-line) children's story script
def get_long_script():
    stanzas = [
        "Twinkle, twinkle, little star, How I wonder what you are!",
        "Up above the world so high, Like a diamond in the sky.",
        "When the blazing sun is gone, When he nothing shines upon,",
        "Then you show your little light, Twinkle, twinkle, all the night.",
        "Then the traveler in the dark, Thanks you for your tiny spark,",
        "He could not see which way to go, If you did not twinkle so.",
        "In the dark blue sky you keep, And often through my curtains peep,",
        "For you never shut your eye, Till the sun is in the sky.",
        "As your bright and tiny spark, Lights the traveler in the dark,",
        "Though I know not what you are, Twinkle, twinkle, little star."
    ]
    
    # Repeat the 10 stanzas (20 lines) 6.5 times to create a 130-line script
    long_script = []
    for i in range(13):
        for stanza in stanzas:
            # We append slightly varied text lines to make it a continuous story
            var_stanza = stanza.replace("Twinkle, twinkle", f"Twinkle Starlight Stanza {i+1}")
            long_script.append(var_stanza)
            
    return "\n".join(long_script)

def split_text_to_safe_phrases(text: str) -> list:
    import re
    raw_lines = re.split(r'[\n.!?;]+', text)
    text_lines = []
    
    for line in raw_lines:
        line_strip = line.strip()
        if not line_strip:
            continue
            
        comma_parts = [p.strip() for p in line_strip.split(",") if p.strip()]
        temp_phrase = ""
        for part in comma_parts:
            part_clean = part.replace("[singing]", "").strip()
            if not part_clean:
                continue
                
            if not temp_phrase:
                temp_phrase = part_clean
            else:
                word_count = len(temp_phrase.split())
                if word_count < 3:
                    temp_phrase = f"{temp_phrase}, {part_clean}"
                else:
                    text_lines.append(temp_phrase)
                    temp_phrase = part_clean
        if temp_phrase:
            text_lines.append(temp_phrase)
            
    return text_lines

def main():
    print("\n📦 Loading ModelsLab/omnivoice-singing checkpoint...")
    t_start = time.time()
    
    # Initialize OmniVoice model on GPU
    model = OmniVoice.from_pretrained(
        "ModelsLab/omnivoice-singing", 
        device_map="cuda:0", 
        dtype=torch.float32
    )
    print(f"✅ Loaded checkpoint in {time.time() - t_start:.2f}s")
    
    script_text = get_long_script()
    phrases = split_text_to_safe_phrases(script_text)
    
    print(f"\n📝 Long Script Loaded: {len(phrases)} synthesis segments.")
    print("🚀 Starting 5-Minute sequential synthesis stress test...")
    
    config = OmniVoiceGenerationConfig(
        num_step=70,
        guidance_scale=2.5,
        denoise=True,
        postprocess_output=True
    )
    
    instruct = "female, child, high pitch"
    audio_segments = []
    
    t_gen_start = time.time()
    
    try:
        for i, phrase in enumerate(phrases, 1):
            # Preprocess to prevent trailing word cutoff bugs
            formatted_phrase = phrase.replace(",", " , ")
            formatted_phrase = formatted_phrase.replace(".", " . ")
            formatted_phrase = formatted_phrase.replace("!", " ! ")
            formatted_phrase = formatted_phrase.replace("?", " ? ")
            formatted_phrase = " ".join(formatted_phrase.split())
            if not formatted_phrase.endswith("."):
                formatted_phrase = f"{formatted_phrase} ."
                
            print(f"🎙️ Segment {i:03d}/{len(phrases):03d}: '{formatted_phrase}'")
            
            # Explicitly specify target language "english" for better phoneme priors
            audio = model.generate(
                text=formatted_phrase, 
                instruct=instruct, 
                language="English", 
                config=config
            )
            
            audio_data = audio[0].cpu().numpy() if hasattr(audio[0], "cpu") else audio[0]
            audio_segments.append(audio_data)
            
            # Clean CUDA cache dynamically to prevent OOM
            if i % 10 == 0:
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"\n❌ Error during synthesis at segment {i}: {e}")
        sys.exit(1)
        
    print(f"\n✅ All {len(phrases)} segments synthesized successfully!")
    print(f"Total GPU synthesis time: {time.time() - t_gen_start:.2f}s")
    
    # Merge audio segments with 0.2s natural pauses
    print("\n🎛️ Merging audio segments...")
    sample_rate = 24000
    pause_samples = int(0.2 * sample_rate)
    pause_interval = np.zeros(pause_samples, dtype=np.float32)
    
    final_audio = []
    for i, segment in enumerate(audio_segments):
        if i > 0:
            final_audio.append(pause_interval)
        final_audio.append(segment)
        
    merged_audio = np.concatenate(final_audio)
    duration_s = len(merged_audio) / sample_rate
    print(f"🎬 Combined Audio Duration: {duration_s/60:.2f} minutes ({duration_s:.2f} seconds)")
    
    # Write output WAV file
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/five_minute_test.wav"
    
    import wave
    # Normalize to 16-bit PCM
    audio_int16 = (merged_audio * 32767).astype(np.int16)
    
    with wave.open(out_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
        
    print(f"🎉 Success! Generated: {out_path}")
    print(f"File size: {os.path.getsize(out_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
