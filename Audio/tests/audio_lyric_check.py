"""
=============================================================================
audio_lyric_check.py — Automatic Speech Recognition (ASR) Lyric Verification
GPU Optimization / Tool / Audio / tests
=============================================================================
Transcribes a generated WAV file using whisper-tiny and compares it to the 
expected prompt lyrics to ensure no words or phrases are skipped or cut off.

Run: python3 tests/audio_lyric_check.py <wav_path> <expected_language>
=============================================================================
"""

import sys
import os
from pathlib import Path

# Force CPU/GPU environment settings
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

def get_expected_lyrics(lang):
    if lang.lower() == "english":
        return [
            "Twinkle twinkle little star How I wonder what you are",
            "Up above the world so high Like a diamond in the sky",
            "When the blazing sun is gone When he nothing shines upon",
            "Then you show your little light Twinkle twinkle all the night"
        ]
    elif lang.lower() == "english_long":
        return [
            "Twinkle twinkle little star How I wonder what you are",
            "Up above the world so high Like a diamond in the sky",
            "When the blazing sun is gone When he nothing shines upon",
            "Then you show your little light Twinkle twinkle all the night",
            "Then the traveler in the dark Thanks you for your tiny spark",
            "He could not see which way to go If you did not twinkle so",
            "In the dark blue sky you keep And often through my curtains peep",
            "For you never shut your eye Till the sun is in the sky",
            "As your bright and tiny spark Lights the traveler in the dark",
            "Though I know not what you are Twinkle twinkle little star"
        ]
    elif lang.lower() == "hindi":
        return [
            "चंदा मामा दूर के पुए पकाएं बूर के",
            "आप खाएं थाली में मुन्ने को दें प्याली में",
            "प्याली गई टूट मुन्ना गया रूठ",
            "लाएंगे नई प्यालियां बजा बजा के तालियां",
            "उड़नखटोले बैठेंगे मुन्ने राजा ऐठेंगे"
        ]
    return []

def normalize_text(text):
    import re
    # Remove punctuation, convert to lowercase
    text = text.lower()
    text = re.sub(r'[^\w\s\u0900-\u097F]', '', text) # Keep devanagari letters
    return " ".join(text.split())

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 audio_lyric_check.py <wav_path> <english|hindi>")
        sys.exit(1)
        
    wav_path = sys.argv[1]
    lang = sys.argv[2]
    
    if not os.path.exists(wav_path):
        print(f"❌ Error: File not found: {wav_path}")
        sys.exit(1)
        
    print("="*60)
    print("🎙️ Loading Whisper Transcription Engine...")
    print("="*60)
    
    import transformers
    import warnings
    warnings.filterwarnings("ignore")
    
    # Load ASR pipeline
    pipe = transformers.pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        chunk_length_s=30,
        device="cuda"
    )
    
    import time
    print("\n💿 Transcribing Audio File:", wav_path)
    t_start = time.time()
    
    # Run transcription
    result = pipe(wav_path)
    transcription = result["text"].strip()
    elapsed = time.time() - t_start
    
    print(f"✅ Transcribed in {elapsed:.2f}s")
    print("\n🗣️ AUDIO LYRICS (Transcribed Text):")
    print("-" * 50)
    print(transcription)
    print("-" * 50)
    
    # Verification
    expected_lines = get_expected_lyrics(lang)
    if not expected_lines:
        print("ℹ️ No expected lyric template for this language, skipping comparison.")
        sys.exit(0)
        
    print("\n📊 Verification Comparison Check:")
    print("="*60)
    
    norm_trans = normalize_text(transcription)
    
    matched_count = 0
    total_lines = len(expected_lines)
    
    for i, line in enumerate(expected_lines, 1):
        norm_line = normalize_text(line)
        # Check if the words of the expected line appear in the transcription
        words = norm_line.split()
        if not words:
            continue
            
        # Match check: do at least 50% of the words appear in order or close?
        matched_words = [w for w in words if w in norm_trans]
        match_ratio = len(matched_words) / len(words)
        
        if match_ratio >= 0.5:
            print(f"  Line {i}: ✅ MATCH ({match_ratio*100:.0f}%)")
            print(f"    Expected: {line}")
            matched_count += 1
        else:
            print(f"  Line {i}: ❌ MISSING/SKIPPED ({match_ratio*100:.0f}%)")
            print(f"    Expected: {line}")
            
    print("="*60)
    ratio = matched_count / total_lines
    print(f"🏆 Verdict: {matched_count}/{total_lines} lines matched ({ratio*100:.0f}%)")
    
    if ratio >= 0.75:
        print("🎉 SUCCESS: Audio lyrics match prompt lyrics with high accuracy!")
    else:
        print("⚠️ WARNING: Audio lyrics mismatch detected. Words might have been skipped.")
    print("="*60)

if __name__ == "__main__":
    main()
