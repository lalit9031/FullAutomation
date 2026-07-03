"""
=============================================================================
audio_quality_check.py — Comprehensive Audio Quality Analyser
GPU Optimization / Tool / Audio / tests
=============================================================================
Analyses a WAV file across 8 quality dimensions and gives a verdict.

Usage:
  python3 tests/audio_quality_check.py outputs/quality_test.wav
  python3 tests/audio_quality_check.py outputs/quality_test.wav --play

Checks:
  1. File validity & format
  2. Sample rate (should be 24000 Hz for OmniVoice)
  3. Duration (not too short/long)
  4. Clipping detection (distortion)
  5. Silence ratio (too much silence = bad generation)
  6. SNR — Signal-to-Noise Ratio (higher = cleaner audio)
  7. RMS energy (robotic/flat = too low, distorted = too high)
  8. Spectral analysis (frequency distribution — checks for robotic tone)
=============================================================================
"""

import sys
import wave
import struct
import argparse
import numpy as np
from pathlib import Path


# ── ANSI colors ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}✅ PASS{RESET}"
FAIL = f"{RED}❌ FAIL{RESET}"
WARN = f"{YELLOW}⚠️  WARN{RESET}"
INFO = f"{CYAN}ℹ️  INFO{RESET}"


def section(title: str):
    print(f"\n{BOLD}{'━'*56}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'━'*56}{RESET}")


def result(label: str, value: str, status: str, detail: str = ""):
    print(f"  {status}  {label:<32} {BOLD}{value}{RESET}")
    if detail:
        print(f"          {CYAN}{detail}{RESET}")


# =============================================================================
# 1. Load WAV
# =============================================================================
def load_wav(path: str):
    """Load WAV file, return (samples as float32, sample_rate, channels)."""
    with wave.open(path, 'rb') as w:
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()
        frame_rate = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    # Convert raw bytes to numpy
    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 3:
        # 24-bit — manual unpack
        ints = []
        for i in range(0, len(raw), 3):
            b = raw[i:i+3]
            val = struct.unpack('<i', b + (b'\xff' if b[2] & 0x80 else b'\x00'))[0] >> 8
            ints.append(val)
        samples = np.array(ints, dtype=np.float32) / 8388608.0
    else:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    # If stereo, take mean of channels
    if n_channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)

    return samples, frame_rate, n_channels


# =============================================================================
# Quality Check Functions
# =============================================================================

def check_format(path: str, samples, sr: int, channels: int):
    """Check 1: File format, sample rate, channels."""
    section("1. File Format & Properties")

    file_size = Path(path).stat().st_size / 1024
    duration  = len(samples) / sr

    result("File readable", "✓", PASS)
    result("File size", f"{file_size:.1f} KB", PASS if file_size > 10 else WARN,
           "Very small — may indicate failed generation" if file_size < 10 else "")

    # Sample rate check
    expected_sr = 24000
    sr_ok = sr == expected_sr
    result("Sample rate", f"{sr:,} Hz",
           PASS if sr_ok else WARN,
           f"OmniVoice outputs {expected_sr} Hz — mismatch may cause pitch issues" if not sr_ok else "")

    result("Channels", f"{channels} ({'mono' if channels == 1 else 'stereo'})", PASS)
    result("Duration", f"{duration:.2f} seconds",
           PASS if duration > 0.5 else FAIL,
           "Too short — generation may have failed" if duration < 0.5 else "")

    return duration


def check_clipping(samples):
    """Check 2: Clipping detection (values at ±1.0 = distortion)."""
    section("2. Clipping Detection (Distortion Check)")

    clip_threshold = 0.99
    clipped = np.sum(np.abs(samples) >= clip_threshold)
    total = len(samples)
    clip_pct = (clipped / total) * 100

    if clip_pct < 0.01:
        status = PASS
        detail = "No audible distortion"
    elif clip_pct < 0.5:
        status = WARN
        detail = "Mild clipping — slight distortion possible"
    else:
        status = FAIL
        detail = "Heavy clipping — distorted output. Try dtype=bfloat16 or reduce guidance_scale"

    result("Clipped samples", f"{clipped:,} ({clip_pct:.3f}%)", status, detail)
    result("Max amplitude", f"{np.max(np.abs(samples)):.4f}", PASS if np.max(np.abs(samples)) < 0.99 else WARN)


def check_silence(samples, sr: int):
    """Check 3: Silence ratio — too much silence = bad generation."""
    section("3. Silence Analysis")

    silence_threshold = 0.01  # below this = effectively silent
    silent_samples = np.sum(np.abs(samples) < silence_threshold)
    total = len(samples)
    silence_pct = (silent_samples / total) * 100

    if silence_pct < 30:
        status = PASS
        detail = "Good speech density"
    elif silence_pct < 60:
        status = WARN
        detail = "Moderate silence — may include natural pauses"
    else:
        status = FAIL
        detail = "Too much silence — possible failed generation or pure static output"

    result("Silence ratio", f"{silence_pct:.1f}%", status, detail)

    # Leading / trailing silence
    non_silent = np.where(np.abs(samples) >= silence_threshold)[0]
    if len(non_silent) > 0:
        lead_sil = non_silent[0] / sr
        trail_sil = (total - non_silent[-1]) / sr
        result("Leading silence", f"{lead_sil:.2f}s",
               PASS if lead_sil < 1.0 else WARN)
        result("Trailing silence", f"{trail_sil:.2f}s",
               PASS if trail_sil < 1.0 else WARN)


def check_snr(samples):
    """Check 4: Signal-to-Noise Ratio estimation."""
    section("4. Signal-to-Noise Ratio (SNR)")

    # Estimate noise floor from quietest 10% of frames
    frame_size = 512
    n_frames = len(samples) // frame_size
    frames = samples[:n_frames * frame_size].reshape(n_frames, frame_size)
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1))

    noise_floor  = np.percentile(rms_per_frame, 10)  # quietest frames
    signal_level = np.percentile(rms_per_frame, 90)  # loudest frames (speech)

    if noise_floor < 1e-10:
        noise_floor = 1e-10  # prevent log(0)

    snr_db = 20 * np.log10(signal_level / noise_floor)

    if snr_db > 30:
        status = PASS
        detail = "Excellent clarity — clean voice generation"
    elif snr_db > 20:
        status = PASS
        detail = "Good clarity — minor background noise"
    elif snr_db > 10:
        status = WARN
        detail = "Moderate noise — try increasing num_step to 60+"
    else:
        status = FAIL
        detail = "Low SNR — robotic/noisy output. Check ROCm env and dtype settings"

    result("SNR estimate", f"{snr_db:.1f} dB", status, detail)
    result("Noise floor", f"{noise_floor:.6f} RMS", INFO)
    result("Signal level", f"{signal_level:.6f} RMS", INFO)


def check_rms_energy(samples, sr: int):
    """Check 5: RMS energy profile — detects flat/robotic output."""
    section("5. RMS Energy Profile")

    overall_rms = np.sqrt(np.mean(samples ** 2))
    peak_db     = 20 * np.log10(np.max(np.abs(samples)) + 1e-10)
    rms_db      = 20 * np.log10(overall_rms + 1e-10)

    # Dynamic range check (difference between loudest and quietest moments)
    frame_size = int(0.02 * sr)  # 20ms frames
    n_frames = len(samples) // frame_size
    frames = samples[:n_frames * frame_size].reshape(n_frames, frame_size)
    rms_frames = np.sqrt(np.mean(frames ** 2, axis=1))
    dynamic_range = 20 * np.log10((rms_frames.max() + 1e-10) / (rms_frames.min() + 1e-10))

    # Overall RMS should be in a reasonable range for speech
    if -30 < rms_db < -5:
        rms_status = PASS
        rms_detail = "Normal speech energy level"
    elif rms_db <= -30:
        rms_status = FAIL
        rms_detail = "Too quiet — possible empty/near-silent output"
    else:
        rms_status = WARN
        rms_detail = "Very loud — check for clipping"

    result("Overall RMS", f"{rms_db:.1f} dBFS", rms_status, rms_detail)
    result("Peak level", f"{peak_db:.1f} dBFS",
           PASS if peak_db < -1 else WARN)

    if dynamic_range > 20:
        dr_status = PASS
        dr_detail = "Natural speech variation ✓"
    elif dynamic_range > 10:
        dr_status = WARN
        dr_detail = "Moderate variation — slightly flat delivery"
    else:
        dr_status = FAIL
        dr_detail = "Very flat — possible robotic/monotone output"

    result("Dynamic range", f"{dynamic_range:.1f} dB", dr_status, dr_detail)


def check_spectral(samples, sr: int):
    """Check 6: Spectral analysis — detects robotic tone / no speech content."""
    section("6. Spectral Analysis (Frequency Check)")

    # FFT analysis
    fft = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(len(samples), 1 / sr)
    magnitude = np.abs(fft)

    # Speech lives mostly in 100Hz–8000Hz
    speech_mask = (freqs >= 100) & (freqs <= 8000)
    speech_energy = np.sum(magnitude[speech_mask] ** 2)
    total_energy  = np.sum(magnitude ** 2)
    speech_ratio  = speech_energy / (total_energy + 1e-10)

    # Fundamental frequency estimate (pitch — should be in human range 80-300Hz)
    voice_mask = (freqs >= 80) & (freqs <= 400)
    if voice_mask.any():
        dominant_freq = freqs[voice_mask][np.argmax(magnitude[voice_mask])]
    else:
        dominant_freq = 0

    if speech_ratio > 0.5:
        spec_status = PASS
        spec_detail = "Strong speech-band content — human voice characteristics detected"
    elif speech_ratio > 0.2:
        spec_status = WARN
        spec_detail = "Moderate speech content — check for artifacts"
    else:
        spec_status = FAIL
        spec_detail = "Low speech-band energy — may be noise/static"

    result("Speech-band energy", f"{speech_ratio*100:.1f}%", spec_status, spec_detail)

    pitch_ok = 60 < dominant_freq < 400
    result("Dominant pitch", f"{dominant_freq:.0f} Hz",
           PASS if pitch_ok else WARN,
           "Human voice range: 80–300 Hz" if not pitch_ok else "Within human voice range ✓")

    # High frequency noise check (hiss: energy > 8kHz should be low)
    hiss_mask = freqs > 8000
    hiss_energy = np.sum(magnitude[hiss_mask] ** 2) / (total_energy + 1e-10)
    result("High-freq noise (hiss)", f"{hiss_energy*100:.1f}%",
           PASS if hiss_energy < 0.3 else WARN,
           "Hiss detected — try denoise=True in config" if hiss_energy >= 0.3 else "")


# =============================================================================
# Overall Verdict
# =============================================================================
def verdict(samples, sr: int, duration: float):
    section("📊 OVERALL VERDICT")

    checks = []

    # Duration
    checks.append(("Duration OK",       duration > 0.5))

    # Clipping
    clip_pct = np.sum(np.abs(samples) >= 0.99) / len(samples) * 100
    checks.append(("No clipping",       clip_pct < 0.5))

    # Silence
    silence_pct = np.sum(np.abs(samples) < 0.01) / len(samples) * 100
    checks.append(("Reasonable silence", silence_pct < 70))

    # RMS energy
    rms_db = 20 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-10)
    checks.append(("Energy in range",   -30 < rms_db < -3))

    # SNR
    frame_size = 512
    n_frames = len(samples) // frame_size
    frames = samples[:n_frames * frame_size].reshape(n_frames, frame_size)
    rms_frames = np.sqrt(np.mean(frames ** 2, axis=1))
    noise_floor  = np.percentile(rms_frames, 10)
    signal_level = np.percentile(rms_frames, 90)
    snr_db = 20 * np.log10((signal_level + 1e-10) / (noise_floor + 1e-10))
    checks.append(("SNR > 20 dB",       snr_db > 20))

    passed = sum(1 for _, ok in checks if ok)
    total  = len(checks)

    print()
    for name, ok in checks:
        icon = f"{GREEN}✅{RESET}" if ok else f"{RED}❌{RESET}"
        print(f"    {icon}  {name}")

    print()
    score_pct = passed / total * 100
    if score_pct == 100:
        grade = f"{GREEN}{BOLD}EXCELLENT — Audio quality is outstanding ✨{RESET}"
    elif score_pct >= 80:
        grade = f"{GREEN}{BOLD}GOOD — Audio quality is acceptable ✅{RESET}"
    elif score_pct >= 60:
        grade = f"{YELLOW}{BOLD}FAIR — Some quality issues, check warnings ⚠️{RESET}"
    else:
        grade = f"{RED}{BOLD}POOR — Audio has significant issues ❌{RESET}"

    print(f"  Score: {passed}/{total} ({score_pct:.0f}%)")
    print(f"  Grade: {grade}")
    print()
    print(f"  📁 WAV file: {CYAN}{sys.argv[1]}{RESET}")
    print(f"  🎧 To listen: aplay {sys.argv[1]}  OR  open in any audio player")
    print()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Audio Quality Analyser for OmniVoice output")
    parser.add_argument("wav_file", help="Path to WAV file to analyse")
    parser.add_argument("--play", action="store_true", help="Play audio after analysis (requires aplay)")
    args = parser.parse_args()

    path = args.wav_file
    if not Path(path).exists():
        print(f"{RED}❌ File not found: {path}{RESET}")
        print("   Run: python3 scripts/generate_audio.py first")
        sys.exit(1)

    print(f"\n{BOLD}{'═'*56}{RESET}")
    print(f"{BOLD}  🎙️  OmniVoice Audio Quality Analyser{RESET}")
    print(f"{BOLD}  File: {path}{RESET}")
    print(f"{BOLD}{'═'*56}{RESET}")

    samples, sr, channels = load_wav(path)

    duration = check_format(path, samples, sr, channels)
    check_clipping(samples)
    check_silence(samples, sr)
    check_snr(samples)
    check_rms_energy(samples, sr)
    check_spectral(samples, sr)
    verdict(samples, sr, duration)

    if args.play:
        import subprocess
        print(f"🔊 Playing audio...")
        subprocess.run(["aplay", path])


if __name__ == "__main__":
    main()
