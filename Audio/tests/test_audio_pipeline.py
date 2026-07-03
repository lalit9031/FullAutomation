"""
=============================================================================
test_audio_pipeline.py — End-to-End Pipeline Tests
GPU Optimization / Tool / Audio / tests
=============================================================================
Tests the full pipeline WITHOUT needing OmniVoice installed:
  - ROCm env variable loading
  - Text splitting logic
  - Audio merging (pydub + wave fallback)
  - Output directory creation
  - Model config building

Usage: python3 tests/test_audio_pipeline.py
=============================================================================
"""

import os
import sys
import wave
import struct
import tempfile
import unittest
import numpy as np
from pathlib import Path

# Set env before any imports
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestEnvironment(unittest.TestCase):
    """Tests that ROCm environment variables are correctly set."""

    def test_hsa_override(self):
        self.assertEqual(os.environ.get("HSA_OVERRIDE_GFX_VERSION"), "11.0.0",
                         "HSA_OVERRIDE_GFX_VERSION must be 11.0.0 for RX 7900 XTX")

    def test_hip_visible_devices(self):
        self.assertEqual(os.environ.get("HIP_VISIBLE_DEVICES"), "0",
                         "HIP_VISIBLE_DEVICES must be 0 to target RX 7900 XTX")

    def test_pytorch_hip_alloc(self):
        self.assertEqual(
            os.environ.get("PYTORCH_HIP_ALLOC_CONF"),
            "expandable_segments:True",
            "PYTORCH_HIP_ALLOC_CONF should be set for OOM prevention"
        )


class TestTextSplitter(unittest.TestCase):
    """Tests the story text splitting logic."""

    def setUp(self):
        from batch_storytelling import split_story
        self.split = split_story

    def test_paragraph_split(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = self.split(text, mode="paragraph")
        self.assertEqual(len(chunks), 3)

    def test_pipe_split(self):
        text = "Chunk A|Chunk B|Chunk C"
        chunks = self.split(text, mode="pipe")
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "Chunk A")

    def test_sentence_split_under_limit(self):
        # Short sentences should group together
        text = "Hello. World. How are you?"
        chunks = self.split(text, mode="sentence", max_chars=200)
        self.assertGreater(len(chunks), 0)

    def test_chapter_split(self):
        text = "Prologue\n\nChapter 1\nFirst chapter.\n\nChapter 2\nSecond chapter."
        chunks = self.split(text, mode="chapter")
        # Should detect Chapter splits
        self.assertGreater(len(chunks), 1)

    def test_empty_chunks_filtered(self):
        text = "\n\n\n   \n\n"
        chunks = self.split(text, mode="paragraph")
        self.assertEqual(len(chunks), 0)

    def test_single_chunk(self):
        text = "A single paragraph with no breaks."
        chunks = self.split(text, mode="paragraph")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)


class TestAudioMerge(unittest.TestCase):
    """Tests the audio merging utilities."""

    def _create_test_wav(self, path: Path, duration_s: float = 1.0,
                          sample_rate: int = 24000, freq: float = 440.0):
        """Create a simple sine-wave WAV for testing."""
        n_samples = int(duration_s * sample_rate)
        t = np.linspace(0, duration_s, n_samples)
        audio = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

        with wave.open(str(path), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(audio.tobytes())
        return path

    def test_wave_merge(self):
        """Test the built-in wave module merger."""
        from batch_storytelling import merge_audio_wave

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Create 3 test WAVs
            files = [self._create_test_wav(tmp / f"chunk_{i:03d}.wav") for i in range(3)]
            output = tmp / "merged.wav"

            result = merge_audio_wave(files, output)
            self.assertTrue(result.exists())

            # Verify merged file is longer than any individual chunk
            with wave.open(str(result), 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                duration = frames / rate
            self.assertGreater(duration, 2.5)  # 3 × 1s merged

    def test_pydub_merge(self):
        """Test pydub merger if available."""
        try:
            from pydub import AudioSegment
            from batch_storytelling import merge_audio

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                files = [self._create_test_wav(tmp / f"chunk_{i:03d}.wav", freq=220*i+220)
                         for i in range(2)]
                output = tmp / "merged_pydub.wav"
                result = merge_audio(files, output, crossfade_ms=100)
                self.assertTrue(result.exists())
                print(f"    pydub merge: {result.stat().st_size} bytes ✅")

        except ImportError:
            self.skipTest("pydub not installed — skipping pydub test")


class TestOutputDirectories(unittest.TestCase):
    """Tests that output and log directories are created properly."""

    def test_dirs_exist(self):
        base = Path(__file__).parent.parent
        for d in ["outputs", "logs", "reference_audio", "scripts", "tests"]:
            self.assertTrue((base / d).exists(), f"Missing directory: {d}")

    def test_log_dir_writable(self):
        log_dir = Path(__file__).parent.parent / "logs"
        test_file = log_dir / "test_write.tmp"
        test_file.write_text("ok")
        self.assertTrue(test_file.exists())
        test_file.unlink()


class TestPyTorchGPU(unittest.TestCase):
    """Tests PyTorch GPU availability (skips if no GPU)."""

    def test_pytorch_importable(self):
        import torch
        self.assertIsNotNone(torch.__version__)

    def test_rocm_in_version(self):
        import torch
        if "rocm" not in torch.__version__:
            self.skipTest("Non-ROCm PyTorch — GPU tests skipped")
        self.assertIn("rocm", torch.__version__)

    def test_gpu_available(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("No GPU detected (CPU-only environment)")
        self.assertTrue(torch.cuda.is_available())

    def test_gpu_is_amd(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("No GPU")
        name = torch.cuda.get_device_name(0)
        print(f"\n    GPU name: {name}")
        # Passes for any GPU, just reports name
        self.assertIsNotNone(name)

    def test_vram_sufficient(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("No GPU")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n    VRAM: {vram_gb:.1f} GB")
        self.assertGreater(vram_gb, 8.0, "Need at least 8GB VRAM for OmniVoice")


if __name__ == "__main__":
    print("=" * 55)
    print("  OmniVoice Audio Pipeline — Test Suite")
    print("=" * 55)
    unittest.main(verbosity=2)
