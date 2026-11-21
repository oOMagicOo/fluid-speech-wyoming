"""Unit tests for the recorder (stdlib only, no model download needed)."""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
import unittest
import wave
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wyoming_onnx_asr.recorder import Recorder  # noqa: E402


def make_wav(path: Path, frames: int = 16000) -> bytes:
    """Write a tiny 16 kHz mono PCM WAV and return its bytes."""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * frames)
    return path.read_bytes()


class RecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.wav"
        self.wav_bytes = make_wav(self.source)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_creates_wav_and_txt(self) -> None:
        recorder = Recorder(
            enabled=True,
            directory=self.root / "out",
            model_name="FluidVoice-Labs/fluid-speech-pro-onnx",
            quantization="int8",
        )
        result = recorder.save(
            self.source,
            text="Wie sind die Temperaturen im Haus?",
            language="de",
            satellite_id="assist_satellite.echo_2",
            conversation_id="conv-42",
        )
        self.assertIsNotNone(result)
        wav_target, txt_target = result
        self.assertRegex(wav_target.parent.name, r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(wav_target.name, r"^\d{8}T\d{6}\.\d+Z_.*\.wav$")
        self.assertEqual(wav_target.read_bytes(), self.wav_bytes)

        self.assertIsNotNone(txt_target)
        text = txt_target.read_text(encoding="utf-8")
        self.assertIn("Wie sind die Temperaturen im Haus?", text)
        self.assertIn("Fluid Speech Pro by FluidVoice", text)
        self.assertIn("language_requested: de", text)
        self.assertIn("duration_s: 1.0", text)

    def test_disabled(self) -> None:
        recorder = Recorder(enabled=False, directory=self.root / "out")
        self.assertIsNone(recorder.save(self.source, text="test"))
        self.assertFalse((self.root / "out").exists())

    def test_retention_removes_old_files(self) -> None:
        old_dir = self.root / "out" / "2000-01-01"
        old_dir.mkdir(parents=True)
        old_wav = old_dir / "old.wav"
        old_txt = old_dir / "old.txt"
        old_wav.write_bytes(self.wav_bytes)
        old_txt.write_text("old text", encoding="utf-8")
        old_ts = time.time() - 10 * 86400
        os.utime(old_wav, (old_ts, old_ts))
        os.utime(old_txt, (old_ts, old_ts))

        recorder = Recorder(
            enabled=True, directory=self.root / "out", retention_days=1
        )
        result = recorder.save(self.source, text="new")
        self.assertIsNotNone(result)
        self.assertTrue(result[0].exists())
        self.assertFalse(old_wav.exists())
        self.assertFalse(old_txt.exists())

    def test_custom_template_is_safe(self) -> None:
        recorder = Recorder(
            enabled=True,
            directory=self.root / "out",
            filename_template="{language}/{satellite}_{shortid}",
            subdirs_by_date=False,
        )
        result = recorder.save(
            self.source, text="hallo", language="de", satellite_id="echo/../../etc"
        )
        self.assertIsNotNone(result)
        wav_target = result[0]
        self.assertEqual(wav_target.parent, self.root / "out")
        self.assertNotIn("..", wav_target.name)
        self.assertTrue(wav_target.name.startswith("de_echo_"))


    def test_retention_count_keeps_newest_utterances(self) -> None:
        out = self.root / "out"
        out.mkdir()
        now = time.time()
        for index in range(1, 5):
            wav = out / f"old{index}.wav"
            txt = out / f"old{index}.txt"
            wav.write_bytes(self.wav_bytes)
            txt.write_text(f"old {index}", encoding="utf-8")
            stamp = now - (5 - index) * 3600
            os.utime(wav, (stamp, stamp))
            os.utime(txt, (stamp, stamp))

        recorder = Recorder(
            enabled=True,
            directory=out,
            subdirs_by_date=False,
            retention_count=2,
        )
        result = recorder.save(self.source, text="new utterance")
        self.assertIsNotNone(result)
        wavs = sorted(p.name for p in out.glob("*.wav"))
        txts = sorted(p.name for p in out.glob("*.txt"))
        self.assertEqual(len(wavs), 2, wavs)
        self.assertEqual(len(txts), 2, txts)
        self.assertIn("old4.wav", wavs)
        self.assertNotIn("old1.wav", wavs)
        self.assertNotIn("old1.txt", txts)
        self.assertIn(result[0].name, wavs)


if __name__ == "__main__":
    unittest.main()
