"""Wyoming event handler: PCM chunks -> WAV -> Fluid Speech Pro -> transcript.

Based on the handler from tboby/wyoming-onnx-asr (MIT), extended with a
Recorder that persists every utterance and its transcript.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import wave
from typing import Optional

import numpy as np
import soundfile as sf
from onnx_asr.adapters import AsrAdapter
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe
from wyoming.server import AsyncEventHandler

from .recorder import Recorder

_LOGGER = logging.getLogger(__name__)


class NemoAsrEventHandler(AsyncEventHandler):
    """One connection: collect audio, transcribe it, optionally record it."""

    def __init__(
        self,
        wyoming_info,
        models: dict[str, AsrAdapter],
        model_lock: asyncio.Lock,
        recorder: Recorder,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.models = models
        self.model_lock = model_lock
        self.recorder = recorder
        self.request_language: Optional[str] = None
        self.context: dict = {}

        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None
        self._rate = 16000
        self._channels = 1

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)
                self._rate = chunk.rate
                self._channels = chunk.channels
            self._wav_file.writeframes(chunk.audio)
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self.request_language = transcribe.language
            self.context = event.data.get("context") or {}
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        if AudioStop.is_type(event.type):
            language = self.request_language or "de"
            text = ""
            if self._wav_file is not None:
                self._wav_file.close()
                self._wav_file = None

                try:
                    waveform, sample_rate = sf.read(self._wav_path, dtype="float32")
                    if len(waveform.shape) > 1:
                        waveform = np.mean(waveform, axis=1)

                    model = self.models.get("multi") or next(
                        iter(self.models.values()), None
                    )
                    if model is None:
                        raise RuntimeError("no ASR model loaded")

                    async with self.model_lock:
                        text = model.recognize(
                            waveform,
                            sample_rate=sample_rate,
                            language=language,
                        )
                except Exception as exc:
                    _LOGGER.exception("Transcription failed")
                    await self.write_event(
                        Transcript(text=f"ERROR: {exc}").event()
                    )
                    await self._record(text="", language=language)
                    self.request_language = None
                    return False

            # The temp WAV is removed when this handler dies, so record first.
            await self._record(text=text, language=language)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.info("%s: %s", language, text)
            self.request_language = None
            return False

        return True

    async def _record(self, *, text: str, language: str) -> None:
        """Persist the WAV + transcript (errors must never break STT)."""
        if not self.recorder.enabled:
            return
        satellite_id = (
            self.context.get("satellite_id")
            or self.context.get("device_id")
            or "unknown"
        )
        conversation_id = self.context.get("conversation_id") or "unknown"
        try:
            self.recorder.save(
                self._wav_path,
                text=text,
                language=language,
                satellite_id=str(satellite_id),
                conversation_id=str(conversation_id),
                sample_rate=self._rate,
                channels=self._channels,
            )
        except Exception:
            _LOGGER.exception("Recording failed (transcript still returned)")
