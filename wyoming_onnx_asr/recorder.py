"""Configurable recording of Wyoming audio and its transcript.

Wyoming delivers raw PCM chunks, not WAV files.  The handler assembles them
into a temporary WAV; this module copies that WAV into a configurable target
directory and writes a text file beside it.  Both file names share a UTC
timecode so a recording and its transcript always belong together.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_LOGGER = logging.getLogger(__name__)

MODEL_CREDIT = "Fluid Speech Pro by FluidVoice (CC BY 4.0)"

DEFAULT_FILENAME = "{utc:%Y%m%dT%H%M%S.%fZ}_{satellite}_{shortid}"


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable (1/true/yes/on vs 0/false/no/off)."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to default."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        _LOGGER.warning("%s=%r is not an integer, using %s", name, value, default)
        return default


def _safe(value: object, max_len: int = 48) -> str:
    """Make a value safe for a single path component."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    # A single dot is fine (the UTC fraction), but never leave ".." behind.
    while ".." in text:
        text = text.replace("..", "_")
    return (text.strip("._-") or "unknown")[:max_len]


def _wav_duration(path: str | Path) -> float:
    """Return the duration of a PCM WAV in seconds, or 0.0 when unreadable."""
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            return round(wav.getnframes() / rate, 3) if rate else 0.0
    except Exception:  # pragma: no cover - defensive, bad audio must not break STT
        _LOGGER.debug("Could not determine duration of %s", path, exc_info=True)
        return 0.0


class Recorder:
    """Save each utterance as ``<timecode>.wav`` plus an optional ``.txt``."""

    def __init__(
        self,
        *,
        enabled: bool,
        directory: str | Path,
        filename_template: str = DEFAULT_FILENAME,
        subdirs_by_date: bool = True,
        retention_days: int = 0,
        retention_count: int = 20,
        metadata: bool = True,
        model_name: str = "",
        quantization: str = "",
    ) -> None:
        self.enabled = enabled
        self.directory = Path(directory)
        self.filename_template = filename_template or DEFAULT_FILENAME
        self.subdirs_by_date = subdirs_by_date
        self.retention_days = max(0, retention_days)
        self.retention_count = max(0, retention_count)
        self.metadata = metadata
        self.model_name = model_name
        self.quantization = quantization

    @classmethod
    def from_env(cls, model_name: str, quantization: str) -> "Recorder":
        """Build a Recorder from the RECORD_* environment variables."""
        return cls(
            enabled=env_bool("RECORD_ENABLED", True),
            directory=os.environ.get("RECORD_DIR", "/recordings"),
            filename_template=os.environ.get("RECORD_FILENAME", DEFAULT_FILENAME),
            subdirs_by_date=env_bool("RECORD_SUBDIRS_BY_DATE", True),
            retention_days=env_int("RECORD_RETENTION_DAYS", 0),
            retention_count=env_int("RECORD_RETENTION_COUNT", 20),
            metadata=env_bool("RECORD_METADATA", True),
            model_name=model_name,
            quantization=quantization,
        )

    def save(
        self,
        wav_path: str | Path,
        *,
        text: str,
        language: Optional[str] = None,
        satellite_id: str = "",
        conversation_id: str = "",
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> Optional[tuple[Path, Optional[Path]]]:
        """Copy the WAV and write the transcript. Returns the paths, or None."""
        if not self.enabled:
            return None

        now = datetime.now(timezone.utc)
        shortid = os.urandom(4).hex()
        fields = {
            "utc": now,
            "utc_iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "satellite": _safe(satellite_id or "unknown"),
            "conversation": _safe(conversation_id or "none"),
            "shortid": shortid,
            "language": _safe(language or "auto"),
        }
        try:
            name = self.filename_template.format(**fields)
        except (KeyError, ValueError, IndexError):
            _LOGGER.warning("Invalid RECORD_FILENAME, using default template")
            name = DEFAULT_FILENAME.format(**fields)

        wav_name = _safe(name, 160) + ".wav"
        target_dir = (
            self.directory / now.strftime("%Y-%m-%d")
            if self.subdirs_by_date
            else self.directory
        )
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            wav_target = target_dir / wav_name
            txt_target = wav_target.with_suffix(".txt")

            duration = _wav_duration(wav_path)
            self._atomic_copy(wav_path, wav_target)

            if self.metadata:
                contents = [
                    f"timestamp_utc: {now.isoformat().replace('+00:00', 'Z')}",
                    f"duration_s: {duration}",
                    f"sample_rate_hz: {sample_rate}",
                    f"channels: {channels}",
                    f"model: {self.model_name}",
                    f"quantization: {self.quantization}",
                    f"credits: {MODEL_CREDIT}",
                    f"language_requested: {language or 'auto'}",
                    f"satellite_id: {satellite_id or 'unknown'}",
                    f"conversation_id: {conversation_id or 'unknown'}",
                    f"text: {str(text).replace(chr(10), ' ').replace(chr(13), ' ')}",
                ]
                self._atomic_text(txt_target, "\n".join(contents) + "\n")

            _LOGGER.info("Recorded utterance: %s", wav_target)
            if self.retention_days or self.retention_count:
                self._cleanup(now)
            return wav_target, txt_target if self.metadata else None
        except OSError:
            _LOGGER.exception("Could not save recording to %s", target_dir)
            return None

    @staticmethod
    def _atomic_copy(source: str | Path, target: Path) -> None:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            tmp = Path(handle.name)
        try:
            shutil.copyfile(source, tmp)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()

    @staticmethod
    def _atomic_text(target: Path, contents: str) -> None:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            tmp = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()

    def _cleanup(self, now: datetime) -> None:
        """Apply retention: by age (days) and/or by number of utterances.

        The count limit keeps the newest N WAV files across all date folders
        and removes each one's transcript beside it.  A limit of 0 disables
        that rule; both rules can be active at the same time.
        """
        removed: set[Path] = set()
        try:
            files: list[tuple[float, Path]] = []
            for path in self.directory.rglob("*"):
                if not path.is_file() or path.suffix not in (".wav", ".txt"):
                    continue
                try:
                    files.append((path.stat().st_mtime, path))
                except OSError:  # pragma: no cover - concurrent deletion
                    continue

            if self.retention_days:
                cutoff = (now - timedelta(days=self.retention_days)).timestamp()
                for mtime, path in files:
                    if mtime < cutoff:
                        removed.add(path)

            if self.retention_count:
                wavs = sorted(
                    ((mtime, path) for mtime, path in files if path.suffix == ".wav"),
                    key=lambda item: (item[0], str(item[1])),
                    reverse=True,
                )
                expired = wavs[self.retention_count:]
                kept_stems = {path.stem for _, path in wavs[:self.retention_count]}
                for _, path in expired:
                    removed.add(path)
                    removed.add(path.with_suffix(".txt"))
                for _, path in files:
                    if path.suffix == ".txt" and path.stem not in kept_stems:
                        removed.add(path)

            for path in removed:
                try:
                    path.unlink()
                except OSError:  # pragma: no cover - already gone
                    continue

            for path in sorted(
                self.directory.rglob("*"), key=lambda p: len(p.parts), reverse=True
            ):
                if path.is_dir():
                    try:
                        path.rmdir()  # only succeeds when empty
                    except OSError:
                        pass
        except OSError:
            _LOGGER.debug("Retention cleanup skipped", exc_info=True)
            return
        if removed:
            _LOGGER.info("Retention removed %d file(s)", len(removed))
