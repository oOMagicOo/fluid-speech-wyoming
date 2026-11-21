#!/usr/bin/env python3
"""Wyoming speech-to-text server for Fluid Speech Pro (ONNX)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import onnx_asr
import onnxruntime
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import NemoAsrEventHandler
from .recorder import Recorder, env_bool, env_int

_LOGGER = logging.getLogger(__name__)

# The 25 languages of Fluid Speech Pro, as published in the model card.
DEFAULT_LANGUAGES = ",".join(
    (
        "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu",
        "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv",
        "uk",
    )
)

ATTRIBUTION = Attribution(
    name="FluidVoice",
    url="https://huggingface.co/FluidVoice-Labs/fluid-speech-pro",
)


def _resolve_model_path(model_dir: Optional[str], model_name: str) -> Optional[str]:
    if not model_dir:
        return None
    safe_name = model_name.replace("/", "_").replace(":", "_")
    return os.path.join(model_dir, safe_name)


def _parse_languages(value: str) -> list[str]:
    languages = [code.strip() for code in value.split(",") if code.strip()]
    if not languages:
        raise ValueError("at least one language is required")
    return languages


def _build_info(model: str, languages: list[str]) -> Info:
    """Describe the ASR service to Home Assistant."""
    return Info(
        asr=[
            AsrProgram(
                name="fluid-speech-pro",
                description="Fluid Speech Pro speech-to-text (ONNX Runtime)",
                attribution=ATTRIBUTION,
                installed=True,
                version=__version__,
                requires_external_vad=True,
                models=[
                    AsrModel(
                        name=model,
                        description="Fluid Speech Pro 1.0 (FluidVoice)",
                        attribution=ATTRIBUTION,
                        installed=True,
                        languages=languages,
                        version="1.0",
                    )
                ],
            )
        ],
    )


def _load_model(args: argparse.Namespace) -> object:
    """Load the ONNX model; optionally enable 'clean mode' from the model card."""
    model_name = args.model
    quantization = args.quantization
    path = _resolve_model_path(args.model_dir, model_name)

    if args.clean_mode:
        # Clean mode uses tiny extra graph files that share the int8 weights.
        # They are not part of the standard hub layout, so fetch them first.
        from huggingface_hub import snapshot_download

        _LOGGER.info("Downloading clean-mode files for %s", model_name)
        path = snapshot_download(
            model_name,
            allow_patterns=["*int8*", "vocab.txt", "config.json"],
            local_dir=path,
        )
        model_name = "nemo-conformer-tdt"
        quantization = "clean.int8" if quantization == "int8" else "clean"

    providers = ["CPUExecutionProvider"]
    sess_options = onnxruntime.SessionOptions()
    _LOGGER.info("Loading %s (quantization=%s, path=%s)", model_name, quantization, path)
    return onnx_asr.load_model(
        model_name,
        path=path,
        quantization=quantization,
        providers=providers,
        sess_options=sess_options,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default=os.environ.get("FLUID_URI", "tcp://0.0.0.0:10301"))
    parser.add_argument(
        "--model",
        default=os.environ.get("FLUID_MODEL", "FluidVoice-Labs/fluid-speech-pro-onnx"),
    )
    parser.add_argument(
        "-q", "--quantization",
        default=os.environ.get("FLUID_QUANTIZATION", "int8"),
        help="int8 (default, 801 MB) or fp32 (2.67 GB); with --clean-mode: clean.int8/clean",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("ONNX_ASR_MODEL_DIR", "/data"),
        help="Directory used to cache model files (mount a volume here)",
    )
    parser.add_argument(
        "--languages",
        default=os.environ.get("FLUID_LANGUAGES", DEFAULT_LANGUAGES),
        help="Comma-separated language codes advertised to Home Assistant",
    )
    parser.add_argument(
        "--clean-mode",
        action="store_true",
        default=env_bool("FLUID_CLEAN_MODE", False),
        help="Use Fluid Speech Pro's clean mode (removes fillers/repetitions)",
    )

    parser.add_argument(
        "--record-enabled",
        action="store_true",
        default=None,
        help="Save WAV + transcript (normally controlled by RECORD_ENABLED)",
    )
    parser.add_argument(
        "--record-dir",
        default=None,
        help="Recording directory (normally controlled by RECORD_DIR)",
    )
    parser.add_argument(
        "--record-retention-days",
        type=int,
        default=env_int("RECORD_RETENTION_DAYS", 0),
        help="Delete recordings older than N days (0 keeps them forever)",
    )

    parser.add_argument("--debug", action="store_true", default=env_bool("FLUID_DEBUG", False))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _LOGGER.debug("Arguments: %s", args)

    try:
        languages = _parse_languages(args.languages)
        model = _load_model(args)
    except Exception:
        _LOGGER.exception("Startup failed")
        sys.exit(1)

    recorder = Recorder.from_env(args.model, args.quantization)
    # Explicit CLI values win over the environment.
    if args.record_enabled is not None:
        recorder.enabled = args.record_enabled
    if args.record_dir:
        recorder.directory = Path(args.record_dir)
    if args.record_retention_days is not None:
        recorder.retention_days = max(0, args.record_retention_days)

    wyoming_info = _build_info(args.model, languages)
    model_lock = asyncio.Lock()
    models = {"multi": model}

    try:
        server = AsyncServer.from_uri(args.uri)
    except Exception:
        _LOGGER.exception("Invalid --uri %r", args.uri)
        sys.exit(1)

    _LOGGER.info("Ready on %s with %d language(s)", args.uri, len(languages))
    await server.run(
        partial(NemoAsrEventHandler, wyoming_info, models, model_lock, recorder)
    )


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
