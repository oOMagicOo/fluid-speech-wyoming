#!/usr/bin/env python3
"""Manual smoke test: send a WAV to a Wyoming STT server and print the result.

Usage:
    python tests/wyoming_client.py tcp://127.0.0.1:10301 audio.wav [de]

Uses only the standard library, so it can run on the server without installing
anything.  It also proves the recorder end-to-end: after a successful run, a
WAV and a TXT with a UTC timecode appear in RECORD_DIR.
"""

from __future__ import annotations

import json
import socket
import sys
import wave
from urllib.parse import urlparse

RECV_SIZE = 65536


def send_event(conn: socket.socket, event_type: str, data: dict | None = None,
               payload: bytes = b"") -> None:
    header = {"type": event_type, "data": data or {}, "payload_length": len(payload)}
    conn.sendall(
        json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\n" + payload
    )


def read_event(stream) -> dict:
    line = stream.readline()
    if not line:
        raise EOFError("server closed the connection")
    header = json.loads(line.decode("utf-8"))
    data = header.get("data") or {}
    data_length = int(header.get("data_length", 0))
    if data_length:
        data.update(json.loads(stream.read(data_length).decode("utf-8")))
    payload = stream.read(int(header.get("payload_length", 0)))
    return {"type": header["type"], "data": data, "payload": payload}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    uri = urlparse(sys.argv[1])
    wav_path = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) > 3 else "de"

    with wave.open(wav_path, "rb") as wav:
        rate, width, channels = (
            wav.getframerate(), wav.getsampwidth(), wav.getnchannels()
        )
        frames = wav.readframes(wav.getnframes())

    with socket.create_connection((uri.hostname, uri.port or 10301), timeout=30) as conn:
        conn.settimeout(120)
        stream = conn.makefile("rb")
        send_event(conn, "describe")
        info = read_event(stream)
        if info["type"] != "info":
            raise RuntimeError(f"expected info, got {info['type']}")
        print("Server info:", json.dumps(info["data"], ensure_ascii=False)[:300])

        send_event(conn, "transcribe", {"language": language})
        send_event(conn, "audio-start", {"rate": rate, "width": width, "channels": channels})
        send_event(conn, "audio-chunk",
                   {"rate": rate, "width": width, "channels": channels}, frames)
        send_event(conn, "audio-stop")

        while True:
            event = read_event(stream)
            if event["type"] == "transcript":
                print("Transcript:", event["data"].get("text", ""))
                return 0
            if event["type"] == "error":
                print("Error:", event["data"], file=sys.stderr)
                return 1


if __name__ == "__main__":
    sys.exit(main())
