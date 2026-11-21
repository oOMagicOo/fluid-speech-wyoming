# fluid-speech-wyoming

Wyoming-Spracherkennung für Home Assistant mit [**Fluid Speech Pro**](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-onnx) (0,63B Parakeet-TDT, 25 europäische Sprachen, Deutsch dabei). Der Container ersetzt einen faster-whisper-Container in der Assist-Pipeline und speichert zusätzlich **jede Äußerung als WAV-Datei samt Textdatei** ab — konfigurierbar und mit UTC-Timecode im Dateinamen.

> **Modell-Credit:** Fluid Speech Pro by FluidVoice, CC BY 4.0. Das Modell ist nicht in diesem Repo enthalten; es wird beim ersten Start heruntergeladen.
>
> Der Wyoming-Teil basiert auf [tboby/wyoming-onnx-asr](https://github.com/tboby/wyoming-onnx-asr) (MIT) und wurde um den Recorder, die Sprachliste und die Modellkonfiguration erweitert.

## Docker / Compose

```sh
mkdir -p models recordings
docker compose pull
docker compose up -d
docker compose logs -f
```

- Port **10301/TCP** (Wyoming, nur im LAN — das Protokoll hat keine Authentifizierung).
- `./models` → Modell-Cache (beim ersten Start lädt onnx-asr ~801 MB für `int8` bzw. ~2,67 GB für `fp32` von Hugging Face).
- `./recordings` → WAV + TXT, standardmäßig in Tagesordnern `YYYY-MM-DD/`.

## Konfiguration

Alle Optionen sind Umgebungsvariablen in `compose.yaml`:

| Variable | Default | Bedeutung |
|---|---|---|
| `FLUID_URI` | `tcp://0.0.0.0:10301` | Wyoming-Listen-Adresse |
| `FLUID_MODEL` | `FluidVoice-Labs/fluid-speech-pro-onnx` | Hugging-Face-Repo des ONNX-Modells |
| `FLUID_QUANTIZATION` | `int8` | `int8` (801 MB) oder `fp32` (2,67 GB) |
| `FLUID_CLEAN_MODE` | `false` | „Clean mode" des Modells (Füllwörter/Wiederholungen entfernen); nutzt `clean.int8` |
| `FLUID_LANGUAGES` | 25 Codes inkl. `de` | Was HA als unterstützte Sprachen gemeldet wird |
| `ONNX_ASR_MODEL_DIR` | `/data` | Modell-Cache im Container |
| `FLUID_DEBUG` | `false` | Debug-Logging |
| `RECORD_ENABLED` | `true` | Aufnahmen an/aus |
| `RECORD_DIR` | `/recordings` | Zielverzeichnis im Container |
| `RECORD_FILENAME` | `{utc:%Y%m%dT%H%M%S.%fZ}_{satellite}_{shortid}` | Dateiname ohne Endung; `utc`, `date`, `satellite`, `conversation`, `shortid`, `language` verfügbar |
| `RECORD_SUBDIRS_BY_DATE` | `true` | Tagesordner `YYYY-MM-DD/` anlegen |
| `RECORD_RETENTION_DAYS` | `0` | Löschen alter Aufnahmen (0 = nie löschen) |
| `RECORD_METADATA` | `true` | `.txt` neben der WAV schreiben |

**Wichtig:** Wyoming überträgt keine WAV-Dateien, sondern rohe PCM-Chunks. Der Container fügt sie zu einer WAV zusammen und speichert genau diese. Die Textdatei enthält unter anderem:

```
timestamp_utc: 2026-11-21T14:30:52.123Z
duration_s: 3.42
model: FluidVoice-Labs/fluid-speech-pro-onnx
credits: Fluid Speech Pro by FluidVoice (CC BY 4.0)
language_requested: de
satellite_id: assist_satellite.echo_2_...
text: Wie sind die Temperaturen im Haus?
```

## Home Assistant

1. Einstellung → Geräte & Dienste → Integration hinzufügen → **Wyoming**, Host = Docker-Server, Port `10301`.
2. In der Assist-Pipeline die STT-Engine auf die neue Entity umstellen (die alte Pipeline als Rollback behalten).
3. Test: „Alexa, wie sind die Temperaturen im Haus?" → Antwort und neue Dateien in `./recordings` prüfen.

## Testen

```sh
python -m unittest discover -s tests -p 'test_*.py' -v          # Recorder-Unit-Tests
python tests/wyoming_client.py tcp://127.0.0.1:10301 audio.wav de  # Smoke-Test gegen den Container
```

Der CI-Workflow `.github/workflows/docker.yml` führt die Unit-Tests aus und baut/pusht das Image nach `ghcr.io/oomagicoo/fluid-speech-wyoming:latest`.

## Lizenz

Eigener Code und der übernommene Wyoming-Teil: MIT (siehe `LICENSE`, `NOTICE`). Das Modell unterliegt CC BY 4.0 und bleibt bei FluidVoice; die Namensnennung ist in jeder Textdatei enthalten.
