FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    ONNX_ASR_MODEL_DIR=/data \
    RECORD_DIR=/recordings

WORKDIR /app

# libsndfile1 is required by soundfile; libgomp1 by onnxruntime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir \
      "wyoming>=1.5.3" \
      "onnx-asr[cpu,hub]==0.12.0" \
      "soundfile>=0.12.1" \
      "numpy>=1.24"

COPY wyoming_onnx_asr/ ./wyoming_onnx_asr/

VOLUME ["/data", "/recordings"]
EXPOSE 10301

ENTRYPOINT ["python", "-m", "wyoming_onnx_asr"]
CMD ["--uri", "tcp://0.0.0.0:10301"]
