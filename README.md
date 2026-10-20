# WhisperX API (Docker)

## Build

```bash
docker build -t whisperx-api .
```

## Run (GPU)

Requires NVIDIA Container Toolkit installed on the host.

```bash
docker run --rm --gpus all -p 8001:8001 -e WHISPERX_DEVICE=cuda -e WHISPERX_MODEL=base -e WHISPERX_COMPUTE_TYPE=float16 -e HF_TOKEN=YOUR_TOKEN_HERE whisperx-api
```

## Run (CPU)

```bash
docker run --rm -p 8001:8001 -e WHISPERX_DEVICE=cpu -e WHISPERX_MODEL=base whisperx-api
```

## Compose

```bash
docker compose up --build
```

## Test

```bash
curl -X POST "http://localhost:8001/transcribe" -F "file=@sample.mp3"
```
