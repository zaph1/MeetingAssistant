# Meeting Assistant

An AI-powered meeting assistant that transcribes audio, checks grammar, summarizes content, and performs fact-checking.

## Project Structure

```
MeetingAssistant/
├── whisperx_api.py          # WhisperX transcription API (main service)
├── Dockerfile               # Docker config for WhisperX service
├── docker-compose.yml       # Compose for running all services
├── requirements.txt
├── Grammar checker, Summarizer, Fact-Check (8).json  # Langflow backend pipeline
└── 164/                     # Web frontend
    ├── server.py            # Flask/backend server
    ├── templates/           # HTML templates
    ├── Dockerfile
    └── requirements.txt
```

## Components

### 🎙️ WhisperX Transcription API (`/` root)
Runs a WhisperX-powered speech-to-text REST API on port `8001`.

- **Model**: WhisperX with GPU (CUDA) or CPU support
- **Endpoint**: `POST /transcribe` — accepts audio file, returns transcript

### 🧠 Langflow Backend (`Grammar checker, Summarizer, Fact-Check (8).json`)
A [Langflow](https://github.com/logspace-ai/langflow) pipeline JSON that handles:
- **Grammar checking** of transcripts
- **Summarization** of meeting content
- **Fact-checking** of statements

Import this JSON into your Langflow instance to deploy the AI pipeline.

### 🌐 Web Frontend (`164/`)
A web interface that connects the transcription and Langflow pipeline together.

- **Server**: `server.py` (Flask)
- **UI**: `templates/index.html`
- Runs on port `8000` by default

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit (for CUDA mode)
- A running [Langflow](https://github.com/logspace-ai/langflow) instance with the pipeline JSON imported
- HuggingFace token (`HF_TOKEN`) for WhisperX diarization models

### Run with Docker Compose

```bash
docker compose up --build
```

### Run WhisperX API manually

**GPU (CUDA):**
```bash
docker build -t whisperx-api .
docker run --rm --gpus all -p 8001:8001 \
  -e WHISPERX_DEVICE=cuda \
  -e WHISPERX_MODEL=base \
  -e WHISPERX_COMPUTE_TYPE=float16 \
  -e HF_TOKEN=YOUR_TOKEN_HERE \
  whisperx-api
```

**CPU:**
```bash
docker run --rm -p 8001:8001 \
  -e WHISPERX_DEVICE=cpu \
  -e WHISPERX_MODEL=base \
  whisperx-api
```

### Run Web Frontend (`164/`)

```bash
cd 164
pip install -r requirements.txt
python server.py
```

---

## API Reference

### WhisperX — `POST /transcribe`
```bash
curl -X POST "http://localhost:8001/transcribe" -F "file=@meeting.mp3"
```

Returns a JSON transcript with word-level timestamps.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WHISPERX_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPERX_MODEL` | `base` | WhisperX model size |
| `WHISPERX_COMPUTE_TYPE` | `int8` | `float16`, `int8`, etc. |
| `HF_TOKEN` | — | HuggingFace token for diarization |
