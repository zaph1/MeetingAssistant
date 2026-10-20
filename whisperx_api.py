from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import tempfile
import os
import whisperx
from whisperx.diarize import DiarizationPipeline
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv("WHISPERX_MODEL", "base")
DEVICE = os.getenv("WHISPERX_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "float16")


def _pick_device_and_compute_type(device: str, compute_type: str) -> tuple[str, str]:
    wanted = (device or "").strip().lower()
    if wanted == "cuda" and not torch.cuda.is_available():
        return "cpu", "int8"
    if wanted != "cuda":
        return "cpu", "int8"
    return "cuda", (compute_type or "float16")


@app.on_event("startup")
def _startup_load_models() -> None:
    device, compute_type = _pick_device_and_compute_type(DEVICE, COMPUTE_TYPE)
    app.state.whisperx_device = device
    app.state.whisperx_compute_type = compute_type

    app.state.whisperx_model = whisperx.load_model(MODEL_NAME, device, compute_type=compute_type)

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    app.state.whisperx_diarize = DiarizationPipeline(device=device, token=hf_token)
    app.state.align_models = {}
    app.state.align_metadata = {}


def _get_alignment_model(language_code: str, device: str):
    if language_code not in app.state.align_models:
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        app.state.align_models[language_code] = model_a
        app.state.align_metadata[language_code] = metadata
    return app.state.align_models[language_code], app.state.align_metadata[language_code]


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None)
):
    suffix = os.path.splitext(file.filename)[1] or ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        model = getattr(app.state, "whisperx_model", None)
        diarize = getattr(app.state, "whisperx_diarize", None)
        if model is None or diarize is None:
            raise HTTPException(status_code=500, detail="Models not loaded. Check server logs.")

        audio = whisperx.load_audio(temp_path)
        result = model.transcribe(audio)

        # 1. Align whisper output (crucial for highly accurate diarization)
        language = result.get("language", "en")
        device = getattr(app.state, "whisperx_device", "cpu")
        
        try:
            model_a, metadata = _get_alignment_model(language, device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        except Exception as e:
            print(f"Warning: Alignment failed: {e}")

        # 2. Diarize using Pyannote
        diarize_kwargs = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

        # Prefer in-memory audio (numpy array) to avoid decoder issues. Fall back to file path if needed.
        try:
            diarize_segments = diarize(audio, **diarize_kwargs)
        except Exception:
            diarize_segments = diarize(temp_path, **diarize_kwargs)
            
        # 3. Assign word speakers
        result = whisperx.assign_word_speakers(diarize_segments, result)

        segments = result.get("segments", []) or []
        
        # Post-process speaker labels from "SPEAKER_01" to "Person 1"
        for seg in segments:
            if "speaker" in seg and seg["speaker"].startswith("SPEAKER_"):
                try:
                    speaker_num = int(seg["speaker"].split("_")[1])
                    seg["speaker"] = f"Person {speaker_num}"
                except ValueError:
                    pass
            
            for word in seg.get("words", []):
                if "speaker" in word and word["speaker"].startswith("SPEAKER_"):
                    try:
                        speaker_num = int(word["speaker"].split("_")[1])
                        word["speaker"] = f"Person {speaker_num}"
                    except ValueError:
                        pass
        
        # Format text to include speaker labels
        formatted_segments = []
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            seg_text = seg.get("text", "").strip()
            formatted_segments.append(f"[{speaker}]: {seg_text}")
        
        diarized_text = "\n".join(formatted_segments)
        raw_text = " ".join(seg.get("text", "").strip() for seg in segments).strip()

        return {
            "success": True,
            "device": getattr(app.state, "whisperx_device", None),
            "compute_type": getattr(app.state, "whisperx_compute_type", None),
            "language": result.get("language"),
            "text": diarized_text,
            "raw_text": raw_text,
            "segments": segments,
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
