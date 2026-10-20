from pathlib import Path
import os
import requests

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

LANGFLOW_URL = os.getenv("LANGFLOW_URL", "http://127.0.0.1:7860")
FLOW_ID = "bf7b2ea6-b57f-4370-ad52-3d21eee253cc"
LANGFLOW_API_KEY = os.getenv("LANGFLOW_API_KEY", "sk-uueLgzblHqrFFdlOrE7w8_R3BV8puK9-ZwDNHh99ioQ")
WHISPERX_API_URL = os.getenv("WHISPERX_API_URL", "http://127.0.0.1:8001/transcribe")

# Component that receives uploaded file path
TARGET_COMPONENT_ID = "WhisperXAPICaller-2punf"

# Your custom email component ID inside Langflow
EMAIL_COMPONENT_ID = "SendEmailWithAttachments-DCxix"

ALLOWED_EXTENSIONS = {
    ".mp4": "video/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".opus": "audio/opus",
}

ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/aac",
    "audio/ogg",
    "audio/webm",
    "audio/opus",
}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


def upload_file_to_langflow(temp_file_path: Path, original_name: str, mime_type: str) -> str:
    url = f"{LANGFLOW_URL}/api/v1/files/upload/{FLOW_ID}"
    headers = {
        "x-api-key": LANGFLOW_API_KEY,
    }

    with temp_file_path.open("rb") as f:
        files = {
            "file": (original_name, f, mime_type)
        }
        response = requests.post(url, headers=headers, files=files, timeout=300)

    if not response.ok:
        raise HTTPException(
            status_code=500,
            detail=f"Langflow upload failed: {response.status_code} {response.text}",
        )

    data = response.json()
    file_path = data.get("file_path")
    if not file_path:
        raise HTTPException(
            status_code=500,
            detail="No file_path returned from Langflow.",
        )

    return file_path


def resolve_mime_type(upload: UploadFile) -> str | None:
    extension = Path(upload.filename or "").suffix.lower()
    mime_type = ALLOWED_EXTENSIONS.get(extension)
    if mime_type:
        return mime_type

    normalized_content_type = (upload.content_type or "").split(";", maxsplit=1)[0].strip().lower()
    if normalized_content_type in ALLOWED_CONTENT_TYPES:
        return normalized_content_type

    return None


def run_flow(file_path: str, to_email: str) -> dict:
    url = f"{LANGFLOW_URL}/api/v1/run/{FLOW_ID}"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": LANGFLOW_API_KEY,
    }

    payload = {
        "input_value": "",
        "input_type": "chat",
        "output_type": "chat",
        "tweaks": {
            TARGET_COMPONENT_ID: {
                "path": file_path,
                "flow_id": FLOW_ID
            },
            EMAIL_COMPONENT_ID: {
                "to_email": to_email
            }
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=600)

    if not response.ok:
        raise HTTPException(
            status_code=500,
            detail=f"Langflow run failed: {response.status_code} {response.text}",
        )

    return response.json()


def extract_result_text(data: dict) -> str:
    try:
        return data["outputs"][0]["outputs"][0]["results"]["message"]["text"]
    except Exception:
        return str(data)


@app.post("/upload-and-run")
async def upload_and_run(
    file: UploadFile = File(...),
    to_email: str = Form(...),
):
    to_email = to_email.strip()

    if not to_email:
        raise HTTPException(status_code=400, detail="Recipient email is required.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    mime_type = resolve_mime_type(file)
    if not mime_type:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload MP4, MP3, WAV, M4A, AAC, OGG, WEBM, or OPUS.",
        )

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    safe_name = Path(file.filename).name
    temp_file_path = temp_dir / safe_name

    try:
        contents = await file.read()
        temp_file_path.write_bytes(contents)

        file_path = upload_file_to_langflow(temp_file_path, safe_name, mime_type)
        flow_response = run_flow(file_path, to_email)
        result_text = extract_result_text(flow_response)

        return JSONResponse({
            "ok": True,
            "file_path": file_path,
            "to_email": to_email,
            "result_text": result_text,
            "raw_langflow_response": flow_response
        })

    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()

@app.post("/transcribe-meeting")
async def transcribe_meeting(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    safe_name = Path(file.filename).name
    temp_file_path = temp_dir / safe_name

    try:
        contents = await file.read()
        temp_file_path.write_bytes(contents)

        with temp_file_path.open("rb") as f:
            files = {"file": (safe_name, f, file.content_type)}
            response = requests.post(WHISPERX_API_URL, files=files, timeout=600)

        if not response.ok:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Transcription failed: {response.text}",
            )

        return JSONResponse(response.json())

    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()
