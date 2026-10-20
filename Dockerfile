FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

COPY whisperx_api.py .

EXPOSE 8001

CMD ["uvicorn", "whisperx_api:app", "--host", "0.0.0.0", "--port", "8001"]
