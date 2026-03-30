FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY worker/handler.py .
COPY worker/qwen3_runner.py .

# HuggingFace cache path — RunPod Flashboot snapshots this layer after first run.
# First cold start downloads Qwen3-TTS (~3.5GB). All subsequent cold starts restore
# from the Flashboot snapshot — no re-download, no Network Volume needed.
ENV HF_HOME=/root/.cache/huggingface

CMD ["python", "-u", "handler.py"]
