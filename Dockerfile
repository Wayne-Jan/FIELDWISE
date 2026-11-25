FROM python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /workspace/FIELDWISE
RUN apt-get update && apt-get install -y --no-install-recommends fonts-liberation && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# CUDA build of torch first; requirements.txt (also used by the CPU-only CI) then finds torch 2.5.1 installed
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
