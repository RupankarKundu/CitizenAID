# ── CitizenAid — Dockerfile (HuggingFace Spaces) ─────────────────────────────

# Stage 1: Install Python dependencies only
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# Stage 2: Runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY backend/  ./backend/
COPY data/     ./data/
COPY frontend/ ./frontend/

# Pre-download embedding model in FINAL stage
# (builder stage cache is not carried over — must do it here)
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
_ = m.encode(['test']); \
print('✅ Embedding model cached in final image')"

# Create chroma directory (NO VOLUME — HF Spaces doesn't support it)
RUN mkdir -p /app/chroma_db

ENV PYTHONUNBUFFERED=1

WORKDIR /app/backend

# HuggingFace Spaces uses port 7860
EXPOSE 7860

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]