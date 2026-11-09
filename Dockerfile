# One container: FastAPI serves both the API and the built Next.js frontend.

# ---------- 1. Build the frontend into static files ----------
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# STATIC_EXPORT: see next.config.ts.
# NEXT_PUBLIC_API_URL is empty so the browser calls the API on the same address as the page.
ENV STATIC_EXPORT=1 NEXT_PUBLIC_API_URL= NEXT_TELEMETRY_DISABLED=1
RUN npm run build


# ---------- 2. Install the Python app and download the embedding model ----------
FROM python:3.13-slim AS backend
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never HF_HOME=/opt/huggingface
WORKDIR /app

# Dependencies first, so changing the code doesn't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN uv sync --frozen --no-dev --no-editable

# Included in the image, so the container starts without downloading it.
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# ---------- 3. Final image: only what's needed to run ----------
FROM python:3.13-slim
RUN useradd --system --uid 10001 --no-create-home app \
    && mkdir /data \
    && chown app:app /data

WORKDIR /app
COPY --from=backend /app/.venv .venv
COPY --from=backend /opt/huggingface /opt/huggingface
COPY --from=frontend /frontend/out frontend

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    FRONTEND_DIR=/app/frontend \
    USERS_DB=/data/users.db \
    UPLOAD_DIR=/data/uploads \
    QDRANT_PATH=/data/qdrant

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "agentic_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
