# syntax=docker/dockerfile:1.7
#
# One image, two processes (run as two containers, see README "Docker"): the API (`python -m uvicorn api.main:app`, needs the index, the models and the keys) and the
# Streamlit UI (the default CMD; a client of the API, it needs neither). The image contains the code and the pinned Python packages only:
#   - no API keys (pass them with --env-file at run time),
#   - no embedding / reranker models (mount your local HuggingFace cache; the app never downloads),
#   - no vector index (build it once inside the /data volume with scripts/ingest.py).
# Build and run commands: README.md, section "Docker".

# ---- builder: install the locked packages into a virtualenv ------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
# Optional: an approved index that serves CPU-only torch wheels. Without it the Linux torch wheel is the (very large) CUDA build.
ARG TORCH_INDEX_URL=""

WORKDIR /build
COPY requirements.lock ./

# The pip configuration (index URL and credentials) is passed as a BuildKit secret, so it is never written into an image layer.
# `required=true` makes the build fail instead of silently falling back to the public index.
RUN --mount=type=secret,id=pipconf,target=/etc/pip.conf,required=true \
    python -m venv /opt/venv \
 && grep -viE '^(pytest|pytest-cov|hypothesis|ruff)==' requirements.lock > runtime.lock \
 && if [ -n "$TORCH_INDEX_URL" ]; then \
      /opt/venv/bin/pip install --index-url "$TORCH_INDEX_URL" "$(grep -i '^torch==' runtime.lock)"; \
    fi \
 && /opt/venv/bin/pip install -r runtime.lock

# ---- runtime ------------------------------------------------------------------------------------------------------------------------
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    RAGAS_DO_NOT_TRACK=true \
    ADVRAG_INDEX_DIR=/data/chroma \
    CHROMA_LOCAL_PATH=/data/chroma

# /data  = the index volume (Chroma store, SQLite FTS5 index and ingest manifests)
# /models = the HuggingFace cache, bind-mounted from the host
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /data/runs/ingest /models /app/runs \
 && ln -s /data/runs/ingest /app/runs/ingest \
 && chown -R app:app /data /models /app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app config.py config.yaml schemas.py ./
COPY --chown=app:app app ./app
COPY --chown=app:app api ./api
COPY --chown=app:app app_data ./app_data
COPY --chown=app:app db ./db
COPY --chown=app:app rag ./rag
COPY --chown=app:app rag_core ./rag_core
COPY --chown=app:app eval ./eval
COPY --chown=app:app observability ./observability
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app utils ./utils

USER app
VOLUME ["/data"]
EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
