# syntax=docker/dockerfile:1.7
#
# One image, run by compose.yaml as two services: the API (`uvicorn api.main:app`, needs the index, the models and the keys) and the
# Streamlit UI (the default CMD; a client of the API, it needs none of them). `docker compose up --build` is the whole build.
# Packages come from public PyPI, at the versions in requirements.lock; torch comes from PyTorch's public CPU-only wheel index (no CUDA).
# No credentials are needed or accepted. The image holds code and packages only:
#   - no API keys (compose passes the named variables at run time),
#   - no embedding / reranker models (compose mounts the host's HuggingFace cache read-only; the app never downloads),
#   - no vector index (built once into the data volume: `docker compose run --rm ingest`).

# ---- builder: install the locked packages into a virtualenv ------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
# The index torch is installed from. The default serves CPU-only wheels for linux/amd64 and linux/arm64.
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

WORKDIR /build
COPY requirements.lock ./

# 1) the runtime lock = requirements.lock without the dev tools (CR stripped: the file may have Windows line endings)
# 2) torch first, alone, from the CPU index; its dependencies come from PyPI with everything else (a local "+cpu" version satisfies the pin)
# 3) fail the build if the environment is inconsistent, or if a CUDA torch or an NVIDIA package got in
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv \
 && tr -d '\r' < requirements.lock | grep -viE '^(pytest|pytest-cov|hypothesis|ruff)==' > runtime.lock \
 && /opt/venv/bin/pip install --no-deps --index-url "$TORCH_INDEX_URL" "$(grep -E '^torch==' runtime.lock)" \
 && /opt/venv/bin/pip install -r runtime.lock \
 && /opt/venv/bin/pip check \
 && /opt/venv/bin/python -c "import torch, sys; sys.exit(0 if torch.version.cuda is None else 'a CUDA build of torch was installed')" \
 && ! /opt/venv/bin/pip list --format=freeze | grep -iE '^nvidia'

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

# /data   = the index volume (Chroma store, SQLite FTS5 index, ingest manifests, feedback file)
# /models = the HuggingFace cache, bind-mounted from the host
# The two symlinks put what the code writes under the volume: runs/ingest (manifests) and ~/.local/share/advrag (feedback.sqlite3).
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /data/runs/ingest /data/advrag /models /app/runs /home/app/.local/share \
 && ln -s /data/runs/ingest /app/runs/ingest \
 && ln -s /data/advrag /home/app/.local/share/advrag \
 && chown -R app:app /data /models /app /home/app/.local

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
EXPOSE 8501 8000

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
