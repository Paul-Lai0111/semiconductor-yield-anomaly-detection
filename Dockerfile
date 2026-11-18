# Fab inspection-priority service: FastAPI (default) or Streamlit dashboard.
#   docker build -t fab-monitor .
#   docker run --rm -p 8000:8000 fab-monitor                          # API docs: http://localhost:8000/docs
#   docker run --rm -p 8501:8501 fab-monitor streamlit run dashboard/app.py --server.address=0.0.0.0
FROM python:3.12-slim

# xgboost (used by the v2 reference pipeline) needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /app

# Frozen research lock first (byte-identical to the studies), then the pinned serving overlay.
COPY pyproject.toml uv.lock requirements-service.txt ./
RUN uv sync --locked --no-dev --no-install-project \
    && uv pip install --no-cache -r requirements-service.txt

COPY secom/ secom/
COPY smartfab/ smartfab/
COPY fabops/ fabops/
COPY dashboard/ dashboard/
COPY .streamlit/ .streamlit/
COPY manufacturing_results/selection.json manufacturing_results/metrics.json manufacturing_results/
COPY fab_monitoring_results/mspc_selection.json fab_monitoring_results/sensor_stability.csv fab_monitoring_results/

# Downloads the hash-checked UCI files and refits the frozen recipes on the development window.
RUN useradd --create-home app && chown -R app /app
USER app
RUN python -m fabops.service build

EXPOSE 8000 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "fabops.api:app", "--host", "0.0.0.0", "--port", "8000"]
