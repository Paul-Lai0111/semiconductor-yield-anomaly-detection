"""FastAPI batch-scoring service.

    uv pip install -r requirements-service.txt   # serving overlay, once
    uv run uvicorn fabops.api:app --port 8000

Endpoints
---------
GET  /health        liveness and whether a bundle is loaded
GET  /model         provenance of the frozen models and control limits
POST /score         JSON records -> ranked inspection queue
POST /score/csv     CSV upload -> ranked inspection queue
POST /drift         JSON records -> per-sensor drift summary vs training window

The bundle path comes from ``FABOPS_BUNDLE`` (default:
``fab_monitoring_results/artifacts/service_bundle.joblib``).
"""

import io
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from fabops.service import (
    DEFAULT_BUNDLE,
    POLICIES,
    batch_drift,
    load_bundle,
    score_batch,
)

MAX_ROWS = 10_000
Policy = Literal["v3_selected_knn", "mspc", "hybrid_knn_mspc"]


class Batch(BaseModel):
    records: list[dict[str, Any]] = Field(
        ..., min_length=1, max_length=MAX_ROWS,
        description="One object per unit: optional integer unit_id plus every sensor column; null = missing.",
    )


def _frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records).replace({None: np.nan})


def create_app(bundle: dict | None = None) -> FastAPI:
    app = FastAPI(
        title="Fab inspection-priority service",
        version="1.0.0",
        description="Ranks a completed batch for limited inspection capacity, with MSPC alarms and drift checks.",
    )
    state = {"bundle": bundle}

    def get_bundle() -> dict:
        if state["bundle"] is None:
            path = Path(os.environ.get("FABOPS_BUNDLE", DEFAULT_BUNDLE))
            if not path.is_file():
                raise HTTPException(503, f"No service bundle at {path}; run `python -m fabops.service build`.")
            state["bundle"] = load_bundle(path)
        return state["bundle"]

    def respond(frame: pd.DataFrame, review_fraction: float | None, policy: str) -> dict:
        bundle = get_bundle()
        try:
            queue = score_batch(bundle, frame, review_fraction, policy)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        return {
            "policy": policy,
            "rows": len(queue),
            "reviews": int(queue.review.sum()),
            "mspc_alarms": int(((queue.t2_alarm == 1) | (queue.spe_alarm == 1)).sum()),
            "score_meaning": "Uncalibrated priority score; larger means inspect sooner. Not a failure probability.",
            "queue": queue.to_dict(orient="records"),
        }

    @app.get("/health")
    def health():
        try:
            get_bundle()
            return {"status": "ok", "bundle_loaded": True}
        except HTTPException as error:
            return {"status": "degraded", "bundle_loaded": False, "detail": error.detail}

    @app.get("/model")
    def model():
        bundle = get_bundle()
        return {
            "policies": list(POLICIES),
            "default_review_fraction": bundle["default_review_fraction"],
            "sensor_columns": len(bundle["sensor_columns"]),
            "watchlist": bundle["watchlist"],
            **bundle["provenance"],
        }

    Fraction = Annotated[float | None, Query(gt=0, le=1, description="Share of the batch to inspect.")]

    @app.post("/score")
    def score(batch: Batch, review_fraction: Fraction = None, policy: Policy = "v3_selected_knn"):
        return respond(_frame(batch.records), review_fraction, policy)

    @app.post("/score/csv")
    async def score_csv(file: Annotated[UploadFile, File()], review_fraction: Fraction = None,
                        policy: Policy = "v3_selected_knn"):
        content = await file.read()
        try:
            frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
        except (ValueError, pd.errors.ParserError) as error:
            raise HTTPException(422, f"Unreadable CSV: {error}") from None
        if len(frame) > MAX_ROWS:
            raise HTTPException(413, f"At most {MAX_ROWS} rows per request.")
        return respond(frame, review_fraction, policy)

    @app.post("/drift")
    def drift(batch: Batch):
        bundle = get_bundle()
        try:
            return batch_drift(bundle, _frame(batch.records))
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    return app


app = create_app()
