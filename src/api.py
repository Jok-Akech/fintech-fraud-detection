"""
FastAPI REST Interface for the Fraud Detection System
Author: Jok Akech Atem Mabior

Endpoints
---------
  GET  /              → welcome + links
  GET  /health        → liveness check + model status
  POST /predict       → score a single transaction
  POST /predict/batch → score up to 100 transactions
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .detect import FraudDetector, Transaction, generate_synthetic_transactions

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/isolation_forest.pkl")

# ── Shared state ──────────────────────────────────────────────────────────────

detector: Optional[FraudDetector] = None
_startup_time: float = 0.0


def _load_or_train() -> FraudDetector:
    """Load saved model; train on synthetic data if none exists."""
    if MODEL_PATH.exists():
        logger.info("Loading model from %s", MODEL_PATH)
        return FraudDetector.load(MODEL_PATH)

    logger.info("No saved model found — training on synthetic data…")
    d = FraudDetector()
    df, _ = generate_synthetic_transactions(n_legit=10_000, n_fraud=150)
    d.train_from_dataframe(df)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    d.save(MODEL_PATH)
    return d


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, _startup_time
    _startup_time = time.time()
    detector = _load_or_train()
    logger.info("Fraud detector ready (version %s)", detector.model_version)
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fintech Fraud Detection API",
    description=(
        "Real-time mobile money fraud detection powered by Isolation Forest. "
        "Author: Jok Akech Atem Mabior."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="TX0001234")
    sender_id: str = Field(..., example="S0042")
    recipient_id: str = Field(..., example="R0099")
    amount: float = Field(..., gt=0, example=4500.00)
    currency: str = Field("USD", example="USD")
    transaction_type: str = Field("TRANSFER", example="TRANSFER")
    timestamp: Optional[float] = Field(None, example=1711234567.0)
    sender_balance_before: float = Field(0.0, example=9000.0)
    sender_balance_after: float = Field(0.0, example=4500.0)
    recipient_balance_before: float = Field(0.0, example=200.0)
    recipient_balance_after: float = Field(0.0, example=4700.0)
    geo_distance_km: float = Field(0.0, ge=0, example=340.5)
    cross_border_flag: int = Field(0, ge=0, le=1, example=0)
    recipient_new_flag: int = Field(0, ge=0, le=1, example=1)

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("amount must be positive")
        return v

    def to_transaction(self) -> Transaction:
        return Transaction(
            transaction_id=self.transaction_id,
            sender_id=self.sender_id,
            recipient_id=self.recipient_id,
            amount=self.amount,
            currency=self.currency,
            transaction_type=self.transaction_type,
            timestamp=self.timestamp or time.time(),
            sender_balance_before=self.sender_balance_before,
            sender_balance_after=self.sender_balance_after,
            recipient_balance_before=self.recipient_balance_before,
            recipient_balance_after=self.recipient_balance_after,
            geo_distance_km=self.geo_distance_km,
            cross_border_flag=self.cross_border_flag,
            recipient_new_flag=self.recipient_new_flag,
        )


class PredictResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    risk_level: str
    fraud_probability: Optional[float]
    anomaly_score: Optional[float]
    triggered_features: list[str]
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    results: list[PredictResponse]
    total: int
    flagged: int
    latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "Fintech Fraud Detection API",
        "author": "Jok Akech Atem Mabior",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_version": detector.model_version,
        "threshold": detector.threshold,
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: TransactionRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    tx = request.to_transaction()
    alert = detector.predict(tx)
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    if alert:
        return PredictResponse(
            transaction_id=tx.transaction_id,
            is_fraud=True,
            risk_level=alert.risk_level,
            fraud_probability=alert.fraud_probability,
            anomaly_score=alert.anomaly_score,
            triggered_features=alert.triggered_features,
            model_version=detector.model_version,
            latency_ms=latency_ms,
        )
    else:
        score = detector.score(tx)
        return PredictResponse(
            transaction_id=tx.transaction_id,
            is_fraud=False,
            risk_level="NONE",
            fraud_probability=None,
            anomaly_score=round(score, 6),
            triggered_features=[],
            model_version=detector.model_version,
            latency_ms=latency_ms,
        )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(request.transactions) > 100:
        raise HTTPException(status_code=422, detail="Maximum batch size is 100")

    t0 = time.perf_counter()
    results = []
    for req in request.transactions:
        tx = req.to_transaction()
        alert = detector.predict(tx)
        if alert:
            results.append(PredictResponse(
                transaction_id=tx.transaction_id,
                is_fraud=True,
                risk_level=alert.risk_level,
                fraud_probability=alert.fraud_probability,
                anomaly_score=alert.anomaly_score,
                triggered_features=alert.triggered_features,
                model_version=detector.model_version,
                latency_ms=0.0,
            ))
        else:
            score = detector.score(tx)
            results.append(PredictResponse(
                transaction_id=tx.transaction_id,
                is_fraud=False,
                risk_level="NONE",
                fraud_probability=None,
                anomaly_score=round(score, 6),
                triggered_features=[],
                model_version=detector.model_version,
                latency_ms=0.0,
            ))

    total_ms = round((time.perf_counter() - t0) * 1000, 3)
    flagged = sum(1 for r in results if r.is_fraud)
    return BatchResponse(
        results=results,
        total=len(results),
        flagged=flagged,
        latency_ms=total_ms,
    )
