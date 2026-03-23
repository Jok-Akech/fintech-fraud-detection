"""
Kafka Streaming Pipeline for Real-Time Fraud Detection
Author: Jok Akech Atem Mabior

Consumes raw mobile-money transactions from a Kafka topic, runs them through
the Isolation Forest detector, and publishes FraudAlerts to an output topic.

Topics
------
  Input  : transactions.raw      (JSON-serialized transaction objects)
  Output : transactions.alerts   (JSON-serialized FraudAlert objects)
  Dead-Letter : transactions.dlq (malformed / unprocessable messages)

Usage
-----
  python -m src.pipeline --broker localhost:9092 --model models/isolation_forest.pkl
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from .detect import FraudDetector, Transaction, FraudAlert

# ── Logging ───────────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

MESSAGES_CONSUMED = Counter(
    "fraud_pipeline_messages_consumed_total",
    "Total Kafka messages consumed",
)
MESSAGES_FLAGGED = Counter(
    "fraud_pipeline_messages_flagged_total",
    "Total transactions flagged as fraudulent",
)
MESSAGES_ERRORED = Counter(
    "fraud_pipeline_messages_errored_total",
    "Total messages that failed processing",
)
PROCESSING_LATENCY = Histogram(
    "fraud_pipeline_processing_seconds",
    "End-to-end processing latency per message",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
ALERT_QUEUE_DEPTH = Gauge(
    "fraud_pipeline_alert_queue_depth",
    "Number of unacknowledged alerts in the output queue",
)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    broker: str = "localhost:9092"
    input_topic: str = "transactions.raw"
    output_topic: str = "transactions.alerts"
    dlq_topic: str = "transactions.dlq"
    consumer_group: str = "fraud-detection-v1"
    model_path: Path = Path("models/isolation_forest.pkl")
    metrics_port: int = 8000
    auto_offset_reset: str = "latest"
    max_poll_records: int = 500
    session_timeout_ms: int = 30_000
    heartbeat_interval_ms: int = 10_000
    fetch_max_bytes: int = 52_428_800  # 50 MB

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            broker=os.getenv("KAFKA_BROKER", "localhost:9092"),
            input_topic=os.getenv("KAFKA_INPUT_TOPIC", "transactions.raw"),
            output_topic=os.getenv("KAFKA_OUTPUT_TOPIC", "transactions.alerts"),
            dlq_topic=os.getenv("KAFKA_DLQ_TOPIC", "transactions.dlq"),
            consumer_group=os.getenv("KAFKA_CONSUMER_GROUP", "fraud-detection-v1"),
            model_path=Path(os.getenv("MODEL_PATH", "models/isolation_forest.pkl")),
            metrics_port=int(os.getenv("METRICS_PORT", "8000")),
        )


# ── Message serializers ───────────────────────────────────────────────────────


def _deserialize_transaction(raw: bytes) -> Transaction:
    """Parse raw Kafka bytes → Transaction dataclass."""
    data = json.loads(raw.decode("utf-8"))
    return Transaction(
        transaction_id=data["transaction_id"],
        sender_id=data["sender_id"],
        recipient_id=data["recipient_id"],
        amount=float(data["amount"]),
        currency=data.get("currency", "USD"),
        transaction_type=data.get("transaction_type", "TRANSFER"),
        timestamp=float(data.get("timestamp", time.time())),
        sender_balance_before=float(data.get("sender_balance_before", 0)),
        sender_balance_after=float(data.get("sender_balance_after", 0)),
        recipient_balance_before=float(data.get("recipient_balance_before", 0)),
        recipient_balance_after=float(data.get("recipient_balance_after", 0)),
        geo_distance_km=float(data.get("geo_distance_km", 0)),
        cross_border_flag=int(data.get("cross_border_flag", 0)),
        recipient_new_flag=int(data.get("recipient_new_flag", 0)),
    )


def _serialize_alert(alert: FraudAlert) -> bytes:
    return alert.to_json().encode("utf-8")


# ── Pipeline ──────────────────────────────────────────────────────────────────


class FraudDetectionPipeline:
    """
    Kafka → Isolation Forest → Kafka streaming pipeline.

    Lifecycle
    ---------
    1. start()      — connect to Kafka, warm up detector, begin polling
    2. _process()   — inner loop: consume → detect → produce
    3. stop()       — graceful shutdown on SIGTERM/SIGINT
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.detector: Optional[FraudDetector] = None
        self._consumer: Optional[KafkaConsumer] = None
        self._producer: Optional[KafkaProducer] = None
        self._running = False
        self._shutdown_event = threading.Event()
        self._stats = {
            "consumed": 0,
            "flagged": 0,
            "errors": 0,
            "start_time": 0.0,
        }

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _load_detector(self) -> None:
        log.info("loading_model", path=str(self.config.model_path))
        if self.config.model_path.exists():
            self.detector = FraudDetector.load(self.config.model_path)
        else:
            log.warning(
                "model_not_found",
                path=str(self.config.model_path),
                action="training_on_synthetic_data",
            )
            from .detect import generate_synthetic_transactions
            detector = FraudDetector()
            df, _ = generate_synthetic_transactions(n_legit=10_000, n_fraud=150)
            detector.train_from_dataframe(df)
            detector.save(self.config.model_path)
            self.detector = detector

    def _connect_kafka(self) -> None:
        log.info("connecting_kafka", broker=self.config.broker)
        retry_delay = 2
        for attempt in range(1, 6):
            try:
                self._consumer = KafkaConsumer(
                    self.config.input_topic,
                    bootstrap_servers=self.config.broker,
                    group_id=self.config.consumer_group,
                    auto_offset_reset=self.config.auto_offset_reset,
                    enable_auto_commit=False,
                    max_poll_records=self.config.max_poll_records,
                    session_timeout_ms=self.config.session_timeout_ms,
                    heartbeat_interval_ms=self.config.heartbeat_interval_ms,
                    fetch_max_bytes=self.config.fetch_max_bytes,
                    value_deserializer=None,  # manual deserialization
                )
                self._producer = KafkaProducer(
                    bootstrap_servers=self.config.broker,
                    value_serializer=None,
                    acks="all",
                    retries=5,
                    max_in_flight_requests_per_connection=1,
                )
                log.info("kafka_connected", attempt=attempt)
                return
            except NoBrokersAvailable:
                log.warning(
                    "kafka_unavailable",
                    attempt=attempt,
                    retry_in_seconds=retry_delay,
                )
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)
        raise RuntimeError(
            f"Could not connect to Kafka broker at {self.config.broker} after 5 attempts."
        )

    # ── Core processing loop ──────────────────────────────────────────────────

    def _process_message(self, raw_value: bytes) -> Optional[FraudAlert]:
        """Parse, detect, and return an alert (or None if benign)."""
        tx = _deserialize_transaction(raw_value)
        return self.detector.predict(tx)

    def _publish_alert(self, alert: FraudAlert) -> None:
        future = self._producer.send(
            self.config.output_topic,
            key=alert.transaction_id.encode("utf-8"),
            value=_serialize_alert(alert),
        )
        future.add_errback(
            lambda exc: log.error(
                "publish_failed",
                transaction_id=alert.transaction_id,
                error=str(exc),
            )
        )
        ALERT_QUEUE_DEPTH.inc()

    def _publish_dlq(self, raw: bytes, reason: str) -> None:
        payload = json.dumps({"reason": reason, "raw": raw.decode("utf-8", errors="replace")})
        self._producer.send(self.config.dlq_topic, value=payload.encode("utf-8"))

    def _run_loop(self) -> None:
        assert self._consumer is not None
        log.info("pipeline_started", topic=self.config.input_topic)

        while not self._shutdown_event.is_set():
            records = self._consumer.poll(timeout_ms=500)
            if not records:
                continue

            for _tp, messages in records.items():
                for msg in messages:
                    t0 = time.perf_counter()
                    MESSAGES_CONSUMED.inc()
                    self._stats["consumed"] += 1
                    try:
                        alert = self._process_message(msg.value)
                        if alert is not None:
                            self._publish_alert(alert)
                            MESSAGES_FLAGGED.inc()
                            self._stats["flagged"] += 1
                            log.info(
                                "fraud_detected",
                                transaction_id=alert.transaction_id,
                                risk=alert.risk_level,
                                score=alert.anomaly_score,
                                amount=alert.amount,
                            )
                    except (KeyError, ValueError, json.JSONDecodeError) as exc:
                        MESSAGES_ERRORED.inc()
                        self._stats["errors"] += 1
                        log.warning(
                            "message_parse_error",
                            error=str(exc),
                            offset=msg.offset,
                        )
                        self._publish_dlq(msg.value, str(exc))
                    finally:
                        elapsed = time.perf_counter() - t0
                        PROCESSING_LATENCY.observe(elapsed)

            try:
                self._consumer.commit()
            except KafkaError as exc:
                log.error("commit_failed", error=str(exc))

        log.info("pipeline_stopped")

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self, metrics_server: bool = True) -> None:
        if metrics_server:
            start_http_server(self.config.metrics_port)
            log.info("metrics_server_started", port=self.config.metrics_port)

        self._load_detector()
        self._connect_kafka()

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        self._stats["start_time"] = time.time()
        self._running = True
        self._run_loop()

    def stop(self) -> None:
        log.info("graceful_shutdown_initiated")
        self._shutdown_event.set()
        if self._producer:
            self._producer.flush(timeout=10)
            self._producer.close()
        if self._consumer:
            self._consumer.close()
        self._running = False
        self._log_final_stats()

    def _handle_signal(self, signum, _frame) -> None:
        log.info("signal_received", signal=signum)
        self.stop()
        sys.exit(0)

    def _log_final_stats(self) -> None:
        elapsed = time.time() - self._stats["start_time"]
        rate = self._stats["consumed"] / max(elapsed, 1)
        log.info(
            "final_stats",
            consumed=self._stats["consumed"],
            flagged=self._stats["flagged"],
            errors=self._stats["errors"],
            elapsed_seconds=round(elapsed, 1),
            throughput_msg_per_sec=round(rate, 2),
            flag_rate_pct=round(
                self._stats["flagged"] / max(self._stats["consumed"], 1) * 100, 3
            ),
        )

    # ── Health check ──────────────────────────────────────────────────────────

    def health(self) -> dict:
        elapsed = time.time() - self._stats["start_time"] if self._stats["start_time"] else 0
        return {
            "status": "running" if self._running else "stopped",
            "consumed": self._stats["consumed"],
            "flagged": self._stats["flagged"],
            "errors": self._stats["errors"],
            "uptime_seconds": round(elapsed, 1),
            "model_version": getattr(self.detector, "model_version", "N/A"),
        }


# ── Transaction producer (for testing / simulation) ──────────────────────────


class TransactionProducer:
    """
    Simulates a mobile-money transaction source by publishing synthetic
    transactions to the input Kafka topic. Useful for end-to-end testing.
    """

    def __init__(self, broker: str, topic: str = "transactions.raw"):
        self.broker = broker
        self.topic = topic
        self._producer = KafkaProducer(
            bootstrap_servers=broker,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def send(self, tx_dict: dict) -> None:
        self._producer.send(self.topic, value=tx_dict, key=tx_dict["transaction_id"].encode())

    def send_batch(self, tx_dicts: list[dict]) -> None:
        for tx in tx_dicts:
            self.send(tx)
        self._producer.flush()
        log.info("batch_sent", count=len(tx_dicts), topic=self.topic)

    def simulate(self, n: int = 1000, fraud_rate: float = 0.015, delay: float = 0.01) -> None:
        """Stream n synthetic transactions with a configurable fraud rate."""
        from .detect import generate_synthetic_transactions
        n_fraud = max(1, int(n * fraud_rate))
        df, _ = generate_synthetic_transactions(n_legit=n - n_fraud, n_fraud=n_fraud)
        records = df.to_dict(orient="records")
        log.info("simulation_starting", total=n, fraud_count=n_fraud, delay_ms=delay * 1000)
        for tx in records:
            self.send(tx)
            time.sleep(delay)
        self._producer.flush()
        log.info("simulation_complete", sent=n)

    def close(self) -> None:
        self._producer.close()


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud Detection Kafka Pipeline")
    parser.add_argument("--broker", default="localhost:9092", help="Kafka broker address")
    parser.add_argument("--model", default="models/isolation_forest.pkl", help="Path to model file")
    parser.add_argument("--input-topic", default="transactions.raw")
    parser.add_argument("--output-topic", default="transactions.alerts")
    parser.add_argument("--metrics-port", type=int, default=8000)
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run a producer simulation instead of the consumer pipeline",
    )
    parser.add_argument("--simulate-n", type=int, default=500)
    args = parser.parse_args()

    if args.simulate:
        producer = TransactionProducer(args.broker, args.input_topic)
        producer.simulate(n=args.simulate_n)
        producer.close()
    else:
        config = PipelineConfig(
            broker=args.broker,
            model_path=Path(args.model),
            input_topic=args.input_topic,
            output_topic=args.output_topic,
            metrics_port=args.metrics_port,
        )
        pipeline = FraudDetectionPipeline(config)
        pipeline.start()
