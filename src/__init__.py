"""
fintech-fraud-detection
=======================
Real-time Isolation Forest fraud detection for mobile money transactions.

Author: Jok Akech Atem Mabior
"""

from .detect import FraudDetector, FraudAlert, Transaction, FeatureEngineer
from .pipeline import FraudDetectionPipeline, PipelineConfig, TransactionProducer

__all__ = [
    "FraudDetector",
    "FraudAlert",
    "Transaction",
    "FeatureEngineer",
    "FraudDetectionPipeline",
    "PipelineConfig",
    "TransactionProducer",
]
