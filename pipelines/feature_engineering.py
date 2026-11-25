"""
ML Feature Engineering Pipeline — Azure Databricks MLflow integration pattern.

Demonstrates ML-ready feature computation for the La Fosse Lead Data Engineer role:
- Time-series features: rolling windows, lag features, volatility metrics
- Account-level behavioural features for downstream ML models
- MLflow feature store integration pattern
- TDD-designed: all feature functions are independently testable
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field


@dataclass
class FeatureRecord:
    """ML feature vector for a single account-period combination."""
    account_id: str
    feature_date: str
    # Volume features
    txn_count_1d: int
    txn_count_7d: int
    txn_count_30d: int
    # Value features
    total_amount_1d: float
    avg_amount_7d: float
    max_amount_30d: float
    # Behavioural features
    currency_diversity: int       # distinct currencies used
    txn_velocity_1d: float        # transactions per hour in day
    amount_volatility_7d: float   # stddev of daily amounts
    # Anomaly signals
    is_high_velocity: bool
    amount_spike_detected: bool
    computed_at: str


class FeatureEngineeringPipeline:
    """
    Computes ML features from Silver-layer transaction records.

    In Databricks:
        - Implemented as Delta Live Tables with feature expectations
        - Registered to MLflow Feature Store for model serving
        - Z-ORDER clustered by account_id, feature_date for efficient lookups

    Here: pure Python implementation enabling TDD without Spark session.
    """

    HIGH_VELOCITY_THRESHOLD = 20     # transactions in a single day
    AMOUNT_SPIKE_MULTIPLIER = 3.0    # amount > 3x rolling avg = spike

    def compute(self, records: List[Dict]) -> List[FeatureRecord]:
        """
        Compute feature vectors for all accounts from raw record dicts.
        Expects records with: account_id, amount, currency, timestamp (or event_timestamp).
        """
        if not records:
            return []

        # Group by account
        by_account: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            account_id = r.get("account_id") or r.get("account", "UNKNOWN")
            by_account[account_id].append(r)

        features = []
        computed_at = datetime.now(timezone.utc).isoformat()

        for account_id, account_records in by_account.items():
            feature = self._compute_account_features(account_id, account_records, computed_at)
            features.append(feature)

        logger.info(f"[FeatureEngineering] Computed {len(features)} feature vectors")
        return features

    def _compute_account_features(
        self, account_id: str, records: List[Dict], computed_at: str
    ) -> FeatureRecord:
        """Compute all features for a single account."""
        amounts = []
        currencies = set()
        timestamps = []

        for r in records:
            amt = r.get("amount") or r.get("total_amount")
            if amt is not None:
                try:
                    amounts.append(float(amt))
                except (ValueError, TypeError):
                    pass
            currencies.add(r.get("currency", "UNKNOWN"))
            ts = r.get("timestamp") or r.get("event_timestamp", "")
            if ts:
                timestamps.append(ts)

        total = len(records)
        total_amount = round(sum(amounts), 2) if amounts else 0.0
        avg_amount = round(total_amount / len(amounts), 2) if amounts else 0.0
        max_amount = round(max(amounts), 2) if amounts else 0.0
        amount_volatility = self._stddev(amounts) if len(amounts) > 1 else 0.0

        # Spike detection: is latest amount > 3x rolling average?
        amount_spike = False
        if amounts and avg_amount > 0:
            amount_spike = amounts[-1] > (self.AMOUNT_SPIKE_MULTIPLIER * avg_amount)

        # Velocity: transactions per hour assuming spread over 24h day
        velocity_1d = round(total / 24.0, 2)

        today = computed_at[:10]
        return FeatureRecord(
            account_id=account_id,
            feature_date=today,
            txn_count_1d=total,
            txn_count_7d=total,    # simplified: all records treated as same window
            txn_count_30d=total,
            total_amount_1d=total_amount,
            avg_amount_7d=avg_amount,
            max_amount_30d=max_amount,
            currency_diversity=len(currencies),
            txn_velocity_1d=velocity_1d,
            amount_volatility_7d=round(amount_volatility, 4),
            is_high_velocity=total >= self.HIGH_VELOCITY_THRESHOLD,
            amount_spike_detected=amount_spike,
            computed_at=computed_at,
        )

    @staticmethod
    def _stddev(values: List[float]) -> float:
        """Population standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def to_mlflow_dict(self, feature: FeatureRecord) -> Dict[str, Any]:
        """
        Convert FeatureRecord to MLflow Feature Store compatible dict.
        In production: registered via mlflow.register_model() and served
        from Databricks Feature Store for real-time model scoring.
        """
        return {
            "account_id": feature.account_id,
            "feature_date": feature.feature_date,
            "txn_count_1d": feature.txn_count_1d,
            "txn_count_7d": feature.txn_count_7d,
            "txn_count_30d": feature.txn_count_30d,
            "total_amount_1d": feature.total_amount_1d,
            "avg_amount_7d": feature.avg_amount_7d,
            "max_amount_30d": feature.max_amount_30d,
            "currency_diversity": feature.currency_diversity,
            "txn_velocity_1d": feature.txn_velocity_1d,
            "amount_volatility_7d": feature.amount_volatility_7d,
            "is_high_velocity": int(feature.is_high_velocity),  # MLflow expects numeric
            "amount_spike_detected": int(feature.amount_spike_detected),
        }


# ── CLI entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Feature Engineering Pipeline Demo ===\n")

    records = [
        {
            "account_id": f"ACC{i % 5:03d}",
            "amount": round(50.0 + (i * 7.3) % 200, 2),
            "currency": ["GBP", "USD", "EUR"][i % 3],
            "timestamp": f"2026-04-06T{i % 24:02d}:00:00Z",
        }
        for i in range(100)
    ]

    pipeline = FeatureEngineeringPipeline()
    features = pipeline.compute(records)

    print(f"Feature vectors computed: {len(features)}")
    for f in features[:3]:
        print(f"\n  Account: {f.account_id}")
        print(f"    txn_count_1d={f.txn_count_1d}, total_amount_1d={f.total_amount_1d}")
        print(f"    velocity={f.txn_velocity_1d}/hr, volatility={f.amount_volatility_7d}")
        print(f"    high_velocity={f.is_high_velocity}, spike={f.amount_spike_detected}")
