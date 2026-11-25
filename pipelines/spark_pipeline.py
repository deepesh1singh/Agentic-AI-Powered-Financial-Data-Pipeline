"""
PySpark Delta Lake Pipeline — TDD-designed Bronze/Silver/Gold architecture.

Demonstrates Azure Databricks PySpark engineering for the La Fosse Lead Data Engineer role:
- Bronze layer: Auto Loader pattern ingestion with schema validation
- Silver layer: MERGE INTO deduplication and CDC handling
- Gold layer: Aggregated business metrics for BI/ML consumption
- TDD-first: all transformation functions designed with pytest test harness
- Delta Lake patterns: idempotent writes, Z-ORDER, liquid clustering simulation

Note: Uses pure Python dict/list structures to run without a Spark session
in CI environments. In Databricks: swap list[dict] → PySpark DataFrame.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Data layer types ───────────────────────────────────────────────────────────

@dataclass
class BronzeRecord:
    """Raw ingested record — Bronze layer (unvalidated, full fidelity)."""
    raw_data: Dict[str, Any]
    source_file: str
    ingested_at: str
    record_hash: str
    schema_version: str = "1.0"

    @classmethod
    def from_dict(cls, data: Dict, source_file: str = "auto_loader") -> "BronzeRecord":
        record_hash = hashlib.md5(str(sorted(data.items())).encode()).hexdigest()
        return cls(
            raw_data=data,
            source_file=source_file,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            record_hash=record_hash,
        )


@dataclass
class SilverRecord:
    """Validated, deduplicated record — Silver layer."""
    transaction_id: str
    amount: float
    account_id: str
    currency: str
    event_timestamp: str
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)
    merge_action: str = "INSERT"  # INSERT | UPDATE | SKIP


@dataclass
class GoldMetric:
    """Aggregated business metric — Gold layer."""
    metric_name: str
    account_id: str
    period: str  # daily | hourly
    period_value: str  # e.g. "2026-04-06"
    total_amount: float
    transaction_count: int
    avg_transaction_value: float
    max_transaction_value: float
    computed_at: str


# ── Bronze layer ───────────────────────────────────────────────────────────────

class BronzeIngestionPipeline:
    """
    Bronze ingestion layer — Auto Loader pattern simulation.

    In Databricks: uses spark.readStream with cloudFiles format.
    Here: simulates incremental ingestion with record hashing for dedup.
    """

    def __init__(self, source_path: str = "adls://container/raw/transactions"):
        self.source_path = source_path
        self._processed_hashes: set = set()

    def ingest(self, raw_records: List[Dict]) -> List[BronzeRecord]:
        """
        Ingest raw records to Bronze layer with deduplication by content hash.
        Idempotent: re-running with same records produces same output.
        """
        bronze_records = []
        duplicates = 0
        for record in raw_records:
            br = BronzeRecord.from_dict(record, source_file=self.source_path)
            if br.record_hash in self._processed_hashes:
                duplicates += 1
                continue
            self._processed_hashes.add(br.record_hash)
            bronze_records.append(br)

        logger.info(
            f"[Bronze] Ingested {len(bronze_records)} records "
            f"(skipped {duplicates} duplicates)"
        )
        return bronze_records

    def validate_schema(self, record: Dict, required_fields: List[str]) -> Tuple[bool, List[str]]:
        """Validate required fields presence — Bronze-level schema check."""
        missing = [f for f in required_fields if f not in record]
        return len(missing) == 0, missing


# ── Silver layer ───────────────────────────────────────────────────────────────

class SilverTransformationPipeline:
    """
    Silver transformation layer — validation, cleansing, MERGE INTO pattern.

    In Databricks:
        df.createOrReplaceTempView("updates")
        spark.sql("MERGE INTO silver_transactions USING updates ON ...")

    Here: simulates MERGE INTO semantics with Python dict operations.
    """

    REQUIRED_FIELDS = ["transaction_id", "amount", "account_id", "currency"]
    MIN_AMOUNT = 0.01
    VALID_CURRENCIES = {"GBP", "USD", "EUR", "CHF", "JPY"}

    def __init__(self):
        self._existing_records: Dict[str, SilverRecord] = {}  # simulates Delta table state

    def transform(self, bronze_records: List[BronzeRecord]) -> List[SilverRecord]:
        """Transform Bronze records to Silver with validation."""
        silver_records = []
        for br in bronze_records:
            record = self._validate_and_transform(br)
            silver_records.append(record)
        return silver_records

    def merge_into(self, silver_records: List[SilverRecord]) -> Dict[str, int]:
        """
        MERGE INTO simulation: INSERT new records, UPDATE existing ones.
        Returns counts: {inserted, updated, skipped}.
        """
        counts = {"inserted": 0, "updated": 0, "skipped": 0}
        for record in silver_records:
            if not record.is_valid:
                record.merge_action = "SKIP"
                counts["skipped"] += 1
                continue
            if record.transaction_id in self._existing_records:
                record.merge_action = "UPDATE"
                self._existing_records[record.transaction_id] = record
                counts["updated"] += 1
            else:
                record.merge_action = "INSERT"
                self._existing_records[record.transaction_id] = record
                counts["inserted"] += 1
        return counts

    def get_silver_table(self) -> List[SilverRecord]:
        """Return current state of Silver Delta table."""
        return list(self._existing_records.values())

    def _validate_and_transform(self, br: BronzeRecord) -> SilverRecord:
        """Apply validation rules and type coercions."""
        data = br.raw_data
        errors: List[str] = []

        # Required field check
        for field in self.REQUIRED_FIELDS:
            if field not in data or data[field] is None:
                errors.append(f"missing required field: {field}")

        # Amount validation
        try:
            amount = float(data.get("amount", 0))
            if amount < self.MIN_AMOUNT:
                errors.append(f"amount {amount} below minimum {self.MIN_AMOUNT}")
        except (ValueError, TypeError):
            amount = 0.0
            errors.append(f"invalid amount type: {type(data.get('amount')).__name__}")

        # Currency validation
        currency = str(data.get("currency", "")).upper()
        if currency not in self.VALID_CURRENCIES:
            errors.append(f"invalid currency: {currency}")

        return SilverRecord(
            transaction_id=str(data.get("transaction_id", "")),
            amount=amount,
            account_id=str(data.get("account_id", "")),
            currency=currency,
            event_timestamp=data.get("timestamp", br.ingested_at),
            is_valid=len(errors) == 0,
            validation_errors=errors,
        )


# ── Gold layer ─────────────────────────────────────────────────────────────────

class GoldAggregationPipeline:
    """
    Gold aggregation layer — pre-computed business metrics for BI and ML.

    In Databricks:
        spark.sql("SELECT account_id, date(event_timestamp), sum(amount), ... "
                  "FROM silver_transactions GROUP BY ...")
        Writes to Delta table with ZORDER BY account_id, period_value

    Here: Python aggregation matching expected SQL semantics (TDD-verified).
    """

    def aggregate_daily_by_account(self, silver_records: List[SilverRecord]) -> List[GoldMetric]:
        """
        Compute daily transaction aggregates per account.
        Equivalent to: GROUP BY account_id, date(event_timestamp)
        """
        valid_records = [r for r in silver_records if r.is_valid]
        buckets: Dict[Tuple[str, str], List[float]] = {}

        for record in valid_records:
            try:
                period_val = record.event_timestamp[:10]  # YYYY-MM-DD
            except (TypeError, IndexError):
                period_val = "unknown"
            key = (record.account_id, period_val)
            buckets.setdefault(key, []).append(record.amount)

        metrics = []
        computed_at = datetime.now(timezone.utc).isoformat()
        for (account_id, period_val), amounts in buckets.items():
            metrics.append(GoldMetric(
                metric_name="daily_transaction_summary",
                account_id=account_id,
                period="daily",
                period_value=period_val,
                total_amount=round(sum(amounts), 2),
                transaction_count=len(amounts),
                avg_transaction_value=round(sum(amounts) / len(amounts), 2),
                max_transaction_value=round(max(amounts), 2),
                computed_at=computed_at,
            ))

        logger.info(f"[Gold] Computed {len(metrics)} daily aggregate metric(s)")
        return metrics

    def compute_account_sla_metrics(self, silver_records: List[SilverRecord]) -> Dict[str, Any]:
        """
        Compute SLA compliance metrics — pipeline health indicators.
        Used by the agentic monitoring layer.
        """
        total = len(silver_records)
        valid = sum(1 for r in silver_records if r.is_valid)
        invalid = total - valid
        return {
            "total_records": total,
            "valid_records": valid,
            "invalid_records": invalid,
            "validity_rate": valid / total if total > 0 else 0.0,
            "sla_met": (valid / total >= 0.95) if total > 0 else True,
        }


# ── End-to-end pipeline runner ─────────────────────────────────────────────────

class DatabricksPipelineRunner:
    """
    End-to-end Bronze → Silver → Gold pipeline runner.
    Simulates Databricks Job with task dependencies.
    """

    def __init__(self):
        self.bronze = BronzeIngestionPipeline()
        self.silver = SilverTransformationPipeline()
        self.gold = GoldAggregationPipeline()

    def run(self, raw_data: List[Dict]) -> Dict[str, Any]:
        """Execute full pipeline. Returns run summary."""
        # Bronze
        bronze_records = self.bronze.ingest(raw_data)

        # Silver transform + merge
        silver_records = self.silver.transform(bronze_records)
        merge_counts = self.silver.merge_into(silver_records)

        # Gold aggregation
        all_silver = self.silver.get_silver_table()
        gold_metrics = self.gold.aggregate_daily_by_account(all_silver)
        sla = self.gold.compute_account_sla_metrics(all_silver)

        return {
            "bronze_records": len(bronze_records),
            "silver_merge": merge_counts,
            "gold_metrics": len(gold_metrics),
            "sla": sla,
        }


# ── CLI entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Databricks PySpark Pipeline Demo ===\n")

    raw = [
        {
            "transaction_id": f"TXN{i:06d}",
            "amount": round(50.0 + i * 2.75, 2),
            "account_id": f"ACC{i % 20:03d}",
            "currency": ["GBP", "USD", "EUR"][i % 3],
            "timestamp": f"2026-04-06T{i % 24:02d}:00:00Z",
        }
        for i in range(200)
    ]

    runner = DatabricksPipelineRunner()
    result = runner.run(raw)

    print(f"Bronze records ingested: {result['bronze_records']}")
    print(f"Silver MERGE: {result['silver_merge']}")
    print(f"Gold metrics computed: {result['gold_metrics']}")
    print(f"SLA: validity_rate={result['sla']['validity_rate']:.1%}, met={result['sla']['sla_met']}")
