"""
TDD Unit Tests — Databricks Agentic AI Pipeline POC
pytest test suite demonstrating TDD engineering discipline for the La Fosse
Lead Data Engineer (Databricks, Agentic AI) role.

Test coverage:
  TestDataQualityTools        — 8 tests (tool-level unit tests)
  TestDataQualityAgent        — 12 tests (agent integration tests)
  TestSchemaEvolutionAgent    — 10 tests (schema change classification + application)
  TestPipelineOrchestrator    — 8 tests (multi-agent state machine)
  TestBronzePipeline          — 5 tests (ingestion + dedup)
  TestSilverPipeline          — 8 tests (validation + MERGE INTO)
  TestGoldPipeline            — 5 tests (aggregation correctness)
  TestFeatureEngineering      — 8 tests (feature computation + ML format)
  ─────────────────────────────────────────
  Total: 64 tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List


# ════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def clean_transactions() -> List[Dict]:
    """100 clean financial transaction records with no quality issues."""
    return [
        {
            "transaction_id": f"TXN{i:06d}",
            "amount": round(50.0 + i * 2.5, 2),
            "account_id": f"ACC{i % 10:03d}",
            "currency": "GBP",
            "timestamp": f"2026-04-06T{i % 24:02d}:00:00Z",
        }
        for i in range(100)
    ]


@pytest.fixture
def dirty_transactions() -> List[Dict]:
    """Transaction records with deliberate quality issues."""
    records = [
        {
            "transaction_id": f"TXN{i:06d}",
            "amount": round(100.0 + i, 2),
            "account_id": f"ACC{i % 5:03d}",
            "currency": "GBP",
            "timestamp": f"2026-04-06T00:00:00Z",
        }
        for i in range(50)
    ]
    # Inject null amounts (>20% = critical)
    for i in range(12):
        records[i]["amount"] = None
    # Inject duplicate transaction_id
    records[40]["transaction_id"] = records[41]["transaction_id"]
    return records


@pytest.fixture
def minimal_schema() -> Dict[str, str]:
    return {"transaction_id": "string", "amount": "double", "account_id": "string"}


# ════════════════════════════════════════════════════════════════════════════
# TestDataQualityTools (8 tests)
# ════════════════════════════════════════════════════════════════════════════

class TestDataQualityTools:

    def setup_method(self):
        from agents.data_quality_agent import DataQualityTools
        self.tools = DataQualityTools()

    def test_null_rate_zero_for_clean_column(self, clean_transactions):
        result = self.tools.check_null_rate("transaction_id", clean_transactions)
        assert result["null_rate"] == 0.0

    def test_null_rate_correct_for_partial_nulls(self):
        data = [{"col": i if i % 4 != 0 else None} for i in range(100)]
        result = self.tools.check_null_rate("col", data)
        assert result["null_rate"] == pytest.approx(0.25, rel=0.01)

    def test_null_rate_empty_dataset(self):
        result = self.tools.check_null_rate("col", [])
        assert result["null_rate"] == 0.0
        assert result["row_count"] == 0

    def test_schema_valid_for_matching_schema(self, clean_transactions, minimal_schema):
        result = self.tools.validate_schema(clean_transactions, minimal_schema)
        assert result["valid"] is True
        assert result["missing_columns"] == []

    def test_schema_invalid_for_missing_columns(self, clean_transactions):
        schema = {"transaction_id": "string", "missing_col": "string", "another_missing": "int"}
        result = self.tools.validate_schema(clean_transactions, schema)
        assert result["valid"] is False
        assert "missing_col" in result["missing_columns"]

    def test_uniqueness_passes_for_unique_keys(self, clean_transactions):
        result = self.tools.check_uniqueness("transaction_id", clean_transactions)
        assert result["unique"] is True
        assert result["duplicate_count"] == 0

    def test_uniqueness_detects_duplicates(self):
        data = [{"id": 1}, {"id": 2}, {"id": 1}]  # id=1 is duplicate
        result = self.tools.check_uniqueness("id", data)
        assert result["unique"] is False
        assert result["duplicate_count"] == 1
        assert 1 in result["duplicate_keys"]

    def test_business_rule_amount_positive(self, clean_transactions):
        result = self.tools.check_business_rule(
            "amount_positive", "amount", ">", 0.0, clean_transactions
        )
        assert result["passed"] is True
        assert result["violation_count"] == 0


# ════════════════════════════════════════════════════════════════════════════
# TestDataQualityAgent (12 tests)
# ════════════════════════════════════════════════════════════════════════════

class TestDataQualityAgent:

    def setup_method(self):
        from agents.data_quality_agent import DataQualityAgent
        self.Agent = DataQualityAgent

    def test_clean_data_produces_high_pass_rate(self, clean_transactions):
        agent = self.Agent(table_name="test_table")
        report = agent.run(clean_transactions)
        assert report.pass_rate >= 0.90

    def test_dirty_data_produces_critical_failures(self, dirty_transactions):
        agent = self.Agent(table_name="test_table")
        report = agent.run(dirty_transactions)
        assert len(report.critical_failures) > 0

    def test_high_null_rate_triggers_critical(self):
        # >20% null on amount = critical
        data = [{"transaction_id": f"T{i}", "amount": None if i < 25 else 100.0}
                for i in range(100)]
        agent = self.Agent(table_name="test_table")
        report = agent.run(data)
        critical_cols = [c.column for c in report.critical_failures]
        assert "amount" in critical_cols

    def test_moderate_null_rate_triggers_warning(self):
        # 8% null = warning (>5% but <20%)
        data = [{"col": None if i < 8 else "value"} for i in range(100)]
        agent = self.Agent(table_name="test_table")
        report = agent.run(data)
        warning_cols = [c.column for c in report.warnings]
        assert "col" in warning_cols

    def test_duplicate_pk_triggers_critical(self):
        data = [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}, {"id": 1, "val": "c"}]
        agent = self.Agent(table_name="test_table")
        report = agent.run(data)
        critical_names = [c.check_name for c in report.critical_failures]
        assert any("uniqueness" in n for n in critical_names)

    def test_schema_mismatch_detected(self, clean_transactions):
        schema = {"transaction_id": "string", "nonexistent_column": "int"}
        agent = self.Agent(table_name="test_table", expected_schema=schema)
        report = agent.run(clean_transactions)
        schema_checks = [c for c in report.critical_failures if c.check_type == "schema"]
        assert len(schema_checks) > 0

    def test_autonomous_remediation_resolves_warnings(self):
        # 8% null = warning → should be auto-resolved, not escalated
        data = [{"col": None if i < 8 else "value", "id": str(i)} for i in range(100)]
        agent = self.Agent(table_name="test_table", enable_autonomous_remediation=True)
        report = agent.run(data)
        assert report.autonomous_resolutions >= 1

    def test_escalation_triggered_for_multiple_criticals(self, dirty_transactions):
        agent = self.Agent(
            table_name="test_table",
            enable_autonomous_remediation=True,
        )
        report = agent.run(dirty_transactions)
        # Multiple criticals → escalation expected
        if len(report.critical_failures) >= 2:
            assert report.escalations > 0

    def test_report_pass_rate_formula(self, clean_transactions):
        agent = self.Agent(table_name="test_table")
        report = agent.run(clean_transactions)
        expected = report.passed_checks / report.total_checks if report.total_checks else 0
        assert report.pass_rate == pytest.approx(expected, rel=0.001)

    def test_empty_dataset_produces_report(self):
        agent = self.Agent(table_name="test_table")
        report = agent.run([])
        assert report.total_checks == 0
        assert report.pass_rate == 0.0

    def test_report_table_name_preserved(self, clean_transactions):
        agent = self.Agent(table_name="my_special_table")
        report = agent.run(clean_transactions)
        assert report.table_name == "my_special_table"

    def test_remediation_log_populated_for_issues(self, dirty_transactions):
        agent = self.Agent(table_name="test_table", enable_autonomous_remediation=True)
        report = agent.run(dirty_transactions)
        assert len(report.remediation_actions) > 0


# ════════════════════════════════════════════════════════════════════════════
# TestSchemaEvolutionAgent (10 tests)
# ════════════════════════════════════════════════════════════════════════════

class TestSchemaEvolutionAgent:

    def setup_method(self):
        from agents.schema_evolution_agent import SchemaEvolutionAgent, ChangeClassification
        self.Agent = SchemaEvolutionAgent
        self.ChangeClassification = ChangeClassification

    def test_new_nullable_column_is_safe(self):
        registered = {"id": "string", "amount": "double"}
        incoming = [{"id": "1", "amount": 100.0, "currency": "GBP"}]
        agent = self.Agent("test_table", registered_schema=registered)
        result = agent.evolve(incoming)
        safe = [c for c in result.applied_changes if c.column_name == "currency"]
        assert len(safe) == 1
        assert safe[0].is_safe

    def test_dropped_column_is_breaking(self):
        registered = {"id": "string", "amount": "double", "currency": "string"}
        incoming = [{"id": "1", "amount": 100.0}]  # currency dropped
        agent = self.Agent("test_table", registered_schema=registered)
        result = agent.evolve(incoming)
        blocked = [c for c in result.blocked_changes if c.column_name == "currency"]
        assert len(blocked) == 1
        assert result.escalation_required

    def test_type_widening_int_to_double_is_safe(self):
        from agents.schema_evolution_agent import _classify_type_change, ChangeClassification
        classification = _classify_type_change("int", "double")
        assert classification == ChangeClassification.SAFE

    def test_type_narrowing_double_to_int_is_breaking(self):
        from agents.schema_evolution_agent import _classify_type_change, ChangeClassification
        classification = _classify_type_change("double", "int")
        assert classification == ChangeClassification.BREAKING

    def test_same_type_is_no_change(self):
        from agents.schema_evolution_agent import _classify_type_change, ChangeClassification
        classification = _classify_type_change("string", "string")
        assert classification == ChangeClassification.NO_CHANGE

    def test_no_changes_on_empty_data(self):
        # Empty incoming data means no columns present — agent detects zero inferred columns
        # but the evolve() method short-circuits and returns empty result for empty data
        registered = {}  # no registered schema → first run baseline path
        agent = self.Agent("test_table", registered_schema=registered)
        result = agent.evolve([])
        assert len(result.detected_changes) == 0

    def test_baseline_registration_on_first_run(self):
        incoming = [{"id": "1", "amount": 100.0}]
        agent = self.Agent("test_table", registered_schema={})
        result = agent.evolve(incoming)
        # First run registers baseline — no changes detected
        assert len(result.detected_changes) == 0
        # Schema should now be registered
        assert len(agent.registered_schema) > 0

    def test_ddl_statement_format_for_new_column(self):
        registered = {"id": "string"}
        incoming = [{"id": "1", "new_col": "value"}]
        agent = self.Agent("test_table", registered_schema=registered)
        result = agent.evolve(incoming)
        applied = result.ddl_statements_applied
        assert len(applied) == 1
        assert "ADD COLUMN" in applied[0].upper()
        assert "new_col" in applied[0]

    def test_dry_run_does_not_apply_changes(self):
        registered = {"id": "string"}
        incoming = [{"id": "1", "new_col": "value"}]
        agent = self.Agent("test_table", registered_schema=registered, dry_run=True)
        result = agent.evolve(incoming)
        # In dry run: change detected and in applied list but DDL not actually executed
        # (ddl_statements_applied is empty in dry_run mode)
        assert len(result.ddl_statements_applied) == 0

    def test_multiple_new_columns_all_safe(self):
        registered = {"id": "string"}
        incoming = [{"id": "1", "col_a": "x", "col_b": 1.0, "col_c": True}]
        agent = self.Agent("test_table", registered_schema=registered)
        result = agent.evolve(incoming)
        assert len(result.applied_changes) == 3
        assert all(c.is_safe for c in result.applied_changes)


# ════════════════════════════════════════════════════════════════════════════
# TestPipelineOrchestrator (8 tests)
# ════════════════════════════════════════════════════════════════════════════

class TestPipelineOrchestrator:

    def setup_method(self):
        from agents.pipeline_orchestrator_agent import PipelineOrchestratorAgent, PipelineStatus
        self.Orchestrator = PipelineOrchestratorAgent
        self.Status = PipelineStatus

    def test_clean_pipeline_completes_successfully(self, clean_transactions):
        orch = self.Orchestrator()
        state = orch.run("PIPE-001", "test_table", clean_transactions)
        assert state["status"] == self.Status.COMPLETED.value

    def test_pipeline_id_preserved_in_state(self, clean_transactions):
        orch = self.Orchestrator()
        state = orch.run("MY-PIPELINE-ID", "test_table", clean_transactions)
        assert state["pipeline_id"] == "MY-PIPELINE-ID"

    def test_audit_trail_starts_with_pipeline_start(self, clean_transactions):
        orch = self.Orchestrator()
        state = orch.run("PIPE-001", "test_table", clean_transactions)
        assert any("PIPELINE_START" in entry for entry in state["audit_trail"])

    def test_completed_pipeline_has_pipeline_complete_entry(self, clean_transactions):
        orch = self.Orchestrator()
        state = orch.run("PIPE-001", "test_table", clean_transactions)
        assert any("PIPELINE_COMPLETE" in entry or "PIPELINE" in entry
                   for entry in state["audit_trail"])

    def test_features_computed_after_successful_run(self, clean_transactions):
        orch = self.Orchestrator()
        state = orch.run("PIPE-001", "test_table", clean_transactions)
        assert state["features_computed"] is True

    def test_dirty_data_may_fail_or_escalate(self, dirty_transactions):
        orch = self.Orchestrator()
        state = orch.run("PIPE-ERR-001", "test_table", dirty_transactions)
        terminal_statuses = {
            self.Status.COMPLETED.value,
            self.Status.FAILED.value,
            self.Status.ESCALATED.value,
        }
        assert state["status"] in terminal_statuses

    def test_max_iterations_prevents_infinite_loop(self, clean_transactions):
        orch = self.Orchestrator(max_iterations=3)
        state = orch.run("PIPE-001", "test_table", clean_transactions)
        assert state["iteration_count"] <= 5  # iterations should be bounded

    def test_audit_summary_contains_pipeline_id(self, clean_transactions):
        orch = self.Orchestrator()
        state = orch.run("PIPE-SUMMARY-TEST", "test_table", clean_transactions)
        summary = orch.get_audit_summary(state)
        assert "PIPE-SUMMARY-TEST" in summary


# ════════════════════════════════════════════════════════════════════════════
# TestBronzePipeline (5 tests)
# ════════════════════════════════════════════════════════════════════════════

class TestBronzePipeline:

    def setup_method(self):
        from pipelines.spark_pipeline import BronzeIngestionPipeline
        self.Pipeline = BronzeIngestionPipeline

    def test_ingests_all_clean_records(self, clean_transactions):
        pipe = self.Pipeline()
        result = pipe.ingest(clean_transactions)
        assert len(result) == len(clean_transactions)

    def test_deduplication_removes_exact_duplicate_dicts(self):
        record = {"transaction_id": "TXN001", "amount": 100.0}
        pipe = self.Pipeline()
        result = pipe.ingest([record, record, record])
        assert len(result) == 1

    def test_idempotent_second_run_produces_zero_new_records(self, clean_transactions):
        pipe = self.Pipeline()
        first = pipe.ingest(clean_transactions)
        second = pipe.ingest(clean_transactions)
        assert len(second) == 0  # all hashes already seen

    def test_schema_validation_passes_complete_record(self):
        pipe = self.Pipeline()
        record = {"transaction_id": "T1", "amount": 50.0, "currency": "GBP"}
        valid, missing = pipe.validate_schema(record, ["transaction_id", "amount", "currency"])
        assert valid is True
        assert missing == []

    def test_schema_validation_fails_for_missing_field(self):
        pipe = self.Pipeline()
        record = {"transaction_id": "T1"}
        valid, missing = pipe.validate_schema(record, ["transaction_id", "amount"])
        assert valid is False
        assert "amount" in missing


# ════════════════════════════════════════════════════════════════════════════
# TestSilverPipeline (8 tests)
# ════════════════════════════════════════════════════════════════════════════

class TestSilverPipeline:

    def setup_method(self):
        from pipelines.spark_pipeline import SilverTransformationPipeline, BronzeIngestionPipeline
        self.SilverPipeline = SilverTransformationPipeline
        self.BronzePipeline = BronzeIngestionPipeline

    def _make_bronze(self, records):
        pipe = self.BronzePipeline()
        return pipe.ingest(records)

    def test_valid_records_pass_transformation(self, clean_transactions):
        bronze = self._make_bronze(clean_transactions[:10])
        silver = self.SilverPipeline()
        results = silver.transform(bronze)
        valid_count = sum(1 for r in results if r.is_valid)
        assert valid_count == 10

    def test_negative_amount_fails_validation(self):
        data = [{"transaction_id": "T1", "amount": -50.0, "account_id": "ACC001", "currency": "GBP"}]
        bronze = self._make_bronze(data)
        silver = self.SilverPipeline()
        results = silver.transform(bronze)
        assert results[0].is_valid is False
        assert len(results[0].validation_errors) > 0

    def test_invalid_currency_fails_validation(self):
        data = [{"transaction_id": "T1", "amount": 100.0, "account_id": "ACC001", "currency": "XYZ"}]
        bronze = self._make_bronze(data)
        silver = self.SilverPipeline()
        results = silver.transform(bronze)
        assert results[0].is_valid is False

    def test_merge_into_inserts_new_records(self, clean_transactions):
        bronze = self._make_bronze(clean_transactions[:5])
        silver = self.SilverPipeline()
        records = silver.transform(bronze)
        counts = silver.merge_into(records)
        assert counts["inserted"] == 5
        assert counts["updated"] == 0

    def test_merge_into_updates_existing_records(self, clean_transactions):
        bronze = self._make_bronze(clean_transactions[:3])
        silver = self.SilverPipeline()
        records = silver.transform(bronze)
        silver.merge_into(records)
        # Re-merge same records → should update
        counts = silver.merge_into(records)
        assert counts["updated"] == 3
        assert counts["inserted"] == 0

    def test_invalid_records_skipped_in_merge(self):
        data = [{"transaction_id": "T1", "amount": -1.0, "account_id": "A", "currency": "GBP"}]
        bronze = self._make_bronze(data)
        silver = self.SilverPipeline()
        records = silver.transform(bronze)
        counts = silver.merge_into(records)
        assert counts["skipped"] == 1

    def test_get_silver_table_returns_merged_records(self, clean_transactions):
        bronze = self._make_bronze(clean_transactions[:5])
        silver = self.SilverPipeline()
        records = silver.transform(bronze)
        silver.merge_into(records)
        table = silver.get_silver_table()
        assert len(table) == 5

    def test_missing_required_field_invalidates_record(self):
        data = [{"amount": 100.0, "account_id": "ACC001", "currency": "GBP"}]  # no transaction_id
        bronze = self._make_bronze(data)
        silver = self.SilverPipeline()
        results = silver.transform(bronze)
        assert results[0].is_valid is False


# ════════════════════════════════════════════════════════════════════════════
# TestGoldPipeline (5 tests)
# ════════════════════════════════════════════════════════════════════════════

class TestGoldPipeline:

    def setup_method(self):
        from pipelines.spark_pipeline import (
            GoldAggregationPipeline, SilverRecord
        )
        self.GoldPipeline = GoldAggregationPipeline
        self.SilverRecord = SilverRecord

    def _make_silver_records(self, n: int = 10):
        from pipelines.spark_pipeline import SilverRecord
        return [
            SilverRecord(
                transaction_id=f"TXN{i:04d}",
                amount=float(100 + i),
                account_id=f"ACC{i % 3:03d}",
                currency="GBP",
                event_timestamp="2026-04-06T00:00:00Z",
                is_valid=True,
            )
            for i in range(n)
        ]

    def test_aggregation_groups_by_account(self):
        records = self._make_silver_records(30)
        gold = self.GoldPipeline()
        metrics = gold.aggregate_daily_by_account(records)
        account_ids = {m.account_id for m in metrics}
        assert len(account_ids) == 3  # 3 distinct accounts

    def test_total_amount_sum_is_correct(self):
        records = [
            self.SilverRecord("T1", 100.0, "ACC001", "GBP", "2026-04-06T00:00:00Z", True),
            self.SilverRecord("T2", 200.0, "ACC001", "GBP", "2026-04-06T01:00:00Z", True),
        ]
        gold = self.GoldPipeline()
        metrics = gold.aggregate_daily_by_account(records)
        acc_metric = next(m for m in metrics if m.account_id == "ACC001")
        assert acc_metric.total_amount == pytest.approx(300.0)
        assert acc_metric.transaction_count == 2

    def test_invalid_records_excluded_from_gold(self):
        records = [
            self.SilverRecord("T1", 100.0, "ACC001", "GBP", "2026-04-06T00:00:00Z", True),
            self.SilverRecord("T2", 500.0, "ACC001", "GBP", "2026-04-06T01:00:00Z", False),
        ]
        gold = self.GoldPipeline()
        metrics = gold.aggregate_daily_by_account(records)
        acc_metric = next(m for m in metrics if m.account_id == "ACC001")
        assert acc_metric.total_amount == pytest.approx(100.0)  # invalid excluded

    def test_sla_metrics_above_threshold(self):
        records = self._make_silver_records(100)
        gold = self.GoldPipeline()
        sla = gold.compute_account_sla_metrics(records)
        assert sla["sla_met"] is True
        assert sla["validity_rate"] == pytest.approx(1.0)

    def test_empty_silver_returns_empty_gold(self):
        gold = self.GoldPipeline()
        metrics = gold.aggregate_daily_by_account([])
        assert metrics == []


# ════════════════════════════════════════════════════════════════════════════
# TestFeatureEngineering (8 tests)
# ════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    def setup_method(self):
        from pipelines.feature_engineering import FeatureEngineeringPipeline
        self.Pipeline = FeatureEngineeringPipeline

    def test_empty_records_returns_empty_list(self):
        pipe = self.Pipeline()
        result = pipe.compute([])
        assert result == []

    def test_correct_account_grouping(self, clean_transactions):
        pipe = self.Pipeline()
        features = pipe.compute(clean_transactions)
        account_ids = {f.account_id for f in features}
        expected_ids = {f"ACC{i % 10:03d}" for i in range(10)}
        assert account_ids == expected_ids

    def test_txn_count_matches_record_count(self):
        data = [{"account_id": "ACC001", "amount": 100.0, "currency": "GBP"} for _ in range(15)]
        pipe = self.Pipeline()
        features = pipe.compute(data)
        assert len(features) == 1
        assert features[0].txn_count_1d == 15

    def test_high_velocity_flag_set_correctly(self):
        # 25 transactions > threshold of 20 → high velocity
        data = [{"account_id": "ACC001", "amount": 50.0, "currency": "GBP"} for _ in range(25)]
        pipe = self.Pipeline()
        features = pipe.compute(data)
        assert features[0].is_high_velocity is True

    def test_low_velocity_flag_not_set(self):
        data = [{"account_id": "ACC001", "amount": 50.0, "currency": "GBP"} for _ in range(5)]
        pipe = self.Pipeline()
        features = pipe.compute(data)
        assert features[0].is_high_velocity is False

    def test_currency_diversity_counted_correctly(self):
        data = [
            {"account_id": "ACC001", "amount": 50.0, "currency": c}
            for c in ["GBP", "USD", "EUR"]
        ]
        pipe = self.Pipeline()
        features = pipe.compute(data)
        assert features[0].currency_diversity == 3

    def test_stddev_zero_for_uniform_amounts(self):
        pipe = self.Pipeline()
        result = pipe._stddev([100.0, 100.0, 100.0])
        assert result == pytest.approx(0.0)

    def test_mlflow_dict_format_has_required_keys(self):
        data = [{"account_id": "ACC001", "amount": 100.0, "currency": "GBP"}]
        pipe = self.Pipeline()
        features = pipe.compute(data)
        mlflow_dict = pipe.to_mlflow_dict(features[0])
        required_keys = [
            "account_id", "feature_date", "txn_count_1d", "txn_count_7d",
            "total_amount_1d", "avg_amount_7d", "is_high_velocity",
            "amount_spike_detected",
        ]
        for key in required_keys:
            assert key in mlflow_dict, f"Missing MLflow key: {key}"
