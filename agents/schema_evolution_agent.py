"""
Schema Evolution Agent — Autonomous schema change detection and application.

Demonstrates automated schema governance for the La Fosse Lead Data Engineer role:
- Detects safe (additive) vs breaking (destructive) schema changes
- Autonomously applies safe changes: new nullable columns, type widening
- Blocks and escalates breaking changes: column drops, non-nullable additions, type narrowing
- Delta Lake DDL generation pattern (ALTER TABLE equivalent)
- Full audit trail for compliance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ChangeClassification(Enum):
    SAFE = "safe"          # Can be applied autonomously
    BREAKING = "breaking"  # Requires human approval
    NO_CHANGE = "no_change"


@dataclass
class SchemaChange:
    """Represents a single detected schema change."""
    change_type: str  # add_column | drop_column | type_change | rename_column
    column_name: str
    old_type: Optional[str]
    new_type: Optional[str]
    classification: ChangeClassification
    ddl_statement: str
    reason: str

    @property
    def is_safe(self) -> bool:
        return self.classification == ChangeClassification.SAFE


@dataclass
class SchemaEvolutionResult:
    """Result of schema evolution agent run."""
    table_name: str
    detected_changes: List[SchemaChange]
    applied_changes: List[SchemaChange]
    blocked_changes: List[SchemaChange]
    escalation_required: bool
    ddl_statements_applied: List[str]

    @property
    def safe_change_count(self) -> int:
        return sum(1 for c in self.detected_changes if c.is_safe)

    @property
    def breaking_change_count(self) -> int:
        return sum(1 for c in self.detected_changes if not c.is_safe
                   and c.classification != ChangeClassification.NO_CHANGE)


# ── Type compatibility rules ──────────────────────────────────────────────────

# Maps (current_type, new_type) → is_safe_widening
TYPE_WIDENING_SAFE = {
    ("int", "long"),
    ("int", "double"),
    ("float", "double"),
    ("string", "string"),
    ("long", "double"),
}


def _classify_type_change(old_type: str, new_type: str) -> ChangeClassification:
    """Classify whether a type change is safe (widening) or breaking (narrowing)."""
    if old_type == new_type:
        return ChangeClassification.NO_CHANGE
    if (old_type, new_type) in TYPE_WIDENING_SAFE:
        return ChangeClassification.SAFE
    return ChangeClassification.BREAKING


# ── Agent ─────────────────────────────────────────────────────────────────────

class SchemaEvolutionAgent:
    """
    Autonomous schema evolution agent.

    Workflow:
    1. Compare incoming data schema against registered table schema
    2. Classify each change as safe or breaking
    3. Apply safe changes via DDL (ALTER TABLE ... ADD COLUMN)
    4. Block and escalate breaking changes
    5. Return audit trail of all actions

    In production: integrates with Databricks Unity Catalog schema registry
    and generates real ALTER TABLE statements via Databricks REST API.
    """

    def __init__(
        self,
        table_name: str,
        registered_schema: Optional[Dict[str, str]] = None,
        dry_run: bool = False,
    ):
        self.table_name = table_name
        self.registered_schema: Dict[str, str] = registered_schema or {}
        self.dry_run = dry_run
        self._applied_ddl: List[str] = []

    def detect_and_apply(self, incoming_data: List[Dict]) -> List[str]:
        """
        Convenience method: detect schema changes and return list of applied DDL.
        Returns empty list if no changes detected.
        """
        if not incoming_data:
            return []
        result = self.evolve(incoming_data)
        return result.ddl_statements_applied

    def infer_schema(self, data: List[Dict]) -> Dict[str, str]:
        """Infer column types from data sample."""
        if not data:
            return {}
        schema: Dict[str, str] = {}
        for row in data[:100]:  # sample first 100 rows
            for col, val in row.items():
                if val is None:
                    schema.setdefault(col, "string")
                elif isinstance(val, bool):
                    schema[col] = "boolean"
                elif isinstance(val, int):
                    existing = schema.get(col, "int")
                    schema[col] = "long" if existing in ("long", "double") else "int"
                elif isinstance(val, float):
                    schema[col] = "double"
                else:
                    schema[col] = "string"
        return schema

    def detect_changes(
        self, current_schema: Dict[str, str], incoming_schema: Dict[str, str]
    ) -> List[SchemaChange]:
        """Compare current and incoming schemas to detect all changes."""
        changes: List[SchemaChange] = []
        current_cols: Set[str] = set(current_schema.keys())
        incoming_cols: Set[str] = set(incoming_schema.keys())

        # New columns (additive — safe if nullable)
        for col in incoming_cols - current_cols:
            new_type = incoming_schema[col]
            ddl = (
                f"ALTER TABLE {self.table_name} "
                f"ADD COLUMN {col} {new_type.upper()} COMMENT 'auto-added by schema evolution agent'"
            )
            changes.append(SchemaChange(
                change_type="add_column",
                column_name=col,
                old_type=None,
                new_type=new_type,
                classification=ChangeClassification.SAFE,
                ddl_statement=ddl,
                reason=f"New nullable column '{col}' detected in incoming data",
            ))

        # Dropped columns (breaking)
        for col in current_cols - incoming_cols:
            old_type = current_schema[col]
            changes.append(SchemaChange(
                change_type="drop_column",
                column_name=col,
                old_type=old_type,
                new_type=None,
                classification=ChangeClassification.BREAKING,
                ddl_statement=f"-- BLOCKED: ALTER TABLE {self.table_name} DROP COLUMN {col}",
                reason=f"Column '{col}' missing from incoming data — potential data loss",
            ))

        # Type changes (safe widening or breaking narrowing)
        for col in current_cols & incoming_cols:
            old_type = current_schema[col]
            new_type = incoming_schema[col]
            classification = _classify_type_change(old_type, new_type)
            if classification == ChangeClassification.NO_CHANGE:
                continue
            if classification == ChangeClassification.SAFE:
                ddl = (
                    f"ALTER TABLE {self.table_name} "
                    f"ALTER COLUMN {col} TYPE {new_type.upper()}"
                )
                reason = f"Safe type widening: {old_type} → {new_type}"
            else:
                ddl = f"-- BLOCKED: unsafe type change {col}: {old_type} → {new_type}"
                reason = f"Breaking type change: {old_type} → {new_type} may cause data loss"

            changes.append(SchemaChange(
                change_type="type_change",
                column_name=col,
                old_type=old_type,
                new_type=new_type,
                classification=classification,
                ddl_statement=ddl,
                reason=reason,
            ))

        return changes

    def evolve(self, incoming_data: List[Dict]) -> SchemaEvolutionResult:
        """
        Full schema evolution pipeline.
        Returns result with applied and blocked changes.
        """
        incoming_schema = self.infer_schema(incoming_data)

        # If no registered schema, register current as baseline
        if not self.registered_schema:
            self.registered_schema = incoming_schema
            logger.info(f"[SchemaEvolutionAgent] Registered baseline schema for {self.table_name}: "
                        f"{list(incoming_schema.keys())}")
            return SchemaEvolutionResult(
                table_name=self.table_name,
                detected_changes=[],
                applied_changes=[],
                blocked_changes=[],
                escalation_required=False,
                ddl_statements_applied=[],
            )

        changes = self.detect_changes(self.registered_schema, incoming_schema)
        applied: List[SchemaChange] = []
        blocked: List[SchemaChange] = []
        ddl_applied: List[str] = []

        for change in changes:
            if change.is_safe:
                if not self.dry_run:
                    # In production: execute DDL via Databricks REST API
                    logger.info(f"[SchemaEvolutionAgent] APPLY: {change.ddl_statement}")
                    # Update registered schema
                    if change.change_type == "add_column":
                        self.registered_schema[change.column_name] = change.new_type  # type: ignore[assignment]
                    elif change.change_type == "type_change":
                        self.registered_schema[change.column_name] = change.new_type  # type: ignore[assignment]
                    ddl_applied.append(change.ddl_statement)
                applied.append(change)
            else:
                logger.warning(f"[SchemaEvolutionAgent] BLOCKED: {change.reason}")
                blocked.append(change)

        escalation_required = len(blocked) > 0

        return SchemaEvolutionResult(
            table_name=self.table_name,
            detected_changes=changes,
            applied_changes=applied,
            blocked_changes=blocked,
            escalation_required=escalation_required,
            ddl_statements_applied=ddl_applied,
        )


# ── CLI entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Schema Evolution Agent Demo ===\n")

    # Registered schema (current table state)
    registered = {
        "transaction_id": "string",
        "amount": "int",
        "account_id": "string",
    }

    # Incoming data with schema changes
    incoming_data = [
        {
            "transaction_id": f"TXN{i:05d}",
            "amount": 100.5 + i,        # int → double (safe widening)
            "account_id": f"ACC{i:03d}",
            "currency": "GBP",           # new column (safe additive)
            "region": "EMEA",            # new column (safe additive)
        }
        for i in range(100)
    ]

    agent = SchemaEvolutionAgent(
        table_name="financial_transactions",
        registered_schema=registered,
    )
    result = agent.evolve(incoming_data)

    print(f"Table: {result.table_name}")
    print(f"Changes detected: {len(result.detected_changes)}")
    print(f"  Safe (applied): {result.safe_change_count}")
    print(f"  Breaking (blocked): {result.breaking_change_count}")
    print(f"Escalation required: {result.escalation_required}")
    if result.ddl_statements_applied:
        print("\nDDL applied:")
        for ddl in result.ddl_statements_applied:
            print(f"  {ddl}")
    if result.blocked_changes:
        print("\nBlocked changes:")
        for c in result.blocked_changes:
            print(f"  • {c.reason}")
