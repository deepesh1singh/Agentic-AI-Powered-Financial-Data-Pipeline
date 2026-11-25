"""
Pipeline Orchestrator Agent — LangGraph multi-agent state machine.

Demonstrates LangGraph-based agentic pipeline orchestration for the La Fosse
Lead Data Engineer (Databricks, Agentic AI) role:

- StateGraph with typed state channels
- Supervisor agent routing between: data_quality_agent, schema_agent, feature_agent
- Conditional edges based on quality report outcomes
- Databricks REST API integration pattern (mocked for local execution)
- Full audit trail via state accumulation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)


# ── State definition ──────────────────────────────────────────────────────────

class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    QUALITY_CHECK = "quality_check"
    SCHEMA_EVOLUTION = "schema_evolution"
    FEATURE_ENGINEERING = "feature_engineering"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class OrchestratorState(TypedDict):
    """LangGraph state channel — passed between agent nodes."""
    pipeline_id: str
    table_name: str
    input_data: List[Dict]
    status: str
    quality_passed: bool
    schema_valid: bool
    schema_changes_applied: List[str]
    features_computed: bool
    error_messages: List[str]
    audit_trail: List[str]
    escalation_required: bool
    autonomous_actions_taken: int
    iteration_count: int


# ── Agent nodes ────────────────────────────────────────────────────────────────

class SupervisorAgent:
    """
    Supervisor agent — decides next action based on current pipeline state.
    In production: LLM (GPT-4o) drives routing decisions using ReAct pattern.
    """

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations

    def decide_next_action(self, state: OrchestratorState) -> str:
        """
        Routing logic: maps state to next agent node.
        Production version uses LLM reasoning chain.
        """
        if state["iteration_count"] >= self.max_iterations:
            return "terminal"

        if state["escalation_required"]:
            return "terminal_escalate"

        current = state["status"]

        if current == PipelineStatus.PENDING.value:
            return "quality_check"
        if current == PipelineStatus.QUALITY_CHECK.value:
            if not state["quality_passed"]:
                return "terminal_failed"
            if not state["schema_valid"]:
                return "schema_evolution"
            return "feature_engineering"
        if current == PipelineStatus.SCHEMA_EVOLUTION.value:
            return "feature_engineering"
        if current == PipelineStatus.FEATURE_ENGINEERING.value:
            return "terminal_success"

        return "terminal_failed"


class QualityCheckNode:
    """Data quality agent node in the LangGraph state machine."""

    def __call__(self, state: OrchestratorState) -> OrchestratorState:
        from agents.data_quality_agent import DataQualityAgent

        logger.info(f"[QualityCheckNode] Running on {state['table_name']}")
        agent = DataQualityAgent(
            table_name=state["table_name"],
            enable_autonomous_remediation=True,
        )
        report = agent.run(state["input_data"])

        audit_entry = (
            f"QUALITY_CHECK: pass_rate={report.pass_rate:.1%}, "
            f"critical={len(report.critical_failures)}, "
            f"auto_resolved={report.autonomous_resolutions}"
        )

        new_state = dict(state)
        new_state["status"] = PipelineStatus.QUALITY_CHECK.value
        new_state["quality_passed"] = not report.requires_escalation and len(report.critical_failures) == 0
        new_state["escalation_required"] = report.requires_escalation
        new_state["autonomous_actions_taken"] += report.autonomous_resolutions
        new_state["audit_trail"] = state["audit_trail"] + [audit_entry]
        new_state["iteration_count"] = state["iteration_count"] + 1

        if not new_state["quality_passed"] and not new_state["escalation_required"]:
            new_state["error_messages"] = state["error_messages"] + [
                f"Quality check failed: {len(report.critical_failures)} critical issues"
            ]
        return new_state  # type: ignore[return-value]


class SchemaEvolutionNode:
    """Schema evolution agent node — applies safe schema changes autonomously."""

    def __call__(self, state: OrchestratorState) -> OrchestratorState:
        from agents.schema_evolution_agent import SchemaEvolutionAgent

        logger.info(f"[SchemaEvolutionNode] Checking schema for {state['table_name']}")

        # Detect schema from input data
        if not state["input_data"]:
            changes = []
        else:
            agent = SchemaEvolutionAgent(table_name=state["table_name"])
            changes = agent.detect_and_apply(state["input_data"])

        audit_entry = f"SCHEMA_EVOLUTION: applied {len(changes)} change(s): {changes}"
        new_state = dict(state)
        new_state["status"] = PipelineStatus.SCHEMA_EVOLUTION.value
        new_state["schema_valid"] = True
        new_state["schema_changes_applied"] = state["schema_changes_applied"] + changes
        new_state["autonomous_actions_taken"] += len(changes)
        new_state["audit_trail"] = state["audit_trail"] + [audit_entry]
        new_state["iteration_count"] = state["iteration_count"] + 1
        return new_state  # type: ignore[return-value]


class FeatureEngineeringNode:
    """ML feature engineering node — computes aggregated features for downstream ML."""

    def __call__(self, state: OrchestratorState) -> OrchestratorState:
        from pipelines.feature_engineering import FeatureEngineeringPipeline

        logger.info(f"[FeatureEngineeringNode] Computing features for {state['table_name']}")

        pipeline = FeatureEngineeringPipeline()
        features = pipeline.compute(state["input_data"])
        feature_count = len(features)

        audit_entry = f"FEATURE_ENGINEERING: computed {feature_count} feature record(s)"
        new_state = dict(state)
        new_state["status"] = PipelineStatus.FEATURE_ENGINEERING.value
        new_state["features_computed"] = True
        new_state["audit_trail"] = state["audit_trail"] + [audit_entry]
        new_state["iteration_count"] = state["iteration_count"] + 1
        return new_state  # type: ignore[return-value]


# ── Orchestrator ───────────────────────────────────────────────────────────────

class PipelineOrchestratorAgent:
    """
    LangGraph-pattern multi-agent pipeline orchestrator.

    Uses a state machine (mimicking LangGraph StateGraph) to coordinate:
    1. Data quality validation
    2. Schema evolution
    3. Feature engineering

    Each node updates shared state and the supervisor routes to the next node.
    In production: implemented as LangGraph StateGraph with LLM-driven supervisor.
    """

    def __init__(self, max_iterations: int = 10):
        self.supervisor = SupervisorAgent(max_iterations=max_iterations)
        self.nodes = {
            "quality_check": QualityCheckNode(),
            "schema_evolution": SchemaEvolutionNode(),
            "feature_engineering": FeatureEngineeringNode(),
        }

    def run(self, pipeline_id: str, table_name: str, input_data: List[Dict]) -> OrchestratorState:
        """
        Execute the agentic pipeline orchestration loop.
        Returns final state with full audit trail.
        """
        state: OrchestratorState = {
            "pipeline_id": pipeline_id,
            "table_name": table_name,
            "input_data": input_data,
            "status": PipelineStatus.PENDING.value,
            "quality_passed": False,
            "schema_valid": True,
            "schema_changes_applied": [],
            "features_computed": False,
            "error_messages": [],
            "audit_trail": [f"PIPELINE_START: {pipeline_id} on {table_name}"],
            "escalation_required": False,
            "autonomous_actions_taken": 0,
            "iteration_count": 0,
        }

        logger.info(f"[Orchestrator] Starting pipeline {pipeline_id}")

        while True:
            next_action = self.supervisor.decide_next_action(state)

            if next_action.startswith("terminal"):
                if next_action == "terminal_success":
                    state["status"] = PipelineStatus.COMPLETED.value
                    state["audit_trail"].append(
                        f"PIPELINE_COMPLETE: {state['autonomous_actions_taken']} autonomous actions"
                    )
                elif next_action == "terminal_escalate":
                    state["status"] = PipelineStatus.ESCALATED.value
                    state["audit_trail"].append("PIPELINE_ESCALATED: requires engineer review")
                else:
                    state["status"] = PipelineStatus.FAILED.value
                    state["audit_trail"].append("PIPELINE_FAILED: quality gate not met")
                break

            node = self.nodes.get(next_action)
            if node is None:
                state["status"] = PipelineStatus.FAILED.value
                break

            state = node(state)

        logger.info(f"[Orchestrator] Pipeline {pipeline_id} → {state['status']}")
        return state

    def get_audit_summary(self, state: OrchestratorState) -> str:
        """Format audit trail as human-readable summary."""
        lines = [f"Pipeline: {state['pipeline_id']}", f"Status: {state['status']}", ""]
        lines.extend(f"  {i+1}. {entry}" for i, entry in enumerate(state["audit_trail"]))
        lines.extend([
            "",
            f"Autonomous actions taken: {state['autonomous_actions_taken']}",
            f"Schema changes applied: {state['schema_changes_applied']}",
            f"Features computed: {state['features_computed']}",
        ])
        if state["error_messages"]:
            lines.append("\nErrors:")
            lines.extend(f"  • {e}" for e in state["error_messages"])
        return "\n".join(lines)


# ── CLI entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Pipeline Orchestrator Agent Demo ===\n")

    # Simulate clean financial data
    data = [
        {
            "transaction_id": f"TXN{i:06d}",
            "amount": round(100.0 + i * 1.5, 2),
            "account_id": f"ACC{i % 50:03d}",
            "currency": "GBP",
            "timestamp": f"2026-04-06T{i % 24:02d}:00:00Z",
        }
        for i in range(500)
    ]

    orchestrator = PipelineOrchestratorAgent(max_iterations=10)
    final_state = orchestrator.run(
        pipeline_id="PIPE-2026-04-06-001",
        table_name="financial_transactions_silver",
        input_data=data,
    )

    print(orchestrator.get_audit_summary(final_state))
