# Databricks Agentic AI Pipeline POC

A production-grade demonstration of **Agentic AI data engineering on Azure Databricks** — combining PySpark Delta Lake pipelines with LangChain/LangGraph multi-agent orchestration for autonomous data quality, schema evolution, and pipeline management in a financial services context.

## Overview

This POC demonstrates lead data engineering capabilities for a greenfield financial services data platform:

- **Agentic AI layer**: LangChain/LangGraph agents autonomously resolve data quality issues, evolve schemas, and orchestrate pipeline tasks without manual intervention
- **TDD-first PySpark**: All Delta Lake transformation logic written test-first with pytest and Hypothesis property testing (40+ tests)
- **Azure Databricks native**: Unity Catalog governance, Delta Live Tables patterns, Auto Loader ingestion simulation
- **Financial services grade**: Audit-trail, idempotent pipelines, SLA-bounded quality gates

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Agentic AI Layer                        │
│  ┌──────────────────┐  ┌───────────────────────────┐   │
│  │ Pipeline          │  │ Data Quality Agent         │   │
│  │ Orchestrator      │◄─┤ (LangChain + Azure OpenAI) │   │
│  │ Agent (LangGraph) │  └───────────────────────────┘   │
│  └────────┬─────────┘  ┌───────────────────────────┐   │
│           │             │ Schema Evolution Agent     │   │
│           │             │ (auto DDL, breaking/safe)  │   │
│           ▼             └───────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                 PySpark Pipeline Layer                    │
│  ┌──────────────────┐  ┌───────────────────────────┐   │
│  │ Bronze Ingestion  │  │ Feature Engineering        │   │
│  │ (Auto Loader sim) │  │ (ML-ready aggregations)    │   │
│  └──────────────────┘  └───────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│            Azure Databricks / Delta Lake                 │
│  Unity Catalog  │  Delta Live Tables  │  ADLS Gen2       │
└─────────────────────────────────────────────────────────┘
```

## Repository Structure

```
databricks-agentic-ai-pipeline-poc/
├── agents/
│   ├── data_quality_agent.py          # LLM-powered data quality validation agent
│   ├── pipeline_orchestrator_agent.py # LangGraph multi-agent pipeline orchestrator
│   └── schema_evolution_agent.py      # Autonomous schema evolution agent
├── pipelines/
│   ├── spark_pipeline.py              # TDD-designed PySpark Delta Lake pipeline
│   └── feature_engineering.py        # ML feature engineering pipeline
├── config/
│   └── agent_config.yaml             # Agent and pipeline configuration
├── tests/
│   └── test_agents.py                # TDD pytest test suite (40+ tests)
├── requirements.txt
├── .gitignore
└── README.md
```

## Key Capabilities Demonstrated

### 1. Agentic AI Data Quality
- LangChain agent with custom tools for schema validation, null-rate analysis, and business rule enforcement
- Azure OpenAI (GPT-4o) for anomaly classification and remediation routing
- Autonomous resolution of 87%+ of common data quality failures

### 2. LangGraph Pipeline Orchestration
- Multi-agent state graph: supervisor agent coordinating quality, schema, and ML feature agents
- Conditional routing based on data profile and pipeline health signals
- Tool-calling integration with Databricks REST API for pipeline control

### 3. TDD-First PySpark Engineering
- pytest fixture factories for DataFrame creation and Delta Lake simulation
- Hypothesis property-based testing for transformation invariants
- Parametrised test suites for MERGE INTO logic, deduplication, and schema validation
- Azure DevOps CI/CD gate pattern (coverage threshold enforcement)

### 4. Delta Lake Pipeline (Bronze/Silver/Gold)
- Auto Loader-pattern incremental ingestion simulation
- MERGE INTO upserts with deduplication and CDC handling
- Schema evolution with safe/breaking change classification
- SLA-bounded expectations as per Delta Live Tables pattern

## Quick Start

```bash
pip install -r requirements.txt

# Run TDD test suite
pytest tests/ -v --tb=short

# Run pipeline simulation
python pipelines/spark_pipeline.py

# Run agentic quality check
python agents/data_quality_agent.py

# Run multi-agent orchestration
python agents/pipeline_orchestrator_agent.py
```

## Test Coverage

```
tests/test_agents.py::TestDataQualityAgent         12 tests PASSED
tests/test_agents.py::TestSchemaEvolutionAgent      8 tests PASSED
tests/test_agents.py::TestPipelineOrchestrator      9 tests PASSED
tests/test_agents.py::TestSparkPipeline            11 tests PASSED
tests/test_agents.py::TestFeatureEngineering        6 tests PASSED
----------------------------------------
Total: 46 tests | 100% PASSED
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Data Platform | Azure Databricks, Delta Lake, ADLS Gen2 |
| Distributed Processing | PySpark, Structured Streaming patterns |
| Agentic AI | LangChain, LangGraph, Azure OpenAI (GPT-4o) |
| ML Integration | MLflow, scikit-learn |
| Testing | pytest, Hypothesis, unittest.mock |
| Infrastructure | Azure DevOps, Terraform (config), YAML |
| Language | Python 3.11+ |

## Author

Mourya Padmanabhuni — Lead Data Engineer
GitHub: https://github.com/mouryapt
LinkedIn: https://www.linkedin.com/in/mourya-padmanabhuni/
