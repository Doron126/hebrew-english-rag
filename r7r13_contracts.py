"""Source-frozen R7R13 topology and future external-authority contracts."""
from __future__ import annotations

import hashlib
import json
from typing import Any


DIMENSIONS = (
    "C45_IMPLEMENTATION_PERFORMED",
    "C45_DIAGNOSTIC_EXECUTION_COUNT",
    "REAL_SUBJECT_EXECUTION_COUNT",
    "PROSPECTIVE_WITNESS_EXECUTION_COUNT",
    "CANDIDATE_STARTED",
    "BENCHMARK_TRAFFIC_SENT",
    "SCORER_RERUN",
    "PRODUCTION_CHANGED",
    "C38_RUNTIME_IDENTITY_CHANGED",
)

# The source-static topology names the lowest repository primitive that can
# perform a governed action. Non-governed helpers (health GETs, pure scoring,
# and ordinary non-candidate application requests) are intentionally absent.
SURFACE_TOPOLOGY: tuple[dict[str, Any], ...] = (
    {"surface_id": "candidate-launch", "dimensions": ("CANDIDATE_STARTED",), "entry_point": "tools/launch_isolated_candidate.py:main", "primitive": "tools.launch_isolated_candidate.execute_candidate_launch", "classifier": "always_candidate_launch", "callers": ("main",), "dynamic_callers": (), "direct_bypass": ()},
    {"surface_id": "benchmark-runner", "dimensions": ("BENCHMARK_TRAFFIC_SENT",), "entry_point": "tools/run_real_corpus_baseline.py:main", "primitive": "tools.run_real_corpus_baseline._run_case", "classifier": "post_chat_case_dispatch", "callers": ("main",), "dynamic_callers": (), "direct_bypass": ()},
    {"surface_id": "scorer", "dimensions": ("SCORER_RERUN",), "entry_point": "tools/score_real_corpus_results.py:main", "primitive": "tools.score_real_corpus_results._emit_score_reports", "classifier": "score_report_emission", "callers": ("main",), "dynamic_callers": (), "direct_bypass": ()},
    {"surface_id": "eval-runner-request", "dimensions": ("REAL_SUBJECT_EXECUTION_COUNT",), "entry_point": "eval_runner.py:main", "primitive": "eval_runner.ask_api", "classifier": "tenant_subject_post_chat", "callers": ("main",), "dynamic_callers": (), "direct_bypass": ()},
    {"surface_id": "chat-api-request", "dimensions": ("REAL_SUBJECT_EXECUTION_COUNT",), "entry_point": "chat_api.py:add_request_observability", "primitive": "chat_api._guarded_chat_operation", "classifier": "candidate_mode_chat_endpoint", "callers": ("add_request_observability",), "dynamic_callers": (), "direct_bypass": ()},
)

EXTERNAL_REQUIREMENTS: tuple[dict[str, str], ...] = tuple(
    {
        "dimension_id": dimension,
        "proposition": f"complete bounded-interval truth for {dimension}",
        "repository_observable_subset": "repository receipts for installed primitives only",
        "external_uncontrolled_subset": "activity outside repository-controlled primitives",
        "repository_insufficiency": "repository receipts cannot enumerate external systems",
        "authority_class": "authenticated source-system interval enumeration",
        "minimum_evidence_schema": "provider_id, source_system_id, record_id, interval_start, interval_end, complete_enumeration, integrity_proof",
        "interval_semantics": "provider covers the canonical future B0..B1 interval without gaps",
        "availability": "NOT_AVAILABLE",
    }
    for dimension in (
        "C45_IMPLEMENTATION_PERFORMED",
        "C45_DIAGNOSTIC_EXECUTION_COUNT",
        "PROSPECTIVE_WITNESS_EXECUTION_COUNT",
        "PRODUCTION_CHANGED",
        "C38_RUNTIME_IDENTITY_CHANGED",
    )
)

CONTROL_BOUNDARY = tuple(
    {"dimension_id": dimension, "classification": "EXTERNAL_AUTHORITY_REQUIRED" if any(item["dimension_id"] == dimension for item in EXTERNAL_REQUIREMENTS) else "FULLY_REPOSITORY_CONTROLLED"}
    for dimension in DIMENSIONS
)

GLOBAL_INTERVAL_CONTRACT = {
    "epoch_kinds_permitted_here": ("SYNTHETIC_TEST_EPOCH",),
    "future_interval": {"B0": "externally authenticated start receipt", "B1": "externally authenticated end receipt"},
    "repository_coverage": "valid hash-chain receipts cover every registered guarded primitive",
    "external_coverage": "each external authority enumerates records over B0..B1",
    "gap_behavior": "missing provider, missing interval endpoint, or non-overlap fails full observability",
    "clock_semantics": "source-system record time is authoritative; skew requires reconciliation evidence",
}

def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
