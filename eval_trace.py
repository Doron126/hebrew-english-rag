"""Candidate-only, request-scoped observability for Phase 2C7.

The recorder is deliberately inert unless ``PROJECT_X_EVAL_TRACE=1``.  It
never changes pipeline values and writes only bounded, structured artifacts to
a private directory.  Full prompts, answers, and candidate documents are not
stored.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional


TRACE_SCHEMA_VERSION = "phase2c23-1.0.0"

# The trace record is an observability projection of the one canonical
# admissibility decision.  Some terminal/operator paths legitimately have no
# dimension-specific proof, but they must still state that every canonical
# dimension was non-constraining rather than omit schema-required fields.
_COMPATIBILITY_DECISION_DIMENSIONS = (
    "value",
    "quantity_kind",
    "statistical_role",
    "aggregation_role",
    "subject",
    "metric",
    "predicate",
    "period",
    "population",
    "facet",
    "unit",
    "polarity",
    "target_entity",
    "relation_arguments",
    "fact_role",
    "structural_type",
    "proposition_completeness",
)
TRACE_ENV = "PROJECT_X_EVAL_TRACE"
TRACE_DIR_ENV = "PROJECT_X_EVAL_TRACE_DIR"
DEFAULT_TRACE_DIR = Path("/var/lib/project-x/eval-traces/phase2c7")
MAX_STAGE_ITEMS = 256
MAX_QUERY_CHARS = 600
MAX_TEXT_FIELD_CHARS = 1200
MAX_PREVIEW_CHARS = 240
_TRACE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_CURRENT: ContextVar[Optional["TraceRecorder"]] = ContextVar(
    "project_x_eval_trace", default=None
)


def enabled() -> bool:
    return os.getenv(TRACE_ENV, "0").strip().casefold() in {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _bounded(value: Any, maximum: int = MAX_TEXT_FIELD_CHARS) -> str:
    value = str(value or "")
    return value if len(value) <= maximum else value[:maximum]


def _enum(value: Any) -> Any:
    return getattr(value, "value", value)


def _semantic_enum(value: Any, fallback: str = "unspecified") -> str:
    """Serialize semantic enums explicitly; blank is never an enum value."""
    value = str(_enum(value) or "").strip()
    return value or fallback


def _source_semantic_fields(value: Any) -> dict[str, Any]:
    """Serialize source semantic dimensions without blank enum placeholders."""
    raw = dict(value) if isinstance(value, Mapping) else {}
    raw["quantity_kind"] = _semantic_enum(raw.get("quantity_kind"))
    raw["statistical_role"] = _semantic_enum(raw.get("statistical_role"))
    raw["aggregation_role"] = _semantic_enum(raw.get("aggregation_role"))
    raw["scalar_role"] = _semantic_enum(
        raw.get("scalar_role"), "not_applicable",
    )
    return _safe(raw)


def validate_source_fact_observation_references(
    typed_stage: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    """Dereference serialized observations without creating any authority.

    ``source_fact_observations`` is only a projection of the occurrence
    registry.  This helper checks its direct IDs against authoritative
    ``CREATED`` rows in the same serialized trace.  It never uses semantic
    equality, source text, request state, or ordering to choose a target.
    """
    created_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in typed_stage.get("source_facts", []) or ():
        if not isinstance(row, Mapping) or row.get("outcome") != "CREATED":
            continue
        source_fact_id = str(row.get("source_fact_id") or "")
        if source_fact_id:
            created_by_id.setdefault(source_fact_id, []).append(row)
    errors: list[dict[str, str]] = []
    for index, observation in enumerate(
        typed_stage.get("source_fact_observations", []) or ()
    ):
        if not isinstance(observation, Mapping):
            errors.append({"index": str(index), "reason": "observation_not_mapping"})
            continue
        source_fact_id = str(observation.get("source_fact_id") or "")
        occurrence_id = str(observation.get("source_fact_occurrence_id") or "")
        matches = created_by_id.get(source_fact_id, [])
        if not source_fact_id or not matches:
            errors.append({"index": str(index), "reason": "dangling_source_fact_reference"})
            continue
        if len(matches) != 1:
            errors.append({"index": str(index), "reason": "ambiguous_source_fact_reference"})
            continue
        authoritative = matches[0]
        if not occurrence_id or occurrence_id != str(
            authoritative.get("source_fact_occurrence_id") or ""
        ):
            errors.append({"index": str(index), "reason": "occurrence_fact_identity_mismatch"})
            continue
        if list(observation.get("authoritative_source_fact_occurrence_provenance") or ()) != list(
            authoritative.get("source_fact_occurrence_provenance") or ()
        ):
            errors.append({"index": str(index), "reason": "authoritative_provenance_mismatch"})
            continue
        # C33's optional observation-local atom is provenance evidence only.
        # Validate its direct source-local reference if present, but never use
        # it to discover, select, merge, or replace an occurrence target.
        observed_atom = observation.get("observed_evidence_atom")
        if observed_atom is None:
            continue
        if not isinstance(observed_atom, Mapping) or not observed_atom:
            errors.append({"index": str(index), "reason": "observed_atom_not_mapping"})
            continue
        observed_atom_id = str(observation.get("observed_atom_id") or "")
        if not observed_atom_id or str(observed_atom.get("atom_id") or "") != observed_atom_id:
            errors.append({"index": str(index), "reason": "observed_atom_id_mismatch"})
            continue
        observed_evidence_id = str(observation.get("observed_evidence_id") or "")
        atom_evidence_id = str(observed_atom.get("evidence_unit_id") or "")
        if observed_evidence_id and atom_evidence_id and atom_evidence_id != observed_evidence_id:
            errors.append({"index": str(index), "reason": "observed_atom_evidence_mismatch"})
            continue
        provenance = list(observation.get("observed_source_fact_occurrence_provenance") or ())
        if len(provenance) == 7 and str(provenance[3]) == "atom_source_span":
            atom_source = (
                str(observed_atom.get("source_id") or ""),
                str(observed_atom.get("chunk_id") or ""),
                str(observed_atom.get("page") if observed_atom.get("page") is not None else ""),
                str(observed_atom.get("text_sha256") or ""),
            )
            expected_source = (
                str(provenance[0]), str(provenance[1]), str(provenance[2]), str(provenance[6]),
            )
            if atom_source != expected_source:
                errors.append({"index": str(index), "reason": "observed_atom_provenance_mismatch"})
    return tuple(errors)


def _safe(value: Any, *, depth: int = 0) -> Any:
    """Convert known planning metadata to bounded JSON without arbitrary text."""
    if depth > 5:
        return None
    value = _enum(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _bounded(value)
    if is_dataclass(value):
        return _safe(asdict(value), depth=depth + 1)
    if isinstance(value, Mapping):
        return {
            _bounded(key, 120): _safe(item, depth=depth + 1)
            for key, item in list(value.items())[:64]
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe(item, depth=depth + 1) for item in list(value)[:64]]
    return _bounded(value)


def _content_hash(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _candidate_id(hit: Mapping[str, Any]) -> str:
    metadata = hit.get("metadata") or {}
    identity = "\x1f".join(
        str(item or "")
        for item in (
            metadata.get("source_id"),
            metadata.get("chunk_id") or hit.get("id"),
            metadata.get("page"),
            _content_hash(str(hit.get("document") or "")),
        )
    )
    return "cand-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _normalized_score(raw: Any, kind: Any) -> Optional[float]:
    try:
        score = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(score):
        return None
    if str(kind or "").casefold() == "distance":
        return round(1.0 / (1.0 + max(0.0, score)), 8)
    return round(max(0.0, min(1.0, score)), 8)


def _raw_score(raw: Any) -> Optional[float]:
    try:
        score = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return round(score, 8) if math.isfinite(score) else None


def _rejection_taxonomy(reason_codes: Iterable[Any]) -> list[dict[str, str]]:
    """Keep stable, query-independent classes alongside exact reasons."""
    output: list[dict[str, str]] = []
    for raw in tuple(dict.fromkeys(str(item) for item in reason_codes if item)):
        category = (
            "structural_type_transition"
            if raw.startswith("structural_type_transition")
            else "scope"
            if "scope" in raw or "population" in raw
            else "relation_argument"
            if raw.startswith("relation_argument") or "relational_argument" in raw
            else "structural_binding"
            if raw.startswith(("structural_", "table_", "chart_", "list_", "cross_"))
            else "semantic_compatibility"
            if raw.startswith("evidence_")
            else "fact_binding"
        )
        output.append({"code": raw, "category": category})
    return output


def _semantic_role_rows(clauses: Iterable[Any]) -> list[dict[str, Any]]:
    """Return bounded, request-derived role provenance without raw prompts."""
    rows: list[dict[str, Any]] = []
    for clause in list(clauses or ())[:16]:
        raw = clause.as_dict() if hasattr(clause, "as_dict") else _safe(clause)
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        text = str(raw.get("text") or "")
        endpoints = raw.get("comparison_endpoints") or ()
        rows.append({
            "clause_id": _bounded(raw.get("id"), 120),
            "source_span_hash": _content_hash(text),
            "source_span_excerpt": _bounded(text, MAX_PREVIEW_CHARS),
            "preliminary_operation": _bounded(raw.get("semantic_relation"), 120),
            "preliminary_answer_shape": _bounded(raw.get("answer_shape"), 120),
            "predicate": _bounded(raw.get("predicate"), 160),
            "metrics": _safe(raw.get("objects_or_metrics") or ()),
            "scope_population": _bounded(raw.get("scope_population"), 240),
            "populations": _safe(raw.get("populations") or ()),
            "facets": _safe(raw.get("requested_facets") or ()),
            "periods": _safe(raw.get("periods") or ()),
            "filter_condition": _safe(raw.get("filter_condition")),
            "scope_constraint": _safe(raw.get("scope_constraint")),
            "output_slots": _safe(raw.get("requested_output_slots") or ()),
            "comparison_endpoints": _safe(endpoints),
            "relation_target": _bounded(raw.get("relation_target"), 240),
            "answer_relation": _bounded(raw.get("answer_relation"), 120),
        })
    return rows


def _unique_value_types(requirements: Iterable[Any]) -> list[str]:
    """Return answer-producing value types in a stable bounded order."""
    types: list[str] = []
    for requirement in list(requirements or ())[:MAX_STAGE_ITEMS]:
        raw = requirement.as_dict() if hasattr(requirement, "as_dict") else requirement
        if not isinstance(raw, Mapping) or not raw.get("answer_producing", True):
            continue
        value_type = _bounded(raw.get("value_type"), 80)
        if value_type and value_type not in types:
            types.append(value_type)
    return types


def _unattributed_compatibility_path_count(
    compatibility: Iterable[Mapping[str, Any]],
    terminal_outcomes: Iterable[Mapping[str, Any]],
) -> int:
    """Count source-fact/requirement compatibility paths lacking a terminal."""
    compatibility_paths = {
        (
            str(row.get("requirement_id") or ""),
            str(row.get("evidence_id") or ""),
            str(row.get("source_fact_id") or ""),
        )
        for row in compatibility
        if str(row.get("source_fact_id") or "")
    }
    terminal_paths = {
        (
            str(row.get("requirement_id") or ""),
            str(row.get("evidence_id") or ""),
            str(row.get("source_fact_id") or ""),
        )
        for row in terminal_outcomes
        if str(row.get("source_fact_id") or "")
    }
    return len(compatibility_paths - terminal_paths)


class TraceRecorder:
    def __init__(self, trace_id: str, *, preview_requested: bool = False) -> None:
        self.trace_id = trace_id
        self.preview_requested = bool(preview_requested)
        self.tenant_id = ""
        self.current_requirement_id: Optional[str] = None
        self.execution_contract: Any = None
        self._candidate_keys: set[tuple[str, str]] = set()
        self._selection_indexes: dict[str, int] = {}
        self._authorization_indexes: dict[str, int] = {}
        self._decomposition_indexes: dict[str, int] = {}
        self._evidence_to_candidate: dict[str, str] = {}
        self.semantic_compilation_result: Any = None
        self.artifact: dict[str, Any] = {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            # A recorder is entered only after the API dependency chain has
            # admitted the request.  Declared pre-handler terminals are
            # represented by their runner result row and deliberately never
            # manufacture an application trace.
            "lifecycle_mode": "POST_HANDLER_EXECUTION",
            "trace_id": trace_id,
            "request_id": trace_id,
            "started_utc": _utc_now(),
            "completed_utc": None,
            "effective_tenant": None,
            "stages": {
                "conversation_resolution": {"status": "not_recorded", "reason_code": "not_recorded"},
                "request_understanding": {"status": "not_recorded", "reason_code": "not_recorded"},
                "semantic_contract_validation": {"status": "not_recorded", "reason_code": "not_recorded", "violations": []},
                "requirement_construction": {"status": "not_recorded", "requirements": []},
                "route_execution": {"status": "not_recorded", "reason_code": "not_recorded"},
                "query_generation": {"status": "not_recorded", "queries": [], "query_count": 0},
                "candidate_retrieval": {"status": "not_recorded", "candidates": []},
                "candidate_selection": {"status": "not_recorded", "reason_code": "not_recorded", "decisions": []},
                "structural_evidence_construction": {"status": "not_recorded", "reason_code": "not_recorded", "units": [], "rejections": []},
                "typed_fact_extraction": {
                    "status": "not_recorded",
                    "reason_code": "not_recorded",
                    "facts": [],
                    "source_facts": [],
                    "source_fact_observations": [],
                    "compatibility": [],
                    "atoms": [],
                    "atom_terminal_outcomes": [],
                    "terminal_outcomes": [],
                },
                "fact_admissibility": {
                    "status": "not_recorded",
                    "reason_code": "not_recorded",
                    "complete_source_fact_count": 0,
                    "compatible_source_fact_count": 0,
                },
                "evidence_context_construction": {"status": "not_recorded", "reason_code": "not_recorded", "contexts": []},
                "model_assisted_evidence_semantics": {"status": "skipped", "reason_code": "not_used", "facts": []},
                "evidence_coverage": {"status": "not_recorded", "reason_code": "not_recorded", "requirements": []},
                "operator_execution": {"status": "skipped", "reason_code": "not_applicable"},
                "synthesis": {"status": "not_recorded", "reason_code": "not_recorded", "generation_attempt_count": 0},
                "question_answer_relevance": {"status": "not_recorded", "reason_code": "not_recorded", "reason_codes": []},
                "claim_evidence_gate": {"status": "not_recorded", "reason_code": "not_recorded", "claims": []},
                "final_gate": {"status": "not_recorded", "reason_code": "not_recorded", "downstream_skipped": []},
            },
        }

    def record_semantic_compilation(self, result: Any) -> None:
        """Record both successful and bounded pre-contract compilation."""
        self.semantic_compilation_result = result
        raw = result.as_dict() if hasattr(result, "as_dict") else _safe(result)
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        status = str(raw.get("status") or "invalid")
        draft = getattr(result, "draft", None)
        role_rows = _semantic_role_rows(getattr(draft, "clauses", ()) or ())
        self.stages["semantic_contract_validation"] = {
            "status": "completed" if status in {"valid", "repaired"} else "failed",
            "reason_code": _bounded(
                (raw.get("reason_codes") or ["semantic_compilation_failed"])[0],
                160,
            ),
            "compilation_status": _bounded(status, 80),
            "model_resolution_attempted": bool(raw.get("model_resolution_attempted")),
            "violations": _safe(raw.get("violations") or ()),
            "pre_contract_diagnostic": not bool(raw.get("contract_id")),
            "planner_operation": _bounded(raw.get("planner_operation"), 120),
            "deterministic_operation": _bounded(
                raw.get("deterministic_operation"), 120
            ),
            "deterministic_semantic_agreement": raw.get(
                "deterministic_semantic_agreement"
            ),
            "pre_freeze_disagreement": bool(
                raw.get("pre_freeze_disagreement")
            ),
            "reconciled": bool(raw.get("reconciled")),
            "preliminary_semantic_roles": role_rows,
            "preliminary_scope_population_count": sum(
                bool(row.get("scope_population")) for row in role_rows
            ),
            "preliminary_comparison_endpoint_pair_count": sum(
                len(row.get("comparison_endpoints") or ()) == 2
                for row in role_rows
            ),
            "preliminary_relation_argument_count": sum(
                bool(row.get("relation_target")) for row in role_rows
            ),
        }

    @property
    def stages(self) -> dict[str, Any]:
        return self.artifact["stages"]

    def set_request(self, query: str, tenant_id: str) -> None:
        self.tenant_id = _bounded(tenant_id, 80)
        self.artifact["effective_tenant"] = self.tenant_id
        self.stages["conversation_resolution"]["original_user_turn"] = _bounded(
            query, MAX_TEXT_FIELD_CHARS
        )

    def record_plan(self, understanding: Any, plan: Any) -> None:
        # Planning precedes conversation resolution.  It must not construct a
        # competing semantic contract; record_contract installs the one
        # immutable contract immediately after resolution.
        contract = None
        subquestions = [
            _safe(item.as_dict() if hasattr(item, "as_dict") else item)
            for item in list(getattr(plan, "subquestions", ()) or ())[:16]
        ]
        subject = next(
            (str(getattr(item, "subject", "") or "") for item in getattr(plan, "subquestions", ()) if getattr(item, "subject", "")),
            str((getattr(plan, "entities", ()) or ("",))[0] if getattr(plan, "entities", ()) else ""),
        )
        stage = self.stages["request_understanding"]
        contract_dict = (
            contract.as_dict()
            if contract is not None and hasattr(contract, "as_dict")
            else {}
        )
        stage.update({
            "status": "completed",
            "reason_code": str(getattr(understanding, "fallback_reason", None) or "frame_constructed"),
            "semantic_contract_version": _bounded(contract_dict.get("schema_version"), 80),
            "conversation_operation": _bounded(contract_dict.get("conversation_operation"), 80),
            "document_source_scope": _safe(contract_dict.get("source_scope") or ()),
            "semantic_operation": _bounded(contract_dict.get("semantic_relation") or _enum(getattr(plan, "request_operator", "")), 120),
            "answer_shape": _bounded(contract_dict.get("answer_shape") or _enum(getattr(plan, "answer_shape", "")), 120),
            "normalized_subject": _bounded(" | ".join(contract_dict.get("subjects") or (subject,)), 360),
            "predicate_or_relation": _bounded(" | ".join(contract_dict.get("predicates") or ()), 240),
            "requested_metrics": _safe(contract_dict.get("objects_or_metrics") or ()),
            "requested_values": _safe(contract_dict.get("requested_values") or ()),
            "requested_units": _safe(contract_dict.get("units") or ()),
            "periods": _safe(contract_dict.get("periods") or ()),
            "populations": _safe(contract_dict.get("populations") or ()),
            "polarity_or_negation": _safe(contract_dict.get("polarity")),
            "qualifiers": _safe(contract_dict.get("qualifiers") or ()),
            "requested_cardinality": _safe(contract_dict.get("requested_cardinality")),
            "completeness": _bounded(contract_dict.get("completeness"), 120),
            "planner_confidence": float(getattr(understanding, "deterministic_confidence", 0.0) or 0.0),
            "planning_mode": "model_assisted" if getattr(understanding, "planner_called", False) else "deterministic",
            "route": str(_enum(getattr(understanding, "route", ""))),
            "decomposition": _safe(contract_dict.get("clauses") or subquestions),
        })
        requirements = []
        contract_requirements = contract_dict.get("requirements") or ()
        raw_requirements = (
            contract_requirements
            or tuple(getattr(self.execution_contract, "requirements", ()) or ())
            or list(getattr(plan, "evidence_requirements", ()) or ())
        )
        for index, item in enumerate(list(raw_requirements)[:64], start=1):
            raw = item.as_dict() if hasattr(item, "as_dict") else _safe(item)
            raw = dict(raw) if isinstance(raw, Mapping) else {}
            requirements.append({
                "requirement_id": _bounded(raw.get("id") or f"req-{index}", 160),
                "clause_id": _bounded(raw.get("clause_id"), 160),
                "kind": _bounded(raw.get("kind"), 160),
                "required_subject": _bounded(raw.get("subject") or raw.get("entity") or subject, 360),
                "predicate": _bounded(raw.get("predicate"), 240),
                "metrics": _safe(raw.get("objects_or_metrics") or ([raw.get("metric")] if raw.get("metric") else [])),
                "requested_values": _safe(raw.get("requested_values") or ()),
                "units": _safe(raw.get("units") or ([raw.get("unit")] if raw.get("unit") else [])),
                "periods": _safe(raw.get("periods") or ([raw.get("period")] if raw.get("period") else [])),
                "populations": _safe(raw.get("populations") or ([raw.get("population")] if raw.get("population") else [])),
                "scope_population": _bounded(raw.get("scope_population"), 240),
                "polarity": _safe(raw.get("polarity") or raw.get("negation")),
                "qualifiers": _safe(raw.get("qualifiers") or ([raw.get("qualifier")] if raw.get("qualifier") else [])),
                "cardinality": int(raw.get("cardinality") or 1),
                "completeness_requirement": _bounded(raw.get("completeness") or "complete", 120),
                "linked_requirement_ids": _safe(raw.get("linked_requirement_ids") or ()),
                "satisfaction_mode": _bounded(
                    raw.get("satisfaction_mode") or "SOURCE_FACT", 80
                ),
                "scalar_role": _semantic_enum(raw.get("scalar_role"), "not_applicable"),
                "quantity_kind": _semantic_enum(raw.get("quantity_kind")),
                "statistical_role": _semantic_enum(raw.get("statistical_role")),
                "aggregation_role": _semantic_enum(raw.get("aggregation_role")),
                "dimension_constraints": _safe(raw.get("dimension_constraints") or ()),
                "semantic_accounting": _safe(raw.get("semantic_accounting") or ()),
                "mandatory": bool(raw.get("mandatory", True)),
            })
        if not requirements:
            requirements = [{
                "requirement_id": "req-1",
                "required_subject": _bounded(subject, 360),
                "predicate": "", "metrics": [], "requested_values": [],
                "units": [], "periods": [], "populations": [], "polarity": None,
                "qualifiers": [], "cardinality": 1, "linked_requirement_ids": [],
                "completeness_requirement": "complete", "scalar_role": "not_applicable",
                "quantity_kind": "unspecified", "statistical_role": "unspecified",
                "aggregation_role": "unspecified", "dimension_constraints": [],
                "semantic_accounting": [],
                "satisfaction_mode": "SOURCE_FACT", "mandatory": True,
            }]
        self.stages["requirement_construction"] = {
            "status": "completed", "reason_code": "requirements_constructed", "requirements": requirements
        }

    def record_contract(self, contract: Any) -> None:
        """Install the sole post-resolution request execution contract."""
        if contract is None or not hasattr(contract, "as_dict"):
            raise RuntimeError("trace requires RequestExecutionContract")
        if self.execution_contract is not None and (
            self.execution_contract is not contract
            or self.execution_contract.contract_sha256 != contract.contract_sha256
        ):
            raise RuntimeError("request execution contract changed within request")
        self.execution_contract = contract
        self.artifact["lifecycle_mode"] = "POST_HANDLER_EXECUTION"
        raw = contract.as_dict()
        # Replace the provisional plan projection as a whole.  Mutating
        # selected fields after compilation would create a second semantic
        # authority and could harden an otherwise NOT_REQUIRED dimension.
        validation = dict(raw.get("semantic_validation") or {})
        pre_freeze_provenance = {
            key: self.stages["semantic_contract_validation"].get(key)
            for key in (
                "preliminary_semantic_roles",
                "preliminary_scope_population_count",
                "preliminary_comparison_endpoint_pair_count",
                "preliminary_relation_argument_count",
            )
            if key in self.stages["semantic_contract_validation"]
        }
        validation_ok = bool(validation.get("valid")) and int(
            validation.get("normalization_count") or 0
        ) == 1
        self.stages["semantic_contract_validation"] = {
            "status": "completed" if validation_ok else "failed",
            "reason_code": (
                "semantic_contract_validated_once"
                if validation_ok else "semantic_contract_validation_failed"
            ),
            "draft_operation": _bounded(validation.get("draft_operation"), 120),
            "normalization_applied": bool(validation.get("normalization_applied")),
            "normalization_count": int(validation.get("normalization_count") or 0),
            "violations": _safe(validation.get("violations") or ()),
            "endpoint_types": _safe(validation.get("endpoint_types") or ()),
            "requested_facets": _safe(validation.get("requested_facets") or ()),
            "requested_output_slots": _safe(validation.get("requested_output_slots") or ()),
            "semantic_role_checks": _safe(
                validation.get("semantic_role_checks") or ()
            ),
            "silent_agreement_check_count": len(
                validation.get("semantic_role_checks") or ()
            ),
            "scope_constraint_classification_count": sum(
                str(item).endswith(":scope_constraint")
                for item in validation.get("semantic_role_checks") or ()
            ),
            "output_filter_classification_count": sum(
                str(item).endswith(":output_filter")
                for item in validation.get("semantic_role_checks") or ()
            ),
            "final_semantic_signature": _bounded(
                validation.get("final_semantic_signature"), 160
            ),
            "compilation_status": _bounded(
                getattr(self.semantic_compilation_result, "status", "valid"), 80
            ),
            "model_resolution_attempted": bool(
                getattr(
                    self.semantic_compilation_result,
                    "model_resolution_attempted",
                    False,
                )
            ),
            "planner_operation": _bounded(
                getattr(
                    self.semantic_compilation_result,
                    "planner_operation",
                    "",
                ),
                120,
            ),
            "deterministic_operation": _bounded(
                getattr(
                    self.semantic_compilation_result,
                    "deterministic_operation",
                    "",
                ),
                120,
            ),
            "deterministic_semantic_agreement": getattr(
                self.semantic_compilation_result,
                "deterministic_semantic_agreement",
                None,
            ),
            "pre_freeze_disagreement": bool(getattr(
                self.semantic_compilation_result,
                "pre_freeze_disagreement",
                False,
            )),
            "reconciled": bool(getattr(
                self.semantic_compilation_result,
                "reconciled",
                False,
            )),
            **pre_freeze_provenance,
        }
        stage = self.stages["request_understanding"]
        stage.update({
            "semantic_contract_version": _bounded(raw.get("schema_version"), 80),
            "conversation_operation": _bounded(raw.get("conversation_operation"), 80),
            "execution_route": _bounded(raw.get("execution_route"), 120),
            "support_status": _bounded(raw.get("support_status"), 80),
            "document_source_scope": _safe(raw.get("source_scope") or ()),
            "semantic_operation": _bounded(raw.get("semantic_relation"), 120),
            "answer_shape": _bounded(raw.get("answer_shape"), 120),
            "normalized_subject": _bounded(" | ".join(raw.get("subjects") or ()), 360),
            "predicate_or_relation": _bounded(" | ".join(raw.get("predicates") or ()), 240),
            "requested_metrics": _safe(raw.get("objects_or_metrics") or ()),
            "requested_values": _safe(raw.get("requested_values") or ()),
            "requested_units": _safe(raw.get("units") or ()),
            "periods": _safe(raw.get("periods") or ()),
            "populations": _safe(raw.get("populations") or ()),
            "polarity_or_negation": _safe(raw.get("polarity")),
            "qualifiers": _safe(raw.get("qualifiers") or ()),
            "requested_cardinality": _safe(raw.get("requested_cardinality")),
            "completeness": _bounded(raw.get("completeness"), 120),
            "decomposition": _safe(raw.get("clauses") or ()),
            "output_filter_conditions": _safe(raw.get("filter_conditions") or ()),
            "scope_constraints": _safe(raw.get("scope_constraints") or ()),
            "requested_value_types": _safe(_unique_value_types(
                raw.get("requirements") or ()
            )),
            "comparison_arguments": _safe([
                {
                    "clause_id": item.get("id"),
                    "metric": item.get("objects_or_metrics") or (),
                    "endpoints": item.get("comparison_endpoints") or (),
                    "periods": item.get("periods") or (),
                }
                for item in raw.get("clauses") or ()
                if item.get("comparison_endpoints")
            ]),
            "relation_arguments": _safe([
                {
                    "clause_id": item.get("id"),
                    "relation": item.get("answer_relation"),
                    "target": item.get("relation_target"),
                    "output_slots": item.get("requested_output_slots") or (),
                }
                for item in raw.get("clauses") or ()
                if item.get("relation_target")
            ]),
        })
        requirements = []
        for item in list(raw.get("requirements") or ())[:MAX_STAGE_ITEMS]:
            requirements.append({
                "requirement_id": _bounded(item.get("id"), 160),
                "clause_id": _bounded(item.get("clause_id"), 160),
                "kind": _bounded(item.get("kind"), 160),
                "required_subject": _bounded(item.get("subject"), 360),
                "predicate": _bounded(item.get("predicate"), 240),
                "metrics": _safe(item.get("objects_or_metrics") or ()),
                "requested_values": _safe(item.get("requested_values") or ()),
                "units": _safe(item.get("units") or ()),
                "periods": _safe(item.get("periods") or ()),
                "populations": _safe(item.get("populations") or ()),
                "scope_population": _bounded(item.get("scope_population"), 240),
                "polarity": _safe(item.get("polarity")),
                "qualifiers": _safe(item.get("qualifiers") or ()),
                "cardinality": int(item.get("cardinality") or 1),
                "completeness_requirement": _bounded(item.get("completeness"), 120),
                "linked_requirement_ids": _safe(item.get("linked_requirement_ids") or ()),
                "mandatory": bool(item.get("mandatory", True)),
                "endpoint_type": _bounded(item.get("endpoint_type"), 80),
                "endpoint_value": _bounded(item.get("endpoint_value"), 240),
                "facet": _bounded(item.get("facet"), 120),
                "output_slot": _bounded(item.get("output_slot"), 120),
                "value_type": _bounded(item.get("value_type"), 80),
                "satisfaction_mode": _bounded(
                    item.get("satisfaction_mode") or "SOURCE_FACT", 80
                ),
                "scalar_role": _semantic_enum(item.get("scalar_role"), "not_applicable"),
                "quantity_kind": _semantic_enum(item.get("quantity_kind")),
                "statistical_role": _semantic_enum(item.get("statistical_role")),
                "aggregation_role": _semantic_enum(item.get("aggregation_role")),
                "dimension_constraints": _safe(item.get("dimension_constraints") or ()),
                "semantic_accounting": _safe(item.get("semantic_accounting") or ()),
            })
        self.stages["requirement_construction"] = {
            "status": "completed",
            "reason_code": "immutable_contract_requirements",
            "requirements": requirements,
        }

    def record_route_execution(self, route: str, *, reason_code: str, abstained: bool = False) -> None:
        expected = str(
            getattr(self.execution_contract, "execution_route", "") or ""
        )
        if expected and route != expected:
            raise RuntimeError("runtime route diverged from execution contract")
        self.stages["route_execution"] = {
            "status": "completed",
            "reason_code": _bounded(reason_code, 160),
            "route": _bounded(route, 120),
            "contract_route": _bounded(expected, 120),
            "route_equivalent": not expected or route == expected,
            "early_abstention": bool(abstained),
            "early_abstention_reason": _bounded(reason_code if abstained else "none", 160),
        }

    def record_contract_authority(
        self,
        *,
        planned_route: str,
        contract_route: str,
        runtime_route: str,
        planned_answer_shape: str,
        contract_answer_shape: str,
    ) -> None:
        """Record pre-freeze disagreement and post-freeze equivalence."""
        if runtime_route != contract_route:
            raise RuntimeError("derived runtime plan is not contract-equivalent")
        self.artifact["semantic_authority"] = {
            "authority": "request_execution_contract",
            "planned_route": _bounded(planned_route, 120),
            "contract_route": _bounded(contract_route, 120),
            "runtime_route": _bounded(runtime_route, 120),
            "pre_freeze_route_disagreement": planned_route != contract_route,
            "post_freeze_route_equivalent": True,
            "planned_answer_shape": _bounded(planned_answer_shape, 120),
            "contract_answer_shape": _bounded(contract_answer_shape, 120),
        }

    def record_conversation(self, resolution: Any, plan: Any) -> None:
        state = getattr(resolution, "state", None)
        previous_id = None
        if state is not None and getattr(state, "previous_user_question", None):
            previous_id = "turn-previous"
        self.stages["conversation_resolution"].update({
            "status": "failed" if getattr(resolution, "needs_clarification", False) else "completed",
            "turn_kind": "follow_up" if getattr(plan, "requires_context_resolution", False) else "new_question",
            "resolved_previous_turn_id": previous_id,
            "resolved_source_scope": _safe(getattr(state, "previous_source_scope", ())),
            "operation": str(_enum(getattr(plan, "conversation_operation", "NONE"))),
            "reason_code": _bounded(getattr(resolution, "reason_code", "conversation_resolved"), 160),
        })

    def record_retrieval(
        self,
        channel: str,
        query: Any,
        hits: Iterable[Mapping[str, Any]],
        *,
        filters: Any = None,
        section_hint: str = "",
        expansion_requested: bool = False,
    ) -> None:
        query_text = query if isinstance(query, str) else json.dumps(_safe(query), ensure_ascii=False, sort_keys=True)
        query_stage = self.stages["query_generation"]
        query_items = query_stage["queries"]
        if len(query_items) < MAX_STAGE_ITEMS:
            query_items.append({
                "requirement_id": self.current_requirement_id or (
                    self.stages["requirement_construction"]["requirements"][0]["requirement_id"]
                    if len(self.stages["requirement_construction"].get("requirements", [])) == 1
                    else None
                ),
                "normalized_query": _bounded(" ".join(query_text.split()), MAX_QUERY_CHARS),
                "channel": _bounded(channel, 40),
                "filters": _safe(filters),
                "section_or_heading_hints": _bounded(section_hint, 240),
                "parent_or_adjacent_expansion_requested": bool(expansion_requested),
            })
        query_stage.update({"status": "completed", "reason_code": "queries_generated", "query_count": len(query_items)})

        candidate_stage = self.stages["candidate_retrieval"]
        candidates = candidate_stage["candidates"]
        for rank, hit in enumerate(list(hits or ())[:MAX_STAGE_ITEMS], start=1):
            metadata = hit.get("metadata") or {}
            candidate_id = _candidate_id(hit)
            key = (candidate_id, channel)
            if key in self._candidate_keys or len(candidates) >= MAX_STAGE_ITEMS:
                continue
            self._candidate_keys.add(key)
            document = str(hit.get("document") or "")
            source_tenant = str(metadata.get("tenant_id") or "")
            preview = None
            if self.preview_requested and source_tenant and source_tenant == self.tenant_id:
                preview = document[:MAX_PREVIEW_CHARS]
            candidates.append({
                "candidate_id": candidate_id,
                "chunk_id": _bounded(metadata.get("chunk_id") or hit.get("id"), 160),
                "source_id": _bounded(metadata.get("source_id"), 80),
                "page": metadata.get("page") if isinstance(metadata.get("page"), int) else None,
                "section": _bounded(metadata.get("section") or metadata.get("title"), 240),
                "structural_type": _bounded(metadata.get("structural_type") or metadata.get("unit_type") or "body", 80),
                "retrieval_channel": _bounded(channel, 40),
                "raw_score": _raw_score(hit.get("score")),
                "normalized_score": _normalized_score(hit.get("score"), hit.get("score_kind")),
                "rank": rank,
                "global_rank": len(candidates) + 1,
                "requirement_association": self.current_requirement_id or _safe(hit.get("_requirement_id")),
                "content_hash": _content_hash(document),
                "evidence_preview": preview,
                "tenant_integrity": "verified" if source_tenant == self.tenant_id else "unverified",
            })
        candidate_stage.update({
            "status": "completed",
            "reason_code": "candidates_entered" if candidates else "bounded_candidate_set_empty",
            "candidate_count": len(candidates),
        })

    def record_candidate_decisions(
        self,
        decisions: Iterable[Any],
        *,
        selected_candidate_ids: Iterable[str] = (),
        selection_stage: str = "synthesis",
    ) -> None:
        """Record selector lifecycle state at the decision point.

        This method is intentionally independent of citations.  Later
        lifecycle events update the same decision rows, but never invent the
        original eligibility or selection decision.
        """
        stage = self.stages["candidate_selection"]
        selected = {str(value) for value in selected_candidate_ids if value}
        for decision in list(decisions or ())[:MAX_STAGE_ITEMS]:
            raw = (
                decision.as_dict()
                if hasattr(decision, "as_dict")
                else dict(decision)
                if isinstance(decision, Mapping)
                else {}
            )
            candidate_id = _bounded(raw.get("candidate_id"), 80)
            if not candidate_id:
                continue
            raw["candidate_id"] = candidate_id
            raw["selection_stage"] = _bounded(selection_stage, 80)
            raw["selected_for_fact_extraction"] = bool(
                raw.get("selected_for_fact_extraction")
                or candidate_id in selected
            )
            raw.setdefault("selected_for_synthesis", False)
            raw.setdefault("produced_fact", False)
            raw.setdefault("selected_for_evidence_bundle", False)
            raw.setdefault("supplied_to_generation", False)
            raw.setdefault("used_in_claim", False)
            raw.setdefault("selected_for_citation", False)
            raw.setdefault("omitted_after_synthesis", False)
            raw.setdefault("removed_by_final_gate", False)
            safe = _safe(raw)
            if candidate_id in self._selection_indexes:
                index = self._selection_indexes[candidate_id]
                existing = stage["decisions"][index]
                preserved = {
                    "selected_for_citation": existing.get("selected_for_citation", False),
                    "omitted_after_synthesis": existing.get("omitted_after_synthesis", False),
                    "removed_by_final_gate": existing.get("removed_by_final_gate", False),
                }
                existing.update(safe)
                for key, value in preserved.items():
                    existing[key] = bool(existing.get(key) or value)
                continue
            if len(stage["decisions"]) >= MAX_STAGE_ITEMS:
                break
            self._selection_indexes[candidate_id] = len(stage["decisions"])
            stage["decisions"].append(safe)
        rows = stage["decisions"]
        stage.update({
            "status": "completed" if rows else "failed",
            "reason_code": "selector_decisions_recorded" if rows else "selector_decision_missing",
            "decision_count": len(rows),
            "selected_for_fact_extraction_count": sum(bool(row.get("selected_for_fact_extraction")) for row in rows),
            "selected_count": sum(bool(row.get("selected_for_fact_extraction")) for row in rows),
            "rejected_count": sum(bool(row.get("rejected")) for row in rows),
        })

    def record_source_authorization(self, candidate_id: str, outcome: Mapping[str, Any]) -> None:
        """Attach the actual authorization decision made before decomposition."""
        stage = self.stages["candidate_selection"]
        identifier = _bounded(candidate_id, 80)
        row = {"candidate_id": identifier, **_safe(dict(outcome))}
        rows = stage.setdefault("source_authorization_events", [])
        if identifier in self._authorization_indexes:
            rows[self._authorization_indexes[identifier]] = row
        elif len(rows) < MAX_STAGE_ITEMS:
            self._authorization_indexes[identifier] = len(rows)
            rows.append(row)

    def record_structural_decomposition(
        self,
        candidate_id: str,
        *,
        status: str,
        reason: str,
        units: Iterable[Any] = (),
        entry: Any = None,
        scalar_span_audit: Iterable[Any] = (),
    ) -> None:
        """Record the terminal structural lifecycle without losing prior data."""
        unit_rows = [item.safe_summary() if hasattr(item, "safe_summary") else _safe(item)
                     for item in units]
        scalar_rows = [
            item.as_dict() if hasattr(item, "as_dict") else _safe(item)
            for item in scalar_span_audit
        ]
        identifier = _bounded(candidate_id, 80)
        rows = self.stages["structural_evidence_construction"].setdefault(
            "candidate_decomposition_events", []
        )
        entry_summary = {
            "source_id": "",
            "structural_entry_status": "rejected",
            "source_decoration_detected": False,
            "source_decoration_kind": None,
            "source_decoration_span": None,
            "body_span": None,
            "body_length": 0,
            "coarse_structural_type": "unknown",
            "canonical_decomposition_attempted": False,
        }
        if entry is not None and hasattr(entry, "trace_summary"):
            entry_summary.update(entry.trace_summary())
        authorization_status = "not_recorded"
        if identifier in self._authorization_indexes:
            authorization_rows = self.stages["candidate_selection"].get(
                "source_authorization_events", []
            )
            authorization_index = self._authorization_indexes[identifier]
            if authorization_index < len(authorization_rows):
                authorization_status = _bounded(
                    authorization_rows[authorization_index].get(
                        "source_authorization_status"
                    ),
                    80,
                ) or "not_recorded"
        row = {
            "candidate_id": identifier,
            "source_id": _bounded(entry_summary.get("source_id"), 80),
            "authorization_status": authorization_status,
            **_safe(entry_summary),
            "structural_decomposition_status": _bounded(status, 80),
            "structural_decomposition_reason": _bounded(reason, 160),
            "decomposition_status": _bounded(status, 80),
            "decomposition_reason": _bounded(reason, 160),
            "candidate_lifecycle_terminal_state": _bounded(
                "structural_units_created" if status == "units_created"
                else "structural_rejected", 80,
            ),
            "structural_unit_count": len(unit_rows),
            "scalar_span_audit": _safe(scalar_rows),
            "scalar_spans_total": len(scalar_rows),
            "scalar_spans_recovered": sum(
                item.get("status") == "RECOVERED"
                for item in scalar_rows if isinstance(item, Mapping)
            ),
            "scalar_spans_explicitly_rejected": sum(
                item.get("status") == "REJECTED"
                for item in scalar_rows if isinstance(item, Mapping)
            ),
            # Every scalar audit row has an explicit terminal state.  An
            # explicit rejection is accounted evidence, not an unaccounted
            # gap.  Keep the C18 aliases only for reader compatibility; C19
            # validation consumes the corrected plural fields above.
            "scalar_spans_accounted": len(scalar_rows),
            "scalar_spans_unaccounted": 0,
            "scalar_span_detected_count": len(scalar_rows),
            "scalar_span_recovered_count": sum(
                item.get("status") == "RECOVERED"
                for item in scalar_rows if isinstance(item, Mapping)
            ),
            "scalar_span_unaccounted_count": 0,
            "structural_unit_ids": [_bounded(item.get("structural_unit_id"), 160)
                                    for item in unit_rows if isinstance(item, Mapping)],
            "structural_unit_types": [_bounded(item.get("structural_unit_type"), 80)
                                      for item in unit_rows if isinstance(item, Mapping)],
            "structural_type_authority": "structural_evidence_unit",
            "largest_unit_size": max((len(str(item.get("semantic_proposition_preview") or ""))
                                      for item in unit_rows if isinstance(item, Mapping)), default=0),
        }
        if identifier in self._decomposition_indexes:
            existing = rows[self._decomposition_indexes[identifier]]
            # A later pipeline gate may add its precise terminal state, but
            # must not erase the original entry/body/unit audit record.
            preserved = {
                key: existing.get(key)
                for key in (
                    "source_id", "authorization_status",
                    "source_decoration_detected", "source_decoration_kind",
                    "source_decoration_span", "body_span", "body_length",
                    "coarse_structural_type", "canonical_decomposition_attempted",
                    "structural_unit_count", "structural_unit_ids",
                    "structural_unit_types", "largest_unit_size",
                )
                if existing.get(key) not in (None, "", [], 0, False)
            }
            existing.update(row)
            for key, value in preserved.items():
                if existing.get(key) in (None, "", [], 0, False):
                    existing[key] = value
        elif len(rows) < MAX_STAGE_ITEMS:
            self._decomposition_indexes[identifier] = len(rows)
            rows.append(row)

    def mark_selected_for_citation(self, candidate: Any) -> None:
        candidate_id = ""
        if isinstance(candidate, str):
            candidate_id = candidate
        elif isinstance(candidate, Mapping):
            candidate_id = _candidate_id(candidate)
        else:
            chunk_id = str(
                getattr(candidate, "chunk_id", None)
                or getattr(candidate, "evidence_chunk_id", None)
                or ""
            )
            if chunk_id:
                matching_ids = {
                    str(item.get("candidate_id") or "")
                    for item in self.stages["candidate_retrieval"].get("candidates", [])
                    if str(item.get("chunk_id") or "") == chunk_id
                }
                for matching_id in matching_ids:
                    if matching_id in self._selection_indexes:
                        self.stages["candidate_selection"]["decisions"][
                            self._selection_indexes[matching_id]
                        ]["selected_for_citation"] = True
                return
        if candidate_id in self._selection_indexes:
            self.stages["candidate_selection"]["decisions"][
                self._selection_indexes[candidate_id]
            ]["selected_for_citation"] = True

    def record_downstream_binding_usage(
        self,
        graph: Any,
        typed_facts: Iterable[Any],
        *,
        accepted: Optional[bool] = None,
    ) -> None:
        """Synchronize selector lifecycle from the actual immutable graph."""
        facts_by_id = {
            str(getattr(fact, "fact_id", "") or ""): fact
            for fact in typed_facts
            if getattr(fact, "fact_id", "")
        }
        claims = tuple(getattr(graph, "claims", ()) or ())
        used_fact_ids = {
            str(fact_id)
            for claim in claims
            for fact_id in getattr(claim, "fact_ids", ()) or ()
        }
        cited_fact_ids = {
            str(fact_id)
            for binding in getattr(graph, "citation_bindings", ()) or ()
            if bool(getattr(binding, "direct", False))
            for fact_id in getattr(binding, "fact_ids", ()) or ()
        }

        def candidate_ids(fact: Any) -> set[str]:
            output = {
                str(getattr(fact, "candidate_id", "") or ""),
                *(
                    str(value)
                    for value in getattr(fact, "source_candidate_ids", ()) or ()
                ),
            } - {""}
            evidence_id = str(getattr(fact, "evidence_id", "") or "")
            mapped = self._evidence_to_candidate.get(evidence_id, "")
            if mapped:
                output.add(mapped)
            return output

        used_candidates = {
            candidate_id
            for fact_id in used_fact_ids
            for candidate_id in candidate_ids(facts_by_id.get(fact_id))
            if facts_by_id.get(fact_id) is not None
        }
        cited_candidates = {
            candidate_id
            for fact_id in cited_fact_ids
            for candidate_id in candidate_ids(facts_by_id.get(fact_id))
            if facts_by_id.get(fact_id) is not None
        }
        for candidate_id in used_candidates | cited_candidates:
            index = self._selection_indexes.get(candidate_id)
            if index is None:
                continue
            row = self.stages["candidate_selection"]["decisions"][index]
            if candidate_id in used_candidates:
                row["selected_for_synthesis"] = True
                row["used_in_claim"] = True
            if candidate_id in cited_candidates:
                row["selected_for_citation"] = True
            if accepted is False and (
                candidate_id in used_candidates or candidate_id in cited_candidates
            ):
                row["removed_by_final_gate"] = True

    def record_response_contract(self, result: Any) -> None:
        raw = result.as_dict() if hasattr(result, "as_dict") else _safe(result)
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        relevance_passed = bool(raw.get("relevance_passed"))
        entailment_passed = bool(raw.get("citation_entailment_passed"))
        requirements = list(raw.get("requirements") or ())[:MAX_STAGE_ITEMS]
        self.stages["question_answer_relevance"] = {
            "status": "completed" if relevance_passed else "failed",
            "reason_code": "all_mandatory_requirements_satisfied" if relevance_passed else "mandatory_requirement_not_satisfied",
            "reason_codes": _safe(raw.get("reason_codes") or ()),
            "passed": relevance_passed,
            "material_claim_count": int(raw.get("claim_count") or 0),
            "requirements": _safe(requirements),
        }
        self.stages["claim_evidence_gate"] = {
            "status": "completed" if entailment_passed else "failed",
            "reason_code": "all_material_claims_directly_bound" if entailment_passed else "claim_field_entailment_failed",
            "claims": _safe(requirements),
            "passed": entailment_passed,
        }

    def record_claim_graph(self, graph: Any) -> None:
        raw = graph.as_dict() if hasattr(graph, "as_dict") else _safe(graph)
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        claims = list(raw.get("claims") or ())[:MAX_STAGE_ITEMS]
        passed = bool(raw.get("passed"))
        missing = list(raw.get("missing_requirement_ids") or ())
        self.stages["question_answer_relevance"] = {
            "status": "completed" if not missing else "failed",
            "reason_code": (
                "all_mandatory_requirements_satisfied"
                if not missing else "mandatory_requirement_not_satisfied"
            ),
            "reason_codes": [] if not missing else ["mandatory_requirement_not_satisfied"],
            "passed": not missing,
            "material_claim_count": len(claims),
            "missing_requirement_ids": _safe(missing),
            "contextual_factual_correctness": bool(
                raw.get("contextual_factual_correctness")
            ),
            "standalone_completeness": bool(raw.get("standalone_completeness")),
            "pre_gate_claims": _safe(claims),
        }
        citation_bindings = list(raw.get("citation_bindings") or ())[:MAX_STAGE_ITEMS]
        fact_provenance = list(raw.get("fact_provenance") or ())[:MAX_STAGE_ITEMS]
        self.stages["claim_evidence_gate"] = {
            "status": "completed" if passed else "failed",
            "reason_code": (
                "explicit_claim_graph_valid"
                if passed else "explicit_claim_graph_incomplete"
            ),
            "claims": _safe(claims),
            "citation_bindings": _safe(citation_bindings),
            "fact_provenance": _safe(fact_provenance),
            "operator_provenance_gap_count": sum(
                not bool(item.get("operator_provenance_complete", True))
                for item in fact_provenance if isinstance(item, Mapping)
            ),
            "passed": passed,
        }

    def mark_removed_by_final_gate(self) -> None:
        for row in self.stages["candidate_selection"].get("decisions", []):
            if row.get("supplied_to_generation") or row.get("selected_for_evidence_bundle"):
                row["removed_by_final_gate"] = True

    @staticmethod
    def _serialized_evidence_atom(atom: Any) -> dict[str, Any]:
        """Return the native trace projection of an already-created atom.

        This helper is observability-only.  It never derives an atom from a
        source fact, request, semantic field, or occurrence identity; callers
        may only pass the EvidenceAtom already retained by their source-local
        runtime object.  Keeping the serializer shared prevents an aggregate
        atom row and an observation-local carry-forward row from acquiring
        subtly different role encodings.
        """
        raw = atom.as_dict() if hasattr(atom, "as_dict") else _safe(atom)
        if not isinstance(raw, Mapping):
            return {}
        normalized = dict(raw)
        normalized["scalar_role"] = _semantic_enum(
            normalized.get("scalar_role"), "not_applicable",
        )
        normalized["quantity_kind"] = _semantic_enum(
            normalized.get("quantity_kind"),
        )
        normalized["statistical_role"] = _semantic_enum(
            normalized.get("statistical_role"),
        )
        normalized["aggregation_role"] = _semantic_enum(
            normalized.get("aggregation_role"),
        )
        return dict(_safe(normalized))

    def record_evidence_atom(self, atom: Any) -> None:
        """Record a bounded, source-derived fact atom at creation time."""
        stage = self.stages["typed_fact_extraction"]
        atoms = stage.setdefault("atoms", [])
        normalized = self._serialized_evidence_atom(atom)
        if normalized and normalized not in atoms and len(atoms) < MAX_STAGE_ITEMS:
            atoms.append(normalized)

    def record_evidence_atom_terminal_outcome(
        self,
        *,
        evidence_id: str,
        candidate_id: str,
        outcome: str,
        reason_codes: Iterable[str] = (),
        atom_id: str = "",
        binding_status: str = "",
    ) -> None:
        """Record the source-local atom path before SOURCE_FACT construction.

        An atom rejection is deliberately distinct from a rejected fact: the
        former means no bounded source-local association was proven, while the
        latter means a proven atom could not form a complete source fact.
        """
        if outcome not in {
            "CONSUMED_BY_SOURCE_FACT",
            "OBSERVED_BY_EXISTING_SOURCE_FACT",
            "REJECTED",
        }:
            raise ValueError("invalid evidence-atom terminal outcome")
        row = {
            "evidence_id": _bounded(evidence_id, 160),
            "candidate_id": _bounded(candidate_id, 160),
            "atom_id": _bounded(atom_id, 160),
            "outcome": outcome,
            "binding_status": _bounded(binding_status or (
                "atom_consumed" if outcome == "CONSUMED_BY_SOURCE_FACT"
                else "atom_reobserved" if outcome == "OBSERVED_BY_EXISTING_SOURCE_FACT"
                else "atom_not_constructed"
            ), 120),
            "reason_codes": _safe(tuple(dict.fromkeys(
                str(item) for item in reason_codes
            ))),
        }
        rows = self.stages["typed_fact_extraction"].setdefault(
            "atom_terminal_outcomes", []
        )
        key = (row["evidence_id"], row["atom_id"], row["outcome"])
        if not any(
            (item.get("evidence_id"), item.get("atom_id"), item.get("outcome"))
            == key for item in rows
        ) and len(rows) < MAX_STAGE_ITEMS:
            rows.append(row)

    def record_fact_terminal_outcome(
        self,
        *,
        requirement_id: str,
        evidence_id: str,
        outcome: str,
        reason_codes: Iterable[str] = (),
        fact_id: str = "",
        source_fact_id: str = "",
        atom_id: str = "",
    ) -> None:
        """Record exactly the terminal state of one context/requirement path."""
        if outcome not in {
            "FACT_CREATED", "REJECTED", "INCOMPATIBLE", "NOT_APPLICABLE",
            "DERIVED_REQUIREMENT", "REQUEST_CONDITION",
            "COMPATIBLE_ALTERNATIVE", "EMITTED_DIRECT",
            "USED_BY_DERIVED_FACT", "COMPATIBLE_ALTERNATIVE_NOT_SELECTED",
            "REJECTED_AFTER_SELECTION", "REOBSERVED_SOURCE_FACT",
        }:
            raise ValueError("invalid typed-fact terminal outcome")
        stage = self.stages["typed_fact_extraction"]
        rows = stage.setdefault("terminal_outcomes", [])
        row = {
            "requirement_id": _bounded(requirement_id, 160),
            "evidence_id": _bounded(evidence_id, 160),
            "outcome": outcome,
            "reason_codes": _safe(tuple(dict.fromkeys(
                str(item) for item in reason_codes
            ))),
            "fact_id": _bounded(fact_id, 160),
            "source_fact_id": _bounded(source_fact_id, 160),
            "atom_id": _bounded(atom_id, 160),
        }
        # Terminal ownership is a source-fact/requirement path, not merely a
        # structural evidence unit.  Multiple candidates from one unit must
        # remain independently auditable after a single best fact is chosen.
        key = (
            row["requirement_id"], row["evidence_id"],
            row["source_fact_id"] or row["fact_id"] or row["atom_id"],
        )
        existing_index = next((
            index for index, existing in enumerate(rows)
            if (
                existing.get("requirement_id"), existing.get("evidence_id"),
                existing.get("source_fact_id")
                or existing.get("fact_id")
                or existing.get("atom_id"),
            ) == key
        ), None)
        if existing_index is None:
            if len(rows) < MAX_STAGE_ITEMS:
                rows.append(row)
        else:
            # A concrete selection terminal supersedes only a provisional
            # state for this exact source-fact/requirement path.  It must not
            # be overwritten by a later duplicate observation.
            existing = rows[existing_index]
            priority = {
                "REJECTED": 1, "INCOMPATIBLE": 1, "NOT_APPLICABLE": 1,
                "DERIVED_REQUIREMENT": 1, "REQUEST_CONDITION": 1,
                "FACT_CREATED": 2, "COMPATIBLE_ALTERNATIVE": 2,
                "EMITTED_DIRECT": 3, "USED_BY_DERIVED_FACT": 3,
                "COMPATIBLE_ALTERNATIVE_NOT_SELECTED": 3,
                "REJECTED_AFTER_SELECTION": 3,
                "REOBSERVED_SOURCE_FACT": 3,
            }
            if priority[outcome] >= priority.get(str(existing.get("outcome")), 0):
                rows[existing_index] = row

    def record_fact_rejection(
        self,
        *,
        requirement_id: str,
        evidence_id: str,
        reason_codes: Iterable[str],
        source_fact_id: str = "",
        atom_id: str = "",
    ) -> None:
        """Record a proposed fact's field-level rejection at extraction time."""
        stage = self.stages["typed_fact_extraction"]
        rejections = stage.setdefault("rejections", [])
        reason_codes = tuple(dict.fromkeys(str(item) for item in reason_codes))
        row = {
            "requirement_id": _bounded(requirement_id, 160),
            # C12 overloaded this legacy field with a structural evidence ID.
            # Preserve it for older consumers, but make the actual provenance
            # explicit for Phase 2C13 audits.
            "source_candidate_id": _bounded(evidence_id, 160),
            "source_evidence_unit_id": _bounded(evidence_id, 160),
            "source_fact_id": _bounded(source_fact_id, 160),
            "atom_id": _bounded(atom_id, 160),
            "reason_codes": _safe(reason_codes),
            "rejection_taxonomy": _rejection_taxonomy(reason_codes),
        }
        if row not in rejections and len(rejections) < MAX_STAGE_ITEMS:
            rejections.append(row)
        structural_codes = {
            "proposition_boundary_unresolved",
            "table_row_boundary_unresolved",
            "table_column_role_unresolved",
            "cross_row_binding_rejected",
            "list_item_boundary_unresolved",
            "cross_item_binding_rejected",
            "chart_category_boundary_unresolved",
            "cross_category_binding_rejected",
            "structural_parent_missing",
            "structural_type_transition_mismatch",
            "structural_binding_ambiguous",
            "structural_argument_mismatch",
            "structural_value_unbound",
        }
        structural_reasons = tuple(
            reason for reason in row["reason_codes"]
            if str(reason) in structural_codes
        )
        if structural_reasons:
            structural_stage = self.stages[
                "structural_evidence_construction"
            ]
            structural_rejections = structural_stage.setdefault(
                "rejections", []
            )
            structural_row = {
                "requirement_id": row["requirement_id"],
                "structural_unit_id": row["source_candidate_id"],
                "reason_codes": list(structural_reasons),
            }
            if (
                structural_row not in structural_rejections
                and len(structural_rejections) < MAX_STAGE_ITEMS
            ):
                structural_rejections.append(structural_row)

    def record_source_fact_construction(
        self,
        fact: Any,
        *,
        outcome: str,
        reason_codes: Iterable[str] = (),
    ) -> None:
        """Record a source-only fact before any request compatibility check."""
        if outcome not in {"CREATED", "REJECTED"}:
            raise ValueError("invalid source-fact construction outcome")
        raw = fact.as_prompt_dict() if hasattr(fact, "as_prompt_dict") else _safe(fact)
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        atom = raw.get("evidence_atom") or {}
        row = {
            "source_fact_id": _bounded(
                raw.get("source_fact_id") or raw.get("fact_id"), 160
            ),
            "source_fact_occurrence_id": _bounded(
                raw.get("source_fact_occurrence_id"), 160
            ),
            "source_fact_occurrence_provenance": _safe(
                raw.get("source_fact_occurrence_provenance") or ()
            ),
            "fact_id": _bounded(raw.get("fact_id"), 160),
            "outcome": outcome,
            "reason_codes": _safe(tuple(dict.fromkeys(str(item) for item in reason_codes))),
            "evidence_id": _bounded(raw.get("evidence_id"), 160),
            "candidate_id": _bounded(raw.get("candidate_id"), 160),
            "source_evidence_unit_ids": _safe(raw.get("source_evidence_unit_ids") or ()),
            "atom_id": _bounded(raw.get("evidence_atom_id"), 160),
            "atom_text_sha256": _bounded(
                (atom.get("text_sha256") if isinstance(atom, Mapping) else ""), 160
            ),
            "source_span": _safe(raw.get("evidence_span") or ()),
            "source_span_sha256": _bounded(raw.get("evidence_span_sha256"), 160),
            "local_text_sha256": _bounded(
                raw.get("evidence_span_sha256")
                or (atom.get("text_sha256") if isinstance(atom, Mapping) else ""),
                160,
            ),
            "numeric_binding_provenance": _safe(
                raw.get("numeric_binding_provenance") or {}
            ),
            "structural_type": _bounded(raw.get("structural_type"), 80),
            "value": _bounded(raw.get("value"), 160),
            "unit": _bounded(raw.get("unit"), 120),
            "scalar_role": _semantic_enum(
                raw.get("scalar_role"), "not_applicable",
            ),
            "quantity_kind": _semantic_enum(raw.get("quantity_kind")),
            "statistical_role": _semantic_enum(raw.get("statistical_role")),
            "aggregation_role": _semantic_enum(raw.get("aggregation_role")),
            "source_semantic_fields": _source_semantic_fields(
                raw.get("source_semantic_fields"),
            ),
            "fact_family": _bounded(raw.get("fact_type") or "numeric", 80),
            "semantic_completeness": {
                "status": "complete" if outcome == "CREATED" else "incomplete",
                "reason_codes": _safe(tuple(dict.fromkeys(
                    str(item) for item in reason_codes
                ))),
            },
        }
        stage = self.stages["typed_fact_extraction"]
        rows = stage.setdefault("source_facts", [])
        key = (row["source_fact_id"], row["outcome"])
        if not any((item.get("source_fact_id"), item.get("outcome")) == key for item in rows):
            if len(rows) < MAX_STAGE_ITEMS:
                rows.append(row)

    def record_source_fact_occurrence_observation(
        self,
        *,
        observed_fact: Any,
        authoritative_fact: Any,
        outcome: str,
    ) -> None:
        """Record structural discovery separately from authoritative emission."""
        if outcome not in {
            "EMITTED",
            "REOBSERVED_SAME_STRUCTURAL_VIEW",
            "REOBSERVED_EQUIVALENT_STRUCTURAL_VIEW",
        }:
            raise ValueError("invalid source-fact occurrence observation outcome")
        observed = (
            observed_fact.as_prompt_dict()
            if hasattr(observed_fact, "as_prompt_dict") else _safe(observed_fact)
        )
        authoritative = (
            authoritative_fact.as_prompt_dict()
            if hasattr(authoritative_fact, "as_prompt_dict") else _safe(authoritative_fact)
        )
        observed = dict(observed) if isinstance(observed, Mapping) else {}
        authoritative = (
            dict(authoritative) if isinstance(authoritative, Mapping) else {}
        )
        atom = self._serialized_evidence_atom(observed.get("evidence_atom"))
        canonical_atom = authoritative.get("evidence_atom") or {}
        row = {
            "source_fact_occurrence_id": _bounded(
                observed.get("source_fact_occurrence_id"), 160,
            ),
            "source_fact_id": _bounded(
                authoritative.get("source_fact_id") or authoritative.get("fact_id"),
                160,
            ),
            "outcome": outcome,
            "observed_evidence_id": _bounded(observed.get("evidence_id"), 160),
            "observed_atom_id": _bounded(
                observed.get("evidence_atom_id") or (
                    atom.get("atom_id") if isinstance(atom, Mapping) else ""
                ),
                160,
            ),
            "canonical_evidence_id": _bounded(
                authoritative.get("evidence_id"), 160,
            ),
            "canonical_atom_id": _bounded(
                authoritative.get("evidence_atom_id") or (
                    canonical_atom.get("atom_id")
                    if isinstance(canonical_atom, Mapping) else ""
                ),
                160,
            ),
            "observed_source_fact_occurrence_provenance": _safe(
                observed.get("source_fact_occurrence_provenance") or ()
            ),
            "authoritative_source_fact_occurrence_provenance": _safe(
                authoritative.get("source_fact_occurrence_provenance") or ()
            ),
        }
        # C33: this existing observation/provenance projection is the first
        # repository-native downstream owner that can retain the exact atom
        # after the bounded aggregate atom list reaches its trace cap.  The
        # field is optional: absence remains absence, and this row never
        # participates in occurrence, semantic, compatibility, or ranking
        # authority.  It is copied from the already-present observed atom;
        # no downstream reconstruction or normalization is permitted.
        # Historical/legacy observations may carry only an atom hash.  That
        # partial mapping is not the native discriminator and must remain
        # absent rather than becoming an incomplete or synthetic identity.
        if atom and str(atom.get("atom_id") or "") == row["observed_atom_id"]:
            row["observed_evidence_atom"] = atom
        rows = self.stages["typed_fact_extraction"].setdefault(
            "source_fact_observations", []
        )
        key = (
            row["source_fact_occurrence_id"],
            row["outcome"],
            row["observed_evidence_id"],
            row["observed_atom_id"],
        )
        if not any(
            (
                item.get("source_fact_occurrence_id"),
                item.get("outcome"),
                item.get("observed_evidence_id"),
                item.get("observed_atom_id"),
            ) == key for item in rows
        ) and len(rows) < MAX_STAGE_ITEMS:
            rows.append(row)

    def record_fact_compatibility(
        self,
        *,
        requirement_id: str,
        evidence_id: str,
        source_fact: Any,
        compatible: bool,
        reason_codes: Iterable[str] = (),
        dimension_decisions: Optional[Mapping[str, str]] = None,
        admissibility_proof: Optional[Mapping[str, Mapping[str, Any]]] = None,
        first_material_mismatch: str = "",
    ) -> None:
        """Record the runtime admission decision and its field-level proof."""
        atom = getattr(source_fact, "evidence_atom", None)
        raw = (
            source_fact.as_prompt_dict()
            if hasattr(source_fact, "as_prompt_dict") else {}
        )
        raw = raw if isinstance(raw, Mapping) else {}
        decisions = dict(dimension_decisions or {})
        proof = {
            str(dimension): dict(value or {})
            for dimension, value in dict(admissibility_proof or {}).items()
        }
        # A recorder must never manufacture a compatible path on behalf of a
        # caller that did not provide canonical proof.  The fallback has no
        # required dimensions and is for historical direct-recorder tests;
        # normal runtime selection supplies a full proof from the canonical
        # admissibility authority.
        if not proof:
            proof = {
                "proposition_completeness": {
                    "requirement_value": "",
                    "source_value": "",
                    "decision": "NOT_REQUIRED",
                    "provenance": {},
                },
            }
        for dimension, item in proof.items():
            item.setdefault("semantic_owner", str(dimension))
            item.setdefault("hard_constraint_owner", "")
            item.setdefault("enforcement_path", "none")
            item.setdefault("enforcement_equivalence_status", "NOT_APPLICABLE")
        proof_failures = [
            dimension for dimension, item in proof.items()
            if item.get("requirement_value") and item.get("decision") != "MATCH"
        ]
        reason_list = list(dict.fromkeys(str(item) for item in reason_codes))
        if compatible and proof_failures:
            compatible = False
            reason_list.append("compatibility_proof_required_dimension_failed")
            if not first_material_mismatch:
                first_material_mismatch = proof_failures[0]
        for dimension, item in proof.items():
            decisions.setdefault(str(dimension), str(item.get("decision") or ""))
        for dimension in _COMPATIBILITY_DECISION_DIMENSIONS:
            decisions.setdefault(dimension, "NOT_REQUIRED")
        decisions["final"] = "COMPATIBLE" if compatible else "INCOMPATIBLE"
        row = {
            "requirement_id": _bounded(requirement_id, 160),
            "evidence_id": _bounded(evidence_id, 160),
            "source_fact_id": _bounded(
                getattr(source_fact, "source_fact_id", "")
                or raw.get("source_fact_id")
                or getattr(source_fact, "fact_id", ""), 160
            ),
            "atom_id": _bounded(
                getattr(atom, "atom_id", "")
                or raw.get("evidence_atom_id"), 160
            ),
            "compatible": bool(compatible),
            "reason_codes": _safe(tuple(dict.fromkeys(reason_list))),
            "dimension_decisions": _safe(decisions),
            "admissibility_proof": _safe(proof),
            "first_material_mismatch": _bounded(first_material_mismatch, 80),
            "selection_status": (
                "compatible_pending" if compatible else "incompatible"
            ),
        }
        stage = self.stages["typed_fact_extraction"]
        rows = stage.setdefault("compatibility", [])
        if row not in rows and len(rows) < MAX_STAGE_ITEMS:
            rows.append(row)

    def record_fact_compatibility_selection(
        self,
        *,
        requirement_id: str,
        evidence_id: str,
        source_fact_id: str,
        atom_id: str = "",
        selected: bool,
    ) -> None:
        """Finalize one compatible source-fact path without coalescing it."""
        for row in reversed(self.stages["typed_fact_extraction"].setdefault(
            "compatibility", []
        )):
            if (
                row.get("requirement_id") == _bounded(requirement_id, 160)
                and row.get("evidence_id") == _bounded(evidence_id, 160)
                and row.get("source_fact_id") == _bounded(source_fact_id, 160)
                and (not atom_id or row.get("atom_id") == _bounded(atom_id, 160))
                and row.get("compatible") is True
            ):
                row["selection_status"] = (
                    "selected_for_canonical"
                    if selected else "compatible_alternative"
                )
                return

    def record_source_fact_reobservation_terminal(
        self,
        *,
        requirement_id: str,
        source_fact_id: str,
    ) -> None:
        """Close observation-only paths for an already selected source fact.

        A C26 occurrence may be discovered by another structural view after
        its authoritative SOURCE_FACT is selected.  The compatibility path is
        still observable, but must not become a second candidate or remain a
        silent terminal gap.
        """
        for row in self.stages["typed_fact_extraction"].setdefault(
            "compatibility", []
        ):
            if not (
                row.get("requirement_id") == _bounded(requirement_id, 160)
                and row.get("source_fact_id") == _bounded(source_fact_id, 160)
                and row.get("compatible") is True
                and row.get("selection_status") == "compatible_pending"
            ):
                continue
            row["selection_status"] = "reobserved_source_fact"
            self.record_fact_terminal_outcome(
                requirement_id=requirement_id,
                evidence_id=str(row.get("evidence_id") or ""),
                outcome="REOBSERVED_SOURCE_FACT",
                reason_codes=("source_fact_reobserved_from_structural_view",),
                source_fact_id=source_fact_id,
                atom_id=str(row.get("atom_id") or ""),
            )

    def record_source_fact_rejection(
        self,
        *,
        evidence_id: str,
        candidate_id: str,
        reason_codes: Iterable[str],
        atom_id: str = "",
    ) -> None:
        """Record a source-bound construction rejection without inventing a fact."""
        row = {
            "source_fact_id": "",
            "source_fact_occurrence_id": "",
            # A rejected construction never established an authoritative
            # occurrence.  Preserve that distinction explicitly so the
            # source-fact row remains schema-complete without inventing
            # provenance for an absent fact.
            "source_fact_occurrence_provenance": [],
            "fact_id": "",
            "outcome": "REJECTED",
            "reason_codes": _safe(tuple(dict.fromkeys(str(item) for item in reason_codes))),
            "evidence_id": _bounded(evidence_id, 160),
            "candidate_id": _bounded(candidate_id, 160),
            "source_evidence_unit_ids": _safe((evidence_id,) if evidence_id else ()),
            "atom_id": _bounded(atom_id, 160),
            "atom_text_sha256": "",
            "source_span": [],
            "source_span_sha256": "",
            "local_text_sha256": "",
            "numeric_binding_provenance": {},
            "structural_type": "",
            "value": "",
            "unit": "",
            "scalar_role": "not_applicable",
            "quantity_kind": "unspecified",
            "statistical_role": "unspecified",
            "aggregation_role": "unspecified",
            "source_semantic_fields": _source_semantic_fields({}),
            "fact_family": "unknown",
            "semantic_completeness": {
                "status": "incomplete",
                "reason_codes": _safe(tuple(dict.fromkeys(
                    str(item) for item in reason_codes
                ))),
            },
        }
        rows = self.stages["typed_fact_extraction"].setdefault("source_facts", [])
        if row not in rows and len(rows) < MAX_STAGE_ITEMS:
            rows.append(row)

    def record_model_assisted_semantic_attempt(
        self,
        *,
        evidence_id: str,
        accepted: bool,
        reason_code: str,
        diagnostics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record one bounded evidence-only resolver attempt and validation."""
        stage = self.stages["model_assisted_evidence_semantics"]
        attempts = stage.setdefault("attempts", [])
        row = {
            "evidence_id": _bounded(evidence_id, 160),
            "accepted": bool(accepted),
            "reason_code": _bounded(reason_code, 160),
            **(
                _safe(dict(diagnostics))
                if isinstance(diagnostics, Mapping) else {}
            ),
        }
        if row not in attempts and len(attempts) < MAX_STAGE_ITEMS:
            attempts.append(row)
        stage["attempt_count"] = len(attempts)
        stage["real_call_started_count"] = sum(
            bool(item.get("resolver_call_started")) for item in attempts
        )
        stage["accepted_count"] = sum(
            bool(item.get("accepted")) for item in attempts
        )
        stage["status"] = "completed"
        stage["reason_code"] = (
            "model_assisted_fact_accepted"
            if stage["accepted_count"]
            else "resolver_attempts_rejected_with_precise_diagnostics"
        )

    def record_acceptance_decision(self, decision: Any) -> None:
        """Record the one route-independent post-render acceptance decision."""
        raw = decision.as_dict() if hasattr(decision, "as_dict") else _safe(decision)
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        accepted = bool(raw.get("accepted"))
        reasons = list(raw.get("reason_codes") or ())
        self.stages["question_answer_relevance"].update({
            "status": "completed" if raw.get("contextual_complete") else "failed",
            "reason_code": (
                "contextual_requirements_complete"
                if raw.get("contextual_complete") else "contextual_requirement_incomplete"
            ),
            "passed": bool(raw.get("contextual_complete")),
            "standalone_completeness": bool(raw.get("standalone_complete")),
            "reason_codes": _safe(reasons),
        })
        self.stages["claim_evidence_gate"].update({
            "status": "completed" if (
                raw.get("claim_fact_linkage")
                and raw.get("fact_evidence_linkage")
                and raw.get("claim_citation_linkage")
                and raw.get("factual_bindings")
            ) else "failed",
            "reason_code": (
                "fact_bound_claims_and_citations"
                if accepted else "fact_binding_or_citation_failed"
            ),
            "passed": accepted,
            "field_checks": _safe(raw),
        })
        self.stages["final_gate"].update({
            "status": "completed",
            "reason_code": "accepted" if accepted else (
                _bounded(reasons[0], 160) if reasons else "semantic_acceptance_failed"
            ),
            "final_decision": "answer" if accepted else "abstain",
            "acceptance_decision": _safe(raw),
        })

    def record_adaptive_result(self, result: Any) -> None:
        coverage = getattr(result, "coverage", None)
        if coverage is None:
            return
        contexts = []
        structural_units: list[dict[str, Any]] = []
        for unit in getattr(result, "evidence_units", ()) or ():
            raw_hit = getattr(unit, "raw_hit", None)
            evidence_id = str(getattr(unit, "id", "") or "")
            if evidence_id and isinstance(raw_hit, Mapping):
                self._evidence_to_candidate[evidence_id] = _candidate_id(raw_hit)
            context = getattr(unit, "evidence_context", None)
            if context is not None and hasattr(context, "as_dict"):
                contexts.append(_safe(context.as_dict()))
            structural_unit = getattr(unit, "structural_evidence_unit", None)
            if structural_unit is not None and hasattr(
                structural_unit, "safe_summary"
            ):
                structural_units.append(dict(structural_unit.safe_summary()))
        self.stages["evidence_context_construction"] = {
            "status": "completed" if contexts else "failed",
            "reason_code": (
                "evidence_contexts_constructed" if contexts
                else "evidence_context_missing"
            ),
            "context_count": len(contexts),
            "contexts": contexts[:MAX_STAGE_ITEMS],
        }
        facts = []
        requirement_by_id = {
            str(item.get("requirement_id") or ""): item
            for item in self.stages["requirement_construction"].get(
                "requirements", []
            )
        }
        for index, fact in enumerate(list(getattr(coverage, "typed_facts", ()) or ())[:MAX_STAGE_ITEMS], start=1):
            raw = fact.as_prompt_dict() if hasattr(fact, "as_prompt_dict") else _safe(fact)
            raw = dict(raw) if isinstance(raw, Mapping) else {}
            requirement = requirement_by_id.get(
                str(raw.get("requirement_id") or ""), {}
            )
            required_scope = str(
                requirement.get("scope_population")
                or (requirement.get("populations") or [""])[0]
                or ""
            )
            observed_scope = str(
                raw.get("evidence_population") or raw.get("population") or ""
            )
            from semantic_contracts import population_scope_compatibility
            scope_compatible, scope_result = population_scope_compatibility(
                required_scope,
                observed_scope,
            )
            facts.append({
                "fact_id": _bounded(raw.get("fact_id") or f"fact-{index}", 160),
                "source_fact_id": _bounded(
                    raw.get("source_fact_id") or raw.get("fact_id") or f"fact-{index}",
                    160,
                ),
                "requirement_id": _bounded(raw.get("requirement_id"), 160),
                "requirement_ids": _safe(raw.get("requirement_ids") or (raw.get("requirement_id"),)),
                "origin_requirement_id": _bounded(raw.get("origin_requirement_id"), 160),
                "fact_type": _bounded(raw.get("fact_type") or "numeric", 80),
                "subject": _bounded(raw.get("subject"), 360),
                "predicate": _bounded(raw.get("predicate") or raw.get("metric") or raw.get("facet"), 240),
                "object_text_hash": _content_hash(str(raw.get("object_text") or "")) if raw.get("object_text") else None,
                "value": _bounded(raw.get("value"), 160),
                "unit": _bounded(raw.get("unit"), 120),
                "numeric_role": _bounded(raw.get("numeric_role"), 80),
                "structural_type": _bounded(raw.get("structural_type"), 80),
                "fact_constructor": _bounded(
                    raw.get("fact_constructor")
                    or ("source_derived_text_relation" if raw.get("fact_type") == "text_proposition" else "source_derived_numeric"),
                    120,
                ),
                "evidence_atom_id": _bounded(raw.get("evidence_atom_id"), 160),
                "evidence_atom": _safe(raw.get("evidence_atom") or {}),
                "source_semantic_fields": _safe(raw.get("source_semantic_fields") or {
                    "subject": raw.get("evidence_subject"),
                    "predicate": raw.get("evidence_predicate"),
                    "target_entity": raw.get("evidence_target_entity"),
                    "population": raw.get("evidence_population"),
                    "period": raw.get("evidence_period"),
                    "unit": raw.get("evidence_unit"),
                    "numeric_role": raw.get("evidence_numeric_role"),
                    "polarity": raw.get("evidence_polarity"),
                }),
                "relation_arguments": _safe(raw.get("relation_arguments") or ()),
                "row_column_bound": bool(raw.get("row_column_bound", True)),
                "row_header": _bounded(raw.get("row_header"), 240),
                "column_header": _bounded(raw.get("column_header"), 240),
                "evidence_span": _safe(raw.get("evidence_span") or ()),
                "evidence_span_sha256": _bounded(
                    raw.get("evidence_span_sha256")
                    or raw.get("proposition_sha256"), 160
                ),
                "proposition_start": _safe(raw.get("proposition_start")),
                "proposition_end": _safe(raw.get("proposition_end")),
                "period": _bounded(raw.get("period"), 120),
                "population": _bounded(raw.get("population") or raw.get("facet"), 240),
                "required_scope_population": _bounded(required_scope, 240),
                "observed_scope_population": _bounded(observed_scope, 240),
                "scope_compatibility_result": scope_result,
                "scope_compatibility_passed": scope_compatible,
                "direction": _bounded(raw.get("direction"), 80),
                "negation": bool(raw.get("negation", False)),
                "qualifiers": _safe(raw.get("qualifiers") or ()),
                "source_evidence_id": _bounded(raw.get("evidence_id"), 160),
                "source_candidate_id": _bounded(
                    raw.get("candidate_id")
                    or self._evidence_to_candidate.get(str(raw.get("evidence_id") or "")),
                    160,
                ),
                "source_evidence_unit_ids": _safe(raw.get("source_evidence_unit_ids") or ()),
                "source_candidate_ids": _safe(raw.get("source_candidate_ids") or ()),
                "source_typed_fact_ids": _safe(raw.get("source_typed_fact_ids") or ()),
                "operator_derivation_id": _bounded(raw.get("operator_derivation_id"), 160),
                "evidence_context": _safe(raw.get("evidence_context") or {}),
                "model_assisted_semantics": bool(raw.get("model_assisted_semantics")),
                "structural_binding_valid": bool(
                    raw.get("structural_binding_valid", True)
                ),
                "structural_validation_errors": _safe(
                    raw.get("structural_validation_errors") or ()
                ),
                "extraction_confidence": _safe(getattr(fact, "association_quality", None)),
            })
        fact_ids_by_unit: dict[str, list[str]] = {}
        for fact in facts:
            unit_ids = list(fact.get("source_evidence_unit_ids") or ())
            if not unit_ids and fact.get("source_evidence_id"):
                unit_ids = [fact["source_evidence_id"]]
            for unit_id in unit_ids:
                fact_ids_by_unit.setdefault(str(unit_id), []).append(
                    str(fact.get("fact_id") or "")
                )
        for unit in structural_units:
            unit_id = str(unit.get("structural_unit_id") or "")
            unit["derived_fact_ids"] = list(dict.fromkeys(
                fact_ids_by_unit.get(unit_id, ())
            ))
        structural_rejections = list(
            self.stages["structural_evidence_construction"].get(
                "rejections", []
            )
        ) + [
            {
                "structural_unit_id": str(fact.get("source_evidence_id") or ""),
                "reason_codes": fact.get("structural_validation_errors") or [],
            }
            for fact in facts
            if not fact.get("structural_binding_valid", True)
        ]
        structural_stage = self.stages["structural_evidence_construction"]
        structural_stage.update({
            "status": "completed" if structural_units else "failed",
            "reason_code": (
                "bounded_structural_units_constructed"
                if structural_units else "structural_evidence_units_missing"
            ),
            "unit_count": len(structural_units),
            "units": _safe(structural_units[:MAX_STAGE_ITEMS]),
            "rejections": _safe(structural_rejections[:MAX_STAGE_ITEMS]),
        })
        for fact in facts:
            candidate_id = str(fact.get("source_candidate_id") or "")
            if candidate_id in self._selection_indexes:
                self.stages["candidate_selection"]["decisions"][
                    self._selection_indexes[candidate_id]
                ]["produced_fact"] = True
        prior_rejections = list(
            self.stages["typed_fact_extraction"].get("rejections", [])
        )
        prior_atoms = list(
            self.stages["typed_fact_extraction"].get("atoms", [])
        )
        prior_source_facts = list(
            self.stages["typed_fact_extraction"].get("source_facts", [])
        )
        # C27: this is a direct runtime projection recorded when the
        # occurrence registry resolves an observed structural view.  Preserve
        # it verbatim through finalization; do not reconstruct or reselect a
        # target from source semantics here.
        prior_source_fact_observations = list(
            self.stages["typed_fact_extraction"].get(
                "source_fact_observations", []
            )
        )
        prior_compatibility = list(
            self.stages["typed_fact_extraction"].get("compatibility", [])
        )
        prior_outcomes = list(
            self.stages["typed_fact_extraction"].get("terminal_outcomes", [])
        )
        prior_atom_outcomes = list(
            self.stages["typed_fact_extraction"].get("atom_terminal_outcomes", [])
        )
        fact_atom_ids = {
            str(fact.get("evidence_atom_id") or "") for fact in facts
        } - {""}
        terminal_fact_ids = {
            str(row.get("fact_id") or "") for row in prior_outcomes
        } - {""}
        terminal_gaps = sum(
            bool(fact.get("evidence_atom_id"))
            and str(fact.get("fact_id") or "") not in terminal_fact_ids
            for fact in facts
        )
        unattributed_no_compatible = sum(
            any(str(reason) == "no_compatible_source_fact" for reason in (
                row.get("reason_codes") or ()
            ))
            and not any(
                str(reason) != "no_compatible_source_fact" for reason in (
                    row.get("reason_codes") or ()
                ))
            for row in prior_outcomes
        )
        selected_paths = {
            (
                str(row.get("requirement_id") or ""),
                str(row.get("evidence_id") or ""),
                str(row.get("source_fact_id") or ""),
            )
            for row in prior_compatibility
            if row.get("selection_status") == "selected_for_canonical"
            and str(row.get("source_fact_id") or "")
        }
        selected_terminals = {
            (
                str(row.get("requirement_id") or ""),
                str(row.get("evidence_id") or ""),
                str(row.get("source_fact_id") or ""),
            ): str(row.get("outcome") or "")
            for row in prior_outcomes
            if str(row.get("source_fact_id") or "")
            and str(row.get("outcome") or "") in {
                "EMITTED_DIRECT", "USED_BY_DERIVED_FACT",
                "COMPATIBLE_ALTERNATIVE_NOT_SELECTED",
                "REJECTED_AFTER_SELECTION",
            }
        }
        unaccounted_selected = selected_paths - set(selected_terminals)
        complete_source_facts = [
            row for row in prior_source_facts
            if row.get("outcome") == "CREATED"
            and str((row.get("semantic_completeness") or {}).get("status") or "")
            == "complete"
        ]
        compatible_source_facts = [
            row for row in prior_compatibility if row.get("compatible") is True
        ]
        fact_admissibility_failed = bool(
            complete_source_facts and not compatible_source_facts
        )
        self.stages["fact_admissibility"] = {
            "status": (
                "failed" if fact_admissibility_failed else "completed"
                if complete_source_facts else "skipped"
            ),
            "reason_code": (
                "complete_source_fact_not_admissible"
                if fact_admissibility_failed else "compatible_source_fact_available"
                if compatible_source_facts else "no_complete_source_fact"
            ),
            "complete_source_fact_count": len(complete_source_facts),
            "compatible_source_fact_count": len(compatible_source_facts),
            "first_material_mismatches": _safe(tuple(dict.fromkeys(
                str(row.get("first_material_mismatch") or "")
                for row in prior_compatibility
                if row.get("compatible") is False
                and str(row.get("first_material_mismatch") or "")
            ))),
        }
        self.stages["typed_fact_extraction"] = {
            "status": (
                "completed" if facts or complete_source_facts else "skipped"
                if all(
                    str(getattr(item, "kind", "")) in {"list", "list_attributes"}
                    for item in getattr(coverage, "requirements", ())
                ) else "failed"
            ),
            "reason_code": (
                "typed_facts_extracted" if facts else
                "source_facts_constructed_no_admissible_requirement"
                if complete_source_facts else "typed_facts_not_required"
                if all(
                    str(getattr(item, "kind", "")) in {"list", "list_attributes"}
                    for item in getattr(coverage, "requirements", ())
                ) else "required_typed_facts_missing"
            ),
            "facts": facts,
            "source_facts": prior_source_facts,
            "source_fact_observations": prior_source_fact_observations,
            "compatibility": prior_compatibility,
            "rejections": prior_rejections,
            "atoms": prior_atoms,
            "atom_terminal_outcomes": prior_atom_outcomes,
            "terminal_outcomes": prior_outcomes,
            "atom_count": len(prior_atoms),
            "source_fact_count": sum(
                item.get("outcome") == "CREATED" for item in prior_source_facts
            ),
            "compatibility_count": len(prior_compatibility),
            "fact_atom_count": len(fact_atom_ids),
            "terminal_outcome_count": len(prior_outcomes),
            "atom_terminal_outcome_count": len(prior_atom_outcomes),
            "silent_terminal_gap_count": terminal_gaps,
            "selected_for_canonical_count": len(selected_paths),
            "selected_fact_terminal_counts": {
                outcome: sum(
                    observed == outcome
                    for observed in selected_terminals.values()
                )
                for outcome in (
                    "EMITTED_DIRECT", "USED_BY_DERIVED_FACT",
                    "COMPATIBLE_ALTERNATIVE_NOT_SELECTED",
                    "REJECTED_AFTER_SELECTION",
                )
            },
            "unaccounted_selected_fact_count": len(unaccounted_selected),
            "unattributed_no_compatible_source_fact_count": unattributed_no_compatible,
            "unattributed_compatibility_path_count": (
                _unattributed_compatibility_path_count(
                    prior_compatibility, prior_outcomes
                )
            ),
            "non_selected_compatible_alternative_count": sum(
                row.get("selection_status") == "compatible_alternative"
                for row in prior_compatibility
            ),
        }
        assisted_facts = [fact for fact in facts if fact.get("model_assisted_semantics")]
        prior_assisted = self.stages.get(
            "model_assisted_evidence_semantics", {}
        )
        assisted_attempts = list(prior_assisted.get("attempts", []))
        real_call_started_count = sum(
            bool(item.get("resolver_call_started"))
            for item in assisted_attempts
        )
        accepted_count = sum(
            bool(item.get("accepted")) for item in assisted_attempts
        )
        self.stages["model_assisted_evidence_semantics"] = {
            "status": (
                "completed" if assisted_facts else "failed"
                if assisted_attempts else "skipped"
            ),
            "reason_code": (
                "bounded_evidence_semantics_used" if assisted_facts
                else "bounded_evidence_semantics_rejected"
                if assisted_attempts else "not_used"
            ),
            "attempt_count": len(assisted_attempts),
            "real_call_started_count": real_call_started_count,
            "accepted_count": accepted_count,
            "attempts": assisted_attempts,
            "fact_count": len(assisted_facts),
            "facts": assisted_facts,
        }
        requirements = []
        for item in list(getattr(coverage, "requirements", ()) or ())[:MAX_STAGE_ITEMS]:
            requirements.append({
                "requirement_id": _bounded(getattr(item, "id", ""), 160),
                "satisfaction_mode": _bounded(
                    getattr(item, "satisfaction_mode", "SOURCE_FACT"), 80
                ),
                "status": str(_enum(getattr(item, "status", "missing"))),
                "evidence_ids": _safe(getattr(item, "evidence_ids", ())),
                "fact_ids": [
                    fact["fact_id"] for fact in facts
                    if getattr(item, "id", "") in fact.get("requirement_ids", ())
                ],
                "missing_fields": [],
                "partial_or_complete": "complete" if str(_enum(getattr(item, "status", ""))) == "covered" else "partial_or_missing",
                "reason_code": "requirement_" + str(_enum(getattr(item, "status", "missing"))),
            })
        self.stages["evidence_coverage"] = {
            "status": "completed" if getattr(coverage, "complete", False) else "failed",
            "reason_code": str(_enum(getattr(coverage, "reason_code", "coverage_unknown"))),
            "complete": bool(getattr(coverage, "complete", False)),
            "partial": 0 < float(getattr(coverage, "coverage_ratio", 0.0) or 0.0) < 1,
            "coverage_ratio": float(getattr(coverage, "coverage_ratio", 0.0) or 0.0),
            "mandatory_requirement_count": len(requirements),
            "coverage_row_count": len(requirements),
            "coverage_invariant_valid": bool(
                not getattr(coverage, "complete", False)
                or float(getattr(coverage, "coverage_ratio", 0.0) or 0.0) == 1.0
            ),
            "requirements": requirements,
            "targeted_repair_attempted": bool(
                getattr(coverage, "targeted_repair_attempted", False)
            ),
            "targeted_repair_requirement_ids": _safe(
                getattr(coverage, "targeted_repair_requirement_ids", ())
            ),
        }

    def record_operator(self, execution: Any, *, candidate_set_size: int = 0) -> None:
        accepted = bool(getattr(execution, "accepted", getattr(execution, "complete", True)))
        operator_facts = []
        operator_source_facts = list(
            getattr(execution, "facts", ()) or ()
        )[:MAX_STAGE_ITEMS]
        operator_sources_by_id: dict[str, Any] = {}
        operator_atom_ids: dict[str, str] = {}
        for source_fact in operator_source_facts:
            raw_source = (
                source_fact.as_prompt_dict()
                if hasattr(source_fact, "as_prompt_dict") else {}
            )
            raw_source = dict(raw_source) if isinstance(raw_source, Mapping) else {}
            source_fact_id = str(
                raw_source.get("source_fact_id")
                or raw_source.get("fact_id")
                or getattr(source_fact, "fact_id", "")
            )
            if not source_fact_id:
                continue
            operator_sources_by_id[source_fact_id] = source_fact
            operator_atom_ids[source_fact_id] = str(
                raw_source.get("evidence_atom_id") or ""
            )
            atom = raw_source.get("evidence_atom")
            if isinstance(atom, Mapping):
                self.record_evidence_atom(atom)
            self.record_source_fact_construction(
                source_fact,
                outcome="CREATED",
            )
        bound_facts: Iterable[Any] = ()
        precovered: set[str] = set()
        if self.execution_contract is not None:
            from semantic_contracts import bind_operator_output_facts
            bound_facts, internal_ids = bind_operator_output_facts(
                self.execution_contract, execution
            )
            precovered = set(internal_ids)
        canonical_requirements = {
            str(item.id): item
            for item in getattr(self.execution_contract, "requirements", ()) or ()
        }

        def source_admission(requirement_id: str, source_fact: Any) -> Any:
            """Route operator source facts through the direct-fact authority."""
            from enterprise_retrieval import (
                canonical_fact_admissibility,
                coverage_requirement_from_canonical_requirement,
            )
            canonical_requirement = canonical_requirements.get(str(requirement_id))
            if canonical_requirement is None:
                return None
            return canonical_fact_admissibility(
                coverage_requirement_from_canonical_requirement(
                    canonical_requirement
                ),
                canonical_requirement,
                source_fact,
            )
        requirement_scopes = {
            str(item.get("requirement_id") or ""): str(
                item.get("scope_population")
                or (item.get("populations") or [""])[0]
                or ""
            )
            for item in self.stages["requirement_construction"].get(
                "requirements", []
            )
        }
        for index, fact in enumerate(list(bound_facts)[:MAX_STAGE_ITEMS], start=1):
            raw = fact.as_dict() if hasattr(fact, "as_dict") else _safe(fact)
            if hasattr(fact, "as_prompt_dict"):
                raw = fact.as_prompt_dict()
            raw = dict(raw) if isinstance(raw, Mapping) else {}
            required_scope = requirement_scopes.get(
                str(raw.get("requirement_id") or ""),
                "",
            )
            observed_scope = str(
                raw.get("evidence_population") or raw.get("population") or ""
            )
            from semantic_contracts import population_scope_compatibility
            scope_compatible, scope_result = population_scope_compatibility(
                required_scope,
                observed_scope,
            )
            operator_facts.append({
                "fact_id": _bounded(raw.get("fact_id") or f"operator-fact-{index}", 160),
                "requirement_id": _bounded(raw.get("requirement_id"), 160),
                "requirement_ids": _safe(raw.get("requirement_ids") or (raw.get("requirement_id"),)),
                "subject": _bounded(raw.get("entity") or raw.get("subject"), 360),
                "predicate": _bounded(raw.get("predicate"), 240),
                "metric": _bounded(raw.get("metric"), 240),
                "value": _bounded(raw.get("value"), 160),
                "unit": _bounded(raw.get("unit"), 120),
                "period": _bounded(raw.get("period"), 120),
                "population": _bounded(raw.get("population"), 240),
                "direction": _bounded(raw.get("direction"), 80),
                "negation": bool(raw.get("negated", False)),
                "qualifiers": _safe(raw.get("qualifiers") or ()),
                "source_evidence_id": _bounded(raw.get("evidence_id"), 160),
                "source_candidate_id": _bounded(raw.get("candidate_id"), 160),
                "source_evidence_unit_ids": _safe(raw.get("source_evidence_unit_ids") or ()),
                "source_candidate_ids": _safe(raw.get("source_candidate_ids") or ()),
                "source_typed_fact_ids": _safe(raw.get("source_typed_fact_ids") or ()),
                "operator_derivation_id": _bounded(raw.get("operator_derivation_id"), 160),
                "derivation_description": _bounded(raw.get("derivation_description"), 480),
                "origin_debug_id": _bounded(raw.get("origin_debug_id"), 160),
                "evidence_context": _safe(raw.get("evidence_context") or {}),
                "structural_binding_valid": bool(
                    raw.get("structural_binding_valid", True)
                ),
                "structural_validation_errors": _safe(
                    raw.get("structural_validation_errors") or ()
                ),
                "fact_constructor": _bounded(
                    raw.get("fact_constructor")
                    or "operator_derived_from_source_facts",
                    120,
                ),
                "source_semantic_fields": _safe(raw.get("source_semantic_fields") or {
                    "subject": raw.get("evidence_subject"),
                    "predicate": raw.get("evidence_predicate"),
                    "target_entity": raw.get("evidence_target_entity"),
                    "population": raw.get("evidence_population"),
                    "period": raw.get("evidence_period"),
                    "unit": raw.get("evidence_unit"),
                    "numeric_role": raw.get("evidence_numeric_role"),
                    "polarity": raw.get("evidence_polarity"),
                }),
                "relation_arguments": _safe(raw.get("relation_arguments") or ()),
                "required_scope_population": _bounded(required_scope, 240),
                "observed_scope_population": _bounded(observed_scope, 240),
                "scope_compatibility_result": scope_result,
                "scope_compatibility_passed": scope_compatible,
            })
        matrix_requirement_ids = [
            str(item.get("requirement_id") or "")
            for item in self.stages["requirement_construction"].get(
                "requirements", []
            )
            if str(item.get("kind") or "") in {
                "category_set", "category_value_bindings",
                "common_metric_unit_period_population",
            }
        ]
        selected_source_ids: set[str] = set()
        for fact in operator_facts:
            source_ids = [
                str(value) for value in fact.get("source_typed_fact_ids", [])
                if str(value) in operator_sources_by_id
            ]
            if not source_ids:
                source_ids = [
                    source_id for source_id, source_fact in operator_sources_by_id.items()
                    if str(getattr(source_fact, "evidence_id", "") or "")
                    == str(fact.get("source_evidence_id") or "")
                ][:1]
            for source_id in source_ids:
                source_fact = operator_sources_by_id[source_id]
                selected_source_ids.add(source_id)
                canonical_requirement = canonical_requirements.get(
                    str(fact.get("requirement_id") or "")
                )
                if (
                    canonical_requirement is not None
                    and str(getattr(
                        getattr(canonical_requirement, "satisfaction_mode", ""), "value", ""
                    ) or getattr(canonical_requirement, "satisfaction_mode", ""))
                    != "SOURCE_FACT"
                ):
                    # A winner/composite requirement is not a source-fact
                    # requirement.  Preserve its source lifecycle without
                    # fabricating a compatibility PASS for a different
                    # admission mode.
                    self.record_fact_terminal_outcome(
                        requirement_id=str(fact.get("requirement_id") or ""),
                        evidence_id=str(fact.get("source_evidence_id") or ""),
                        outcome="USED_BY_DERIVED_FACT",
                        reason_codes=(
                            "operator_composite_requirement_uses_source_fact",
                        ),
                        fact_id=str(fact.get("fact_id") or ""),
                        source_fact_id=source_id,
                        atom_id=operator_atom_ids.get(source_id, ""),
                    )
                    continue
                admission = source_admission(
                    str(fact.get("requirement_id") or ""), source_fact,
                )
                compatible = bool(admission and admission.compatible)
                self.record_fact_compatibility(
                    requirement_id=str(fact.get("requirement_id") or ""),
                    evidence_id=str(fact.get("source_evidence_id") or ""),
                    source_fact=source_fact,
                    compatible=compatible,
                    reason_codes=(
                        admission.reason_codes
                        if admission is not None else ("operator_requirement_unresolved",)
                    ),
                    dimension_decisions=(
                        admission.dimension_decisions if admission is not None else None
                    ),
                    admissibility_proof=(
                        admission.dimension_proof if admission is not None else None
                    ),
                    first_material_mismatch=(
                        admission.first_material_mismatch if admission is not None else ""
                    ),
                )
                if compatible:
                    self.record_fact_compatibility_selection(
                        requirement_id=str(fact.get("requirement_id") or ""),
                        evidence_id=str(fact.get("source_evidence_id") or ""),
                        source_fact_id=source_id,
                        atom_id=operator_atom_ids.get(source_id, ""),
                        selected=True,
                    )
                self.record_fact_terminal_outcome(
                    requirement_id=str(fact.get("requirement_id") or ""),
                    evidence_id=str(fact.get("source_evidence_id") or ""),
                    outcome=("USED_BY_DERIVED_FACT" if compatible else "INCOMPATIBLE"),
                    reason_codes=(
                        ("operator_derived_fact_uses_source_fact",)
                        if compatible else admission.reason_codes
                        if admission is not None else ("operator_requirement_unresolved",)
                    ),
                    fact_id=str(fact.get("fact_id") or ""),
                    source_fact_id=source_id,
                    atom_id=operator_atom_ids.get(source_id, ""),
                )
        matrix_requirement_id = next(iter(matrix_requirement_ids), "")
        for source_id, source_fact in operator_sources_by_id.items():
            if source_id in selected_source_ids:
                continue
            evidence_id = str(getattr(source_fact, "evidence_id", "") or "")
            if matrix_requirement_id:
                canonical_requirement = canonical_requirements.get(
                    matrix_requirement_id
                )
                if (
                    canonical_requirement is not None
                    and str(getattr(
                        getattr(canonical_requirement, "satisfaction_mode", ""), "value", ""
                    ) or getattr(canonical_requirement, "satisfaction_mode", ""))
                    != "SOURCE_FACT"
                ):
                    self.record_fact_compatibility(
                        requirement_id=matrix_requirement_id,
                        evidence_id=evidence_id,
                        source_fact=source_fact,
                        compatible=False,
                        reason_codes=("operator_composite_requirement_not_source_fact",),
                        dimension_decisions={"final": "INCOMPATIBLE"},
                        admissibility_proof={
                            "proposition_completeness": {
                                "requirement_value": "",
                                "source_value": "",
                                "decision": "NOT_REQUIRED",
                                "provenance": {},
                            },
                        },
                    )
                    self.record_fact_terminal_outcome(
                        requirement_id=matrix_requirement_id,
                        evidence_id=evidence_id,
                        outcome="COMPATIBLE_ALTERNATIVE",
                        reason_codes=(
                            "operator_composite_requirement_candidate_not_selected",
                        ),
                        fact_id=str(getattr(source_fact, "fact_id", "") or ""),
                        source_fact_id=source_id,
                        atom_id=operator_atom_ids.get(source_id, ""),
                    )
                    continue
                admission = source_admission(matrix_requirement_id, source_fact)
                compatible = bool(admission and admission.compatible)
                self.record_fact_compatibility(
                    requirement_id=matrix_requirement_id,
                    evidence_id=evidence_id,
                    source_fact=source_fact,
                    compatible=compatible,
                    reason_codes=(
                        admission.reason_codes if admission is not None else (
                            "operator_requirement_unresolved",
                        )
                    ),
                    dimension_decisions=(
                        admission.dimension_decisions if admission is not None else None
                    ),
                    admissibility_proof=(
                        admission.dimension_proof if admission is not None else None
                    ),
                    first_material_mismatch=(
                        admission.first_material_mismatch if admission is not None else ""
                    ),
                )
                if compatible:
                    self.record_fact_compatibility_selection(
                        requirement_id=matrix_requirement_id,
                        evidence_id=evidence_id,
                        source_fact_id=source_id,
                        atom_id=operator_atom_ids.get(source_id, ""),
                        selected=False,
                    )
                self.record_fact_terminal_outcome(
                    requirement_id=matrix_requirement_id,
                    evidence_id=evidence_id,
                    outcome=(
                        "COMPATIBLE_ALTERNATIVE" if compatible else "INCOMPATIBLE"
                    ),
                    reason_codes=(
                        ("operator_candidate_not_selected",) if compatible else
                        admission.reason_codes if admission is not None else (
                            "operator_requirement_unresolved",
                        )
                    ),
                    fact_id=str(getattr(source_fact, "fact_id", "") or ""),
                    source_fact_id=source_id,
                    atom_id=operator_atom_ids.get(source_id, ""),
                )
        typed_stage = self.stages["typed_fact_extraction"]
        typed_stage.update({
            "status": "completed" if operator_facts else "failed" if accepted else "skipped",
            "reason_code": "operator_typed_facts_extracted" if operator_facts else "operator_typed_facts_missing",
            "facts": operator_facts,
            "operator_source_fact_count": len(operator_sources_by_id),
            "operator_fact_count": len(operator_facts),
            "unattributed_compatibility_path_count": (
                _unattributed_compatibility_path_count(
                    typed_stage.get("compatibility", []),
                    typed_stage.get("terminal_outcomes", []),
                )
            ),
            "non_selected_compatible_alternative_count": sum(
                row.get("selection_status") == "compatible_alternative"
                for row in typed_stage.get("compatibility", [])
            ),
        })
        operator_contexts = [
            fact["evidence_context"] for fact in operator_facts
            if fact.get("evidence_context")
        ]
        self.stages["evidence_context_construction"] = {
            "status": "completed" if operator_contexts else "failed",
            "reason_code": (
                "operator_evidence_contexts_constructed"
                if operator_contexts else "operator_evidence_context_missing"
            ),
            "context_count": len(operator_contexts),
            "contexts": operator_contexts,
        }
        structural_units_by_id: dict[str, dict[str, Any]] = {}
        for fact in operator_facts:
            context = fact.get("evidence_context") or {}
            structural_unit = context.get("structural_evidence_unit") or {}
            unit_id = str(structural_unit.get("structural_unit_id") or "")
            if not unit_id:
                continue
            row = structural_units_by_id.setdefault(unit_id, dict(structural_unit))
            row.setdefault("derived_fact_ids", [])
            row["derived_fact_ids"].append(fact["fact_id"])
        structural_rejections = [
            {
                "structural_unit_id": fact.get("source_evidence_id"),
                "reason_codes": fact.get("structural_validation_errors") or [],
            }
            for fact in operator_facts
            if not fact.get("structural_binding_valid", True)
        ]
        prior_structural_stage = dict(
            self.stages.get("structural_evidence_construction") or {}
        )
        self.stages["structural_evidence_construction"] = {
            **{
                key: value for key, value in prior_structural_stage.items()
                if key in {"candidate_decomposition_events"}
            },
            "status": "completed" if structural_units_by_id else "failed",
            "reason_code": (
                "operator_structural_units_constructed"
                if structural_units_by_id else "operator_structural_units_missing"
            ),
            "unit_count": len(structural_units_by_id),
            "units": _safe(tuple(structural_units_by_id.values())),
            "rejections": _safe(structural_rejections),
        }
        contract_requirements = self.stages["requirement_construction"].get("requirements", [])
        self.stages["evidence_coverage"] = {
            "status": "completed" if accepted else "failed",
            "reason_code": "operator_typed_coverage_complete" if accepted else _bounded(getattr(execution, "reason_code", "operator_coverage_failed"), 160),
            "complete": accepted,
            "partial": bool(operator_facts) and not accepted,
            "coverage_ratio": 1.0 if accepted else 0.0,
            "mandatory_requirement_count": len(contract_requirements),
            "coverage_row_count": len(contract_requirements),
            "coverage_invariant_valid": bool(not accepted or contract_requirements),
            "requirements": [
                {
                    "requirement_id": item.get("requirement_id"),
                    "status": "covered" if accepted else "missing",
                    "fact_ids": [
                        fact["fact_id"] for fact in operator_facts
                        if item.get("requirement_id") in fact.get("requirement_ids", ())
                    ],
                    "operator_internal_precoverage": item.get("requirement_id") in precovered,
                    "reason_code": "operator_requirement_covered" if accepted else "operator_requirement_missing",
                }
                for item in contract_requirements[:MAX_STAGE_ITEMS]
            ],
        }
        self.stages["operator_execution"] = {
            "status": "completed" if accepted else "failed",
            "reason_code": _bounded(getattr(execution, "reason_code", "operator_executed"), 160),
            "operator_name": str(_enum(getattr(execution, "operator", getattr(execution, "operation", "")))),
            "candidate_set_size": int(candidate_set_size),
            "deterministic_inputs": {
                "fact_count": len(operator_facts),
                "evidence_hit_indexes": _safe(getattr(execution, "evidence_hit_indexes", ())),
            },
            "deterministic_result": {
                "present": getattr(execution, "result", None) is not None,
                "winner_attribute_count": len(getattr(execution, "result_attributes", ()) or ()),
                "candidate_order_count": len(getattr(execution, "candidate_order", ()) or ()),
                "operator_provenance_gap_count": sum(
                    not (
                        fact.get("source_evidence_id")
                        and not str(fact.get("source_evidence_id")).startswith("operator-hit-")
                        and fact.get("source_candidate_id")
                        and fact.get("source_typed_fact_ids")
                        and fact.get("operator_derivation_id")
                        and fact.get("evidence_context")
                    )
                    for fact in operator_facts
                ),
            },
            "rejected_associations": [],
            "completeness_decision": accepted,
        }

    def record_synthesis(self, result: Any, *, bundle: Any = None, deterministic_repair: bool = False) -> None:
        budget_fields = {
            key: self.stages["synthesis"][key]
            for key in ("prompt_chars", "prompt_budget")
            if key in self.stages["synthesis"]
        }
        bindings = list(getattr(result, "claim_bindings", ()) or ())[:MAX_STAGE_ITEMS]
        bundle_evidence_ids = {
            str(getattr(getattr(entry, "unit", None), "id", "") or "")
            for entry in getattr(bundle, "evidence", ()) or ()
        }
        supplied_evidence_ids = {
            str(value) for value in getattr(result, "evidence_unit_ids", ()) or ()
            if value
        }
        used_evidence_ids = {
            str(value)
            for binding in bindings
            for value in getattr(binding, "evidence_ids", ()) or ()
            if value
        }
        for evidence_id, field in (
            *((value, "selected_for_evidence_bundle") for value in bundle_evidence_ids),
            *((value, "supplied_to_generation") for value in supplied_evidence_ids),
            *((value, "used_in_claim") for value in used_evidence_ids),
        ):
            candidate_id = self._evidence_to_candidate.get(evidence_id, "")
            if candidate_id in self._selection_indexes:
                self.stages["candidate_selection"]["decisions"][
                    self._selection_indexes[candidate_id]
                ][field] = True
        self.stages["synthesis"] = {
            **budget_fields,
            "status": "completed" if getattr(result, "complete", False) or getattr(result, "partial", False) else "failed",
            "reason_code": str(_enum(getattr(result, "reason_code", "synthesis_observed"))),
            "fact_ids_supplied": [
                _bounded(getattr(fact, "fact_id", ""), 160)
                for fact in getattr(getattr(bundle, "coverage", None), "typed_facts", ()) or ()
                if getattr(fact, "fact_id", "")
            ],
            "evidence_ids_supplied": _safe(getattr(result, "evidence_unit_ids", ())),
            "generation_attempt_count": int(getattr(result, "generation_attempts", 0) or 0),
            "deterministic_repair_used": bool(
                deterministic_repair
                or getattr(result, "deterministic_repair_used", False)
            ),
            "answer_shape_validation": "passed" if getattr(result, "complete", False) else "failed_or_partial",
        }
        claims = []
        for index, binding in enumerate(bindings, start=1):
            claims.append({
                "claim_id": f"claim-{index}",
                "citation_id": None,
                "source_fact_id": None,
                "subject_check": "not_exposed",
                "predicate_check": "not_exposed",
                "value_check": "not_exposed",
                "unit_check": "not_exposed",
                "period_check": "not_exposed",
                "population_check": "not_exposed",
                "polarity_check": "not_exposed",
                "passed": bool(getattr(binding, "supported", False)),
                "reason_codes": ["claim_supported" if getattr(binding, "supported", False) else "claim_not_supported"],
            })
        self.stages["claim_evidence_gate"] = {
            "status": "completed" if claims and all(item["passed"] for item in claims) else "failed" if claims else "not_recorded",
            "reason_code": "all_claims_bound" if claims and all(item["passed"] for item in claims) else "claim_binding_incomplete",
            "claims": claims,
        }

    def record_context_budget(
        self,
        *,
        prompt_chars: int,
        prompt_budget: int,
        accepted: bool,
    ) -> None:
        self.stages["synthesis"] = {
            "status": "completed" if accepted else "skipped",
            "reason_code": (
                "realization_context_budget_valid"
                if accepted else "realization_context_budget_exceeded"
            ),
            "prompt_chars": int(prompt_chars),
            "prompt_budget": int(prompt_budget),
            "generation_attempt_count": 0,
        }

    def record_response(
        self,
        answer: str,
        citations: Iterable[Any],
        *,
        final_reason_code: str = "",
        response_decision: Any = None,
    ) -> None:
        citations = list(citations or ())[:64]
        selected_chunks = {
            str(identifier)
            for item in citations
            for identifier in (
                getattr(item, "evidence_chunk_id", None),
                getattr(item, "chunk_id", None),
            )
            if identifier
        }
        decisions = self.stages["candidate_selection"].get("decisions", [])
        for row in decisions:
            if row.get("supplied_to_generation") and not row.get("selected_for_citation"):
                row["omitted_after_synthesis"] = True
        if not decisions and self.stages["candidate_retrieval"].get("candidates"):
            self.stages["candidate_selection"].update({
                "status": "failed",
                "reason_code": "selector_decision_not_instrumented",
            })
        response_raw = (
            response_decision.as_dict()
            if hasattr(response_decision, "as_dict") else {}
        )
        response_mode = str(response_raw.get("mode") or "abstain").casefold()
        # Keep traces canonical even if an older caller still supplies the
        # former public spelling. The live ResponseDecision normalizes this
        # before serialization; this defensive conversion protects trace-only
        # callers and fixture adapters.
        if response_mode == "unsupported_refusal":
            response_mode = "deny"
        elif response_mode not in {"answer", "abstain", "clarify", "deny"}:
            response_mode = "abstain"
        response_disposition = str(
            response_raw.get("disposition")
            or {
                "answer": "ANSWER",
                "abstain": "ABSTAIN",
                "clarify": "CLARIFY",
                "deny": "DENY",
            }[response_mode]
        ).upper()
        abstained = response_disposition != "ANSWER"
        synthesis = self.stages["synthesis"]
        if synthesis.get("status") == "not_recorded":
            synthesis.update({
                "status": "skipped" if abstained else "completed",
                "reason_code": "upstream_abstention" if abstained else "existing_route_realization",
                "evidence_ids_supplied": sorted(selected_chunks),
                "answer_shape_validation": "not_instrumented_on_existing_route",
            })
        if self.stages["question_answer_relevance"].get("status") == "not_recorded":
            self.stages["question_answer_relevance"] = {
                "status": "skipped" if abstained else "failed",
                "reason_code": "upstream_abstention" if abstained else "relevance_gate_not_instrumented",
                "reason_codes": ["upstream_abstention" if abstained else "relevance_gate_not_instrumented"],
                "passed": False,
            }
        if self.stages["claim_evidence_gate"].get("status") == "not_recorded" and not abstained:
            self.stages["claim_evidence_gate"] = {
                "status": "failed",
                "reason_code": "claim_binding_not_instrumented",
                "claims": [],
            }
        first = "no_failure"
        for stage_name in (
            "conversation_resolution", "request_understanding", "semantic_contract_validation",
            "requirement_construction", "route_execution",
            "query_generation", "candidate_retrieval", "candidate_selection",
            "structural_evidence_construction", "evidence_context_construction",
            "typed_fact_extraction", "fact_admissibility",
            "model_assisted_evidence_semantics", "evidence_coverage",
            "operator_execution", "synthesis",
            "question_answer_relevance", "claim_evidence_gate",
        ):
            stage = self.stages[stage_name]
            if stage.get("status") == "failed":
                first = {
                    "question_answer_relevance": "synthesis_relevance",
                    "claim_evidence_gate": "citation_binding",
                    "synthesis": "synthesis_relevance",
                }.get(stage_name, stage_name)
                break
        if first == "no_failure" and abstained:
            if not self.stages["candidate_retrieval"].get("candidates"):
                first = "candidate_retrieval"
            elif not any(item.get("selected_for_fact_extraction") for item in decisions):
                first = "candidate_selection"
            else:
                first = "final_gate"
        entered_evidence_processing = any(
            self.stages[name].get("status") not in {"not_recorded", "skipped"}
            for name in (
                "structural_evidence_construction",
                "evidence_context_construction",
                "typed_fact_extraction",
                "evidence_coverage",
            )
        )
        for stage_name, stage in self.stages.items():
            if stage_name == "final_gate" or stage.get("status") != "not_recorded":
                continue
            if stage_name == "evidence_coverage" and not entered_evidence_processing:
                reason = "request_did_not_enter_evidence_processing"
                stage.setdefault("requirements", [])
            elif abstained:
                reason = f"skipped_after_{first}"
            else:
                reason = "route_not_applicable"
            stage.update({"status": "skipped", "reason_code": reason})
        ordered = [
            "conversation_resolution", "request_understanding", "semantic_contract_validation",
            "requirement_construction", "route_execution",
            "query_generation", "candidate_retrieval", "candidate_selection",
            "structural_evidence_construction", "evidence_context_construction",
            "typed_fact_extraction", "fact_admissibility",
            "model_assisted_evidence_semantics", "evidence_coverage", "operator_execution",
            "synthesis_relevance", "citation_binding", "citation_excerpt_entailment", "final_gate",
        ]
        skipped = ordered[ordered.index(first) + 1:] if first in ordered and first != "no_failure" else []
        existing_final = dict(self.stages.get("final_gate") or {})
        acceptance = dict(existing_final.get("acceptance_decision") or {})
        if acceptance:
            abstained = not bool(acceptance.get("accepted"))
            final_reason_code = str(
                existing_final.get("reason_code") or final_reason_code
            )
        self.stages["final_gate"] = {
            **existing_final,
            "status": "completed",
            "reason_code": _bounded(final_reason_code or ("observed_abstention" if abstained else "accepted"), 160),
            "final_decision": response_disposition,
            "response_mode": response_mode,
            "response_disposition": response_disposition,
            "response_decision": _safe(response_raw),
            "abstention_reason": _bounded(final_reason_code or ("observed_abstention" if abstained else "none"), 160),
            "first_observed_failing_stage": first,
            "attribution_confidence": "actual_phase2c7_lifecycle_events",
            "supporting_trace_facts": [
                f"candidate_count={len(self.stages['candidate_retrieval'].get('candidates', []))}",
                f"selected_citation_count={len(citations)}",
            ],
            "downstream_skipped": skipped,
            "answer_sha256": _content_hash(answer),
            "answer_chars": len(answer or ""),
        }

    def write(self) -> Optional[Path]:
        if self.execution_contract is None:
            if os.getenv("PROJECT_X_CANDIDATE_MODE", "0").strip().casefold() in {
                "1", "true", "yes", "on",
            } and self.semantic_compilation_result is None:
                raise RuntimeError("candidate trace missing execution contract")
            self.artifact["lifecycle_mode"] = "POST_HANDLER_EXECUTION"
            self.artifact["pre_contract_diagnostic"] = True
        else:
            self.artifact["lifecycle_mode"] = "POST_HANDLER_EXECUTION"
            contract_id = self.execution_contract.contract_id
            contract_hash = self.execution_contract.contract_sha256
            requirement_ids = sorted(self.execution_contract.requirement_ids)
            for stage_name, stage in self.stages.items():
                prior_hash = stage.get("contract_sha256")
                if prior_hash and prior_hash != contract_hash:
                    raise RuntimeError(
                        f"execution contract hash mismatch at stage {stage_name}"
                    )
                stage["contract_id"] = contract_id
                stage["contract_sha256"] = contract_hash
                stage["canonical_requirement_ids"] = requirement_ids
            self.artifact["contract_id"] = contract_id
            self.artifact["contract_sha256"] = contract_hash
        self.artifact["completed_utc"] = _utc_now()
        root = Path(os.getenv(TRACE_DIR_ENV, str(DEFAULT_TRACE_DIR))).expanduser()
        if not root.is_absolute():
            raise ValueError("PROJECT_X_EVAL_TRACE_DIR must be absolute")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        target = root / f"{self.trace_id}.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(target, flags, 0o600)
        try:
            payload = json.dumps(self.artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            os.write(fd, payload.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(target, 0o600)
        return target


def begin(trace_id: str, *, preview_requested: bool = False) -> Optional[Token]:
    if not enabled():
        return None
    trace_id = str(trace_id or "")
    if not _TRACE_ID_RE.fullmatch(trace_id):
        raise ValueError("invalid evaluation trace ID")
    # A development run may trace only its own requests while the normal
    # application remains available to other users of the local lab.
    prefix = os.getenv("PROJECT_X_EVAL_TRACE_ID_PREFIX", "").strip()
    if prefix and not trace_id.startswith(prefix):
        return None
    if os.getenv("APP_BIND_HOST", "127.0.0.1").strip() not in {
        "127.0.0.1", "localhost", "::1"
    }:
        raise ValueError("evaluation tracing requires a loopback-only API")
    return _CURRENT.set(TraceRecorder(trace_id, preview_requested=preview_requested))


def current() -> Optional[TraceRecorder]:
    return _CURRENT.get()


def finish(token: Optional[Token]) -> Optional[Path]:
    if token is None:
        return None
    recorder = _CURRENT.get()
    try:
        return recorder.write() if recorder is not None else None
    finally:
        _CURRENT.reset(token)


def trace_retrieval(channel: str):
    """Decorator for existing retrieval calls; return values are untouched."""
    def decorate(function):
        @wraps(function)
        def wrapper(self, query=None, *args, **kwargs):
            result = function(self, query, *args, **kwargs)
            recorder = current()
            if recorder is not None:
                hits = result[0] if isinstance(result, tuple) else result
                if isinstance(hits, list):
                    recorder.record_retrieval(
                        channel,
                        query,
                        hits,
                        filters=kwargs.get("where"),
                    )
            return result
        return wrapper
    return decorate


def trace_synthesis_method(function):
    """Record a synthesis result without observing prompts or answer text."""
    @wraps(function)
    def wrapper(self, bundle, *args, **kwargs):
        result = function(self, bundle, *args, **kwargs)
        recorder = current()
        if recorder is not None:
            recorder.record_synthesis(result, bundle=bundle)
        return result
    return wrapper
