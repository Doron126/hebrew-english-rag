"""Request-scoped, content-free performance observability.

The collector deliberately accepts only bounded numeric counters and short
enumerated metadata.  It never receives questions, answers, evidence text,
source names, tenant identifiers, prompts, or physical paths.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import logging
import math
import threading
import time
from typing import Dict, Iterator, Mapping, Optional


STAGE_NAMES = (
    "request_validation_ms",
    "tenant_resolution_ms",
    "language_detection_ms",
    "intent_classification_ms",
    "planner_ms",
    "request_understanding_ms",
    "context_resolution_ms",
    "conversation_resolution_ms",
    "decomposition_ms",
    "query_normalization_ms",
    "query_rewrite_ms",
    "embedding_ms",
    "lexical_retrieval_ms",
    "vector_retrieval_ms",
    "exact_retrieval_ms",
    "rank_fusion_ms",
    "rerank_ms",
    "context_expansion_ms",
    "structural_expansion_ms",
    "multi_hop_ms",
    "source_integrity_ms",
    "evidence_coverage_ms",
    "sentence_reconstruction_ms",
    "structured_fact_extraction_ms",
    "answer_generation_ms",
    "auxiliary_generation_ms",
    "final_answer_generation_ms",
    "synthesis_ms",
    "grounding_ms",
    "final_gate_ms",
    "quote_extraction_ms",
    "citation_ms",
    "serialization_ms",
)

_SAFE_ENUM_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.:"
)
_CURRENT: ContextVar[Optional["RequestPerformance"]] = ContextVar(
    "project_x_request_performance",
    default=None,
)
_GENERATION_ROLE: ContextVar[str] = ContextVar("project_x_generation_role", default="final")


@contextmanager
def generation_role(role: str):
    """Mark a caller explicitly; never infer its role from prompt contents."""
    if role not in {"auxiliary", "final"}:
        raise ValueError("invalid generation role")
    token = _GENERATION_ROLE.set(role)
    try:
        yield
    finally:
        _GENERATION_ROLE.reset(token)


def _safe_enum(value: object, default: str = "unknown") -> str:
    text = str(value or "").strip()[:64]
    if not text or any(character not in _SAFE_ENUM_CHARS for character in text):
        return default
    return text


def _bounded_number(value: object, *, maximum: float = 86_400_000.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(maximum, max(0.0, number))


@dataclass
class RequestPerformance:
    """Mutable metrics shared by one request and its nested sync calls."""

    request_id: str
    started: float = field(default_factory=time.monotonic)
    stages: Dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in STAGE_NAMES}
    )
    route: str = "unknown"
    retrieval_profile: str = "unknown"
    cold_warm_state: str = "unknown"
    candidate_count: int = 0
    selected_evidence_count: int = 0
    subquestion_count: int = 0
    query_rewrite_count: int = 0
    coverage_requirement_count: int = 0
    coverage_ratio_milli: int = 0
    deterministic_realization_used: bool = False
    generation_used: bool = False
    fast_abstention_used: bool = False
    generation_attempts: int = 0
    auxiliary_generation_attempts: int = 0
    final_generation_attempts: int = 0
    auxiliary_generation_completed: int = 0
    final_generation_completed: int = 0
    generation_circuit_skips: int = 0
    generation_terminal_error: str = "none"
    prompt_tokens: int = 0
    generated_tokens: int = 0
    final_reason_code: str = "completed"
    cache_events: Dict[str, Dict[str, int]] = field(default_factory=dict)
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    def add_stage(self, name: str, duration_ms: object) -> None:
        if name not in self.stages:
            return
        with self._lock:
            self.stages[name] = _bounded_number(
                self.stages[name] + _bounded_number(duration_ms)
            )

    def add_tokens(self, *, prompt: object = 0, generated: object = 0) -> None:
        with self._lock:
            self.prompt_tokens = min(
                10_000_000,
                self.prompt_tokens + int(_bounded_number(prompt, maximum=10_000_000)),
            )
            self.generated_tokens = min(
                10_000_000,
                self.generated_tokens
                + int(_bounded_number(generated, maximum=10_000_000)),
            )

    def record_generation_attempt(self) -> None:
        with self._lock:
            self.generation_used = True
            self.generation_attempts = min(100, self.generation_attempts + 1)
            name = _GENERATION_ROLE.get() + "_generation_attempts"
            setattr(self, name, min(100, getattr(self, name) + 1))

    def record_generation_completed(self) -> None:
        with self._lock:
            name = _GENERATION_ROLE.get() + "_generation_completed"
            setattr(self, name, min(100, getattr(self, name) + 1))

    def record_generation_circuit_skip(self) -> None:
        with self._lock:
            self.generation_circuit_skips = min(100, self.generation_circuit_skips + 1)

    def record_generation_terminal_error(self, error_class: str) -> None:
        with self._lock:
            if self.generation_terminal_error == "none":
                self.generation_terminal_error = _safe_enum(error_class)

    def record_cache(self, category: object, state: object) -> None:
        safe_category = _safe_enum(category)
        safe_state = _safe_enum(state)
        if safe_state not in {"hit", "miss"}:
            return
        with self._lock:
            counts = self.cache_events.setdefault(
                safe_category,
                {"hit": 0, "miss": 0},
            )
            counts[safe_state] = min(1_000_000, counts[safe_state] + 1)

    def set_metadata(self, **values: object) -> None:
        with self._lock:
            if "route" in values:
                self.route = _safe_enum(values["route"])
            if "retrieval_profile" in values:
                self.retrieval_profile = _safe_enum(values["retrieval_profile"])
            if "cold_warm_state" in values:
                self.cold_warm_state = _safe_enum(values["cold_warm_state"])
            if "final_reason_code" in values:
                self.final_reason_code = _safe_enum(values["final_reason_code"])
            if "candidate_count" in values:
                self.candidate_count = int(
                    _bounded_number(values["candidate_count"], maximum=1_000_000)
                )
            if "selected_evidence_count" in values:
                self.selected_evidence_count = int(
                    _bounded_number(
                        values["selected_evidence_count"],
                        maximum=100_000,
                    )
                )
            for name, maximum in (
                ("subquestion_count", 100),
                ("query_rewrite_count", 1_000),
                ("coverage_requirement_count", 1_000),
                ("coverage_ratio_milli", 1_000),
            ):
                if name in values:
                    setattr(
                        self,
                        name,
                        int(_bounded_number(values[name], maximum=maximum)),
                    )
            for name in (
                "deterministic_realization_used",
                "generation_used",
                "fast_abstention_used",
            ):
                if name in values:
                    setattr(self, name, bool(values[name]))

    def snapshot(self) -> Mapping[str, object]:
        with self._lock:
            cache = ",".join(
                f"{category}:hit={counts['hit']}:miss={counts['miss']}"
                for category, counts in sorted(self.cache_events.items())
            ) or "none"
            values: Dict[str, object] = {
                name: round(self.stages[name], 1)
                for name in STAGE_NAMES
            }
            values.update(
                total_ms=round(
                    (time.monotonic() - self.started) * 1000.0,
                    1,
                ),
                route=self.route,
                retrieval_profile=self.retrieval_profile,
                cold_warm_state=self.cold_warm_state,
                candidate_count=self.candidate_count,
                selected_evidence_count=self.selected_evidence_count,
                subquestion_count=self.subquestion_count,
                query_rewrite_count=self.query_rewrite_count,
                coverage_requirement_count=self.coverage_requirement_count,
                coverage_ratio_milli=self.coverage_ratio_milli,
                deterministic_realization_used=self.deterministic_realization_used,
                generation_used=self.generation_used,
                fast_abstention_used=self.fast_abstention_used,
                generation_attempts=self.generation_attempts,
                auxiliary_generation_attempts=self.auxiliary_generation_attempts,
                final_generation_attempts=self.final_generation_attempts,
                auxiliary_generation_completed=self.auxiliary_generation_completed,
                final_generation_completed=self.final_generation_completed,
                generation_circuit_skips=self.generation_circuit_skips,
                generation_terminal_error=self.generation_terminal_error,
                prompt_tokens=self.prompt_tokens,
                generated_tokens=self.generated_tokens,
                cache=cache,
                final_reason_code=self.final_reason_code,
            )
            return values

    def log(self, logger: logging.Logger) -> None:
        values = self.snapshot()
        ordered = [f"request_id={_safe_enum(self.request_id)}"]
        ordered.extend(f"{name}={values[name]}" for name in STAGE_NAMES)
        ordered.extend(
            f"{name}={values[name]}"
            for name in (
                "total_ms",
                "route",
                "retrieval_profile",
                "cold_warm_state",
                "candidate_count",
                "selected_evidence_count",
                "subquestion_count",
                "query_rewrite_count",
                "coverage_requirement_count",
                "coverage_ratio_milli",
                "deterministic_realization_used",
                "generation_used",
                "fast_abstention_used",
                "generation_attempts",
                "auxiliary_generation_attempts",
                "final_generation_attempts",
                "auxiliary_generation_completed",
                "final_generation_completed",
                "generation_circuit_skips",
                "generation_terminal_error",
                "prompt_tokens",
                "generated_tokens",
                "cache",
                "final_reason_code",
            )
        )
        logger.info("REQUEST_PERFORMANCE %s", " ".join(ordered))


def bind(collector: RequestPerformance) -> Token:
    return _CURRENT.set(collector)


def reset(token: Token) -> None:
    _CURRENT.reset(token)


def current() -> Optional[RequestPerformance]:
    return _CURRENT.get()


def add_stage(name: str, duration_ms: object) -> None:
    collector = current()
    if collector is not None:
        collector.add_stage(name, duration_ms)


def add_tokens(*, prompt: object = 0, generated: object = 0) -> None:
    collector = current()
    if collector is not None:
        collector.add_tokens(prompt=prompt, generated=generated)


def record_cache(category: object, state: object) -> None:
    collector = current()
    if collector is not None:
        collector.record_cache(category, state)


def record_generation_attempt() -> None:
    collector = current()
    if collector is not None:
        collector.record_generation_attempt()


def record_generation_completed() -> None:
    collector = current()
    if collector is not None:
        collector.record_generation_completed()


def record_generation_circuit_skip() -> None:
    collector = current()
    if collector is not None:
        collector.record_generation_circuit_skip()


def record_generation_terminal_error(error_class: str) -> None:
    collector = current()
    if collector is not None:
        collector.record_generation_terminal_error(error_class)


def timed_generation(function):
    """Keep the historical total while exposing auxiliary and final timings."""
    def wrapped(*args, **kwargs):
        stage = (
            "auxiliary_generation_ms"
            if _GENERATION_ROLE.get() == "auxiliary" else "final_answer_generation_ms"
        )
        with measure("answer_generation_ms"), measure(stage):
            return function(*args, **kwargs)
    wrapped.__name__ = function.__name__
    wrapped.__doc__ = function.__doc__
    wrapped.__wrapped__ = function
    return wrapped


@contextmanager
def measure(name: str) -> Iterator[None]:
    started = time.monotonic()
    try:
        yield
    finally:
        add_stage(name, (time.monotonic() - started) * 1000.0)


def timed_stage(name: str):
    """Decorate a synchronous stage while preserving all return paths."""

    def decorate(function):
        def wrapped(*args, **kwargs):
            with measure(name):
                return function(*args, **kwargs)

        wrapped.__name__ = function.__name__
        wrapped.__doc__ = function.__doc__
        wrapped.__module__ = function.__module__
        wrapped.__wrapped__ = function
        return wrapped

    return decorate
