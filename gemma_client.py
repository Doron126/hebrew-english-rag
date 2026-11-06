"""Shared generation interface backed exclusively by the Gemma lifecycle owner.

The request context carries an already-authorized tenant and cancellation event.
This client never opens an HTTP model endpoint, loads weights, or starts a model.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import re
from typing import Iterable

import gemma_app
from gemma_output_policy import MAX_OUTPUT_TOKENS
from llm_client import (
    LlamaCppContextLengthError, LlamaCppIncompleteError, LlamaCppProtocolError,
    LlamaCppTimeoutError, LlamaCppUnavailableError,
)
from request_performance import (
    add_tokens, record_generation_attempt, record_generation_completed,
    record_generation_circuit_skip, timed_generation,
    record_generation_terminal_error, current as current_request_performance,
)

MODEL_ID = "gemma-4-31b"
_REQUEST = ContextVar("projectx_gemma_generation_request", default=None)


@dataclass
class _GenerationRequest:
    tenant_id: str
    cancel_event: object
    terminal_error: Exception | None = None
    last_error: Exception | None = None

    def note_failure(self, error):
        self.last_error = error
        return error

    def fail(self, error):
        self.note_failure(error)
        if self.terminal_error is None:
            self.terminal_error = error
            record_generation_terminal_error(type(error).__name__)
        return error


@contextmanager
def generation_context(tenant_id, cancel_event):
    """Bind a resolved tenant only for the lifetime of its serving worker."""
    tenant_id = gemma_app._tenant_identity(tenant_id)
    context = _GenerationRequest(tenant_id, cancel_event)
    token = _REQUEST.set(context)
    try:
        yield context
    finally:
        _REQUEST.reset(token)


def _validated_messages(messages):
    normalized = list(messages)
    if not 1 <= len(normalized) <= 6:
        raise LlamaCppProtocolError("Gemma requires between 1 and 6 complete messages")
    chars = 0
    for message in normalized:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str) or not message["content"]):
            raise LlamaCppProtocolError("invalid Gemma message")
        chars += len(message["content"])
    if chars > 80000 or normalized[-1]["role"] != "user":
        raise LlamaCppProtocolError("invalid Gemma prompt bound")
    return normalized


def _validate_output_budget(max_tokens):
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
        raise LlamaCppProtocolError("Gemma output budget exceeds the supported context")


def _active_request():
    context = _REQUEST.get()
    if context is None:
        raise LlamaCppUnavailableError("Gemma generation requires an authorized request context")
    if context.terminal_error is not None:
        record_generation_circuit_skip()
        raise type(context.terminal_error)(str(context.terminal_error))
    if context.cancel_event is not None and context.cancel_event.is_set():
        raise LlamaCppUnavailableError("Gemma request was cancelled")
    return context


def _correlation_fields():
    performance = current_request_performance()
    request_id = getattr(performance, "request_id", None)
    if isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id):
        return {"request_id": request_id}
    return {}


class GemmaSupervisedClient:
    model_id = MODEL_ID
    base_url = "unix://" + gemma_app.SOCKET_PATH
    supervised = True
    context_size_tokens = 4096

    def __init__(self, *, request_timeout_seconds=1200, context_reserve_tokens=256):
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.context_reserve_tokens = int(context_reserve_tokens)
        if not 1 <= self.request_timeout_seconds <= 2400:
            raise ValueError("Gemma request timeout must be between 1 and 2400 seconds")
        if not 128 <= self.context_reserve_tokens <= 1024:
            raise ValueError("Gemma context reserve must be between 128 and 1024 tokens")

    def is_ready(self):
        context = _REQUEST.get()
        if context is None:
            return False
        if context.terminal_error is not None:
            return False
        try:
            state = gemma_app.model_status(tenant_id=context.tenant_id)
        except gemma_app.GemmaAppError:
            return False
        return (state.get("selected") == MODEL_ID and state.get("ready") is True
                and state.get("loaded") is True)

    def measure_chat(self, *, messages: Iterable[dict], max_tokens: int) -> dict:
        """Measure the exact framed messages with the owner's tokenizer, without generation.

        The caller must include the same trusted suffix and history as the
        later completion. This uses the same tenant/cancellation context and
        the supervisor serializes it with all other model operations.
        """
        context = _active_request()
        normalized = _validated_messages(messages)
        _validate_output_budget(max_tokens)
        try:
            result = gemma_app._supervisor_request(
                {"tenant_id": context.tenant_id, "action": "measure", "messages": normalized, **_correlation_fields(),
                 "settings": {"max_tokens": max_tokens,
                              "context_reserve_tokens": self.context_reserve_tokens}},
                cancel_event=context.cancel_event, timeout_seconds=60)
        except gemma_app.GemmaAppError as exc:
            if exc.code.startswith(("invalid_", "request_too_large")):
                raise context.note_failure(LlamaCppProtocolError("Gemma measurement was rejected")) from exc
            failure = LlamaCppUnavailableError("Gemma measurement is unavailable: " + exc.code)
            if exc.code != "model_busy":
                context.fail(failure)
            else:
                context.note_failure(failure)
            raise failure from exc
        tokens = result.get("prompt_tokens")
        if (result.get("selected") != MODEL_ID or result.get("ready") is not True
                or result.get("loaded") is not True or type(tokens) is not int or tokens < 0
                or result.get("max_tokens") != max_tokens
                or result.get("context_reserve_tokens") != self.context_reserve_tokens
                or result.get("context_size_tokens") != self.context_size_tokens
                or type(result.get("fits")) is not bool
                or result["fits"] != (tokens + max_tokens + self.context_reserve_tokens <= self.context_size_tokens)):
            raise context.fail(LlamaCppProtocolError("Gemma returned inconsistent measurement"))
        return {key: result[key] for key in ("prompt_tokens", "max_tokens", "context_reserve_tokens",
                                             "context_size_tokens", "fits")}

    @timed_generation
    def chat_completion(self, *, messages: Iterable[dict], max_tokens: int,
                        temperature: float, top_p: float, top_k: int,
                        repeat_penalty: float, stop=None, seed=None,
                        request_timeout_seconds=None) -> str:
        context = _active_request()
        tenant_id, cancel_event = context.tenant_id, context.cancel_event
        normalized = _validated_messages(messages)
        _validate_output_budget(max_tokens)
        numeric = {"temperature": temperature, "top_p": top_p, "repeat_penalty": repeat_penalty}
        for key, (low, high) in {"temperature": (0, 5), "top_p": (0, 1), "repeat_penalty": (0, 10)}.items():
            value = numeric[key]
            if type(value) not in {int, float} or not math.isfinite(value) or not low <= value <= high:
                raise LlamaCppProtocolError("invalid Gemma generation settings")
        if type(top_k) is not int or not 0 <= top_k <= 1000:
            raise LlamaCppProtocolError("invalid Gemma top_k")
        stop_values = list(stop or ())
        if (len(stop_values) > 8 or any(not isinstance(value, str) or not 1 <= len(value) <= 256
                                        for value in stop_values)):
            raise LlamaCppProtocolError("invalid Gemma stop sequences")
        if seed is not None and (type(seed) is not int or not -1 <= seed <= 2147483647):
            raise LlamaCppProtocolError("invalid Gemma seed")
        timeout = self.request_timeout_seconds if request_timeout_seconds is None else request_timeout_seconds
        if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 1 <= timeout <= 2400:
            raise LlamaCppProtocolError("invalid Gemma request timeout")
        settings = {**numeric, "top_k": top_k, "max_tokens": max_tokens,
                    "timeout_seconds": timeout, "context_reserve_tokens": self.context_reserve_tokens,
                    "stop": stop_values, "seed": -1 if seed is None else seed}
        record_generation_attempt()
        try:
            result = gemma_app._supervisor_request(
                {"tenant_id": tenant_id, "action": "completion", "messages": normalized, **_correlation_fields(),
                 "settings": settings}, cancel_event=cancel_event,
                # Engine work keeps the caller's exact timeout. The transport
                # additionally waits for the owner's bounded process cleanup.
                timeout_seconds=min(3600, timeout + 360))
        except gemma_app.GemmaAppError as exc:
            if exc.code == "evidence_context_too_large":
                raise context.note_failure(LlamaCppContextLengthError(
                    exc.details.get("prompt_tokens", -1), max_tokens, self.context_size_tokens)) from exc
            if "timeout" in exc.code:
                raise context.fail(LlamaCppTimeoutError("Gemma generation timed out")) from exc
            if exc.code in {"invalid_supervisor_response", "invalid_supervisor_state"}:
                raise context.fail(LlamaCppProtocolError("Gemma supervisor response was invalid")) from exc
            if exc.code.startswith(("invalid_", "request_too_large")):
                raise context.note_failure(LlamaCppProtocolError("Gemma request was rejected")) from exc
            failure = LlamaCppUnavailableError("Gemma generation is unavailable: " + exc.code)
            if exc.code != "model_busy":
                context.fail(failure)
            else:
                context.note_failure(failure)
            raise failure from exc
        if (result.get("selected") != MODEL_ID or result.get("model_id") != MODEL_ID
                or result.get("ready") is not True or result.get("loaded") is not True):
            raise context.fail(LlamaCppProtocolError("Gemma completion identity or readiness was not confirmed"))
        answer = result.get("answer")
        finish_reason = result.get("finish_reason")
        if finish_reason not in {"stop", "length"} or not isinstance(answer, str) or (finish_reason == "stop" and not answer.strip()):
            raise context.fail(LlamaCppProtocolError("Gemma returned an invalid completion state"))
        prompt_tokens, completion_tokens = result.get("prompt_tokens"), result.get("completion_tokens")
        recorder = gemma_app.current_eval_trace()
        if recorder is not None:
            recorder.artifact.setdefault("managed_completion_results", []).append({
                "finish_reason": finish_reason, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "max_tokens": max_tokens,
                "context_reserve_tokens": self.context_reserve_tokens, "raw_text": answer})
        if (type(prompt_tokens) is not int or prompt_tokens < 0
                or prompt_tokens + max_tokens + self.context_reserve_tokens > self.context_size_tokens
                or type(completion_tokens) is not int or not 0 <= completion_tokens <= max_tokens):
            if finish_reason == "length":
                raise context.fail(LlamaCppIncompleteError(
                    "Gemma returned incomplete output with inconsistent token accounting",
                    partial_text=answer, prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens, finish_reason=finish_reason))
            raise context.fail(LlamaCppProtocolError("Gemma returned inconsistent token accounting"))
        add_tokens(prompt=prompt_tokens, generated=completion_tokens)
        if finish_reason == "length":
            raise context.note_failure(LlamaCppIncompleteError(
                "Gemma answer reached its request token limit", partial_text=answer,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                finish_reason=finish_reason))
        record_generation_completed()
        context.last_error = None
        return answer.strip()
