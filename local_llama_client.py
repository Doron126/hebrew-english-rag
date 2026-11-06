"""Tenant-bound generation against the portable, operator-owned llama-server.

The local launcher owns the child process. This adapter only observes its state,
measures its embedded chat template, and sends bounded authenticated requests.
The original Gemma supervisor remains a separate runtime.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import ipaddress
import os
import threading
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from config_env import validate_tenant_id
from llm_client import (
    LlamaCppClient, LlamaCppError, LlamaCppHTTPError,
    LlamaCppContextLengthError, LlamaCppIncompleteError,
    LlamaCppProtocolError, LlamaCppTimeoutError, LlamaCppUnavailableError,
)
from request_performance import (
    record_generation_circuit_skip, record_generation_completed,
    record_generation_terminal_error,
)


_REQUEST = ContextVar("local_llama_generation_request", default=None)


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
    """Bind the already-resolved tenant and keep the operator allow-list guard."""
    tenant_id = validate_tenant_id(tenant_id)
    allowed = {value.strip() for value in os.getenv("LOCAL_ALLOWED_TENANTS", "").split(",")
               if value.strip()}
    if tenant_id not in allowed:
        raise LlamaCppUnavailableError("Local generation is not enabled for this tenant")
    context = _GenerationRequest(tenant_id, cancel_event)
    token = _REQUEST.set(context)
    try:
        yield context
    finally:
        _REQUEST.reset(token)


def _active_request():
    context = _REQUEST.get()
    if context is None:
        raise LlamaCppUnavailableError("Local generation requires an authorized request context")
    if context.terminal_error is not None:
        record_generation_circuit_skip()
        raise context.terminal_error
    if context.cancel_event is not None and context.cancel_event.is_set():
        raise context.fail(LlamaCppUnavailableError("Local model request was cancelled"))
    return context


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward credentials or document text to a second origin.
        return None


class LocalLlamaClient(LlamaCppClient):
    """The shared generation interface for one explicitly configured local model."""

    supervised = False

    def __init__(self, **kwargs):
        parsed = urlsplit(str(kwargs.get("base_url") or ""))
        try:
            loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            loopback = False
        if (parsed.scheme != "http" or not loopback or parsed.path not in {"", "/"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Portable LLAMA_BASE_URL must be an HTTP loopback IP origin")
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError("Portable local generation requires LLAMA_API_KEY_FILE")
        # System proxy settings must not route local documents off this machine.
        self._local_opener = build_opener(ProxyHandler({}), _NoRedirects())
        self._activity_lock = threading.Lock()
        self._active_generations = 0

    def _open_request(self, request, *, timeout):
        return self._local_opener.open(request, timeout=timeout)

    def model_status(self) -> dict:
        """Observe health, loaded identity, and context capacity without inference."""
        result = {"backend": "llama_cpp", "selected": self.model_id,
                  "ready": False, "loaded": False, "state": "error", "error": None}
        try:
            health = self._request_json("/health", timeout_seconds=self.health_timeout_seconds)
            if str(health.get("status") or "").casefold() != "ok":
                result["error"] = "model_not_ready"
                return result
            models = self._request_json("/v1/models", timeout_seconds=self.health_timeout_seconds)
            data = models.get("data")
            if (not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict)
                    or data[0].get("id") != self.model_id):
                result["error"] = "model_identity_mismatch"
                return result
            result["loaded"] = True
            props = self._request_json("/props", timeout_seconds=self.health_timeout_seconds)
            settings = props.get("default_generation_settings")
            capacity = settings.get("n_ctx") if isinstance(settings, dict) else None
            if type(capacity) is not int or capacity < self.context_size_tokens:
                result["error"] = "model_context_mismatch"
                return result
        except LlamaCppHTTPError as exc:
            result["error"] = ("model_auth_failed" if exc.status_code in {401, 403}
                               else "model_not_ready")
            return result
        except (LlamaCppError, ValueError):
            result["error"] = "model_unavailable"
            return result
        with self._activity_lock:
            busy = self._active_generations > 0
        result.update(ready=not busy, state="busy" if busy else "ready")
        return result

    def is_ready(self) -> bool:
        return self.model_status()["ready"] is True

    def _request_json(self, path, *, payload=None, timeout_seconds=None):
        # Health is deliberately usable without a serving request. Every text
        # operation is bound to the same authorized tenant/cancellation context.
        context = _active_request() if payload is not None else None
        try:
            result = super()._request_json(path, payload=payload, timeout_seconds=timeout_seconds)
            if context is not None:
                _active_request()
            return result
        except LlamaCppError as exc:
            if context is not None:
                if isinstance(exc, (LlamaCppUnavailableError, LlamaCppProtocolError)):
                    context.fail(exc)
                else:
                    context.note_failure(exc)
            raise

    def measure_chat(self, *, messages, max_tokens: int) -> dict:
        context = _active_request()
        try:
            if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
                raise LlamaCppProtocolError("Invalid local model output budget")
            tokens = self.count_chat_tokens(messages)
            return {"prompt_tokens": tokens, "max_tokens": max_tokens,
                    "context_reserve_tokens": self.context_reserve_tokens,
                    "context_size_tokens": self.context_size_tokens,
                    "fits": tokens + max_tokens + self.context_reserve_tokens <= self.context_size_tokens}
        except LlamaCppError as exc:
            context.note_failure(exc)
            raise

    def chat_completion(self, **kwargs) -> str:
        context = _active_request()
        with self._activity_lock:
            self._active_generations += 1
        try:
            answer = super().chat_completion(**kwargs)
            _active_request()
        except (LlamaCppContextLengthError, LlamaCppIncompleteError) as exc:
            context.note_failure(exc)
            raise
        except LlamaCppError as exc:
            if isinstance(exc, (LlamaCppTimeoutError, LlamaCppUnavailableError, LlamaCppProtocolError)):
                context.fail(exc)
            else:
                context.note_failure(exc)
            raise
        finally:
            with self._activity_lock:
                self._active_generations -= 1
        context.last_error = None
        record_generation_completed()
        return answer
