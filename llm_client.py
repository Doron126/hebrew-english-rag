"""Small, dependency-free client for a loopback llama.cpp server.

The application owns retrieval, tenant isolation, grounding, and answer validation.
The native llama.cpp process owns GGUF loading and token generation. Keeping
this boundary HTTP-based prevents the Python worker from loading a second copy
of the model and leaves its lifecycle to the selected runtime owner.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import socket
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:
    from .request_performance import (
        add_tokens,
        record_generation_attempt,
        timed_generation,
    )
except ImportError:
    from request_performance import (
        add_tokens,
        record_generation_attempt,
        timed_generation,
    )


class LlamaCppError(RuntimeError):
    """Base error for the local inference boundary."""


class LlamaCppUnavailableError(LlamaCppError):
    """The local llama.cpp process is unavailable or not ready."""


class LlamaCppTimeoutError(LlamaCppUnavailableError):
    """The local llama.cpp request exceeded its caller-provided time bound."""


class LlamaCppProtocolError(LlamaCppError):
    """The local llama.cpp process returned an invalid response."""


class LlamaCppContextLengthError(LlamaCppProtocolError):
    """The rendered prompt cannot fit beside the requested answer budget."""

    def __init__(self, prompt_tokens: int, max_tokens: int, context_size: int) -> None:
        self.prompt_tokens = int(prompt_tokens)
        self.max_tokens = int(max_tokens)
        self.context_size = int(context_size)
        super().__init__("llama.cpp prompt exceeds the configured context budget")


class LlamaCppIncompleteError(LlamaCppProtocolError):
    """llama.cpp stopped at its token limit before completing the answer."""

    def __init__(self, message: str, *, partial_text: str | None = None,
                 prompt_tokens: int | None = None, completion_tokens: int | None = None,
                 finish_reason: str | None = None) -> None:
        # Diagnostic-only metadata: the message deliberately contains no
        # model output, since callers may expose it in ordinary error logs.
        super().__init__(message)
        self.partial_text = partial_text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.finish_reason = finish_reason


class LlamaCppHTTPError(LlamaCppError):
    """The local llama.cpp process rejected an otherwise valid request."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = int(status_code)
        super().__init__(message)


def _normalize_base_url(value: str) -> str:
    raw = str(value or "").strip().rstrip("/")
    parsed = urlsplit(raw)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("LLAMA_BASE_URL must be an http(s) origin without credentials")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _read_api_key(path_value: str) -> str:
    path_text = str(path_value or "").strip()
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        if path.stat().st_size > 16 * 1024:
            raise ValueError("llama.cpp API key file is unexpectedly large")
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read llama.cpp API key file: {path}") from exc
    keys = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if len(keys) != 1 or len(keys[0]) > 4096 or re.search(r"[\r\n]", keys[0]):
        raise ValueError("llama.cpp API key file must contain exactly one bounded key")
    return keys[0]


def _safe_error_message(payload: Any, fallback: str) -> str:
    value = payload
    if isinstance(payload, dict):
        value = payload.get("error", payload)
    if isinstance(value, dict):
        value = value.get("message") or value.get("type") or fallback
    if not isinstance(value, str):
        value = fallback
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", value).strip()
    return cleaned[:500] or fallback


class LlamaCppClient:
    """Bounded synchronous client for llama-server's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key_file: str = "",
        request_timeout_seconds: float = 1200.0,
        health_timeout_seconds: float = 5.0,
        max_request_bytes: int = 512 * 1024,
        max_response_bytes: int = 4 * 1024 * 1024,
        context_size_tokens: int = 32768,
        context_reserve_tokens: int = 256,
        expected_runtime_model_bytes: int = 0,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.model_id = str(model_id or "").strip()
        if not self.model_id or len(self.model_id) > 512:
            raise ValueError("LLAMA_MODEL_ID must be between 1 and 512 characters")
        self.api_key = _read_api_key(api_key_file)
        self.request_timeout_seconds = max(1.0, min(float(request_timeout_seconds), 3600.0))
        self.health_timeout_seconds = max(0.5, min(float(health_timeout_seconds), 60.0))
        self.max_request_bytes = max(4096, min(int(max_request_bytes), 8 * 1024 * 1024))
        self.max_response_bytes = max(4096, min(int(max_response_bytes), 32 * 1024 * 1024))
        self.context_size_tokens = max(1024, min(int(context_size_tokens), 2_000_000))
        self.context_reserve_tokens = max(0, min(
            int(context_reserve_tokens),
            self.context_size_tokens // 4,
        ))
        self.expected_runtime_model_bytes = max(
            0,
            min(int(expected_runtime_model_bytes), 1_000_000_000_000),
        )

    def _request_json(
        self,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(body) > self.max_request_bytes:
                raise LlamaCppProtocolError(
                    f"llama.cpp request exceeds {self.max_request_bytes} bytes"
                )
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            self.base_url + "/" + path.lstrip("/"),
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        timeout = timeout_seconds or self.request_timeout_seconds
        try:
            with self._open_request(request, timeout=timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
                status = int(getattr(response, "status", 200))
        except HTTPError as exc:
            raw = exc.read(min(self.max_response_bytes, 64 * 1024))
            try:
                error_payload = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                error_payload = {}
            message = _safe_error_message(
                error_payload,
                f"llama.cpp returned HTTP {exc.code}",
            )
            raise LlamaCppHTTPError(exc.code, message) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise LlamaCppTimeoutError("llama.cpp request timed out") from exc
        except URLError as exc:
            if isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout)):
                raise LlamaCppTimeoutError("llama.cpp request timed out") from exc
            raise LlamaCppUnavailableError("llama.cpp is unavailable") from exc
        except OSError as exc:
            raise LlamaCppUnavailableError("llama.cpp is unavailable") from exc

        if len(raw) > self.max_response_bytes:
            raise LlamaCppProtocolError(
                f"llama.cpp response exceeds {self.max_response_bytes} bytes"
            )
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LlamaCppProtocolError("llama.cpp returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise LlamaCppProtocolError("llama.cpp returned a non-object response")
        if status < 200 or status >= 300:
            raise LlamaCppHTTPError(status, f"llama.cpp returned HTTP {status}")
        return decoded

    def _open_request(self, request: Request, *, timeout: float):
        """Transport hook for runtimes with stricter local-network boundaries."""
        return urlopen(request, timeout=timeout)

    def is_ready(self) -> bool:
        try:
            health = self._request_json(
                "/health",
                timeout_seconds=self.health_timeout_seconds,
            )
            if str(health.get("status") or "").casefold() != "ok":
                return False
            models = self._request_json(
                "/v1/models",
                timeout_seconds=self.health_timeout_seconds,
            )
        except (LlamaCppError, ValueError):
            return False
        data = models.get("data")
        if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
            return False
        loaded = data[0]
        if loaded.get("id") != self.model_id:
            return False
        if self.expected_runtime_model_bytes:
            meta = loaded.get("meta")
            if (
                not isinstance(meta, dict)
                or meta.get("size") != self.expected_runtime_model_bytes
            ):
                return False
        return True

    def _normalize_messages(
        self,
        messages: Iterable[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        normalized_messages: List[Dict[str, str]] = []
        total_chars = 0
        for message in messages:
            if len(normalized_messages) >= 128:
                raise LlamaCppProtocolError("too many messages for llama.cpp")
            if not isinstance(message, dict):
                raise LlamaCppProtocolError("invalid chat message for llama.cpp")
            role = str(message.get("role") or "").strip()
            content = message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise LlamaCppProtocolError("invalid chat message for llama.cpp")
            total_chars += len(content)
            if total_chars > 500_000:
                raise LlamaCppProtocolError("chat messages exceed the application prompt bound")
            normalized_messages.append({"role": role, "content": content})
        if not normalized_messages:
            raise LlamaCppProtocolError("at least one chat message is required")
        return normalized_messages

    def count_chat_tokens(self, messages: Iterable[Dict[str, str]]) -> int:
        """Count the exact embedded-template prompt tokens without inference."""
        normalized_messages = self._normalize_messages(messages)
        templated = self._request_json(
            "/apply-template",
            payload={"messages": normalized_messages},
            timeout_seconds=self.health_timeout_seconds,
        )
        prompt = templated.get("prompt")
        if not isinstance(prompt, str):
            raise LlamaCppProtocolError("llama.cpp did not return a rendered chat prompt")
        tokenized = self._request_json(
            "/tokenize",
            payload={"content": prompt, "add_special": False, "parse_special": True},
            timeout_seconds=self.health_timeout_seconds,
        )
        tokens = tokenized.get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise LlamaCppProtocolError("llama.cpp returned invalid prompt tokens")
        return len(tokens)

    @timed_generation
    def chat_completion(
        self,
        *,
        messages: Iterable[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop: Optional[Iterable[str]] = None,
        seed: Optional[int] = None,
        request_timeout_seconds: Optional[float] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        record_generation_attempt()
        normalized_messages = self._normalize_messages(messages)

        try:
            max_tokens_value = max(1, min(int(max_tokens), 8192))
            temperature_value = float(temperature)
            top_p_value = float(top_p)
            top_k_value = max(0, min(int(top_k), 1000))
            repeat_penalty_value = float(repeat_penalty)
        except (TypeError, ValueError, OverflowError) as exc:
            raise LlamaCppProtocolError("invalid llama.cpp generation settings") from exc
        if not all(math.isfinite(value) for value in (
            temperature_value, top_p_value, repeat_penalty_value
        )):
            raise LlamaCppProtocolError("non-finite llama.cpp generation settings")

        stop_values: List[str] = []
        for value in stop or []:
            text = str(value or "")
            if text and text not in stop_values:
                stop_values.append(text[:256])
            if len(stop_values) >= 8:
                break

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": normalized_messages,
            "max_tokens": max_tokens_value,
            "temperature": max(0.0, min(temperature_value, 5.0)),
            "top_p": max(0.0, min(top_p_value, 1.0)),
            "top_k": top_k_value,
            "repeat_penalty": max(0.0, min(repeat_penalty_value, 10.0)),
            "stream": False,
        }
        if stop_values:
            payload["stop"] = stop_values
        if seed is not None and int(seed) >= 0:
            payload["seed"] = int(seed)
        if response_schema is not None:
            if not isinstance(response_schema, dict) or not response_schema:
                raise LlamaCppProtocolError("llama.cpp response schema must be a nonempty object")
            # Native llama.cpp supports this form directly. The normal request
            # byte limit still bounds the complete schema and prompt together.
            payload["response_format"] = {"type": "json_object", "schema": response_schema}

        prompt_tokens = self.count_chat_tokens(normalized_messages)
        add_tokens(prompt=prompt_tokens)
        if (
            prompt_tokens
            + max_tokens_value
            + self.context_reserve_tokens
            > self.context_size_tokens
        ):
            raise LlamaCppContextLengthError(
                prompt_tokens,
                max_tokens_value,
                self.context_size_tokens,
            )

        response = self._request_json(
            "/v1/chat/completions",
            payload=payload,
            timeout_seconds=request_timeout_seconds,
        )
        if response.get("model") != self.model_id:
            raise LlamaCppProtocolError("llama.cpp completion model identity mismatch")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LlamaCppProtocolError("llama.cpp response has no completion choice")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LlamaCppProtocolError("llama.cpp response has no assistant content")
        usage = response.get("usage")
        completion_tokens = (
            usage.get("completion_tokens")
            if isinstance(usage, dict)
            else None
        )
        if type(completion_tokens) is int and completion_tokens >= 0:
            add_tokens(generated=completion_tokens)
        finish_reason = choices[0].get("finish_reason")
        if finish_reason == "length":
            raise LlamaCppIncompleteError(
                "llama.cpp answer reached its token limit",
                partial_text=message["content"],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
            )
        if finish_reason != "stop":
            raise LlamaCppProtocolError("llama.cpp returned an unexpected finish reason")
        if not message["content"].strip():
            raise LlamaCppProtocolError("llama.cpp returned empty assistant content")
        return message["content"].strip()
