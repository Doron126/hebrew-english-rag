"""Exercise the portable inference boundary against a synthetic HTTP server."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import URLError

from llm_client import (
    LlamaCppContextLengthError, LlamaCppHTTPError, LlamaCppIncompleteError,
    LlamaCppProtocolError, LlamaCppUnavailableError,
)
from local_llama_client import LocalLlamaClient, generation_context
import request_performance


class LocalLlamaClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.respond()

            def do_POST(self):
                self.respond()

            def respond(self):
                state = self.server.state
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))) if self.command == "POST" else None
                state["calls"].append((self.path, body))
                if self.headers.get("Authorization") != "Bearer synthetic-test-key":
                    self.send_response(401)
                    self.end_headers()
                    return
                if state.get("redirect"):
                    self.send_response(307)
                    self.send_header("Location", self.server.origin + "/unexpected-target")
                    self.end_headers()
                    return
                if self.path == "/v1/chat/completions" and state.get("completion_started"):
                    state["completion_started"].set()
                    state["release_completion"].wait(timeout=5)
                payloads = {
                    "/health": {"status": "ok"},
                    "/v1/models": {"data": [{"id": state["model"]}]},
                    "/props": {"default_generation_settings": {"n_ctx": state["capacity"]}},
                    "/apply-template": {"prompt": "<user>סיכום המסמך<assistant>"},
                    "/tokenize": {"tokens": list(range(state["tokens"]))},
                    "/v1/chat/completions": {
                        "model": state["completion_model"],
                        "choices": [{"message": {"role": "assistant", "content": "תשובה מהמסמך"},
                                     "finish_reason": state["finish_reason"]}],
                        "usage": {"prompt_tokens": state["tokens"], "completion_tokens": 8},
                    },
                }
                payload = json.dumps(payloads.get(self.path, {}), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.origin = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.key = Path(self.temporary.name) / "model-api.key"
        self.key.write_text("synthetic-test-key\n", encoding="utf-8")
        self.key.chmod(0o600)
        self.server.state = {"model": "test-model", "completion_model": "test-model",
                             "capacity": 4096, "tokens": 17, "finish_reason": "stop", "calls": []}
        self.client = LocalLlamaClient(base_url=self.server.origin, model_id="test-model",
                                       api_key_file=str(self.key), context_size_tokens=4096,
                                       context_reserve_tokens=256)
        self.environment = patch.dict(os.environ, {"LOCAL_ALLOWED_TENANTS": "research,personal"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.messages = [{"role": "user", "content": "סיכום המסמך"}]

    @contextmanager
    def authorized(self, cancel=None):
        with generation_context("research", cancel or threading.Event()) as context:
            yield context

    def complete(self, **kwargs):
        return self.client.chat_completion(messages=self.messages, max_tokens=128,
            temperature=0.0, top_p=1.0, top_k=40, repeat_penalty=1.0, **kwargs)

    def test_response_schema_is_optional_and_sent_in_native_format(self):
        schema = {"type": "object", "properties": {"status": {"const": "checked"}},
                  "required": ["status"], "additionalProperties": False}
        with self.authorized():
            self.complete()
            ordinary = self.server.state["calls"][-1][1]
            self.assertNotIn("response_format", ordinary)
            self.complete(response_schema=schema)
            constrained = self.server.state["calls"][-1][1]
            self.assertEqual(constrained["response_format"], {"type": "json_object", "schema": schema})
            self.assertEqual(constrained["messages"], self.messages)

    def test_schema_cannot_exceed_existing_request_limit(self):
        schema = {"type": "string", "description": "x" * self.client.max_request_bytes}
        with self.authorized(), self.assertRaises(LlamaCppProtocolError):
            self.complete(response_schema=schema)
        self.assertNotIn("/v1/chat/completions", [path for path, _ in self.server.state["calls"]])

    def test_status_requires_actual_model_identity_and_context_capacity(self):
        self.assertEqual(self.client.model_status(), {"backend": "llama_cpp", "selected": "test-model",
            "ready": True, "loaded": True, "state": "ready", "error": None})
        self.server.state["model"] = "different-model"
        self.assertEqual(self.client.model_status()["error"], "model_identity_mismatch")
        self.server.state["model"] = "test-model"
        self.server.state["capacity"] = 2048
        status = self.client.model_status()
        self.assertTrue(status["loaded"])
        self.assertFalse(status["ready"])
        self.assertEqual(status["error"], "model_context_mismatch")

    def test_generation_requires_authorized_request_and_cannot_leak_context(self):
        with self.assertRaises(LlamaCppUnavailableError):
            self.complete()
        with self.assertRaises(LlamaCppUnavailableError):
            with generation_context("unlisted", threading.Event()):
                self.complete()
        self.assertEqual(self.server.state["calls"], [])
        with self.authorized():
            self.assertEqual(self.complete(), "תשובה מהמסמך")
        with self.assertRaises(LlamaCppUnavailableError):
            self.complete()

    def test_exact_measurement_uses_template_and_rejects_oversize_before_inference(self):
        with self.authorized():
            measurement = self.client.measure_chat(messages=self.messages, max_tokens=128)
            self.assertEqual(measurement, {"prompt_tokens": 17, "max_tokens": 128,
                "context_reserve_tokens": 256, "context_size_tokens": 4096, "fits": True})
            tokenize_payload = self.server.state["calls"][-1][1]
            self.assertEqual(tokenize_payload, {"content": "<user>סיכום המסמך<assistant>",
                                               "add_special": False, "parse_special": True})
            self.server.state["tokens"] = 3900
            self.assertFalse(self.client.measure_chat(messages=self.messages, max_tokens=128)["fits"])
            with self.assertRaises(LlamaCppContextLengthError):
                self.complete()
        self.assertNotIn("/v1/chat/completions", [path for path, _ in self.server.state["calls"]])

    def test_truncation_keeps_diagnostics_and_later_success_clears_recoverable_error(self):
        self.server.state["finish_reason"] = "length"
        with self.authorized() as context:
            with self.assertRaises(LlamaCppIncompleteError) as caught:
                self.complete()
            self.assertEqual(caught.exception.partial_text, "תשובה מהמסמך")
            self.assertEqual(caught.exception.prompt_tokens, 17)
            self.assertEqual(caught.exception.completion_tokens, 8)
            self.assertNotIn("תשובה", str(caught.exception))
            self.assertIs(context.last_error, caught.exception)
            self.assertIsNone(context.terminal_error)
            self.server.state["finish_reason"] = "stop"
            self.complete()
            self.assertIsNone(context.last_error)

    def test_wrong_completion_identity_opens_request_circuit(self):
        self.server.state["completion_model"] = "different-model"
        with self.authorized() as context:
            with self.assertRaises(LlamaCppProtocolError):
                self.complete()
            calls = len(self.server.state["calls"])
            with self.assertRaises(LlamaCppProtocolError):
                self.complete()
            self.assertIsNotNone(context.terminal_error)
            self.assertEqual(len(self.server.state["calls"]), calls)

    def test_unavailable_transport_opens_request_circuit(self):
        with self.authorized() as context, patch.object(self.client, "_open_request", side_effect=URLError("offline")) as opener:
            with self.assertRaises(LlamaCppUnavailableError):
                self.complete()
            with self.assertRaises(LlamaCppUnavailableError):
                self.complete()
            self.assertEqual(opener.call_count, 1)
            self.assertIsNotNone(context.terminal_error)

    def test_cancellation_prevents_further_document_requests(self):
        cancel = threading.Event()
        cancel.set()
        with self.authorized(cancel), self.assertRaises(LlamaCppUnavailableError):
            self.complete()
        self.assertEqual(self.server.state["calls"], [])

    def test_local_boundary_rejects_remote_origins_and_missing_credentials(self):
        for origin in ("http://example.com", "https://127.0.0.1", "http://127.0.0.1/forward", "http://user@127.0.0.1"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                LocalLlamaClient(base_url=origin, model_id="test-model", api_key_file=str(self.key))
        with self.assertRaises(ValueError):
            LocalLlamaClient(base_url=self.server.origin, model_id="test-model")
        self.client.api_key = "invalid-key"
        self.assertEqual(self.client.model_status()["error"], "model_auth_failed")

    def test_local_transport_ignores_proxies_and_does_not_follow_redirects(self):
        with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1", "no_proxy": "", "NO_PROXY": ""}):
            self.assertTrue(self.client.is_ready())
        self.server.state["calls"].clear()
        self.server.state["redirect"] = True
        with self.authorized(), self.assertRaises(LlamaCppHTTPError):
            self.complete()
        self.assertEqual([path for path, _ in self.server.state["calls"]], ["/apply-template"])

    def test_live_completion_reports_busy_and_returns_to_ready(self):
        started, release = threading.Event(), threading.Event()
        self.server.state.update(completion_started=started, release_completion=release)
        outcome = []

        def generate():
            try:
                with self.authorized():
                    outcome.append(self.complete())
            except Exception as exc:
                outcome.append(exc)

        worker = threading.Thread(target=generate)
        worker.start()
        try:
            self.assertTrue(started.wait(timeout=2))
            status = self.client.model_status()
            self.assertEqual(status["state"], "busy")
            self.assertTrue(status["loaded"])
            self.assertFalse(status["ready"])
        finally:
            release.set()
            worker.join(timeout=2)
        self.assertEqual(outcome, ["תשובה מהמסמך"])
        self.assertEqual(self.client.model_status()["state"], "ready")

    def test_request_metrics_preserve_generation_roles_and_counts(self):
        performance = request_performance.RequestPerformance("synthetic-request")
        token = request_performance.bind(performance)
        try:
            with self.authorized():
                with request_performance.generation_role("auxiliary"):
                    self.complete()
                self.complete()
        finally:
            request_performance.reset(token)
        self.assertEqual(performance.generation_attempts, 2)
        self.assertEqual(performance.auxiliary_generation_attempts, 1)
        self.assertEqual(performance.auxiliary_generation_completed, 1)
        self.assertEqual(performance.final_generation_attempts, 1)
        self.assertEqual(performance.final_generation_completed, 1)
        self.assertEqual(performance.prompt_tokens, 34)
        self.assertEqual(performance.generated_tokens, 16)
        self.assertGreater(performance.stages["auxiliary_generation_ms"], 0)
        self.assertGreater(performance.stages["final_answer_generation_ms"], 0)


if __name__ == "__main__":
    unittest.main()
