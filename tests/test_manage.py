"""Local setup boundaries, without downloading models or importing the app."""

import argparse
import contextlib
import io
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import unittest
from unittest import mock
from urllib.request import HTTPHandler, build_opener
from urllib.response import addinfourl

import manage


class LocalSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root / "checkout.env"
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"GGUF" + b"\0" * 64)
        self.binary = self.root / "llama-server"
        self.binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.binary.chmod(0o700)
        self.data = self.root / "user-data"
        self.output = io.StringIO()

    def tearDown(self):
        self.temporary.cleanup()

    def initialize(self, **overrides):
        args = dict(config=str(self.config), model=str(self.model), llama_server=str(self.binary),
                    model_id="local-test-model", model_port=18081, port=18080,
                    data_dir=str(self.data), gpu_layers=0, embedding_model="chosen-embedding-model")
        args.update(overrides)
        with contextlib.redirect_stdout(self.output):
            manage.command_init(argparse.Namespace(**args))
        return manage.read_config(self.config)

    def add_tenant(self, tenant, name=None):
        with contextlib.redirect_stdout(self.output):
            return manage.command_tenant_add(argparse.Namespace(config=str(self.config), tenant=tenant, name=name))

    def test_fresh_setup_is_empty_and_credentials_are_private(self):
        values = self.initialize()
        self.assertEqual(manage.tenant_ids(values), [])
        self.assertEqual(list((self.data / "documents").iterdir()), [])
        key = Path(values["LLAMA_API_KEY_FILE"])
        self.assertEqual(key.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.config.stat().st_mode & 0o777, 0o600)
        self.assertGreaterEqual(len(key.read_text().strip()), 32)
        self.assertNotIn(key.read_text().strip(), self.output.getvalue())
        self.assertEqual(values["RAG_MODEL_RUNTIME"], "llama_cpp")

    def test_init_never_overwrites_existing_installation(self):
        values = self.initialize()
        original_config = self.config.read_bytes()
        original_key = Path(values["LLAMA_API_KEY_FILE"]).read_bytes()
        with self.assertRaises(manage.SetupError):
            self.initialize(model_id="different-model")
        self.assertEqual(self.config.read_bytes(), original_config)
        self.assertEqual(Path(values["LLAMA_API_KEY_FILE"]).read_bytes(), original_key)

    def test_explicit_tenants_get_separate_folders(self):
        self.initialize()
        self.add_tenant("research", "מחקר")
        self.add_tenant("personal", "Personal documents")
        values = manage.read_config(self.config)
        self.assertEqual(manage.tenant_ids(values), ["research", "personal"])
        research = manage.tenant_documents(values, "research")
        (research / "notes.txt").write_text("User-owned notes", encoding="utf-8")
        with self.assertRaises(manage.SetupError):
            self.add_tenant("research", "Replacement")
        self.assertEqual((research / "notes.txt").read_text(), "User-owned notes")
        self.assertEqual(list(manage.tenant_documents(values, "personal").iterdir()), [])

    def test_tenant_paths_cannot_escape_documents_directory(self):
        self.initialize()
        with self.assertRaises(ValueError):
            self.add_tenant("../outside")
        (self.data / "documents" / "redirected").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(manage.SetupError):
            self.add_tenant("redirected")
        self.assertEqual(manage.tenant_ids(manage.read_config(self.config)), [])

    def test_bad_model_and_equal_ports_fail_before_configuration(self):
        self.model.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
        with self.assertRaises(manage.SetupError):
            self.initialize()
        self.assertFalse(self.config.exists())
        self.model.write_bytes(b"GGUF" + b"\0" * 64)
        with self.assertRaises(manage.SetupError):
            self.initialize(model_port=18080)
        self.assertFalse(self.config.exists())

    def test_child_process_uses_this_installations_settings(self):
        values = self.initialize()
        with mock.patch.dict(os.environ, {"CHROMA_PATH": "/another-installation/index", "LOCAL_ALLOWED_TENANTS": "unrelated"}):
            env = manage.runtime_environment(values)
        self.assertEqual(env["CHROMA_PATH"], str(self.data / "chroma"))
        self.assertEqual(env["LOCAL_ALLOWED_TENANTS"], "")

    def test_managed_model_must_stay_on_loopback(self):
        values = self.initialize()
        for origin in ("http://example.com:8080", "http://127.0.0.1:8080/private",
                       "http://user:password@127.0.0.1:8080", "http://127.0.0.1:8080?x=1"):
            with self.subTest(origin=origin):
                with self.assertRaises(manage.SetupError):
                    manage.model_origin({**values, "LLAMA_BASE_URL": origin})
        command = manage.model_command(values)
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertIn("--api-key-file", command)
        self.assertNotIn(Path(values["LLAMA_API_KEY_FILE"]).read_text().strip(), command)

    def test_ingest_requires_selected_tenant_and_preserves_partial_exit(self):
        self.initialize()
        args = argparse.Namespace(config=str(self.config), tenant="research", force=False)
        with self.assertRaises(manage.SetupError):
            manage.command_ingest(args)
        self.add_tenant("research")
        with self.assertRaises(manage.SetupError):
            manage.command_ingest(args)
        (self.data / "documents" / "research" / "notes.txt").write_text("Synthetic fixture", encoding="utf-8")
        with mock.patch("manage.subprocess.run", return_value=argparse.Namespace(returncode=2)) as run:
            with contextlib.redirect_stdout(self.output):
                self.assertEqual(manage.command_ingest(args), 2)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--docs-dir") + 1], str(self.data / "documents" / "research"))
        self.assertNotIn("--clear", command)
        self.assertNotIn("--sync", command)

    def test_startup_health_stays_local_with_proxy_environment(self):
        values = self.initialize()
        destinations = []

        class CaptureHTTP(HTTPHandler):
            def http_open(self, request):
                destinations.append((request.host, bool(request.get_header("Authorization"))))
                response = addinfourl(io.BytesIO(b'{"status":"ok"}'), {}, request.full_url, 200)
                response.msg = "OK"
                return response

        def capture_opener(*handlers):
            # Exercise urllib routing without opening any network connection.
            return build_opener(*handlers, CaptureHTTP())

        with (mock.patch.dict(os.environ, {"http_proxy": "http://proxy.invalid:3128",
                                           "HTTP_PROXY": "http://proxy.invalid:3128",
                                           "no_proxy": "", "NO_PROXY": ""}),
              mock.patch("urllib.request.proxy_bypass", return_value=False),
              mock.patch.object(manage, "build_opener", side_effect=capture_opener)):
            manage.wait_for_model(mock.Mock(poll=mock.Mock(return_value=None)), values)
        self.assertEqual(destinations, [("127.0.0.1:18081", True)])

    def test_startup_health_never_follows_redirect_with_credentials(self):
        values = self.initialize()
        destinations = []

        class RedirectHTTP(HTTPHandler):
            def http_open(self, request):
                destinations.append(request.host)
                if request.host != "127.0.0.1:18081":
                    raise AssertionError("The local startup request followed a redirect")
                response = addinfourl(io.BytesIO(b""),
                    {"Location": "http://redirected.invalid/health"}, request.full_url, 307)
                response.msg = "Temporary Redirect"
                return response

        def capture_opener(*handlers):
            return build_opener(*handlers, RedirectHTTP())

        process = mock.Mock(poll=mock.Mock(side_effect=[None, 1]))
        with (mock.patch.object(manage, "build_opener", side_effect=capture_opener),
              mock.patch.object(manage.time, "sleep"),
              self.assertRaises(manage.SetupError)):
            manage.wait_for_model(process, values)
        self.assertEqual(destinations, ["127.0.0.1:18081"])

    def test_port_check_accepts_time_wait_but_rejects_a_listener(self):
        # Uvicorn's listening socket permits reuse after its active close.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
            server.listen()
            self.assertFalse(manage.port_available(port))
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.settimeout(5)
                client.connect(("127.0.0.1", port))
                with server.accept()[0] as connection:
                    connection.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b"")
        self.assertTrue(manage.port_available(port))

    def test_shutdown_race_still_reaps_both_children(self):
        self.initialize()
        self.add_tenant("research")
        model = mock.Mock(pid=11001, poll=mock.Mock(side_effect=[KeyboardInterrupt, None]))
        api = mock.Mock(pid=11002, poll=mock.Mock(return_value=None))
        with (mock.patch.object(manage, "dependency_errors", return_value=[]),
              mock.patch.object(manage, "port_available", return_value=True),
              mock.patch.object(manage, "wait_for_model"),
              mock.patch.object(manage.subprocess, "Popen", side_effect=[model, api]),
              mock.patch.object(manage.signal, "signal"),
              mock.patch.object(manage.os, "killpg", side_effect=[ProcessLookupError, None]) as kill,
              contextlib.redirect_stdout(self.output)):
            result = manage.command_serve(argparse.Namespace(config=str(self.config)))
        self.assertEqual(result, 0)
        self.assertEqual(kill.call_args_list, [mock.call(api.pid, signal.SIGTERM),
                                                mock.call(model.pid, signal.SIGTERM)])
        api.wait.assert_called_once_with()
        model.wait.assert_called_once_with(timeout=10)

    def test_child_exit_during_force_stop_is_reaped(self):
        process = mock.Mock(pid=11003, poll=mock.Mock(return_value=None),
                            wait=mock.Mock(side_effect=[subprocess.TimeoutExpired("child", 10), 0]))
        with mock.patch.object(manage.os, "killpg", side_effect=[None, ProcessLookupError]) as kill:
            manage.stop_child(process)
        self.assertEqual(kill.call_args_list, [mock.call(process.pid, signal.SIGTERM),
                                               mock.call(process.pid, signal.SIGKILL)])
        self.assertEqual(process.wait.call_args_list, [mock.call(timeout=10), mock.call()])


if __name__ == "__main__":
    unittest.main()
