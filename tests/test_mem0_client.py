"""Offline checks for the Mem0 client: validation, request shape, error handling."""

import io
import json
import os
import unittest
import urllib.error
from contextlib import contextmanager
from unittest.mock import patch

from src.clients.mem0_client import (
    Mem0Client,
    Mem0Error,
    Mem0Settings,
    create_mem0_client,
)


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


class Mem0SettingsTests(unittest.TestCase):
    def test_defaults_need_no_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Mem0Settings.from_env(), Mem0Settings())

    def test_environment_overrides_defaults(self):
        with patch.dict(os.environ, {
            "MEM0_BASE_URL": "https://example.invalid",
            "MEM0_TIMEOUT_SECONDS": "12.5",
        }, clear=True):
            settings = Mem0Settings.from_env()
            self.assertEqual(settings.base_url, "https://example.invalid")
            self.assertEqual(settings.timeout_seconds, 12.5)

    def test_invalid_settings_fail_locally(self):
        cases = [
            {"base_url": ""}, {"base_url": "  "},
            {"timeout_seconds": 0}, {"timeout_seconds": -1},
            {"timeout_seconds": float("nan")}, {"timeout_seconds": float("inf")},
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Mem0Settings(**values)


class CreateMem0ClientTests(unittest.TestCase):
    def test_missing_key_fails_locally(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                create_mem0_client()

    def test_environment_key_is_used(self):
        with patch.dict(os.environ, {"MEM0_API_KEY": "env-key"}, clear=True), \
                patch("src.clients.mem0_client.urlopen", return_value=_FakeResponse(b"{}")):
            client = create_mem0_client()
            self.assertEqual(client.search("q", filters={"user_id": "u"}), {})

    def test_explicit_key_overrides_environment(self):
        with patch.dict(os.environ, {"MEM0_API_KEY": "env-key"}), \
                patch("src.clients.mem0_client.urlopen", return_value=_FakeResponse(b"{}")):
            client = create_mem0_client(api_key="explicit-key")
            client.add("hello", user_id="u")

    def test_explicit_settings_override_even_malformed_environment(self):
        with patch.dict(os.environ, {"MEM0_TIMEOUT_SECONDS": "invalid"}), \
                patch("src.clients.mem0_client.urlopen", return_value=_FakeResponse(b"{}")) as urlopen:
            settings = Mem0Settings(base_url="https://example.invalid", timeout_seconds=5)
            create_mem0_client(settings, api_key="test-key").search("q", filters={"user_id": "u"})
            self.assertEqual(urlopen.call_args.kwargs["timeout"], 5)

    def test_empty_explicit_key_fails_locally(self):
        with patch.dict(os.environ, {"MEM0_API_KEY": "env-key"}, clear=True):
            with self.assertRaises(ValueError):
                create_mem0_client(api_key=" ")


class Mem0ClientRequestTests(unittest.TestCase):
    def setUp(self):
        self.client = Mem0Client("test-key", Mem0Settings(base_url="https://example.invalid"))

    @contextmanager
    def _capture_request(self, body: bytes = b"{}"):
        with patch("src.clients.mem0_client.urlopen", return_value=_FakeResponse(body)) as urlopen:
            yield urlopen

    def test_add_posts_messages_and_scope(self):
        with self._capture_request() as urlopen:
            response = self.client.add(
                [{"role": "user", "content": "I'm a vegetarian"}, {"role": "assistant", "content": "Noted."}],
                user_id="user123",
            )
        self.assertEqual(response, {})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_full_url(), "https://example.invalid/v3/memories/add/")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 30.0)
        self.assertEqual(request.get_header("Authorization"), "Token test-key")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(request.get_method(), "POST")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["user_id"], "user123")
        self.assertEqual(payload["messages"][0]["role"], "user")

    def test_add_accepts_plain_user_string(self):
        with self._capture_request() as urlopen:
            self.client.add("I'm allergic to nuts.", agent_id="agent123")
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["messages"], [{"role": "user", "content": "I'm allergic to nuts."}])
        self.assertEqual(payload["agent_id"], "agent123")

    def test_search_posts_query_and_filters(self):
        with self._capture_request() as urlopen:
            self.client.search("dietary restrictions", filters={"user_id": "user123"})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_full_url(), "https://example.invalid/v3/memories/search/")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["query"], "dietary restrictions")
        self.assertEqual(payload["filters"], {"user_id": "user123"})

    def test_add_includes_metadata_when_given(self):
        with self._capture_request() as urlopen:
            self.client.add("note", user_id="u", metadata={"source": "cli"})
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["metadata"], {"source": "cli"})

    def test_verbatim_immutable_add_and_reranked_search_options(self):
        with self._capture_request() as urlopen:
            self.client.add("query", user_id="u", infer=False, immutable=True)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertIs(payload["infer"], False)
        self.assertIs(payload["immutable"], True)
        with self._capture_request() as urlopen:
            self.client.search("query", filters={"user_id": "u"}, top_k=5, rerank=True)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["top_k"], 5)
        self.assertIs(payload["rerank"], True)

    def test_cache_filters_are_sent_inside_filters_without_reranking(self):
        filters = {"AND": [
            {"user_id": "u"}, {"run_id": "context"},
            {"metadata": {"schema": "rag-report-cache-v2", "cache_kind": "query"}},
            {"created_at": {"gte": "2026-11-17T09:00:00+00:00"}},
        ]}
        with self._capture_request() as urlopen:
            self.client.search("VPN?", filters=filters, top_k=5, rerank=False)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["filters"], filters)
        self.assertIs(payload["rerank"], False)
        self.assertNotIn("metadata", payload)

    def test_get_uses_documented_endpoint_without_body(self):
        with self._capture_request(b'{"memory":"saved report"}') as urlopen:
            self.assertEqual(self.client.get("memory-1"), {"memory": "saved report"})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_full_url(), "https://example.invalid/v1/memories/memory-1/")
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.data)
        for value in ("../unsafe", "", "a/b"):
            with self.assertRaises(ValueError):
                self.client.get(value)


class Mem0ClientValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = Mem0Client("test-key")

    def test_add_requires_a_scope(self):
        with self.assertRaises(ValueError):
            self.client.add("hello")

    def test_scope_values_must_be_non_empty(self):
        with self.assertRaises(ValueError):
            self.client.add("hello", user_id="  ")

    def test_messages_must_not_be_empty(self):
        with self.assertRaises(ValueError):
            self.client.add([], user_id="u")

    def test_messages_need_role_and_content(self):
        bad_messages = [
            [{"role": "user"}],
            [{"content": "hi"}],
            [{"role": "", "content": "hi"}],
            [{"role": "user", "content": " "}],
            [{"role": "user", "content": 5}],
        ]
        for messages in bad_messages:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.client.add(messages, user_id="u")

    def test_search_requires_query_and_filters(self):
        with self.assertRaises(ValueError):
            self.client.search("  ", filters={"user_id": "u"})
        with self.assertRaises(ValueError):
            self.client.search("query")
        with self.assertRaises(ValueError):
            self.client.search("query", filters={})

    def test_invalid_options_fail_before_network(self):
        with patch("src.clients.mem0_client.urlopen") as request:
            for value in (0, True, 1001):
                with self.assertRaises(ValueError):
                    self.client.search("q", filters={"user_id": "u"}, top_k=value)
            with self.assertRaises(ValueError):
                self.client.add("q", user_id="u", infer="false")
            with self.assertRaises(ValueError):
                self.client.search("q", filters={"user_id": "u"}, rerank=1)
        request.assert_not_called()


class Mem0ClientErrorTests(unittest.TestCase):
    def setUp(self):
        self.client = Mem0Client("test-key", Mem0Settings(base_url="https://example.invalid"))

    def _http_error(self, code: int, body: bytes) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            "https://example.invalid/v3/memories/add/", code, "Error", {}, io.BytesIO(body)
        )

    def test_http_errors_carry_status_and_detail(self):
        with patch(
            "src.clients.mem0_client.urlopen",
            side_effect=self._http_error(401, json.dumps({"detail": "invalid api key"}).encode()),
        ):
            with self.assertRaises(Mem0Error) as context:
                self.client.add("hello", user_id="u")
        self.assertEqual(context.exception.status_code, 401)
        self.assertIn("invalid api key", str(context.exception))

    def test_http_error_with_plain_body(self):
        with patch(
            "src.clients.mem0_client.urlopen",
            side_effect=self._http_error(500, b"internal failure"),
        ):
            with self.assertRaises(Mem0Error) as context:
                self.client.search("q", filters={"user_id": "u"})
        self.assertEqual(context.exception.status_code, 500)
        self.assertIn("internal failure", str(context.exception))

    def test_network_errors_are_unreachable(self):
        with patch(
            "src.clients.mem0_client.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with self.assertRaises(Mem0Error) as context:
                self.client.add("hello", user_id="u")
        self.assertIsNone(context.exception.status_code)
        self.assertIn("unreachable", str(context.exception))

    def test_malformed_provider_json_raises_mem0_error(self):
        for body in (b"not-json", b"[]", b"\xff"):
            with patch("src.clients.mem0_client.urlopen", return_value=_FakeResponse(body)):
                with self.assertRaises(Mem0Error):
                    self.client.search("q", filters={"user_id": "u"})


if __name__ == "__main__":
    unittest.main()
