"""Offline checks for shared Jev configuration and credential validation."""

import os
import unittest
from unittest.mock import patch

from typesafe_sdk import TypeSafeClient, TypeSafeError

from src.clients.jev_client import JevSettings, create_jev_client


class JevClientTests(unittest.TestCase):
    def test_defaults_need_no_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(JevSettings.from_env(), JevSettings())

    def test_environment_configures_factory(self):
        with patch.dict(os.environ, {
            "TYPESAFE_DEFAULT_MODEL": "test-model",
            "TYPESAFE_BASE_URL": "https://example.invalid",
            "TYPESAFE_TIMEOUT_SECONDS": "12.5",
            "TYPESAFE_MAX_RETRIES": "0",
        }, clear=True), patch("src.clients.jev_client.TypeSafeClient") as constructor:
            create_jev_client()
            options = constructor.call_args.kwargs
            self.assertEqual(options["model"], "test-model")
            self.assertEqual(options["base_url"], "https://example.invalid")
            self.assertEqual(options["timeout"], 12.5)
            self.assertEqual(options["retry"].max_retries, 0)

    def test_explicit_settings_override_even_malformed_environment(self):
        with patch.dict(os.environ, {"TYPESAFE_TIMEOUT_SECONDS": "invalid"}), \
                patch("src.clients.jev_client.TypeSafeClient") as constructor:
            settings = JevSettings(model="explicit-model", timeout_seconds=5, max_retries=1)
            create_jev_client(settings, api_key="test-key")
            options = constructor.call_args.kwargs
            self.assertEqual(options["model"], "explicit-model")
            self.assertEqual(options["timeout"], 5)
            self.assertEqual(options["retry"].max_retries, 1)
            self.assertEqual(options["api_key"], "test-key")

    def test_invalid_settings_fail_before_client_creation(self):
        cases = [
            {"model": " "}, {"base_url": ""}, {"timeout_seconds": 0},
            {"timeout_seconds": -1}, {"timeout_seconds": float("nan")},
            {"timeout_seconds": float("inf")}, {"max_retries": -1},
            {"max_retries": 1.5}, {"max_retries": True},
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                JevSettings(**values)

    def test_missing_or_explicitly_empty_key_fails_locally(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(TypeSafeError):
                create_jev_client()
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}, clear=True):
            with self.assertRaises(TypeSafeError):
                create_jev_client(api_key="")

    def test_real_sdk_client_can_be_constructed_and_closed_offline(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}, clear=True), \
                patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
            with create_jev_client() as client:
                self.assertIsInstance(client, TypeSafeClient)


if __name__ == "__main__":
    unittest.main()
