"""JSON log correlation, redaction, rerun safety and rotating-file behavior."""

import io
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.logger import configure_logging, get_logger


class LoggerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "app.jsonl"
        self.environment = patch.dict(os.environ, {
            "LOG_FILE": str(self.path), "LOG_LEVEL": "INFO", "LOG_MAX_BYTES": "4096", "LOG_BACKUP_COUNT": "2",
            "OPENAI_API_KEY": "test-sensitive-openai-value",
        })
        self.environment.start()
        self.stderr = patch('sys.stderr', io.StringIO())
        self.stderr.start()
        configure_logging()

    def tearDown(self):
        self.stderr.stop()
        self.environment.stop()
        configure_logging()
        self.directory.cleanup()

    def records(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_rerun_configuration_does_not_duplicate_logs(self):
        configure_logging()
        configure_logging()
        get_logger("test").info("agent.started", extra={"run_id": "run-123", "stage": "startup"})
        records = self.records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["run_id"], "run-123")
        self.assertEqual(records[0]["stage"], "startup")
        self.assertEqual(records[0]["level"], "INFO")

    def test_exception_chain_and_traceback_are_recorded_with_credentials_redacted(self):
        try:
            try:
                raise RuntimeError("Bearer bearer-secret-token api_key=another-secret test-sensitive-openai-value")
            except RuntimeError as exc:
                raise ValueError("Failed with sk-private-key-123") from exc
        except ValueError:
            get_logger("test").exception("agent.failed", extra={"run_id": "run-123"})
        record = self.records()[0]
        text = json.dumps(record)
        for value in ("bearer-secret-token", "another-secret", "test-sensitive-openai-value", "sk-private-key-123"):
            self.assertNotIn(value, text)
        self.assertEqual(record["exception"]["type"], "ValueError")
        self.assertEqual(record["exception"]["cause"]["type"], "RuntimeError")
        self.assertTrue(record["exception"]["frames"])
        self.assertNotIn("locals", text)

    def test_only_safe_metadata_is_serialized(self):
        get_logger("test").info("agent.started", extra={"query": "private question", "documents": "private docs",
                                                        "run_id": "run-safe"})
        text = self.path.read_text()
        self.assertNotIn("private question", text)
        self.assertNotIn("private docs", text)

    def test_logs_rotate_and_third_party_handlers_are_untouched(self):
        handlers = list(logging.getLogger().handlers)
        for number in range(30):
            get_logger("test").info("agent.progress." + str(number), extra={"run_id": "run-123"})
        self.assertTrue(Path(str(self.path) + ".1").exists())
        self.assertEqual(logging.getLogger().handlers, handlers)

    def test_console_only_mode(self):
        with patch.dict(os.environ, {"LOG_FILE": ""}):
            configure_logging()
            get_logger("test").info("console.only")
        self.assertEqual(len(logging.getLogger("enterprise_rag").handlers), 1)

    def test_cache_status_and_measured_stage_times_are_preserved(self):
        get_logger("test").info("knowledge.lookup_completed", extra={
            "run_id": "run-cache", "cache_source": "mem0", "cache_status": "hit",
            "pipeline_runs": 0, "equivalence_checks": 0,
            "memory_search_latency_ms": 400.0, "memory_fetch_latency_ms": 200.0,
            "total_latency_ms": 605.0,
        })
        record = self.records()[-1]
        self.assertEqual(record["cache_source"], "mem0")
        self.assertEqual(record["cache_status"], "hit")
        self.assertEqual(record["pipeline_runs"], 0)
        self.assertEqual(record["memory_search_latency_ms"], 400.0)
        self.assertEqual(record["memory_fetch_latency_ms"], 200.0)


if __name__ == "__main__":
    unittest.main()
