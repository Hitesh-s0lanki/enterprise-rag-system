"""Offline Streamlit integration and conversation/access boundary tests."""

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.utils.events import AgentEvent
from src.app.services.conversation import Turn, history_context, restore_turns
from src.app.services.runtime import AppRuntime
from src.db.auth import Role, initialize_database
from src.db.documents.repository import DocumentRecord, initialize_document_catalog


APP = Path(__file__).resolve().parents[1] / "src/app/main.py"


def event(kind, data):
    return AgentEvent(kind, "test-run", 0, "2026-11-16T12:00:00+00:00", data)


class ConversationTests(unittest.TestCase):
    def test_restore_legacy_turn_without_events_uses_defaults(self):
        saved = SimpleNamespace(question="Hello", answer="", status="error", error="Failed",
                                events=[], reports=[], metrics={})
        restored = restore_turns([saved])[0]
        self.assertIsInstance(restored, Turn)
        self.assertIsNone(restored.run_id)

    def test_restore_preserves_evidence_and_marks_abandoned_stream_failed(self):
        evidence = [{"documents": [{"metadata": {"allowed_roles": ["contributor"]}}]}]
        saved = Turn("VPN", "Partial", reports=evidence, events=[event("run_started", {})])
        restored = restore_turns([saved])[0]
        self.assertEqual(restored.reports, evidence)
        self.assertEqual(restored.status, "error")
        self.assertEqual(restored.answer, "")
        self.assertEqual(restored.run_id, "test-run")

    def test_authoritative_answer_replaces_intermediate_text(self):
        turn = Turn("VPN?")
        turn.apply(event("text_delta", {"text": "Searching…"}))
        turn.apply(event("text_delta", {"text": "Draft"}))
        turn.apply(event("run_completed", {"answer": "Final answer"}))
        self.assertEqual(turn.answer, "Final answer")
        self.assertEqual(turn.status, "complete")

    def test_denied_stream_discards_partial_text_and_evidence(self):
        turn = Turn("VPN?")
        turn.apply(event("tool_completed", {"output": {"result": {"documents": [{"text": "Private"}]}}}))
        turn.apply(event("text_delta", {"text": "Private"}))
        turn.apply(event("run_error", {"code": "access_denied", "message": "Access revoked"}))
        self.assertEqual(turn.answer, "")
        self.assertEqual(turn.reports, [])
        self.assertEqual(turn.status, "error")

    def test_history_uses_only_complete_recent_whole_exchanges(self):
        turns = [Turn("Old", "x" * 200, "complete"), Turn("failed", "secret", "error"),
                 Turn("Recent", "Supported", "complete")]
        context = history_context(turns, max_characters=150)
        assert context is not None
        self.assertIn("Recent", context)
        self.assertNotIn("Old", context)
        self.assertNotIn("secret", context)
        self.assertLessEqual(len(context), 150)
        self.assertIsNone(history_context(turns, max_characters=10))


class AppTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "auth.sqlite3"
        initialize_database(self.path)
        initialize_document_catalog(self.path)
        self.environment = patch.dict(os.environ, {"AUTH_DB_PATH": str(self.path), "PINECONE_NAMESPACE": "it-support"})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def test_missing_database_is_initialized_automatically_without_providers(self):
        fresh_path = self.path.parent / "nested" / "fresh.sqlite3"
        with patch.dict(os.environ, {"AUTH_DB_PATH": str(fresh_path)}), \
             patch("src.agents.base_agent.create_chat_client") as provider:
            app = AppTest.from_file(str(APP)).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(fresh_path.is_file())
            runtime = AppRuntime.from_env()
            self.assertEqual(len(runtime.users.list_users()), 3)
            self.assertEqual(runtime.revisions(), {})
            app.run()
            self.assertFalse(app.exception)
            provider.assert_not_called()

    def test_existing_auth_database_gets_catalog_without_resetting_users(self):
        auth_only = self.path.parent / "auth-only.sqlite3"
        initialize_database(auth_only)
        with patch.dict(os.environ, {"AUTH_DB_PATH": str(auth_only)}):
            runtime = AppRuntime.from_env()
            alice = runtime.users.get_by_username("alice")
            runtime.users.save_user(replace(alice, display_name="Edited Alice", is_active=False, namespaces=()))
            with closing(sqlite3.connect(auth_only)) as connection, connection:
                connection.execute("DELETE FROM users WHERE username = ?", ("bob",))
                connection.execute("UPDATE roles SET permissions = ? WHERE name = ?", (0, "employee"))
            runtime = AppRuntime.from_env()
            self.assertEqual(runtime.users.get_by_username("alice").display_name, "Edited Alice")
            self.assertFalse(runtime.users.get_by_username("alice").is_active)
            self.assertEqual(runtime.users.get_by_username("alice").namespaces, ())
            self.assertIsNone(runtime.users.get_by_username("bob"))
            self.assertEqual(runtime.users.get_permissions(Role.EMPLOYEE), 0)
            self.assertEqual(runtime.revisions(), {})

    def test_repeated_startup_preserves_published_and_staged_revisions(self):
        runtime = AppRuntime.from_env()
        administrator = runtime.users.get_by_username("mice")
        record = DocumentRecord("it-support", "a" * 64, "b" * 64, "Original content", {},
                                ("employee",), ("administrator",), None)
        runtime.catalog.register(record, str(administrator.id), lambda record: None)
        update_id = runtime.catalog.stage(replace(record, revision_id="c" * 64, content="Next version"),
                                         record.revision_id, str(administrator.id), lambda record: None)
        restarted = AppRuntime.from_env()
        self.assertEqual(restarted.revisions(), {record.document_id: record.revision_id})
        stored = restarted.catalog.get("it-support", record.document_id)
        self.assertEqual(stored.content, "Original content")
        self.assertEqual(stored.pending_update_id, update_id)

    def test_inaccessible_database_has_actionable_error_without_setup_commands(self):
        with patch.dict(os.environ, {"AUTH_DB_PATH": str(self.path.parent)}):
            app = AppTest.from_file(str(APP)).run()
        self.assertFalse(app.exception)
        self.assertIn("database initialization failed", app.error[0].value)
        self.assertEqual(len(app.code), 0)

    def test_empty_catalog_disables_chat_without_calling_providers(self):
        with patch("src.agents.base_agent.create_chat_client") as provider:
            app = AppTest.from_file(str(APP)).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.chat_input[0].disabled)
        suggestions = [button for button in app.button if button.key and button.key.startswith("suggested_question_")]
        self.assertEqual(len(suggestions), 4)
        self.assertTrue(all(button.disabled for button in suggestions))
        self.assertEqual(len(app.selectbox[0].options), 3)
        provider.assert_not_called()

    def test_suggested_question_streams_once_and_disappears_after_submission(self):
        questions = []

        def fake_stream(agent, question, **kwargs):
            questions.append(question)
            yield event("run_started", {})
            yield event("run_completed", {"answer": "Use the VPN with MFA.", "tool_calls": 0,
                                          "usage": {"total_tokens": 0, "complete": True}})

        with patch.object(AppRuntime, "revisions", return_value={"a" * 64: "b" * 64}), \
             patch("src.agents.root_agent.RootAgent.stream", fake_stream):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="suggested_question_0").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(questions, ["How do I connect to the company VPN from home?"])
            self.assertEqual(app.session_state["turns"][0].question, questions[0])
            self.assertEqual(app.session_state["turns"][0].answer, "Use the VPN with MFA.")
            self.assertEqual(len(app.chat_message), 2)
            self.assertFalse(any(button.key and button.key.startswith("suggested_question_") for button in app.button))
            app.run()
            self.assertEqual(len(questions), 1)

    def test_live_stream_persists_answer_tools_and_followup_context(self):
        contexts = []

        def fake_stream(agent, question, *, context=None):
            contexts.append(context)
            yield event("run_started", {})
            yield event("tool_started", {"name": "search_knowledge"})
            yield event("tool_completed", {"name": "search_knowledge", "output": {"documents": [],
                "grading": {"summary": {"status": "no_documents"}}, "metrics": {}}})
            yield event("text_delta", {"text": "Contact "})
            yield event("text_delta", {"text": "IT support."})
            yield event("run_completed", {"answer": "Contact IT support.", "duration_ms": 50,
                "tool_calls": 1, "usage": {"total_tokens": 10, "complete": True}})

        with patch.object(AppRuntime, "revisions", return_value={"a" * 64: "b" * 64}), \
             patch("src.agents.root_agent.RootAgent.stream", fake_stream):
            app = AppTest.from_file(str(APP)).run()
            app.chat_input[0].set_value("How do I connect to VPN?").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message), 2)
            self.assertEqual(app.session_state["turns"][0].answer, "Contact IT support.")
            app.chat_input[0].set_value("What about MFA?").run()
            self.assertFalse(app.exception)
            self.assertIn("How do I connect to VPN?", contexts[1])
            self.assertIsNone(contexts[0])
            app.selectbox[0].select_index(1).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["turns"], [])

    def test_provider_exception_is_sanitized_and_partial_answer_removed(self):
        def broken_stream(agent, question, **kwargs):
            yield event("text_delta", {"text": "Unfinished answer"})
            raise RuntimeError("secret-provider-key")

        with patch.object(AppRuntime, "revisions", return_value={"a" * 64: "b" * 64}), \
             patch("src.agents.root_agent.RootAgent.stream", broken_stream):
            app = AppTest.from_file(str(APP)).run()
            app.chat_input[0].set_value("VPN help").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["turns"][0].answer, "")
        self.assertTrue(app.session_state["turns"][0].run_id)
        self.assertTrue(any("Reference:" in caption.value for caption in app.caption))
        self.assertNotIn("secret-provider-key", app.error[0].value)

    def test_new_conversation_clears_history(self):
        app = AppTest.from_file(str(APP)).run()
        app.session_state["turns"] = [Turn("Hello", "Hello", "complete")]
        app.button(key="new_conversation").click().run()
        self.assertEqual(app.session_state["turns"], [])

    def test_legacy_failed_turn_survives_reload_and_recovers_event_reference(self):
        app = AppTest.from_file(str(APP)).run()
        # Model an object retained from before run_id was added to Turn.
        app.session_state["turns"] = [SimpleNamespace(
            question="VPN help", answer="", status="error", error="Request failed",
            events=[event("run_error", {"message": "Request failed"})], reports=[], metrics={})]
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["turns"][0].run_id, "test-run")
        self.assertTrue(any("Reference: test-run" in caption.value for caption in app.caption))

    def test_repeated_new_conversations_after_failed_stream(self):
        contexts = []

        def broken_stream(agent, question, *, context=None):
            contexts.append(context)
            raise RuntimeError("provider failed before first event")
            yield  # Keep the same generator contract as RootAgent.stream.

        with patch.object(AppRuntime, "revisions", return_value={"a" * 64: "b" * 64}), \
             patch("src.agents.root_agent.RootAgent.stream", broken_stream):
            app = AppTest.from_file(str(APP)).run()
            for _ in range(3):
                app.chat_input[0].set_value("VPN help").run()
                self.assertFalse(app.exception)
                self.assertTrue(app.session_state["turns"][0].run_id)
                app.run()  # Replay the failed turn before clearing it.
                self.assertFalse(app.exception)
                app.button(key="new_conversation").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["turns"], [])
                self.assertEqual(len(app.chat_message), 0)
            self.assertEqual(contexts, [None, None, None])

    def test_revoked_namespace_denies_ui(self):
        runtime = AppRuntime.from_env()
        alice = runtime.users.get_by_username("alice")
        runtime.users.save_user(replace(alice, namespaces=()))
        app = AppTest.from_file(str(APP)).run()
        self.assertFalse(app.exception)
        self.assertIn("does not have read access", app.error[0].value)
        self.assertEqual(len(app.chat_input), 0)

    def test_history_rechecks_document_roles_and_revision(self):
        runtime = AppRuntime.from_env()
        alice = runtime.users.get_by_username("alice")
        turn = Turn("VPN", "Private answer", "complete", reports=[{"documents": [{"metadata": {
            "document_id": "a" * 64, "revision_id": "b" * 64, "allowed_roles": ["contributor"]}}]}])
        with patch.object(runtime, "revisions", return_value={"a" * 64: "b" * 64}):
            with self.assertRaises(PermissionError):
                runtime.check_history(alice, [turn])
        turn.reports[0]["documents"][0]["metadata"]["allowed_roles"] = ["employee"]
        with patch.object(runtime, "revisions", return_value={"a" * 64: "c" * 64}):
            with self.assertRaises(PermissionError):
                runtime.check_history(alice, [turn])


if __name__ == "__main__":
    unittest.main()
