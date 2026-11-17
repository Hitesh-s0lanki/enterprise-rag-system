"""Offline checks for inheritance, external prompts and shared model helpers."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents import AgentConfig, AgentRunSettings, BaseAgent, RootAgentSettings
from src.clients.openai_client import create_chat_client


class ExampleAgent(BaseAgent):
    prompt_name = "example_agent"

    def run(self, query: str, *, context: str | None = None):
        return self.invoke_model(query, context=context)


class BaseAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "base_agent.md").write_text("Shared instructions.")
        (self.root / "example_agent.md").write_text('Return JSON such as {"answer": "text"}.')
        prompt_patch = patch("src.agents.base_agent.PROMPTS_DIRECTORY", self.root)
        prompt_patch.start()
        self.addCleanup(prompt_patch.stop)

    def test_base_is_abstract_and_child_loads_literal_markdown(self):
        with self.assertRaises(TypeError):
            BaseAgent()
        agent = ExampleAgent(AgentConfig(), llm=Mock())
        messages = agent.build_messages("VPN?", context="Evidence text.")
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIn('{"answer": "text"}', messages[1].content)
        self.assertIsInstance(messages[2], HumanMessage)
        self.assertIn("Evidence text.", messages[2].content)
        self.assertEqual(messages[-1].content, "VPN?")

    def test_invalid_names_missing_empty_and_escaping_prompts_fail(self):
        for name in ("../secret", "/tmp/prompt", "example_agent.md", "", "../base_agent"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                BaseAgent.load_prompt(name)
        with self.assertRaises(FileNotFoundError):
            BaseAgent.load_prompt("missing_agent")
        (self.root / "empty.md").write_text("  ")
        with self.assertRaises(ValueError):
            BaseAgent.load_prompt("empty")
        with tempfile.TemporaryDirectory() as other:
            external = Path(other) / "external.md"
            external.write_text("Outside instructions")
            (self.root / "escape.md").symlink_to(external)
            with self.assertRaises(ValueError):
                BaseAgent.load_prompt("escape")

    def test_constructor_is_lazy_and_explicit_config_overrides_environment(self):
        with patch.dict(os.environ, {"AGENT_MAX_OUTPUT_TOKENS": "invalid"}), \
                patch("src.agents.base_agent.create_chat_client") as factory:
            agent = ExampleAgent(AgentConfig(model="test-model"))
            factory.assert_not_called()
            self.assertEqual(agent.config.model, "test-model")
            self.assertIs(agent.llm, agent.llm)
            factory.assert_called_once()

    def test_invalid_input_fails_before_model_creation(self):
        with patch("src.agents.base_agent.create_chat_client") as factory:
            agent = ExampleAgent(AgentConfig(max_input_characters=10))
            for query, context in [(" ", None), ("x" * 11, None), ("query", "context"), ("query", 123)]:
                with self.subTest(query=query, context=context), self.assertRaises(ValueError):
                    agent.invoke_model(query, context=context)
            factory.assert_not_called()

    def test_sync_and_async_preserve_response_usage_metadata(self):
        message = AIMessage(content="Supported answer", usage_metadata={
            "input_tokens": 10, "output_tokens": 3, "total_tokens": 13,
        })
        model = FakeMessagesListChatModel(responses=[message])
        agent = ExampleAgent(AgentConfig(), llm=model)
        self.assertEqual(agent.run("VPN?").usage_metadata["total_tokens"], 13)
        result = asyncio.run(agent.ainvoke_model("VPN?"))
        self.assertEqual(result.content, "Supported answer")
        self.assertEqual(result.usage_metadata["total_tokens"], 13)

    def test_config_parsing_and_invalid_values(self):
        with patch.dict(os.environ, {"AGENT_MODEL": "test-model", "AGENT_TEMPERATURE": "0.3",
                                    "AGENT_MAX_OUTPUT_TOKENS": "256"}):
            settings = AgentConfig.from_env()
        self.assertEqual(settings.model, "test-model")
        self.assertEqual(settings.temperature, 0.3)
        self.assertEqual(settings.max_output_tokens, 256)
        for values in ({"model": " "}, {"temperature": float("nan")}, {"temperature": True},
                       {"max_output_tokens": 0}, {"max_retries": -1}, {"max_retries": True},
                       {"timeout_seconds": float("inf")}, {"max_input_characters": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AgentConfig(**values)

    def test_client_requires_key_and_forwards_config_without_request(self):
        values = dict(model="test-model", temperature=None, max_output_tokens=256,
                      timeout_seconds=15, max_retries=1)
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
                patch("src.clients.openai_client.ChatOpenAI") as factory:
            with self.assertRaises(ValueError):
                create_chat_client(**values)
            factory.assert_not_called()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-key"}), \
                patch("src.clients.openai_client.ChatOpenAI") as factory, \
                patch("src.clients.openai_client.DefaultHttpxClient") as http, \
                patch("src.clients.openai_client.DefaultAsyncHttpxClient") as async_http:
            create_chat_client(**values)
            factory.assert_called_once_with(model="test-model", temperature=None,
                                             max_tokens=256, timeout=15, max_retries=1,
                                             stream_usage=True,
                                             http_client=http.return_value, http_async_client=async_http.return_value)

    def test_child_inherits_stream_run_async_and_sse_without_implementing_them(self):
        class InheritedAgent(BaseAgent):
            prompt_name = "example_agent"

        message = AIMessage(content="Inherited answer", usage_metadata={
            "input_tokens": 10, "output_tokens": 3, "total_tokens": 13,
        })
        agent = InheritedAgent(AgentConfig(), llm=FakeMessagesListChatModel(responses=[message]))
        events = list(agent.stream("Question"))
        self.assertEqual(events[0].data["agent"], "example_agent")
        self.assertEqual(events[-1].type, "run_completed")
        self.assertEqual(events[-1].data["answer"], "Inherited answer")
        self.assertEqual(agent.run("Question")["answer"], "Inherited answer")
        self.assertEqual(asyncio.run(agent.arun("Question"))["answer"], "Inherited answer")
        self.assertIn("event: run_completed\n", list(agent.stream_sse("Question"))[-1])

    def test_shared_run_settings_and_root_overrides_are_independent(self):
        with patch.dict(os.environ, {"AGENT_RUN_MAX_MODEL_CALLS": "2", "AGENT_RUN_TIMEOUT_SECONDS": "40",
                                    "ROOT_AGENT_MAX_MODEL_CALLS": "5", "ROOT_AGENT_TIMEOUT_SECONDS": "90"}):
            common, root = AgentRunSettings.from_env(), RootAgentSettings.from_env()
        self.assertEqual((common.max_model_calls, common.timeout_seconds), (2, 40))
        self.assertEqual((root.max_model_calls, root.timeout_seconds), (5, 90))
        for values in ({"max_model_calls": 0}, {"max_tool_calls": True}, {"timeout_seconds": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AgentRunSettings(**values)

    def test_owned_model_is_recreated_before_reusing_a_different_sync_event_loop(self):
        class InheritedAgent(BaseAgent):
            prompt_name = "example_agent"

        class LoopBoundModel(FakeMessagesListChatModel):
            event_loop: object | None = None

            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                loop = asyncio.get_running_loop()
                if self.event_loop is not None and self.event_loop is not loop:
                    raise RuntimeError("Event loop is closed")
                self.event_loop = loop
                return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

        def fresh_model(**kwargs):
            return LoopBoundModel(responses=[AIMessage(content="Reusable answer")])

        with patch("src.agents.base_agent.create_chat_client", side_effect=fresh_model) as factory:
            agent = InheritedAgent(AgentConfig())
            self.assertEqual(agent.run("First question")["answer"], "Reusable answer")
            self.assertEqual(agent.run("Second question")["answer"], "Reusable answer")
            self.assertEqual(factory.call_count, 2)


if __name__ == "__main__":
    unittest.main()
