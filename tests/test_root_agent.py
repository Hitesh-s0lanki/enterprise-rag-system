"""Real LangChain tool-loop streaming with scripted models and temporary auth."""

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import StructuredTool

from src.agents import AgentConfig, RootAgent, RootAgentError, RootAgentSettings
from src.db.auth import AuthorizationService, SQLiteUserRepository, initialize_database
from src.query.pipeline import KnowledgeQueryService
from src.clients.firecrawl_client import FirecrawlClient, FirecrawlSettings
from mcp.types import CallToolResult, TextContent


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage]
    position: int = 0

    @property
    def _llm_type(self):
        return "scripted-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _next(self):
        response = self.responses[min(self.position, len(self.responses) - 1)]
        self.position += 1
        return response

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = self._next()
        if message.tool_calls:
            for index, call in enumerate(message.tool_calls):
                args = json.dumps(call["args"])
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[{
                    "name": call["name"], "id": call["id"], "index": index, "args": args[:5],
                }]))
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[{
                    "name": None, "id": None, "index": index, "args": args[5:],
                }]))
        elif isinstance(message.content, str):
            for word in message.content.splitlines(keepends=True):
                yield ChatGenerationChunk(message=AIMessageChunk(content=word))
        else:
            yield ChatGenerationChunk(message=AIMessageChunk(content=message.content))
        yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata={
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        }))


def tool_request(name="search_knowledge", call_id="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "id": call_id, "args": {"query": "VPN?"}}])


class RootAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repository = SQLiteUserRepository(initialize_database(Path(self.temp.name) / "auth.sqlite3"))
        self.user = self.repository.get_by_username("alice")
        self.service = KnowledgeQueryService(
            actor_id=self.user.id, authorization=AuthorizationService(self.repository),
            approved_revisions={"a" * 64: "b" * 64},
        )
        self.service.search_knowledge = Mock(return_value={
            "result": {"grading": {"summary": {"status": "sufficient_evidence"}}},
        })

    def agent(self, responses=None, **kwargs):
        return RootAgent(self.service, AgentConfig(model="scripted"),
                         llm=ScriptedModel(responses=responses or [tool_request(), AIMessage(content="Use the VPN with MFA.")]),
                         **kwargs)

    def test_real_tool_loop_streams_arguments_tools_tokens_usage_and_one_terminal(self):
        events = list(self.agent().stream("VPN?"))
        types = [event.type for event in events]
        self.assertEqual(types[0], "run_started")
        self.assertEqual(types[-1], "run_completed")
        self.assertEqual(types.count("run_completed"), 1)
        self.assertEqual(types.count("model_started"), 2)
        self.assertIn("tool_call_delta", types)
        self.assertEqual(types.count("tool_started"), 1)
        self.assertEqual(types.count("tool_completed"), 1)
        self.assertEqual("".join(e.data["text"] for e in events if e.type == "text_delta"), "Use the VPN with MFA.")
        self.assertEqual(events[-1].data["usage"]["total_tokens"], 30)
        self.assertTrue(events[-1].data["usage"]["complete"])
        self.assertEqual(events[-1].data["tool_calls"], 1)
        self.assertEqual([e.sequence for e in events], list(range(len(events))))
        self.assertEqual(len({e.run_id for e in events}), 1)
        start = next(e for e in events if e.type == "tool_started")
        end = next(e for e in events if e.type == "tool_completed")
        self.assertEqual(start.source_run_id, end.source_run_id)
        self.assertEqual(end.data["tool_call_id"], "call-1")
        self.service.search_knowledge.assert_called_once_with("VPN?")
        for event in events:
            self.assertEqual(json.loads(event.to_json())["type"], event.type)
            self.assertTrue(event.to_sse().endswith("\n\n"))

    def test_run_and_arun_use_the_same_tool_and_event_pipeline(self):
        self.assertEqual(self.agent().run("VPN?")["answer"], "Use the VPN with MFA.")
        self.assertEqual(asyncio.run(self.agent().arun("VPN?"))["answer"], "Use the VPN with MFA.")

    def test_agent_can_retry_cached_report_with_memory_disabled(self):
        retry = AIMessage(content="", tool_calls=[{
            "name": "search_knowledge", "id": "call-fresh",
            "args": {"query": "VPN?", "use_memory": False},
        }])
        self.service.search_knowledge.side_effect = [
            {"source": "mem0", "result": {"grading": {"query": "Old VPN?"}}},
            {"source": "rag_pipeline", "result": {"grading": {"summary": {"status": "sufficient_evidence"}}}},
        ]
        events = list(self.agent([tool_request(), retry, AIMessage(content="Use the VPN with MFA.")]).stream("VPN?"))
        self.assertEqual(events[-1].type, "run_completed")
        self.assertEqual(sum(e.type == "tool_completed" for e in events), 2)
        self.assertEqual(sum(e.type == "run_completed" for e in events), 1)
        self.assertEqual(self.service.search_knowledge.call_args_list[0].args, ("VPN?",))
        self.service.search_knowledge.assert_called_with("VPN?", use_memory=False)

    def test_no_global_history_and_separate_run_identifiers(self):
        agent = self.agent([AIMessage(content="Hello")])
        first, second = agent.run("Hello"), agent.run("Hello again")
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["model_calls"], second["model_calls"])

    def test_tool_error_is_streamed_and_provider_details_are_sanitized(self):
        self.service.search_knowledge.side_effect = RuntimeError("secret-provider-credential")
        events = list(self.agent([tool_request(), AIMessage(content="The knowledge lookup failed.")]).stream("VPN?"))
        errors = [e for e in events if e.type == "tool_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].data["output"]["error"]["code"], "lookup_failed")
        self.assertNotIn("secret-provider-credential", "".join(e.to_json() for e in events))

    def test_denial_before_runtime_and_recheck_after_tool(self):
        agent = self.agent()
        self.repository.save_user(replace(self.user, is_active=False))
        events = list(agent.stream("VPN?"))
        self.assertEqual(events[-1].type, "run_error")
        self.assertEqual(events[-1].data["code"], "access_denied")
        self.assertIsNone(agent._runtime)
        self.service.search_knowledge.assert_not_called()
        self.repository.save_user(self.user)

        def revoke(query):
            self.repository.save_user(replace(self.user, is_active=False))
            return {"sensitive": "do-not-stream"}
        self.service.search_knowledge.side_effect = revoke
        events = list(self.agent().stream("VPN?"))
        self.assertEqual(events[-1].data["code"], "access_denied")
        self.assertNotIn("do-not-stream", "".join(e.to_json() for e in events))

    def test_model_and_tool_call_limits_terminate_without_success(self):
        events = list(self.agent([tool_request()], settings=RootAgentSettings(max_model_calls=1)).stream("VPN?"))
        self.assertEqual(events[-1].type, "run_error")
        self.assertEqual(events[-1].data["code"], "call_limit")
        self.service.search_knowledge.reset_mock()
        events = list(self.agent([tool_request()], settings=RootAgentSettings(max_tool_calls=1)).stream("VPN?"))
        self.assertEqual(events[-1].type, "run_error")
        self.assertEqual(events[-1].data["code"], "call_limit")
        self.assertEqual(self.service.search_knowledge.call_count, 1)

    def test_model_failure_never_leaks_exception_or_claims_completion(self):
        class FailingRuntime:
            async def astream_events(self, *args, **kwargs):
                raise RuntimeError("secret-api-error")
                yield
        agent = self.agent()
        agent._runtime = FailingRuntime()
        events = list(agent.stream("VPN?"))
        self.assertEqual(events[-1].type, "run_error")
        self.assertNotIn("secret-api-error", "".join(e.to_json() for e in events))
        with self.assertRaises(RootAgentError):
            agent.run("VPN?")

    def test_custom_events_and_public_text_only(self):
        def progress(query: str):
            """Emit a progress event."""
            dispatch_custom_event("lookup_progress", {"stage": "retrieving"})
            return {"ok": True}
        tool = StructuredTool.from_function(progress, name="progress_lookup")
        agent = self.agent([tool_request("progress_lookup"), AIMessage(content=[
            {"type": "reasoning", "reasoning": "private-reasoning"},
            {"type": "text", "text": "Public answer"},
        ])], extra_tools=[tool])
        events = list(agent.stream("VPN?"))
        self.assertTrue(any(e.type == "custom" and e.data["name"] == "lookup_progress" for e in events))
        self.assertEqual(events[-1].data["answer"], "Public answer")
        self.assertNotIn("private-reasoning", "".join(e.to_json() for e in events))

    def test_explicit_cancellation_timeout_and_consumer_close_release_stream(self):
        class WaitingRuntime:
            closed = False

            async def astream_events(self, *args, **kwargs):
                try:
                    await asyncio.sleep(10)
                    yield {}
                finally:
                    self.closed = True

        async def cancel_case():
            agent = self.agent()
            runtime = WaitingRuntime()
            agent._runtime = runtime
            cancel = asyncio.Event()
            iterator = agent.astream("VPN?", cancel_event=cancel)
            self.assertEqual((await anext(iterator)).type, "run_started")
            task = asyncio.create_task(anext(iterator))
            await asyncio.sleep(0.01)
            cancel.set()
            self.assertEqual((await task).type, "run_cancelled")
            await iterator.aclose()
            self.assertTrue(runtime.closed)

        asyncio.run(cancel_case())
        agent = self.agent(settings=RootAgentSettings(timeout_seconds=0.02))
        runtime = WaitingRuntime()
        agent._runtime = runtime
        events = list(agent.stream("VPN?"))
        self.assertEqual(events[-1].data["code"], "timeout")
        self.assertTrue(runtime.closed)

        iterator = self.agent().stream("VPN?")
        self.assertEqual(next(iterator).type, "run_started")
        iterator.close()

    def test_duplicate_tools_rejected_and_sync_usage_inside_async_is_explicit(self):
        duplicate = StructuredTool.from_function(lambda query: {}, name="search_knowledge", description="Duplicate")
        with self.assertRaises(ValueError):
            self.agent(extra_tools=[duplicate])

        async def inside_loop():
            with self.assertRaisesRegex(RuntimeError, "astream/arun"):
                self.agent().run("VPN?")
        asyncio.run(inside_loop())

    def test_parallel_tool_results_keep_separate_execution_ids(self):
        request = AIMessage(content="", tool_calls=[
            {"name": "search_knowledge", "id": "call-a", "args": {"query": "VPN?"}},
            {"name": "search_knowledge", "id": "call-b", "args": {"query": "Password?"}},
        ])
        events = list(self.agent([request, AIMessage(content="Completed.")]).stream("Help with VPN and passwords"))
        starts = {event.source_run_id for event in events if event.type == "tool_started"}
        ends = [event for event in events if event.type == "tool_completed"]
        self.assertEqual(len(starts), 2)
        self.assertEqual({event.source_run_id for event in ends}, starts)
        self.assertEqual({event.data["tool_call_id"] for event in ends}, {"call-a", "call-b"})

    def test_cancelled_active_tool_gets_a_terminal_lifecycle_event(self):
        class WaitingToolRuntime:
            async def astream_events(self, *args, **kwargs):
                yield {"event": "on_tool_start", "run_id": "tool-run", "name": "search_knowledge",
                       "data": {"input": {"query": "VPN?"}}, "parent_ids": ["root"]}
                await asyncio.sleep(10)

        async def check():
            agent = self.agent()
            agent._runtime = WaitingToolRuntime()
            cancel = asyncio.Event()
            events = []
            async for event in agent.astream("VPN?", cancel_event=cancel):
                events.append(event)
                if event.type == "tool_started":
                    cancel.set()
            self.assertEqual([event.type for event in events], [
                "run_started", "tool_started", "tool_cancelled", "run_cancelled",
            ])
            self.assertEqual(events[1].source_run_id, events[2].source_run_id)
        asyncio.run(check())

    def test_cleanup_failure_is_sanitized_before_any_success_terminal(self):
        class Iterator:
            sent = False

            async def __anext__(self):
                if self.sent:
                    raise StopAsyncIteration
                self.sent = True
                return {"event": "on_chain_end", "run_id": "root", "parent_ids": [],
                        "data": {"output": {"messages": [AIMessage(content="Answer")]}}}

            async def aclose(self):
                raise RuntimeError("secret-cleanup-error")

        class Runtime:
            def astream_events(self, *args, **kwargs):
                return Iterator()

        agent = self.agent()
        agent._runtime = Runtime()
        events = list(agent.stream("VPN?"))
        self.assertEqual(events[-1].type, "run_error")
        self.assertEqual(events[-1].data["code"], "cleanup_failed")
        self.assertNotIn("run_completed", [event.type for event in events])
        self.assertNotIn("secret-cleanup-error", "".join(event.to_json() for event in events))

    def test_invalid_query_and_sse_wrappers(self):
        agent = self.agent()
        events = list(agent.stream(" "))
        self.assertEqual(events[-1].data["code"], "invalid_input")
        self.assertIsNone(agent._runtime)
        frames = list(self.agent([AIMessage(content="Hello")]).stream_sse("Hello"))
        self.assertIn("event: run_completed\n", frames[-1])

        async def collect():
            return [frame async for frame in self.agent([AIMessage(content="Hello")]).astream_sse("Hello")]
        self.assertIn("event: run_completed\n", asyncio.run(collect())[-1])

    @patch("src.grading.web.WebEvidenceGrader.grade", new_callable=AsyncMock,
           return_value={"accepted": True, "status": "sufficient_evidence"})
    def test_firecrawl_registry_tool_loop_success_and_exhausted_failure(self, grade):
        client = FirecrawlClient("offline-key", FirecrawlSettings(max_retries=0))
        client._call_once = AsyncMock(return_value=CallToolResult(
            content=[TextContent(type="text", text="Public vendor guidance")],
        ))
        agent = self.agent([tool_request("firecrawl_search"), AIMessage(content="Vendor guidance.")],
                           firecrawl_client=client)
        self.assertTrue({"firecrawl_search", "firecrawl_scrape", "firecrawl_map"}.issubset(
            {tool.name for tool in agent.tools}))
        events = list(agent.stream("Search public vendor guidance"))
        self.assertEqual(events[-1].type, "run_completed")
        self.assertEqual(sum(e.type == "tool_completed" for e in events), 1)
        self.assertEqual(sum(e.type.startswith("run_") and e.type != "run_started" for e in events), 1)
        client._call_once = AsyncMock(side_effect=OSError("private-key-and-error"))
        agent = self.agent([tool_request("firecrawl_search"), AIMessage(content="Web lookup failed.")],
                           firecrawl_client=client)
        events = list(agent.stream("Search vendor guidance"))
        errors = [e for e in events if e.type == "tool_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].data["output"]["error"]["code"], "firecrawl_failed")
        self.assertEqual(events[-1].type, "run_completed")
        self.assertNotIn("private-key-and-error", "".join(e.to_json() for e in events))

    def test_firecrawl_enhancement_stays_inside_tool_and_preserves_final_answer(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from src.agents.query_enhancement_agent import QueryEnhancementAgent

        enhancer = QueryEnhancementAgent(llm=FakeListChatModel(responses=["official vendor VPN documentation"]))
        client = FirecrawlClient("offline-key", FirecrawlSettings(max_retries=0))
        client._call_once = AsyncMock(return_value=CallToolResult(
            content=[TextContent(type="text", text="Public vendor guidance")],
        ))
        agent = self.agent([tool_request("firecrawl_search"), AIMessage(content="Verified vendor guidance.")],
                           firecrawl_client=client)
        with patch("src.grading.web.WebEvidenceGrader.grade", new_callable=AsyncMock,
                   side_effect=[{"accepted": False, "status": "insufficient_evidence"},
                                {"accepted": True, "status": "sufficient_evidence"}]), patch(
            "src.agents.query_enhancement_agent.QueryEnhancementAgent", return_value=enhancer,
        ):
            events = list(agent.stream("Search vendor documentation"))
        self.assertEqual(client._call_once.await_count, 2)
        self.assertEqual(events[-1].type, "run_completed")
        self.assertEqual(events[-1].data["answer"], "Verified vendor guidance.")
        self.assertEqual(sum(e.type == "tool_completed" for e in events), 1)
        self.assertEqual(sum(e.type in {"run_completed", "run_error", "run_cancelled"} for e in events), 1)
        self.assertNotIn("official vendor VPN documentation",
                         "".join(e.data["text"] for e in events if e.type == "text_delta"))

    def test_firecrawl_cancellation_closes_active_request_and_emits_terminal(self):
        closed = []

        async def waiting(*args):
            try:
                await asyncio.sleep(10)
            finally:
                closed.append(True)

        async def check():
            client = FirecrawlClient("offline-key")
            client._call_once = AsyncMock(side_effect=waiting)
            agent = self.agent([tool_request("firecrawl_search")], firecrawl_client=client)
            cancel = asyncio.Event()
            events = []
            async for event in agent.astream("Search vendor guidance", cancel_event=cancel):
                events.append(event)
                if event.type == "tool_started":
                    asyncio.get_running_loop().call_later(0.02, cancel.set)
            self.assertEqual(events[-1].type, "run_cancelled")
            self.assertEqual(sum(e.type == "tool_cancelled" for e in events), 1)
            self.assertEqual(client._call_once.await_count, 1)
            self.assertEqual(closed, [True])

        asyncio.run(check())

    def test_firecrawl_denied_user_never_calls_mcp(self):
        client = FirecrawlClient("offline-key")
        client._call_once = AsyncMock()
        agent = self.agent([tool_request("firecrawl_search")], firecrawl_client=client)
        self.repository.save_user(replace(self.user, is_active=False))
        events = list(agent.stream("Search public guidance"))
        self.assertEqual(events[-1].data["code"], "access_denied")
        client._call_once.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
