"""Offline evidence middleware, rewrite bounds, authorization and cancellation."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessageChunk
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.base_agent import BaseAgent
from src.agents.query_enhancement_agent import QueryEnhancementAgent
from src.grading.web import WebEvidenceGrader
from src.tools.firecrawl import create_firecrawl_tools
from src.tools.firecrawl_middleware import FirecrawlSearchMiddleware
from src.utils.streaming import AgentEventStream


def grade(accepted=False):
    return {"accepted": accepted, "status": "sufficient_evidence" if accepted else "insufficient_evidence",
            "relevance": 1.0 if accepted else 0.25}


class QueryEnhancementTests(unittest.IsolatedAsyncioTestCase):
    def setup_loop(self, grades):
        self.client = Mock(call_tool=AsyncMock(return_value={"source": "public_web", "content": []}))
        self.grader = Mock(grade=AsyncMock(side_effect=grades))
        self.enhancer = Mock(enhance=AsyncMock(side_effect=["vendor official vpn MFA", "vendor vpn MFA troubleshooting"]),
                             aclose=AsyncMock())
        self.authorize = Mock()
        self.factory = Mock(return_value=self.enhancer)
        return FirecrawlSearchMiddleware(self.client, authorize=self.authorize,
                                          grader=self.grader, enhancer_factory=self.factory)

    async def test_good_initial_grade_does_not_create_agent(self):
        middleware = self.setup_loop([grade(True)])
        result = await middleware.search("VPN MFA")
        self.factory.assert_not_called()
        self.assertEqual(result["grading"]["enhancements"], 0)
        self.assertEqual(self.client.call_tool.await_count, 1)

    async def test_weak_grades_retry_twice_and_always_grade_original_intent(self):
        middleware = self.setup_loop([grade(), grade(), grade()])
        result = await middleware.search("VPN MFA")
        self.assertEqual(self.client.call_tool.await_count, 3)
        self.assertEqual(self.enhancer.enhance.await_count, 2)
        self.assertEqual(result["grading"]["status"], "insufficient_evidence")
        self.assertEqual(result["grading"]["enhancements"], 2)
        self.assertEqual([c.args[0] for c in self.grader.grade.await_args_list], ["VPN MFA"] * 3)
        self.assertEqual(self.client.call_tool.await_args_list[1].args[1]["query"], "vendor official vpn MFA")
        self.enhancer.aclose.assert_awaited_once()

    async def test_stops_at_first_accepted_retry(self):
        middleware = self.setup_loop([grade(), grade(True)])
        result = await middleware.search("VPN MFA")
        self.assertEqual(result["grading"]["enhancements"], 1)
        self.enhancer.enhance.assert_awaited_once()

    async def test_repeated_query_does_not_retrieve_again(self):
        middleware = self.setup_loop([grade()])
        self.enhancer.enhance.side_effect = None
        self.enhancer.enhance.return_value = " vpn mfa "
        await middleware.search("VPN MFA")
        self.assertEqual(self.client.call_tool.await_count, 1)

    async def test_denied_before_provider_and_revoked_after_grading(self):
        middleware = self.setup_loop([grade()])
        self.authorize.side_effect = PermissionError("denied")
        with self.assertRaises(PermissionError):
            await middleware.search("VPN MFA")
        self.client.call_tool.assert_not_awaited()
        self.grader.grade.assert_not_awaited()
        middleware = self.setup_loop([grade()])
        self.authorize.side_effect = [None, None, PermissionError("revoked")]
        with self.assertRaises(PermissionError):
            await middleware.search("VPN MFA")
        self.factory.assert_not_called()

    async def test_cancellation_during_rewrite_propagates_and_closes_agent(self):
        middleware = self.setup_loop([grade()])
        self.enhancer.enhance.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await middleware.search("VPN MFA")
        self.enhancer.aclose.assert_awaited_once()
        self.assertEqual(self.client.call_tool.await_count, 1)

    async def test_grading_failure_returns_sanitized_tool_error_without_retry(self):
        middleware = self.setup_loop([RuntimeError("private-provider-error")])
        tool = create_firecrawl_tools(self.client, authorize=self.authorize, search_middleware=middleware)[0]
        result = await tool.ainvoke({"query": "VPN MFA"})
        self.assertIn("firecrawl_failed", result)
        self.assertNotIn("private-provider-error", result)
        self.assertEqual(self.client.call_tool.await_count, 1)
        self.factory.assert_not_called()

    async def test_agent_inherits_shared_runtime_and_has_no_tools(self):
        agent = QueryEnhancementAgent(llm=FakeListChatModel(responses=["vendor VPN MFA documentation"]))
        self.assertIsInstance(agent, BaseAgent)
        self.assertEqual(agent.get_tools(), ())
        self.assertEqual(await agent.enhance("VPN MFA", [{"query": "VPN MFA", "grade": grade()}]),
                         "vendor VPN MFA documentation")

    async def test_web_grader_uses_async_jev_and_thresholds(self):
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.system_one.return_value = SimpleNamespace(
            scores={"relevance": SimpleNamespace(score=4), "coverage": SimpleNamespace(score=3)},
            nouls={"sufficient_evidence": SimpleNamespace(noul=0.9)}, model="offline-jev")
        with patch("src.grading.web.create_async_jev_client", return_value=client):
            report = await WebEvidenceGrader().grade("VPN MFA", {"content": [{"text": "evidence"}]})
        self.assertTrue(report["accepted"])
        self.assertEqual(client.system_one.call_args.kwargs["state"]["original_query"], "VPN MFA")
        client.__aexit__.assert_awaited_once()
        client.system_one.return_value.nouls["sufficient_evidence"].noul = float("nan")
        with patch("src.grading.web.create_async_jev_client", return_value=client):
            with self.assertRaises(ValueError):
                await WebEvidenceGrader().grade("VPN MFA", {"content": []})

    def test_private_rewrite_does_not_stream_as_answer(self):
        stream = AgentEventStream(agent="root_agent")
        events = stream.process({"event": "on_chat_model_stream", "tags": ["query_enhancement"],
                                 "data": {"chunk": AIMessageChunk(content="rewritten query")}}, authorize=Mock())
        self.assertEqual(events, [])
