"""Offline MCP transport, retries, cancellation and tool integration checks."""

import asyncio
import os
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import httpx
from mcp.types import CallToolResult, TextContent

from src.clients.firecrawl_client import (
    FirecrawlClient, FirecrawlError, FirecrawlSettings, create_firecrawl_client,
)
from src.tools.firecrawl import create_firecrawl_tools


def result():
    return CallToolResult(content=[TextContent(type="text", text="Vendor documentation")])


class FirecrawlClientTests(unittest.IsolatedAsyncioTestCase):
    def client(self, **kwargs):
        return FirecrawlClient("test-private-key", FirecrawlSettings(retry_delay_seconds=0.001, **kwargs))

    async def test_http_bearer_initialization_call_and_cleanup(self):
        session = AsyncMock()
        session.__aenter__.return_value = session
        session.call_tool.return_value = result()
        closed = []

        @asynccontextmanager
        async def transport(url, *, http_client):
            self.assertEqual(url, "https://mcp.firecrawl.dev/v2/mcp")
            self.assertEqual(http_client.headers["Authorization"], "Bearer test-private-key")
            self.assertFalse(http_client.follow_redirects)
            try:
                yield ("read", "write", lambda: "session")
            finally:
                closed.append(True)

        with patch("src.clients.firecrawl_client.streamable_http_client", transport), patch(
            "src.clients.firecrawl_client.ClientSession", return_value=session,
        ):
            response = await self.client().call_tool("firecrawl_search", {"query": "vendor"})
        session.initialize.assert_awaited_once()
        session.call_tool.assert_awaited_once_with("firecrawl_search", {"query": "vendor"})
        session.__aexit__.assert_awaited_once()
        self.assertEqual(closed, [True])
        self.assertEqual(response["source"], "public_web")
        self.assertEqual(response["content"][0]["text"], "Vendor documentation")

    async def test_initialization_failure_reconnects(self):
        first, second = AsyncMock(), AsyncMock()
        first.__aenter__.return_value = first
        second.__aenter__.return_value = second
        first.initialize.side_effect = OSError("disconnected")
        second.call_tool.return_value = result()

        @asynccontextmanager
        async def transport(*args, **kwargs):
            yield ("read", "write", lambda: None)

        with patch("src.clients.firecrawl_client.streamable_http_client", transport), patch(
            "src.clients.firecrawl_client.ClientSession", side_effect=[first, second],
        ) as sessions:
            await self.client().call_tool("firecrawl_map", {"url": "https://example.com"})
        self.assertEqual(sessions.call_count, 2)
        first.call_tool.assert_not_awaited()
        first.__aexit__.assert_awaited_once()

    async def test_backoff_and_retry_exhaustion_are_bounded_and_sanitized(self):
        client = self.client()
        client._call_once = AsyncMock(side_effect=OSError("test-private-key"))
        with patch("src.clients.firecrawl_client.asyncio.sleep", new_callable=AsyncMock) as sleep:
            with self.assertRaises(FirecrawlError) as error:
                await client.call_tool("firecrawl_search", {"query": "vendor"})
        self.assertNotIn("test-private-key", str(error.exception))
        self.assertEqual(client._call_once.await_count, 3)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [0.001, 0.002])

    async def test_http_permanent_error_is_not_retried_and_429_is_retried(self):
        for status, attempts in ((401, 1), (403, 1), (400, 1), (429, 3), (503, 3)):
            client = self.client()
            response = httpx.Response(status, request=httpx.Request("POST", client.settings.url))
            error = httpx.HTTPStatusError("secret", request=response.request, response=response)
            client._call_once = AsyncMock(side_effect=ExceptionGroup("transport", [error]))
            with self.assertRaises(FirecrawlError):
                await client.call_tool("firecrawl_search", {})
            self.assertEqual(client._call_once.await_count, attempts)

    async def test_mcp_error_result_is_retried_without_exposing_provider_content(self):
        client = self.client(max_retries=1)
        client._call_once = AsyncMock(return_value=CallToolResult(
            isError=True, content=[TextContent(type="text", text="private-provider-error")],
        ))
        with self.assertRaises(FirecrawlError) as error:
            await client.call_tool("firecrawl_search", {})
        self.assertEqual(client._call_once.await_count, 2)
        self.assertNotIn("private-provider-error", str(error.exception))

    async def test_deadline_and_cancellation(self):
        async def wait(*args):
            await asyncio.sleep(10)
        client = self.client(max_retries=0, timeout_seconds=0.01)
        client._call_once = AsyncMock(side_effect=wait)
        with self.assertRaises(FirecrawlError):
            await client.call_tool("firecrawl_search", {})
        client._call_once = AsyncMock(side_effect=asyncio.CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            await client.call_tool("firecrawl_search", {})
        self.assertEqual(client._call_once.await_count, 1)

    async def test_authorization_before_retry_and_after_result(self):
        client = self.client()
        client._call_once = AsyncMock(side_effect=[OSError(), result()])
        authorize = Mock(side_effect=[None, PermissionError("denied")])
        with self.assertRaises(PermissionError):
            await client.call_tool("firecrawl_search", {}, authorize=authorize)
        self.assertEqual(client._call_once.await_count, 1)
        client._call_once = AsyncMock(return_value=result())
        authorize = Mock(side_effect=[None, PermissionError("revoked")])
        with self.assertRaises(PermissionError):
            await client.call_tool("firecrawl_search", {}, authorize=authorize)

    async def test_tools_validate_inputs_and_hide_provider_errors(self):
        client = self.client()
        client._call_once = AsyncMock(return_value=result())
        authorize = Mock()
        tools = {tool.name: tool for tool in create_firecrawl_tools(client, authorize=authorize)}
        for name, args in (("firecrawl_search", {"query": " "}),
                           ("firecrawl_scrape", {"url": "file:///etc/passwd"}),
                           ("firecrawl_scrape", {"url": "https://[invalid"}),
                           ("firecrawl_search", {"query": "vendor", "actor_id": "admin"})):
            output = await tools[name].ainvoke(args)
            self.assertIn("invalid_input", output)
        client._call_once.assert_not_awaited()
        await tools["firecrawl_scrape"].ainvoke({"url": "https://example.com"})
        client._call_once.assert_awaited_once_with("firecrawl_scrape", {
            "url": "https://example.com", "formats": ["markdown"], "onlyMainContent": True,
        })
        client._call_once = AsyncMock(side_effect=RuntimeError("secret-provider-error"))
        output = await tools["firecrawl_search"].ainvoke({"query": "vendor"})
        self.assertIn("firecrawl_failed", output)
        self.assertNotIn("secret-provider-error", output)

    def test_optional_configuration_and_settings(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(create_firecrawl_client())
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "key", "FIRECRAWL_ENABLED": "false"}):
            self.assertIsNone(create_firecrawl_client())
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "key", "FIRECRAWL_ENABLED": "true"}):
            self.assertIsInstance(create_firecrawl_client(), FirecrawlClient)
        for kwargs in ({"max_retries": -1}, {"max_retries": True}, {"timeout_seconds": float("nan")},
                       {"url": "http://example.com"}, {"url": "https://example.com?key=secret"}):
            with self.assertRaises(ValueError):
                FirecrawlSettings(**kwargs)
