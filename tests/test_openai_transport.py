"""Exercise real SDK transports across Streamlit-style loops without paid calls."""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from src.agents.base_agent import AgentConfig, BaseAgent


class LocalAgent(BaseAgent):
    prompt_name = "root_agent"


class CompletionHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        chunks = [
            {"delta": {"role": "assistant", "content": "Local response"}, "finish_reason": None},
            {"delta": {}, "finish_reason": "stop"},
        ]
        frames = []
        for choice in chunks:
            frames.append("data: " + json.dumps({
                "id": "chatcmpl-local", "object": "chat.completion.chunk", "created": 1,
                "model": "gpt-4.1-mini", "choices": [{"index": 0, **choice}],
            }) + "\n\n")
        body = ("".join(frames) + "data: [DONE]\n\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass


class OpenAITransportTests(unittest.TestCase):
    def test_repeated_sync_runs_and_new_agents_use_fresh_transports(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), CompletionHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {
                "OPENAI_API_KEY": "offline-test-key",
                "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
            }):
                for _ in range(3):
                    agent = LocalAgent(config=AgentConfig(max_retries=0))
                    for _ in range(2):
                        events = list(agent.stream("Hello"))
                        self.assertEqual(events[-1].type, "run_completed", events[-1].data)
                        self.assertEqual(events[-1].data["answer"], "Local response")
                        self.assertIsNone(agent._llm)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
