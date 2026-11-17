"""The design reference stays usable without a database or live providers."""

import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class ArchitecturePageTests(unittest.TestCase):
    def test_search_flow_renders_without_application_runtime(self):
        with patch("src.app.services.runtime.AppRuntime.from_env", side_effect=AssertionError("runtime created")), \
             patch("src.agents.base_agent.create_chat_client", side_effect=AssertionError("provider created")):
            app = AppTest.from_file(str(ROOT / "src/app/app_pages/architecture.py"), default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.title[0].value, "System architecture")
            self.assertEqual(len(app.get("iframe")), 1)
            self.assertEqual(app.segmented_control[0].value, "Client and tools")
            app.segmented_control[0].set_value("Inside search knowledge").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.subheader[0].value, "Routing inside search_knowledge")
            app.segmented_control[0].set_value("Client and tools").run()
            self.assertFalse(app.exception)
            self.assertEqual([item.value for item in app.subheader],
                             ["Firecrawl web search", "Search knowledge", "Document updates and RBAC"])
            self.assertEqual(len(app.get("iframe")), 1)
            self.assertEqual(len(app.table), 0)
            app.segmented_control[0].set_value("Inside document updates").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.subheader[0].value, "Document updates and RBAC")
            self.assertEqual(len(app.get("iframe")), 1)

    def test_navigation_reaches_architecture_when_database_is_unavailable(self):
        with patch("src.app.services.runtime.AppRuntime.from_env", side_effect=OSError("unavailable")) as runtime:
            app = AppTest.from_file(str(ROOT / "src/app/main.py"), default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            runtime.reset_mock()
            app.switch_page("app_pages/architecture.py").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.title[0].value, "System architecture")
            runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
