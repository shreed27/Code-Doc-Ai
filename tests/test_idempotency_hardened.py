"""
Tests for Idempotency and Non-Repetitive Rewrites (Priority 7, H4).
Verifies that 3 consecutive runs with varying model output wording produce 0 rewrites
once documentation accurately reflects AST facts, while new code changes trigger updates.
"""

import json

from doc_updater import DocumentationUpdater


def make_varying_ai(responses: list[dict]):
    idx = 0

    def _generator(_system_prompt, _user_prompt):
        nonlocal idx
        resp = responses[min(idx, len(responses) - 1)]
        idx += 1
        return json.dumps(resp)

    return _generator


class TestIdempotencyHardened:
    def test_three_consecutive_runs_prevent_redundant_rewrites(self):
        old_code = "def start_server(port: int = 8080): pass"
        new_code = "def start_server(port: int = 9000): pass"
        doc = (
            "# Server\n\n## Configuration\nCall `start_server(port)`. Server listens on port 8080."
        )

        # Run 1: Model updates to port 9000
        run1_ai = make_varying_ai(
            [
                {
                    "proposed_content": "Call `start_server(port)`. Server listens on port 9000 by default.",
                    "explanation": "Updated port to 9000",
                    "requires_human_review": False,
                }
            ]
        )
        updater1 = DocumentationUpdater(ai_generator=run1_ai)
        report1 = updater1.update_documentation(
            code_changes={"src/server.py": (old_code, new_code)},
            documents={"server.md": doc},
        )
        assert report1.total_sections_applied == 1
        doc_after_run1 = report1.updated_documents["server.md"]
        assert "9000" in doc_after_run1

        # Run 2: Same code, but model proposes paraphrased wording:
        run2_ai = make_varying_ai(
            [
                {
                    "proposed_content": "Call `start_server(port)`. Default server port configuration is set to 9000.",
                    "explanation": "Paraphrased port 9000",
                    "requires_human_review": False,
                }
            ]
        )
        updater2 = DocumentationUpdater(ai_generator=run2_ai)
        report2 = updater2.update_documentation(
            code_changes={"src/server.py": (old_code, new_code)},
            documents={"server.md": doc_after_run1},
        )
        # MUST BE no_change and NOT rewrite doc!
        assert report2.total_sections_applied == 0
        assert len(report2.applied_updates) == 1
        assert report2.applied_updates[0].status == "no_change"
        assert report2.updated_documents["server.md"] == doc_after_run1

        # Run 3: Same code, yet another phrasing from model:
        run3_ai = make_varying_ai(
            [
                {
                    "proposed_content": "Call `start_server(port)`. The service starts listening on port 9000.",
                    "explanation": "Another phrasing",
                    "requires_human_review": False,
                }
            ]
        )
        updater3 = DocumentationUpdater(ai_generator=run3_ai)
        report3 = updater3.update_documentation(
            code_changes={"src/server.py": (old_code, new_code)},
            documents={"server.md": doc_after_run1},
        )
        assert report3.total_sections_applied == 0
        assert len(report3.applied_updates) == 1
        assert report3.applied_updates[0].status == "no_change"
        assert report3.updated_documents["server.md"] == doc_after_run1

        # Run 4: Actual code change happens! Port changes to 9090
        newest_code = "def start_server(port: int = 9090): pass"
        run4_ai = make_varying_ai(
            [
                {
                    "proposed_content": "Call `start_server(port)`. Server listens on port 9090 by default.",
                    "explanation": "Updated port to 9090",
                    "requires_human_review": False,
                }
            ]
        )
        updater4 = DocumentationUpdater(ai_generator=run4_ai)
        report4 = updater4.update_documentation(
            code_changes={"src/server.py": (new_code, newest_code)},
            documents={"server.md": doc_after_run1},
        )
        assert report4.total_sections_applied == 1
        assert "9090" in report4.updated_documents["server.md"]

    def test_d3_stale_8080_port_not_suppressed_by_unrelated_9090_ms(self):
        """
        D3: Code changes PORT from 8080 to 9090.
        Doc contains stale 'PORT: 8080' and an unrelated '9090 ms timeout'.
        The fact-specific matcher must NOT treat '9090 ms' as satisfying the PORT fact.
        It must update the doc.
        """
        old_code = "PORT = 8080"
        new_code = "PORT = 9090"
        doc = "# Config\n\n## Network\nServer runs on PORT = 8080 with latency 9090 ms."

        mock_ai = make_varying_ai(
            [
                {
                    "proposed_content": "Server runs on PORT = 9090 with latency 9090 ms.",
                    "explanation": "Updated PORT to 9090",
                    "requires_human_review": False,
                }
            ]
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/config.py": (old_code, new_code)},
            documents={"config.md": doc},
        )
        assert report.total_sections_applied == 1
        assert "PORT = 9090" in report.updated_documents["config.md"]

    def test_d4_stale_constant_30_not_suppressed_by_unrelated_60_seconds(self):
        """
        D4: Code changes TIMEOUT from 30 to 60.
        Doc contains 'TIMEOUT = 30' and unrelated text mentioning '60 seconds'.
        Must NOT treat as already documented; must update.
        """
        old_code = "TIMEOUT = 30"
        new_code = "TIMEOUT = 60"
        doc = "# Config\n\n## Timeout\nDefault TIMEOUT = 30. Refresh interval is 60 seconds."

        mock_ai = make_varying_ai(
            [
                {
                    "proposed_content": "Default TIMEOUT = 60. Refresh interval is 60 seconds.",
                    "explanation": "Updated TIMEOUT to 60",
                    "requires_human_review": False,
                }
            ]
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/config.py": (old_code, new_code)},
            documents={"config.md": doc},
        )
        assert report.total_sections_applied == 1
        assert "TIMEOUT = 60" in report.updated_documents["config.md"]

    def test_d5_changed_return_annotation_not_suppressed_by_unchanged_parameter(self):
        """
        D5: Function return type changes from int to str, but parameter 'count: int' is unchanged.
        Doc contains 'get_total(count) returns a int'.
        Must NOT be suppressed; must update return type to str.
        """
        old_code = "def get_total(count: int) -> int: return count"
        new_code = "def get_total(count: int) -> str: return str(count)"
        doc = "# API\n\n## Total\nCall `get_total(count)`: function returns a `int`."

        mock_ai = make_varying_ai(
            [
                {
                    "proposed_content": "Call `get_total(count)`: function returns a `str`.",
                    "explanation": "Updated return type to str",
                    "requires_human_review": False,
                }
            ]
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/api.py": (old_code, new_code)},
            documents={"api.md": doc},
        )
        assert report.total_sections_applied == 1
        assert "returns a `str`" in report.updated_documents["api.md"]

    def test_d6_and_d11_stale_documentation_never_reported_as_no_change(self):
        """
        D6 & D11: Stale documentation must never be incorrectly reported as no_change.
        """
        old_code = "DEFAULT_CACHE_SIZE = 100"
        new_code = "DEFAULT_CACHE_SIZE = 500"
        doc = "# Cache\n\n## Settings\nConfigured with DEFAULT_CACHE_SIZE = 100."

        mock_ai = make_varying_ai(
            [
                {
                    "proposed_content": "Configured with DEFAULT_CACHE_SIZE = 500.",
                    "explanation": "Updated cache size",
                    "requires_human_review": False,
                }
            ]
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/cache.py": (old_code, new_code)},
            documents={"cache.md": doc},
        )
        assert report.total_sections_applied == 1
        assert len(report.applied_updates) == 1
        assert report.applied_updates[0].status == "applied"
        assert report.applied_updates[0].status != "no_change"
