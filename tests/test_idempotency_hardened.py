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
