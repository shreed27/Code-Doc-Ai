"""
Adversarial tests for technical claim validation hardening (Priority 2).
Replaces broad numeric-literal authorization with exact entity-tied evidence,
validates semantic equivalence, rejects stale numbers, and blocks unrelated literals.
"""

import json

from doc_updater import DocumentationUpdater, values_are_equivalent


def make_mock_ai(response_dict):
    def _generator(_system_prompt, _user_prompt):
        return json.dumps(response_dict)

    return _generator


class TestClaimValidationHardened:
    def test_values_are_equivalent_semantic_matching(self):
        assert values_are_equivalent("False", "false")
        assert values_are_equivalent("True", "true")
        assert values_are_equivalent("30", "30.0")
        assert values_are_equivalent("0", "0.0")
        assert values_are_equivalent("None", "null")
        assert values_are_equivalent("None", "none")
        assert values_are_equivalent("hello", "hello")
        assert not values_are_equivalent("30", "31")
        assert not values_are_equivalent("true", "false")

    def test_constant_value_mismatch_rejected(self):
        old_code = "DEFAULT_PORT = 8080"
        new_code = "DEFAULT_PORT = 9000"
        doc = "# Config\n\n## Network\nDefault port is 8080."

        # AI hallucinates that port is 7000
        mock_ai = make_mock_ai(
            {
                "proposed_content": "Default port is DEFAULT_PORT = 7000.",
                "explanation": "Updated port",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/config.py": (old_code, new_code)},
            documents={"config.md": doc},
        )
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any(
            "constant_value_mismatch" in f
            for r in report.rejected_updates
            for f in r.validation.failed_checks
        )

    def test_constant_value_semantic_equivalence_accepted(self):
        old_code = "DEBUG_ENABLED: bool = False"
        new_code = "DEBUG_ENABLED: bool = False"
        doc = "# Config\n\n## Debug\nDebug mode info."

        # Doc claims DEBUG_ENABLED = false (lowercase bool)
        mock_ai = make_mock_ai(
            {
                "proposed_content": "Debug mode is configured via `DEBUG_ENABLED = false`.",
                "explanation": "Documented debug flag",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/config.py": (old_code, new_code)},
            documents={"config.md": doc},
        )
        # Should not reject due to constant_value_mismatch
        for r in report.rejected_updates:
            for f in r.validation.failed_checks:
                assert "constant_value_mismatch" not in f

    def test_unrelated_literal_does_not_authorize_numeric_claim(self):
        """
        An unrelated constant SLEEP_DELAY = 7 in code must NOT authorize
        an invented claim that retries is 7 when MAX_RETRIES is 3.
        """
        old_code = """
SLEEP_DELAY = 7
def fetch(): pass
"""
        new_code = """
SLEEP_DELAY = 7
MAX_RETRIES = 3
def fetch(retries: int = MAX_RETRIES): pass
"""
        doc = "# API\n\n## Fetch\nFetches data."

        # Model claims retries is 7 (picking up SLEEP_DELAY)
        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `fetch(retries)`. All requests have retries of 7.",
                "explanation": "Added retries doc",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/api.py": (old_code, new_code)},
            documents={"api.md": doc},
        )
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any(
            "unsupported_numeric_claim" in f
            for r in report.rejected_updates
            for f in r.validation.failed_checks
        )

    def test_stale_numbers_from_old_doc_are_rejected(self):
        """
        Old doc claimed 500 requests. Code changes limit to 100 requests.
        AI proposing stale 500 requests must be rejected.
        """
        old_code = "RATE_LIMIT = 500"
        new_code = "RATE_LIMIT = 100"
        doc = "# API\n\n## Rate Limit\nRate limit of 500 requests per minute."

        mock_ai = make_mock_ai(
            {
                "proposed_content": "Rate limit is 500 requests per minute with updated engine.",
                "explanation": "Stale rate limit claim",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/api.py": (old_code, new_code)},
            documents={"api.md": doc},
        )
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any(
            "unsupported_numeric_claim" in f or "constant_value_mismatch" in f
            for r in report.rejected_updates
            for f in r.validation.failed_checks
        )
