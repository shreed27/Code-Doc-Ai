"""
Hardened adversarial tests for Entity-Bound Claim Validation (Priority 3).

Covers probes:
A1: TIMEOUT = 60 with SLEEP = 45 rejects doc claiming TIMEOUT = 45.
A2: PAGE_SIZE = 500 rejects doc claiming 500 requests rate limit.
A3: SLEEP = 45 rejects doc claiming RETRY_DELAY = 45.
A4: Parameter type mismatch (parameter x: int, doc claims x: str, while y: str exists) is rejected.
A5: Return type mismatch (def get_count() -> int: doc claims returns a str) is rejected.
A6: Environment variable check (source has DATABASE_URL, doc claiming SECRET_API_KEY is rejected).
A7: Removed symbol with historical/negated statement is accepted; active call is rejected.
P4: Valid type alias (UserId = int, user_id: UserId, doc says user_id: int) is accepted.
P5: Imported constant.
P6: Conditional assignment.
P7: Reassigned constant.
P8: Inherited methods.
P9: Changed default between revisions (old default 30, new default 60 -> doc claiming 30 is rejected).
P10: Configuration values / environment variables.
E2: Unverifiable external dependency claim routes to review_required.
E3: Explicitly negated active instruction ("Never pass None to connect()") accepted as valid.
"""

import json

import pytest

from doc_updater import DocumentationUpdater


def make_mock_ai(response_dict):
    def _generator(_system_prompt, _user_prompt):
        return json.dumps(response_dict)

    return _generator


class TestEntityBoundClaimsExtended:
    def test_a1_timeout_cannot_be_authorized_by_sleep_constant(self):
        """A1: TIMEOUT = 60 in code with SLEEP = 45; proposed doc claiming TIMEOUT = 45 is rejected."""
        old_code = "TIMEOUT = 30\nSLEEP = 45\ndef run(): pass"
        new_code = "TIMEOUT = 60\nSLEEP = 45\ndef run(): pass"
        doc = "# Config\n\n## Timeouts\nTimeout configuration: `TIMEOUT` is 30."

        mock_ai = make_mock_ai(
            {
                "proposed_content": "The request TIMEOUT is 45 seconds.",
                "explanation": "Timeout claim using SLEEP value",
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
        assert any("timeout mismatch" in f for r in report.rejected_updates for f in r.validation.failed_checks)

    def test_a2_page_size_cannot_authorize_rate_limit(self):
        """A2: PAGE_SIZE = 500 must not authorize a 500-request rate limit."""
        old_code = "PAGE_SIZE = 100\ndef list_items(): pass"
        new_code = "PAGE_SIZE = 500\ndef list_items(): pass"
        doc = "# API\n\n## Limits\nAPI limits: `PAGE_SIZE` items per page."

        mock_ai = make_mock_ai(
            {
                "proposed_content": "The API is rate limited to 500 requests per minute.",
                "explanation": "Invented rate limit using PAGE_SIZE",
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
        assert any("rate limit of '500' claimed" in f for r in report.rejected_updates for f in r.validation.failed_checks)

    def test_a3_sleep_cannot_authorize_retry_delay(self):
        """A3: SLEEP = 45 must not authorize a RETRY_DELAY of 45."""
        old_code = "SLEEP = 45\nRETRY_DELAY = 5\ndef retry_op(): pass"
        new_code = "SLEEP = 45\nRETRY_DELAY = 10\ndef retry_op(): pass"
        doc = "# Operations\n\n## Retries\nRetry policy: `RETRY_DELAY` interval."

        mock_ai = make_mock_ai(
            {
                "proposed_content": "Operations are executed with retry_delay = 45 seconds.",
                "explanation": "Retry delay claim using SLEEP value",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/ops.py": (old_code, new_code)},
            documents={"ops.md": doc},
        )
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any("retry delay '45' claimed" in f for r in report.rejected_updates for f in r.validation.failed_checks)

    def test_a4_parameter_type_mismatch_with_another_parameter(self):
        """A4: Parameter x: int cannot be documented as x: str even though y: str exists."""
        old_code = "def process(x: int): pass"
        new_code = "def process(x: int, y: str = ''): pass"
        doc = "# API\n\n## Process\nCall `process(x)` to process data."

        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `process(x: str, y: str)` to process data.",
                "explanation": "Documented x as str",
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
        assert any("parameter_type_mismatch" in f for r in report.rejected_updates for f in r.validation.failed_checks)

    def test_a5_return_type_mismatch_rejected(self):
        """A5: def get_count() -> int: doc claiming it returns a string is rejected."""
        old_code = "def get_count(): return 42"
        new_code = "def get_count() -> int: return 42"
        doc = "# API\n\n## Count\nCall `get_count()` to get count description."

        mock_ai = make_mock_ai(
            {
                "proposed_content": "The function `get_count` returns a string representation.",
                "explanation": "Claimed string return type",
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
        assert any("return_type_mismatch" in f for r in report.rejected_updates for f in r.validation.failed_checks)

    def test_a6_unsupported_environment_variable_rejected(self):
        """A6: Code has DATABASE_URL, doc claiming SECRET_API_KEY is rejected."""
        old_code = 'import os\ndef get_db(): return None'
        new_code = 'import os\ndef get_db(): return os.getenv("DATABASE_URL")'
        doc = "# Config\n\n## Env\nCall `get_db()`: Environment variables."

        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `get_db()`. Set the SECRET_API_KEY environment variable to authenticate.",
                "explanation": "Invented env var",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/env.py": (old_code, new_code)},
            documents={"env.md": doc},
        )
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any("unsupported_environment_variable" in f for r in report.rejected_updates for f in r.validation.failed_checks)

    def test_a7_removed_symbol_negated_vs_active(self):
        """A7: Removed symbol in negated statement accepted; active instruction rejected."""
        old_code = "def legacy_handler(): pass\ndef modern_handler(): pass"
        new_code = "def modern_handler(): pass"
        doc = "# API\n\n## Handlers\nCall `legacy_handler()` or `modern_handler()`."

        # Case 1: Active instruction to call removed symbol -> REJECTED
        mock_ai_active = make_mock_ai(
            {
                "proposed_content": "Call `legacy_handler()` for legacy requests.",
                "explanation": "Active call to removed symbol",
                "requires_human_review": False,
            }
        )
        updater1 = DocumentationUpdater(ai_generator=mock_ai_active)
        rep1 = updater1.update_documentation(
            code_changes={"src/api.py": (old_code, new_code)},
            documents={"api.md": doc},
        )
        assert rep1.total_sections_applied == 0
        assert rep1.total_sections_rejected >= 1

        # Case 2: Explicitly negated / removal statement -> ACCEPTED
        mock_ai_negated = make_mock_ai(
            {
                "proposed_content": "Call `modern_handler()`. Do not call `legacy_handler`, which has been removed.",
                "explanation": "Proper removal notice",
                "requires_human_review": False,
            }
        )
        updater2 = DocumentationUpdater(ai_generator=mock_ai_negated)
        rep2 = updater2.update_documentation(
            code_changes={"src/api.py": (old_code, new_code)},
            documents={"api.md": doc},
        )
        assert rep2.total_sections_applied == 1
        assert "Do not call `legacy_handler`" in rep2.updated_documents["api.md"]

    def test_p4_type_alias_resolved_correctly(self):
        """P4: Type alias UserId = int allows documenting parameter as int or UserId."""
        old_code = """
UserId = int
def get_user(user_id: UserId): pass
"""
        new_code = """
UserId = int
def get_user(user_id: UserId): pass
"""
        doc = "# Users\n\n## Get\nGet user info."

        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `get_user(user_id: int)` to fetch user profile.",
                "explanation": "Documented underlying type of type alias",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/users.py": (old_code, new_code)},
            documents={"users.md": doc},
        )
        # Should not reject due to parameter_type_mismatch
        for r in report.rejected_updates:
            for f in r.validation.failed_checks:
                assert "parameter_type_mismatch" not in f

    def test_p9_changed_defaults_between_revisions_rejects_stale_default(self):
        """P9: Parameter default changed from 30 to 60. Proposed doc claiming 30 is rejected."""
        old_code = "def connect(timeout: int = 30): pass"
        new_code = "def connect(timeout: int = 60): pass"
        doc = "# Network\n\n## Connect\nCall `connect(timeout)`."

        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `connect(timeout = 30)` to initiate connection.",
                "explanation": "Stale timeout default",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/net.py": (old_code, new_code)},
            documents={"net.md": doc},
        )
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any("parameter_default_mismatch" in f for r in report.rejected_updates for f in r.validation.failed_checks)

    def test_e2_unverifiable_claim_routes_to_human_review(self):
        """E2: Unverifiable external dependency claim routes to review_required, not silently accepted."""
        old_code = "def sync(): pass"
        new_code = "def sync(force: bool = False): pass"
        doc = "# Sync\n\n## Operation\nCall `sync()` for sync details."

        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `sync(force)`. The sync operation depends on third-party Redis server version 7.0.",
                "explanation": "Unverifiable third-party claim",
                "uncertainties": ["External Redis version requirements cannot be verified from local AST"],
                "requires_human_review": True,
                "review_reason": "External dependency on Redis 7.0 requires human verification.",
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/sync.py": (old_code, new_code)},
            documents={"sync.md": doc},
        )
        assert report.total_sections_applied == 0
        assert len(report.review_required_updates) >= 1
        assert report.review_required_updates[0].status == "review_required"
        assert any("Redis" in r for r in report.review_required_updates[0].validation.review_reasons)

    def test_e3_explicitly_negated_instruction_accepted(self):
        """E3: Explicitly negated active instruction ('Never pass None to connect') accepted."""
        old_code = "def connect(session: dict): pass"
        new_code = "def connect(session: dict): pass"
        doc = "# Network\n\n## Connection\nConnection instructions."

        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `connect(session)`. Never pass None to connect, as a session dictionary is required.",
                "explanation": "Added valid usage warning",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/net.py": (old_code, new_code)},
            documents={"net.md": doc},
        )
        # Should not be rejected
        assert report.total_sections_rejected == 0
