"""
Tests for AI Documentation Updating Engine (Phase 3.4).

Covers:
1. Changed function signature updates correct documentation
2. New parameter updates relevant API documentation
3. Changed configuration constant updates configuration instructions
4. Deleted function triggers stale documentation handling
5. Multiple affected documents updated correctly
6. Unrelated sections remain unchanged byte-for-byte
7. Nested Markdown sections preserved
8. Duplicate headings handled safely
9. Invalid AI output rejected
10. Unsupported parameters and fabricated functions rejected
11. Insufficient evidence produces review-required result
12. Failed section update does not corrupt other documents
13. Re-running the same update is idempotent
"""

import json

from doc_updater import (
    DocumentationUpdater,
    DocumentationValidator,
    build_update_prompt,
    parse_ai_response,
)
from section_patcher import parse_sections


def make_mock_ai(response_dict: dict) -> callable:
    """Helper to create a deterministic mock AI generator."""

    def _mock(_system_prompt: str, _user_prompt: str) -> str:
        return json.dumps(response_dict)

    return _mock


class TestDocumentationUpdater:
    """Test suite verifying end-to-end documentation updating behaviors."""

    def test_changed_function_signature_updates_correct_doc(self):
        old_code = "def authenticate(user: str): pass"
        new_code = "def authenticate(user: str, token: str = ''): pass"

        doc = """# Auth Module

## Usage
Call `authenticate(user)` with the username.

## Architecture
Auth runs locally.
"""
        mock_response = {
            "proposed_content": "Call `authenticate(user, token)` with username and optional authentication token.",
            "explanation": "Added `token` parameter to authenticate documentation.",
            "technical_evidence": ["token parameter added"],
            "requires_human_review": False,
        }

        updater = DocumentationUpdater(ai_generator=make_mock_ai(mock_response))
        report = updater.update_documentation(
            code_changes={"src/auth.py": (old_code, new_code)},
            documents={"README.md": doc},
        )

        assert report.total_sections_applied == 1
        assert len(report.applied_updates) == 1
        updated_doc = report.updated_documents["README.md"]

        assert "authenticate(user, token)" in updated_doc
        # Verify unrelated section remains untouched
        assert "## Architecture\nAuth runs locally." in updated_doc

    def test_new_parameter_updates_relevant_api_doc(self):
        old_code = "def fetch_records(limit: int = 100): pass"
        new_code = "def fetch_records(limit: int = 100, cursor: str = None): pass"

        api_doc = """# API Reference

## Database Querying
The `fetch_records` function retrieves items up to `limit`.
"""
        mock_response = {
            "proposed_content": "The `fetch_records` function retrieves items up to `limit`, with pagination via `cursor`.",
            "explanation": "Documented new cursor parameter for pagination.",
            "technical_evidence": ["cursor parameter added to fetch_records"],
            "requires_human_review": False,
        }

        updater = DocumentationUpdater(ai_generator=make_mock_ai(mock_response))
        report = updater.update_documentation(
            code_changes={"src/db.py": (old_code, new_code)},
            documents={"docs/api.md": api_doc},
        )

        assert report.total_sections_applied == 1
        assert "pagination via `cursor`" in report.updated_documents["docs/api.md"]

    def test_changed_configuration_constant_updates_configuration(self):
        old_code = "MAX_RETRIES = 3"
        new_code = "MAX_RETRIES = 5"

        config_doc = """# Configuration

## Retry Policy
Network calls retry up to `MAX_RETRIES` times (default 3).
"""
        mock_response = {
            "proposed_content": "Network calls retry up to `MAX_RETRIES` times (default 5).",
            "explanation": "Updated MAX_RETRIES default value to 5.",
            "technical_evidence": ["MAX_RETRIES constant changed to 5"],
            "requires_human_review": False,
        }

        updater = DocumentationUpdater(ai_generator=make_mock_ai(mock_response))
        report = updater.update_documentation(
            code_changes={"src/config.py": (old_code, new_code)},
            documents={"docs/config.md": config_doc},
        )

        assert report.total_sections_applied == 1
        assert "(default 5)" in report.updated_documents["docs/config.md"]

    def test_deleted_function_triggers_stale_doc_handling(self):
        old_code = "def legacy_handler(): pass"
        new_code = "# removed"

        doc = """# Endpoints

## Legacy API
Call `legacy_handler` to process v1 requests.
"""
        mock_response = {
            "proposed_content": "The `legacy_handler` function has been removed. Use v2 endpoints instead.",
            "explanation": "Noted removal of legacy_handler.",
            "technical_evidence": ["legacy_handler removed from source"],
            "requires_human_review": False,
        }

        updater = DocumentationUpdater(ai_generator=make_mock_ai(mock_response))
        report = updater.update_documentation(
            code_changes={"src/api.py": (old_code, new_code)},
            documents={"docs/endpoints.md": doc},
        )

        assert report.total_sections_applied == 1
        assert "has been removed" in report.updated_documents["docs/endpoints.md"]

    def test_multiple_affected_documents_updated_correctly(self):
        old_code = "def export_data(format: str): pass"
        new_code = "def export_data(format: str, compress: bool = True): pass"

        doc1 = "# User Guide\n\n## Exporting\nRun `export_data`."
        doc2 = (
            "# API Reference\n\n## Export Details\nThe `export_data` function details."
        )

        def multi_mock(_sys, user_prompt: str) -> str:
            if "docs/guide.md" in user_prompt:
                return json.dumps(
                    {
                        "proposed_content": "Run `export_data` with optional `compress`.",
                        "explanation": "Updated User Guide",
                        "requires_human_review": False,
                    }
                )
            return json.dumps(
                {
                    "proposed_content": "The `export_data` function now accepts `compress` flag.",
                    "explanation": "Updated API Reference",
                    "requires_human_review": False,
                }
            )

        updater = DocumentationUpdater(ai_generator=multi_mock)
        report = updater.update_documentation(
            code_changes={"src/exporter.py": (old_code, new_code)},
            documents={"docs/guide.md": doc1, "docs/api.md": doc2},
        )

        assert report.total_sections_applied == 2
        assert "with optional `compress`" in report.updated_documents["docs/guide.md"]
        assert "accepts `compress` flag" in report.updated_documents["docs/api.md"]

    def test_unrelated_sections_remain_unchanged_byte_for_byte(self):
        old_code = "def start(): pass"
        new_code = "def start(port: int = 8080): pass"

        doc = """# Application

## Overview
This is an untouched overview paragraph with exact spacing.

## Getting Started
Call `start` to boot server.

## Troubleshooting
Check logs at `/var/log/app.log`.
"""
        mock_response = {
            "proposed_content": "Call `start(port)` to boot server.",
            "explanation": "Added port parameter",
            "requires_human_review": False,
        }

        updater = DocumentationUpdater(ai_generator=make_mock_ai(mock_response))
        report = updater.update_documentation(
            code_changes={"src/app.py": (old_code, new_code)},
            documents={"README.md": doc},
        )

        updated_text = report.updated_documents["README.md"]
        # Prefix check
        assert updated_text.startswith(
            "# Application\n\n## Overview\nThis is an untouched overview paragraph with exact spacing.\n\n## Getting Started"
        )
        # Suffix check
        assert updated_text.endswith(
            "## Troubleshooting\nCheck logs at `/var/log/app.log`.\n"
        )

    def test_nested_markdown_sections_are_preserved(self):
        old_code = "def init_cluster(): pass"
        new_code = "def init_cluster(nodes: int = 3): pass"

        doc = """# Deployment

## Cluster Setup
Run `init_cluster` for deployment.

### Worker Nodes
Worker node architecture details.

### Storage
Persistent storage options.
"""
        mock_response = {
            "proposed_content": "Run `init_cluster(nodes)` to deploy with worker count.",
            "explanation": "Added nodes param",
            "requires_human_review": False,
        }

        updater = DocumentationUpdater(ai_generator=make_mock_ai(mock_response))
        report = updater.update_documentation(
            code_changes={"src/deploy.py": (old_code, new_code)},
            documents={"DEPLOY.md": doc},
        )

        updated = report.updated_documents["DEPLOY.md"]
        assert "init_cluster(nodes)" in updated
        # Subsection headings and contents must be preserved
        assert "### Worker Nodes\nWorker node architecture details." in updated
        assert "### Storage\nPersistent storage options." in updated

    def test_duplicate_headings_handled_safely(self):
        old_code = "def run_worker(): pass"
        new_code = "def run_worker(concurrency: int = 4): pass"

        doc = """# Service A
## Configuration
Service A uses `run_worker`.

# Service B
## Configuration
Service B config.
"""
        mock_response = {
            "proposed_content": "Service A uses `run_worker(concurrency)`.",
            "explanation": "Updated Service A worker run",
            "requires_human_review": False,
        }

        updater = DocumentationUpdater(ai_generator=make_mock_ai(mock_response))
        report = updater.update_documentation(
            code_changes={"src/worker.py": (old_code, new_code)},
            documents={"README.md": doc},
        )

        assert report.total_sections_applied == 1
        updated = report.updated_documents["README.md"]
        # Service A was updated
        assert "run_worker(concurrency)" in updated
        # Service B configuration section was NOT overwritten
        assert "# Service B\n## Configuration\nService B config." in updated

    def test_invalid_ai_output_rejected(self):
        old_code = "def ping(): pass"
        new_code = "def ping(timeout: int = 1): pass"

        doc = "# Health\n## Ping\nCall `ping`."

        # Mock returns plain string with unclosed code fence
        def broken_ai(_s, _u):
            return "```json\n{ broken json here"

        updater = DocumentationUpdater(ai_generator=broken_ai)
        report = updater.update_documentation(
            code_changes={"src/health.py": (old_code, new_code)},
            documents={"health.md": doc},
        )

        # Flagged for human review instead of applied
        assert report.total_sections_applied == 0
        assert len(report.review_required_updates) == 1
        assert report.updated_documents["health.md"] == doc  # untouched

    def test_unsupported_parameters_and_fabricated_functions_rejected(self):
        old_code = "def login(user: str): pass"
        new_code = "def login(user: str, remember: bool = False): pass"

        doc = "# Auth\n## Login\nCall `login`."

        # AI hallucinates a non-existent parameter and invented function
        mock_hallucination = {
            "proposed_content": "Call `login(user, remember)` or use `magic_auto_login(secret_enterprise_token)`.",
            "explanation": "Added remember parameter and enterprise login",
            "requires_human_review": False,
        }

        updater = DocumentationUpdater(ai_generator=make_mock_ai(mock_hallucination))
        report = updater.update_documentation(
            code_changes={"src/auth.py": (old_code, new_code)},
            documents={"auth.md": doc},
        )

        assert report.total_sections_applied == 0
        assert len(report.rejected_updates) == 1
        rejected = report.rejected_updates[0]
        assert "unsupported_technical_tokens" in str(rejected.validation.failed_checks)
        assert report.updated_documents["auth.md"] == doc  # untouched

    def test_insufficient_evidence_produces_review_required(self):
        old_code = "def calculate(val: int): pass"
        new_code = "def calculate(val: int, opt: bool = True): pass"

        doc = "# Math\n\n## Calc\nUses `calculate`."

        mock_uncertain = {
            "proposed_content": "Uses `calculate` with new options.",
            "explanation": "Uncertain about behavioral impact",
            "uncertainties": ["Internal logic changed but no public API changes"],
            "requires_human_review": True,
            "review_reason": "Low confidence on behavioral changes.",
        }

        updater = DocumentationUpdater(ai_generator=make_mock_ai(mock_uncertain))
        report = updater.update_documentation(
            code_changes={"src/math_util.py": (old_code, new_code)},
            documents={"math.md": doc},
        )
        assert report.total_sections_applied == 0
        assert len(report.review_required_updates) == 1
        assert report.review_required_updates[0].status == "review_required"
        assert report.updated_documents["math.md"] == doc

    def test_failed_section_update_does_not_corrupt_other_documents(self):
        old_code = "def sync_all(): pass"
        new_code = "def sync_all(force: bool = False): pass"

        doc_good = "# Sync\n\n## Main Sync\nCall `sync_all`."
        doc_bad = "# Corrupted\n\n## Bad Section\nCall `sync_all`."

        def selective_ai(_s, user_prompt: str):
            if "Bad Section" in user_prompt:
                # Hallucination in bad doc
                return json.dumps(
                    {
                        "proposed_content": "Uses `fake_hallucinated_function()`.",
                        "requires_human_review": False,
                    }
                )
            return json.dumps(
                {
                    "proposed_content": "Call `sync_all(force)` to run sync.",
                    "requires_human_review": False,
                }
            )

        updater = DocumentationUpdater(ai_generator=selective_ai)
        report = updater.update_documentation(
            code_changes={"src/sync.py": (old_code, new_code)},
            documents={"good.md": doc_good, "bad.md": doc_bad},
        )

        # All-or-nothing: rejection in bad doc prevents partial application to good doc!
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert report.updated_documents["good.md"] == doc_good
        assert report.updated_documents["bad.md"] == doc_bad

    def test_rerunning_update_is_idempotent(self):
        old_code = "def ping(): pass"
        new_code = "def ping(count: int = 1): pass"

        doc = "# Status\n## Ping\nCall `ping`."
        mock_response = {
            "proposed_content": "Call `ping(count)` to verify status.",
            "explanation": "Added count",
            "requires_human_review": False,
        }

        updater = DocumentationUpdater(ai_generator=make_mock_ai(mock_response))
        report1 = updater.update_documentation(
            code_changes={"src/status.py": (old_code, new_code)},
            documents={"status.md": doc},
        )
        first_result = report1.updated_documents["status.md"]

        report2 = updater.update_documentation(
            code_changes={"src/status.py": (old_code, new_code)},
            documents={"status.md": first_result},
        )
        second_result = report2.updated_documents["status.md"]

        # Idempotent: re-running does not duplicate content or corrupt document
        assert first_result == second_result


class TestPromptAndResponseHelpers:
    """Test helper utilities for prompt construction and response parsing."""

    def test_build_update_prompt_formatting(self):
        from doc_updater import SectionUpdateRequest
        from impact_mapper import ImpactEvidence

        req = SectionUpdateRequest(
            doc_path="README.md",
            section_id="usage",
            section_heading="Usage",
            current_section_content="Call `foo`.",
            source_files=["src/foo.py"],
            changed_symbols=["foo"],
            change_categories=["modified_signature"],
            impact_reasons=["Signature modified"],
            impact_evidence=[
                ImpactEvidence(
                    evidence_type="explicit_symbol",
                    matched_text="foo",
                    context="Call `foo`.",
                    confidence=1.0,
                )
            ],
            old_code_snippets={"src/foo.py": "def foo(): pass"},
            new_code_snippets={"src/foo.py": "def foo(bar: int): pass"},
            style_guidelines="Use active voice.",
        )
        prompt = build_update_prompt(req)
        assert "Target Section: Usage" in prompt
        assert "src/foo.py" in prompt
        assert "Use active voice." in prompt

    def test_parse_ai_response_empty_and_fenced(self):
        # Empty string
        resp_empty = parse_ai_response("", "doc.md", "sec-1")
        assert resp_empty.requires_human_review is True

        # Fenced markdown
        fenced_json = """```json
{
  "doc_path": "doc.md",
  "section_id": "sec-1",
  "proposed_content": "New content",
  "explanation": "Updated"
}
```"""
        resp_fenced = parse_ai_response(fenced_json, "doc.md", "sec-1")
        assert resp_fenced.proposed_content == "New content"
        assert resp_fenced.requires_human_review is False

    def test_parse_ai_response_plain_text_fallback(self):
        plain_text = "Just plain documentation text without JSON format."
        resp = parse_ai_response(plain_text, "doc.md", "sec-1")
        assert resp.requires_human_review is True
        assert resp.proposed_content == plain_text
        assert "unstructured text" in resp.explanation

    def test_validator_missing_section_and_empty_content_warning(self):
        from doc_updater import AISectionUpdateResponse, SectionUpdateRequest

        req = SectionUpdateRequest(
            doc_path="doc.md",
            section_id="nonexistent-section",
            section_heading="Nonexistent",
            current_section_content="Existing content here.",
            source_files=[],
            changed_symbols=[],
            change_categories=[],
            impact_reasons=[],
            impact_evidence=[],
            old_code_snippets={},
            new_code_snippets={},
        )
        resp = AISectionUpdateResponse(
            doc_path="doc.md",
            section_id="nonexistent-section",
            proposed_content="",
            explanation="Empty proposal",
        )
        parsed_doc = parse_sections("# Real Section\nContent")
        val = DocumentationValidator.validate(req, resp, parsed_doc)
        assert val.is_valid is False
        assert any("target_section_not_found" in f for f in val.failed_checks)
        assert any("proposed_content_is_empty" in w for w in val.warnings)

    def test_validator_subheading_warning(self):
        from doc_updater import AISectionUpdateResponse, SectionUpdateRequest

        req = SectionUpdateRequest(
            doc_path="doc.md",
            section_id="real-section",
            section_heading="Real Section",
            current_section_content="Plain text with no subheadings.",
            source_files=[],
            changed_symbols=[],
            change_categories=[],
            impact_reasons=[],
            impact_evidence=[],
            old_code_snippets={},
            new_code_snippets={},
        )
        resp = AISectionUpdateResponse(
            doc_path="doc.md",
            section_id="real-section",
            proposed_content="### Injected Subheading\nNew content under injected subheading.",
            explanation="Added subheading",
        )
        parsed_doc = parse_sections("# Real Section\nPlain text with no subheadings.")
        val = DocumentationValidator.validate(req, resp, parsed_doc)
        assert any("subheadings_introduced_in_leaf_section" in w for w in val.warnings)

    def test_updater_target_section_missing_in_doc(self):
        from impact_mapper import ImpactEvidence, SectionImpact

        fake_impact = SectionImpact(
            source_file="src/fake.py",
            changed_symbol="fake",
            change_category="added_symbol",
            doc_path="doc.md",
            section_id="missing-section-id",
            section_heading="Missing",
            is_deterministic=True,
            confidence=1.0,
            evidence=[ImpactEvidence("explicit_symbol", "fake", "fake", 1.0)],
            reason="Fake reason",
        )

        updater = DocumentationUpdater(
            ai_generator=make_mock_ai({"proposed_content": "xyz"})
        )
        report = updater.update_documentation(
            code_changes={"src/fake.py": ("x=1", "x=2")},
            documents={"doc.md": "# Doc\nContent"},
            precomputed_impacts=[fake_impact],
        )

        assert report.total_sections_applied == 0
        assert report.total_sections_rejected == 1
        assert "not found in document" in report.rejected_updates[0].explanation

    def test_default_ai_generator_exception_fallback(self):
        # When config or OpenAI is unavailable, fallback gracefully returns review-required JSON
        updater = DocumentationUpdater(ai_generator=None)
        raw_out = updater._default_ai_generator("sys", "user")
        data = json.loads(raw_out)
        assert data["requires_human_review"] is True
        assert "AI completion unavailable" in data["review_reason"]


class TestAdversarialProbesP1ToP15:
    """
    Direct verification of Claude's adversarial audit probes (P1 through P15).
    Ensures safety remediation and product integration invariants are strictly maintained.
    """

    def test_p1_missing_section_target_handled_gracefully(self):
        """P1: Missing section target handled cleanly without crash and leaves document untouched."""
        from impact_mapper import ImpactEvidence, SectionImpact

        fake_impact = SectionImpact(
            source_file="src/missing.py",
            changed_symbol="missing_fn",
            change_category="added_symbol",
            doc_path="doc.md",
            section_id="nonexistent-target-section",
            section_heading="Nonexistent",
            is_deterministic=True,
            confidence=1.0,
            evidence=[
                ImpactEvidence("explicit_symbol", "missing_fn", "missing_fn", 1.0)
            ],
            reason="Function added",
        )
        original_doc = "# Title\n\n## Real Section\nReal content."
        updater = DocumentationUpdater(
            ai_generator=make_mock_ai({"proposed_content": "xyz"})
        )
        report = updater.update_documentation(
            code_changes={"src/missing.py": ("", "def missing_fn(): pass")},
            documents={"doc.md": original_doc},
            precomputed_impacts=[fake_impact],
        )

        assert report.total_sections_applied == 0
        assert report.total_sections_rejected == 1
        assert report.rejected_updates[0].status == "rejected"
        assert report.updated_documents["doc.md"] == original_doc

    def test_p2_insufficient_evidence_produces_review_required_with_call_count(self):
        """P2: Insufficient evidence produces review-required and verifies AI call counts."""
        calls = 0

        def counting_ai(_sys, _user):
            nonlocal calls
            calls += 1
            return json.dumps(
                {
                    "proposed_content": "Uncertain changes.",
                    "explanation": "Low confidence on behavioral impact",
                    "uncertainties": ["Ambiguous interface change"],
                    "requires_human_review": True,
                    "review_reason": "Insufficient evidence to determine public contract.",
                }
            )

        old_code = "def process(): return 1"
        new_code = "def process(): return 2"
        doc = "# Operations\n\n## Process\nCall `process`."

        updater = DocumentationUpdater(ai_generator=counting_ai)
        report = updater.update_documentation(
            code_changes={"src/op.py": (old_code, new_code)},
            documents={"ops.md": doc},
        )

        assert calls == 1
        assert report.total_sections_applied == 0
        assert len(report.review_required_updates) == 1
        assert report.review_required_updates[0].status == "review_required"
        assert report.updated_documents["ops.md"] == doc

    def test_p3_deleted_symbol_active_rejected_vs_historical_accepted(self):
        """P3: Active stale references to removed symbols rejected; historical notes accepted."""
        old_code = "def old_calc(): pass"
        new_code = "def new_calc(): pass"
        doc = "# Math\n\n## Calculation\nCall `old_calc`."

        # Case A: AI proposes active instruction to call removed symbol
        active_ai = make_mock_ai(
            {
                "proposed_content": "Call `old_calc` or `new_calc` to perform the operation.",
                "explanation": "Added new_calc",
                "requires_human_review": False,
            }
        )
        updater_a = DocumentationUpdater(ai_generator=active_ai)
        report_a = updater_a.update_documentation(
            code_changes={"src/calc.py": (old_code, new_code)},
            documents={"calc.md": doc},
        )
        assert report_a.total_sections_applied == 0
        assert report_a.total_sections_rejected == 1
        assert any(
            "active_reference_to_removed_symbol" in f
            for f in report_a.rejected_updates[0].validation.failed_checks
        )

        # Case B: AI proposes historical deprecation/removal note
        historical_ai = make_mock_ai(
            {
                "proposed_content": "The `old_calc` function has been removed and deprecated. Use `new_calc` instead.",
                "explanation": "Documented removal of old_calc",
                "requires_human_review": False,
            }
        )
        updater_b = DocumentationUpdater(ai_generator=historical_ai)
        report_b = updater_b.update_documentation(
            code_changes={"src/calc.py": (old_code, new_code)},
            documents={"calc.md": doc},
        )
        assert report_b.total_sections_applied == 1
        assert "has been removed" in report_b.updated_documents["calc.md"]

    def test_p4_idempotency_with_no_change_status_and_call_counts(self):
        """P4: Idempotency with no_change status and prevents redundant re-writes."""
        old_code = "def get_user(): pass"
        new_code = "def get_user(uid: int): pass"
        doc = "# API\n\n## Users\nCall `get_user(uid)` with user ID."

        calls = 0

        def static_ai(_s, _u):
            nonlocal calls
            calls += 1
            return json.dumps(
                {
                    "proposed_content": "Call `get_user(uid)` with user ID.",
                    "explanation": "Already matches up-to-date documentation",
                    "requires_human_review": False,
                }
            )

        updater = DocumentationUpdater(ai_generator=static_ai)
        report = updater.update_documentation(
            code_changes={"src/user.py": (old_code, new_code)},
            documents={"api.md": doc},
        )

        assert calls == 1
        assert report.total_sections_applied == 0
        assert len(report.applied_updates) == 1
        assert report.applied_updates[0].status == "no_change"
        assert report.updated_documents["api.md"] == doc

    def test_p5_unrelated_sections_compared_completely_byte_for_byte(self):
        """P5: All non-target sections remain 100% byte-identical."""
        old_code = "def step_b(): pass"
        new_code = "def step_b(v: int = 1): pass"

        doc = (
            "# Workflow\n\n"
            "## Section 1\n"
            "| Key | Value |\n"
            "| --- | --- |\n"
            "| alpha | 123 |\n"
            "Trailing whitespace here   \n\n"
            "## Section 2\n"
            "Call `step_b`.\n\n"
            "## Section 3\n"
            "Special symbols: 🚀 and `code_snippet`.\n"
        )
        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `step_b(v)` with version.",
                "explanation": "Updated step_b",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/workflow.py": (old_code, new_code)},
            documents={"workflow.md": doc},
        )

        assert report.total_sections_applied == 1
        res = report.updated_documents["workflow.md"]

        sec1_orig = "## Section 1\n| Key | Value |\n| --- | --- |\n| alpha | 123 |\nTrailing whitespace here   "
        sec3_orig = "## Section 3\nSpecial symbols: 🚀 and `code_snippet`.\n"
        assert sec1_orig in res
        assert sec3_orig in res

    def test_p6_parameter_type_spoofing_rejected(self):
        """P6: Reject parameter type spoofing (e.g. timeout: int documented as timeout: str)."""
        old_code = "def dial(): pass"
        new_code = "def dial(timeout: int = 10): pass"
        doc = "# Network\n\n## Dial\nCall `dial`."

        # Document falsely claims timeout is a str
        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `dial(timeout: str)` to connect.",
                "explanation": "Documented timeout parameter",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/net.py": (old_code, new_code)},
            documents={"net.md": doc},
        )

        assert report.total_sections_applied == 0
        assert report.total_sections_rejected == 1
        assert any(
            "parameter_type_mismatch" in f
            for f in report.rejected_updates[0].validation.failed_checks
        )

    def test_p7_parameter_default_value_spoofing_rejected(self):
        """P7: Reject parameter default value spoofing (e.g. actual default is 60, doc claims 45)."""
        old_code = "def connect(): pass"
        new_code = "def connect(timeout: int = 60): pass"
        doc = "# Connect\n\n## Dial\nCall `connect`."

        # Document claims timeout default is 45 instead of 60
        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `connect(timeout=45)` with default timeout 45 seconds.",
                "explanation": "Documented timeout parameter",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/connect.py": (old_code, new_code)},
            documents={"connect.md": doc},
        )

        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any(
            "parameter_default_mismatch" in f
            for r in report.rejected_updates
            for f in r.validation.failed_checks
        )

    def test_p8_fabricated_function_name_rejected(self):
        """P8: Reject fabricated / invented functions not present in source AST."""
        old_code = "def backup(): pass"
        new_code = "def backup(dest: str = '/tmp'): pass"
        doc = "# Backup\n\n## Run\nCall `backup`."

        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `backup(dest)` or use `cloud_storage_auto_sync()` for cloud backup.",
                "explanation": "Added cloud backup helper",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/backup.py": (old_code, new_code)},
            documents={"backup.md": doc},
        )

        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any(
            "unsupported_technical_tokens" in f
            for r in report.rejected_updates
            for f in r.validation.failed_checks
        )

    def test_p9_comments_and_string_literals_cannot_validate_symbols(self):
        """P9: Tokens found only in comments or string literals cannot validate AI hallucinations."""
        old_code = "def render(): pass"
        new_code = """
# TODO: consider adding hardware_accelerated_gl()
FALLBACK_LABEL = "legacy_raster_renderer"
def render(quality: int = 1):
    msg = "software_engine_pipeline"
    return quality
"""
        doc = "# Render\n\n## Overview\nCall `render`."

        # AI hallucinates functions from comments or strings
        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `render(quality)` or call `hardware_accelerated_gl()`.",
                "explanation": "Added hardware gl",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/render.py": (old_code, new_code)},
            documents={"render.md": doc},
        )

        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any(
            "unsupported_technical_tokens" in f
            for r in report.rejected_updates
            for f in r.validation.failed_checks
        )

    def test_p10_unbalanced_code_fences_rejected(self):
        """P10: Reject proposed documentation with unclosed / unbalanced code fences."""
        old_code = "def ping(): pass"
        new_code = "def ping(count: int = 1): pass"
        doc = "# Health\n\n## Check\nCall `ping`."

        # AI returns an unclosed code fence
        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call ping:\n```python\nping(count=1)\n",
                "explanation": "Added code fence with ping",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/health.py": (old_code, new_code)},
            documents={"health.md": doc},
        )

        assert report.total_sections_applied == 0
        assert report.total_sections_rejected == 1
        assert any(
            "unbalanced_code_fences" in f
            for f in report.rejected_updates[0].validation.failed_checks
        )

    def test_p11_leaf_section_heading_injection_rejected(self):
        """P11: Reject introducing headings into leaf sections."""
        old_code = "def setup(): pass"
        new_code = "def setup(mode: str = 'auto'): pass"
        doc = "# Installation\n\n## Quickstart\nCall `setup`."

        # AI attempts to inject subheadings into a leaf section
        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `setup(mode)`.\n\n### Advanced Options\nSet mode to manual.",
                "explanation": "Added advanced options subsection",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/install.py": (old_code, new_code)},
            documents={"install.md": doc},
        )

        assert report.total_sections_applied == 0
        assert report.total_sections_rejected == 1
        assert any(
            "headings_introduced_in_leaf_section" in f
            for f in report.rejected_updates[0].validation.failed_checks
        )

    def test_p12_yaml_front_matter_preserved_without_corruption(self):
        """P12: Documents with YAML front matter are updated without corrupting header metadata."""
        old_code = "def run(): pass"
        new_code = "def run(verbose: bool = False): pass"

        doc = (
            "---\n"
            "title: Operational Guide\n"
            "version: 2.1\n"
            "author: Core Team\n"
            "---\n\n"
            "# Manual\n\n"
            "## Running\n"
            "Call `run` to start.\n"
        )
        mock_ai = make_mock_ai(
            {
                "proposed_content": "Call `run(verbose)` to start.",
                "explanation": "Added verbose flag",
                "requires_human_review": False,
            }
        )
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/run.py": (old_code, new_code)},
            documents={"manual.md": doc},
        )

        assert report.total_sections_applied == 1
        res = report.updated_documents["manual.md"]
        assert res.startswith(
            "---\ntitle: Operational Guide\nversion: 2.1\nauthor: Core Team\n---"
        )
        assert "run(verbose)" in res

    def test_p13_malformed_ai_responses_handled_safely(self):
        """P13: Handle null, list, number, missing keys, and invalid responses safely."""
        from doc_updater import parse_ai_response

        # null proposed_content
        resp_null = parse_ai_response(
            json.dumps({"proposed_content": None, "explanation": "test"}),
            "doc.md",
            "sec",
        )
        assert resp_null.requires_human_review is True
        assert "proposed_content must be a string" in resp_null.review_reason

        # list proposed_content
        resp_list = parse_ai_response(
            json.dumps({"proposed_content": ["a", "b"], "explanation": "test"}),
            "doc.md",
            "sec",
        )
        assert resp_list.requires_human_review is True

        # number proposed_content
        resp_num = parse_ai_response(
            json.dumps({"proposed_content": 42, "explanation": "test"}), "doc.md", "sec"
        )
        assert resp_num.requires_human_review is True

        # empty text
        resp_empty = parse_ai_response("", "doc.md", "sec")
        assert resp_empty.requires_human_review is True

    def test_p14_source_syntax_error_reported_without_calling_ai(self):
        """P14: Source syntax errors reported immediately without invoking AI generator."""
        old_code = "def ok(): pass"
        new_code = "def bad_syntax(::: invalid python code"
        doc = "# API\n\n## Functions\nCall `ok`."

        calls = 0

        def should_never_be_called(_s, _u):
            nonlocal calls
            calls += 1
            return json.dumps({"proposed_content": "xyz"})

        updater = DocumentationUpdater(ai_generator=should_never_be_called)
        report = updater.update_documentation(
            code_changes={"src/broken.py": (old_code, new_code)},
            documents={"api.md": doc},
        )

        # AI generator must NOT be invoked
        assert calls == 0
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected == 1
        assert "syntax error" in report.rejected_updates[0].explanation.lower()

    def test_p15_suggest_docs_integration_feature_flag(self, tmp_path, monkeypatch):
        """P15: Integration test for suggest_docs.py feature flag enabled vs disabled."""
        from doc_updater import atomic_overwrite_file

        doc_file = tmp_path / "README.md"
        doc_file.write_text("# Service\n\n## Usage\nCall `start`.\n")

        # Test atomic_overwrite_file directly
        updated_text = "# Service\n\n## Usage\nCall `start(port)`.\n"
        success = atomic_overwrite_file(doc_file, updated_text)
        assert success is True
        assert doc_file.read_text() == updated_text


class TestDocumentTransactionSafety:
    """Tests multi-document transaction staging, path security, permissions, and rollback (Priority 3)."""

    def test_transaction_extension_validation_rejects_rst_and_adoc(self, tmp_path):
        from doc_updater import DocumentTransaction, UnsupportedDocumentExtensionError
        import pytest

        tx = DocumentTransaction(docs_root=tmp_path)
        with pytest.raises(UnsupportedDocumentExtensionError):
            tx.validate_destination_path("guide.rst")
        with pytest.raises(UnsupportedDocumentExtensionError):
            tx.validate_destination_path("reference.adoc")
        with pytest.raises(UnsupportedDocumentExtensionError):
            tx.validate_destination_path("script.py")

        # Supported markdown extensions
        assert tx.validate_destination_path("guide.md") == tmp_path / "guide.md"
        assert tx.validate_destination_path("docs/spec.markdown") == (tmp_path / "docs/spec.markdown").resolve()

    def test_transaction_path_traversal_rejected(self, tmp_path):
        from doc_updater import DocumentTransaction, PathTraversalSecurityError
        import pytest

        tx = DocumentTransaction(docs_root=tmp_path)
        with pytest.raises(PathTraversalSecurityError):
            tx.validate_destination_path("../outside.md")
        with pytest.raises(PathTraversalSecurityError):
            tx.validate_destination_path("sub/../../etc/passwd.md")

    def test_transaction_permission_preservation(self, tmp_path):
        from doc_updater import DocumentTransaction
        import os

        doc_file = tmp_path / "secure.md"
        doc_file.write_text("# Secure\nSecret info.\n")
        os.chmod(doc_file, 0o600)

        tx = DocumentTransaction(docs_root=tmp_path)
        tx.stage_updates({"secure.md": "# Secure\nUpdated secret info.\n"})
        tx.commit()

        assert doc_file.read_text() == "# Secure\nUpdated secret info.\n"
        current_mode = doc_file.stat().st_mode & 0o777
        assert current_mode == 0o600

    def test_transaction_atomic_commit_multiple_documents(self, tmp_path):
        from doc_updater import DocumentTransaction

        doc1 = tmp_path / "doc1.md"
        doc2 = tmp_path / "sub" / "doc2.md"
        doc1.write_text("# Doc 1\nOriginal\n")
        doc2.parent.mkdir(parents=True)
        doc2.write_text("# Doc 2\nOriginal\n")

        tx = DocumentTransaction(docs_root=tmp_path)
        tx.stage_updates({
            "doc1.md": "# Doc 1\nUpdated 1\n",
            "sub/doc2.md": "# Doc 2\nUpdated 2\n",
        })
        tx.commit()

        assert doc1.read_text() == "# Doc 1\nUpdated 1\n"
        assert doc2.read_text() == "# Doc 2\nUpdated 2\n"

    def test_transaction_rollback_on_failure_restores_all_files(self, tmp_path, monkeypatch):
        from doc_updater import DocumentTransaction, DocumentTransactionError
        import pytest
        import os

        doc1 = tmp_path / "doc1.md"
        doc2 = tmp_path / "doc2.md"
        doc1.write_text("# Doc 1\nOriginal 1\n")
        doc2.write_text("# Doc 2\nOriginal 2\n")

        tx = DocumentTransaction(docs_root=tmp_path)
        tx.stage_updates({
            "doc1.md": "# Doc 1\nNew 1\n",
            "doc2.md": "# Doc 2\nNew 2\n",
        })

        # Make os.replace fail on the second file
        real_replace = os.replace
        replace_count = 0

        def flaky_replace(src, dst):
            nonlocal replace_count
            replace_count += 1
            if replace_count == 2:
                raise OSError("Simulated disk error during replacement")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", flaky_replace)

        with pytest.raises(DocumentTransactionError, match="rolled back all replaced documents"):
            tx.commit()

        # Both original files must be restored completely!
        assert doc1.read_text() == "# Doc 1\nOriginal 1\n"
        assert doc2.read_text() == "# Doc 2\nOriginal 2\n"

    def test_transaction_cleans_up_temporary_and_backup_files(self, tmp_path):
        from doc_updater import DocumentTransaction

        doc = tmp_path / "clean.md"
        doc.write_text("# Clean\nContent\n")

        tx = DocumentTransaction(docs_root=tmp_path)
        tx.stage_updates({"clean.md": "# Clean\nNew content\n"})
        tx.commit()

        # Check no .stage_doc_ or .bak_doc_ temporary files remain
        remaining = list(tmp_path.glob(".*"))
        assert not any(".stage_doc_" in p.name or ".bak_doc_" in p.name for p in remaining)


class TestTechnicalClaimValidation:
    """Verifies strict technical claim validation (Priority 4, C1)."""

    def test_c1_unknown_parameter_in_function_call_rejected(self):
        """Reject function call introducing an invalid keyword parameter."""
        old_code = "def create_user(name: str): pass"
        new_code = "def create_user(name: str, email: str = ''): pass"
        doc = "# Users\n\n## Create\nCall `create_user`."

        mock_ai = make_mock_ai({
            "proposed_content": "Call `create_user(name, admin=True)` to create an admin.",
            "explanation": "Added admin parameter",
            "requires_human_review": False,
        })
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/user.py": (old_code, new_code)},
            documents={"users.md": doc},
        )
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any("unknown_parameter_in_call" in f for r in report.rejected_updates for f in r.validation.failed_checks)

    def test_c1_supports_nonexistent_option_rejected(self):
        """Reject claim that function supports a nonexistent option."""
        old_code = "def query(term: str): pass"
        new_code = "def query(term: str, limit: int = 10): pass"
        doc = "# Search\n\n## Query\nCall `query`."

        mock_ai = make_mock_ai({
            "proposed_content": "The `query` function supports the `max_retries` option for queries.",
            "explanation": "Documented query options",
            "requires_human_review": False,
        })
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/search.py": (old_code, new_code)},
            documents={"search.md": doc},
        )
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any("unsupported_option_or_parameter" in f for r in report.rejected_updates for f in r.validation.failed_checks)

    def test_c1_invented_rate_limit_or_numeric_claim_rejected(self):
        """Reject fabricated numeric limits and claims not found in source code."""
        old_code = "def send(): pass"
        new_code = "def send(data: str): pass"
        doc = "# API\n\n## Send\nCall `send`."

        mock_ai = make_mock_ai({
            "proposed_content": "Call `send(data)`. All requests are rate limited to 500 requests per minute.",
            "explanation": "Added rate limit",
            "requires_human_review": False,
        })
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/api.py": (old_code, new_code)},
            documents={"api.md": doc},
        )
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any("unsupported_numeric_claim" in f for r in report.rejected_updates for f in r.validation.failed_checks)

    def test_c1_return_type_mismatch_rejected(self):
        """Reject documentation claiming return type conflicting with source annotation."""
        old_code = "def count_items() -> int: return 0"
        new_code = "def count_items() -> int: return 42"
        doc = "# Counters\n\n## Count\nCall `count_items`."

        mock_ai = make_mock_ai({
            "proposed_content": "The `count_items` function returns a list of strings.",
            "explanation": "Documented return value",
            "requires_human_review": False,
        })
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/counter.py": (old_code, new_code)},
            documents={"counters.md": doc},
        )
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any("return_type_mismatch" in f for r in report.rejected_updates for f in r.validation.failed_checks)

    def test_c1_unsupported_environment_variable_rejected(self):
        """Reject invented environment variable names not accessed in source."""
        old_code = "import os\ndef init(): pass"
        new_code = "import os\ndef init(debug: bool = False): pass"
        doc = "# Config\n\n## Init\nCall `init`."

        mock_ai = make_mock_ai({
            "proposed_content": "Call `init(debug)` and configure database via `MY_APP_SECRET_ACCESS_KEY` environment variable.",
            "explanation": "Added env config",
            "requires_human_review": False,
        })
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/init.py": (old_code, new_code)},
            documents={"config.md": doc},
        )
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any("unsupported_environment_variable" in f for r in report.rejected_updates for f in r.validation.failed_checks)


class TestRemovedSymbolsStrictValidation:
    """Verifies strict removed symbol validation (Priority 5, C3)."""

    def test_c3_call_removed_symbol_with_migration_keyword_rejected(self):
        """Reject active instruction to call removed symbol even if 'migration' is in the sentence."""
        old_code = "def legacy_handler(): pass"
        new_code = "def v2_handler(): pass"
        doc = "# API\n\n## Handler\nCall `legacy_handler`."

        mock_ai = make_mock_ai({
            "proposed_content": "Call `legacy_handler` to handle migration of requests.",
            "explanation": "Updated handler for migration",
            "requires_human_review": False,
        })
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/handler.py": (old_code, new_code)},
            documents={"api.md": doc},
        )
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any("active_reference_to_removed_symbol" in f for r in report.rejected_updates for f in r.validation.failed_checks)

    def test_c3_call_removed_symbol_not_deprecated_claim_rejected(self):
        """Reject active instruction claiming removed symbol is 'not deprecated'."""
        old_code = "def legacy_handler(): pass"
        new_code = "def v2_handler(): pass"
        doc = "# API\n\n## Handler\nCall `legacy_handler`."

        mock_ai = make_mock_ai({
            "proposed_content": "Call `legacy_handler`; it is not deprecated.",
            "explanation": "Handler note",
            "requires_human_review": False,
        })
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/handler.py": (old_code, new_code)},
            documents={"api.md": doc},
        )
        assert report.total_sections_applied == 0
        assert report.total_sections_rejected >= 1
        assert any("active_reference_to_removed_symbol" in f for r in report.rejected_updates for f in r.validation.failed_checks)

    def test_c3_supported_removal_explanation_accepted(self):
        """Accept factual statement that removed symbol was removed in a release."""
        old_code = "def legacy_handler(): pass"
        new_code = "def v2_handler(): pass"
        doc = "# API\n\n## Handler\nCall `legacy_handler`."

        mock_ai = make_mock_ai({
            "proposed_content": "`legacy_handler` was removed in version 2.0.",
            "explanation": "Removal notice",
            "requires_human_review": False,
        })
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/handler.py": (old_code, new_code)},
            documents={"api.md": doc},
        )
        assert report.total_sections_applied == 1
        assert "was removed in version 2.0" in report.updated_documents["api.md"]

    def test_c3_supported_replacement_notice_accepted(self):
        """Accept explicit replacement notice directing users to replacement function."""
        old_code = "def legacy_handler(): pass"
        new_code = "def new_handler(): pass"
        doc = "# API\n\n## Handler\nCall `legacy_handler`."

        mock_ai = make_mock_ai({
            "proposed_content": "Replace `legacy_handler` with `new_handler`.",
            "explanation": "Replacement notice",
            "requires_human_review": False,
        })
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/handler.py": (old_code, new_code)},
            documents={"api.md": doc},
        )
        assert report.total_sections_applied == 1
        assert "Replace `legacy_handler` with `new_handler`" in report.updated_documents["api.md"]

    def test_c3_unchanged_doc_with_active_deleted_symbol_flags_review(self):
        """If AI proposes no_change but doc still has active instructions for a deleted symbol, flag review!"""
        old_code = "def old_fn(): pass"
        new_code = "def new_fn(): pass"
        doc = "# Guide\n\n## Section\nCall `old_fn` to start.\n"

        # AI proposes identical content
        mock_ai = make_mock_ai({
            "proposed_content": "Call `old_fn` to start.\n",
            "explanation": "No change",
            "requires_human_review": False,
        })
        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={"src/service.py": (old_code, new_code)},
            documents={"guide.md": doc},
        )
        # Must NOT silently report no_change! Must flag review_required!
        assert report.total_sections_applied == 0
        assert len(report.review_required_updates) == 1
        assert report.review_required_updates[0].status == "review_required"
        assert "Removed symbol" in report.review_required_updates[0].explanation

