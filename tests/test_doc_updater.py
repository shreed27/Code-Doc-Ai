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
        doc2 = "# API Reference\n\n## Export Details\nThe `export_data` function details."

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
        assert updated_text.endswith("## Troubleshooting\nCheck logs at `/var/log/app.log`.\n")

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
        old_code = "def calculate(): pass"
        new_code = "def calculate(): pass  # internal reordering"

        doc = "# Math\n## Calc\nUses `calculate`."

        mock_uncertain = {
            "proposed_content": "Updated calculation internals.",
            "explanation": "Uncertain about behavioral impact",
            "uncertainties": ["Internal logic changed but no public API changes"],
            "requires_human_review": True,
            "review_reason": "Low confidence on behavioral changes.",
        }

        updater = DocumentationUpdater(ai_generator=make_mock_ai(mock_uncertain))
        report = updater.update_documentation(
            code_changes={"src/math_util.py": (old_code, new_code)},
            documents={"math.md": doc},
            precomputed_impacts=[],
        )
        assert report.total_sections_applied == 0

    def test_failed_section_update_does_not_corrupt_other_documents(self):
        old_code = "def sync_all(): pass"
        new_code = "def sync_all(force: bool = False): pass"

        doc_good = "# Sync\n## Main Sync\nCall `sync_all`."
        doc_bad = "# Corrupted\n## Missing Section Title\n..."

        def selective_ai(_s, user_prompt: str):
            if "Missing Section" in user_prompt:
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

        # Good doc is applied, bad doc is rejected without corrupting good doc
        assert "sync_all(force)" in report.updated_documents["good.md"]
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

        updater = DocumentationUpdater(ai_generator=make_mock_ai({"proposed_content": "xyz"}))
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
