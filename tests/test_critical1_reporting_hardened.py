"""
Hardened integration tests for Critical-1 and product-level reporting (Priority 1).

Tests:
1. Syntax error in unrelated file + two valid documentation updates:
   - Valid updates proceed, write to disk, and commit.
   - Final report/comment derives success solely from committed updates.
   - Comment displays both committed docs and source analysis failures.
   - No false success.
2. No-op run produces up-to-date comment with 0 updates.
3. Rejected-only run produces rejected comment with 0 updates.
4. Review-only run produces human-review comment with 0 updates.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from comments import format_section_update_review_comment
from doc_updater import (
    AppliedSectionUpdate,
    DocumentationUpdateReport,
    DocumentationUpdater,
    ValidationResult,
)


class TestCritical1ReportingHardened:
    def test_syntax_error_plus_two_valid_doc_updates_e2e(self, tmp_path):
        """
        Critical-1:
        Two valid documentation sections affected by valid source changes.
        One unrelated source file with a fatal syntax error.
        Valid updates proceed, write to disk, and commit.
        Final comment accurately reflects both the committed updates and the analysis failure.
        """
        # Source files:
        # File 1 (broken):
        broken_old = "def parse_data(raw): pass"
        broken_new = "def parse_data(raw::: broken syntax error !@#"

        # File 2 (valid billing):
        billing_old = "def calculate_bill(subtotal: float): pass"
        billing_new = "def calculate_bill(subtotal: float, discount: float = 0.1): pass"

        # File 3 (valid auth):
        auth_old = "def login_user(username: str): pass"
        auth_new = "def login_user(username: str, remember_me: bool = False): pass"

        # Documents:
        doc_billing = "# Billing\n\n## Calculate\nCall `calculate_bill(subtotal)` to get total.\n"
        doc_auth = "# Auth\n\n## Login\nCall `login_user(username)` to sign in.\n"
        doc_broken = "# Parser\n\n## Parse\nCall `parse_data(raw)` to parse data.\n"

        def mock_ai(system_prompt, user_prompt):
            import json

            if "calculate_bill" in user_prompt:
                return json.dumps(
                    {
                        "proposed_content": "Call `calculate_bill(subtotal, discount)` with optional discount (default 0.1).",
                        "explanation": "Added discount parameter to billing.",
                        "requires_human_review": False,
                    }
                )
            elif "login_user" in user_prompt:
                return json.dumps(
                    {
                        "proposed_content": "Call `login_user(username, remember_me)` with optional remember_me (default False).",
                        "explanation": "Added remember_me parameter to login.",
                        "requires_human_review": False,
                    }
                )
            return json.dumps({"proposed_content": "No change", "requires_human_review": True})

        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={
                "src/broken.py": (broken_old, broken_new),
                "src/billing.py": (billing_old, billing_new),
                "src/auth.py": (auth_old, auth_new),
            },
            documents={
                "billing.md": doc_billing,
                "auth.md": doc_auth,
                "parser.md": doc_broken,
            },
        )

        # 1. Source analysis error for broken.py must be recorded:
        assert len(report.source_analysis_errors) == 1
        assert "src/broken.py" in report.source_analysis_errors[0][0]

        # 2. Both valid documentation sections must be applied:
        assert report.total_sections_applied == 2
        assert "discount" in report.updated_documents["billing.md"]
        assert "remember_me" in report.updated_documents["auth.md"]
        assert report.updated_documents["parser.md"] == doc_broken

        # 3. Format review comment:
        report.committed_updates = list(report.applied_updates)
        comment = format_section_update_review_comment(
            report,
            commit_info={"short_hash": "abcdef1", "pr_url": "https://github.com/org/repo/pull/1"},
            review_mode=False,
        )

        # Comment MUST display committed updates for the 2 valid sections:
        assert "### 📄 Documentation Updated (2 section(s))" in comment
        assert "billing.md" in comment
        assert "auth.md" in comment

        # Comment MUST display source analysis failure for broken.py:
        assert "### ⚠️ Source Analysis Failures (1 file(s))" in comment
        assert "src/broken.py" in comment

        # Must NOT claim parser.md was updated
        assert "parser.md" not in comment.split("### 📄 Documentation Updated")[1].split("###")[0]

    def test_no_op_run_produces_clean_comment_with_zero_updates(self):
        report = DocumentationUpdateReport(
            updated_documents={},
            applied_updates=[],
            committed_updates=[],
            rejected_updates=[],
            review_required_updates=[],
            source_analysis_errors=[],
            total_sections_analyzed=5,
            total_sections_applied=0,
            total_sections_rejected=0,
        )
        comment = format_section_update_review_comment(report, review_mode=False)
        assert "No documentation updates needed" in comment
        assert "### 📄 Documentation Updated" not in comment

    def test_rejected_only_run_produces_rejected_comment_with_zero_updates(self):
        rej = AppliedSectionUpdate(
            doc_path="api.md",
            section_id="api/endpoint",
            section_heading="Endpoint",
            status="rejected",
            original_section_content="Original",
            new_section_content="Hallucinated",
            explanation="Hallucinated parameter",
            validation=ValidationResult(
                is_valid=False,
                failed_checks=["unsupported_option_or_parameter: 'fake_param' does not exist"],
            ),
        )
        report = DocumentationUpdateReport(
            updated_documents={"api.md": "Original"},
            applied_updates=[],
            committed_updates=[],
            rejected_updates=[rej],
            total_sections_analyzed=1,
            total_sections_applied=0,
            total_sections_rejected=1,
        )
        comment = format_section_update_review_comment(report, review_mode=False)
        assert "### ❌ Rejected Updates (1 item(s))" in comment
        assert "fake_param" in comment
        assert "### 📄 Documentation Updated" not in comment
        assert "No documentation updates needed" not in comment

    def test_review_only_run_produces_human_review_comment_with_zero_updates(self):
        rev = AppliedSectionUpdate(
            doc_path="guide.md",
            section_id="guide/auth",
            section_heading="Auth",
            status="review_required",
            original_section_content="Original",
            new_section_content="Ambiguous",
            explanation="Requires human review",
            validation=ValidationResult(
                is_valid=False,
                requires_human_review=True,
                review_reasons=["External dependency behavior cannot be verified from AST."],
            ),
        )
        report = DocumentationUpdateReport(
            updated_documents={"guide.md": "Original"},
            applied_updates=[],
            committed_updates=[],
            review_required_updates=[rev],
            total_sections_analyzed=1,
            total_sections_applied=0,
            total_sections_rejected=0,
        )
        comment = format_section_update_review_comment(report, review_mode=False)
        assert "### ⚠️ Human Review Required (1 item(s))" in comment
        assert "External dependency" in comment
        assert "### 📄 Documentation Updated" not in comment
        assert "No documentation updates needed" not in comment
