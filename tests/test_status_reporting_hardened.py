"""
Hardened tests for Status Reporting and PR Comments (Priority 5).

Covers:
- M1: Start applied, Stop review-required -> both sections appear in final comment.
- M4: One section returns HTTP 500 while another succeeds -> generation failure is not swallowed and appears in PR comment.
- X1: Transaction failure in normal update mode posts a visible PR comment with transaction failure details.
"""

from comments import format_section_update_review_comment
from doc_updater import (
    AppliedSectionUpdate,
    DocumentationUpdateReport,
    DocumentationUpdater,
    ValidationResult,
)


class TestStatusReportingHardened:
    def test_m1_start_applied_and_stop_review_required_both_appear_in_comment(self):
        """
        M1: One section (Start) is applied successfully, while another section (Stop)
        requires human review. Both must be clearly visible in the final PR comment.
        """
        applied_sec = AppliedSectionUpdate(
            doc_path="server.md",
            section_id="server/start",
            section_heading="Start Server",
            status="applied",
            original_section_content="Start server on port 8080.",
            new_section_content="Start server on port 9000.",
            explanation="Updated port to 9000.",
            validation=ValidationResult(is_valid=True),
        )
        review_sec = AppliedSectionUpdate(
            doc_path="server.md",
            section_id="server/stop",
            section_heading="Stop Server",
            status="review_required",
            original_section_content="Stop server gracefully.",
            new_section_content="Stop server with graceful timeout.",
            explanation="Requires human review for graceful shutdown timeout.",
            validation=ValidationResult(
                is_valid=False,
                requires_human_review=True,
                review_reasons=["Timeout value for graceful shutdown cannot be verified from AST."],
            ),
        )

        report = DocumentationUpdateReport(
            updated_documents={"server.md": "# Server\n\n## Start Server\n..."},
            applied_updates=[applied_sec],
            committed_updates=[applied_sec],
            review_required_updates=[review_sec],
            total_sections_analyzed=2,
            total_sections_applied=1,
            total_sections_rejected=0,
        )

        comment = format_section_update_review_comment(report, review_mode=False)

        # Both sections must be present in comment!
        assert "### 📄 Documentation Updated (1 section(s))" in comment
        assert "Start Server" in comment
        assert "### ⚠️ Human Review Required (1 item(s))" in comment
        assert "Stop Server" in comment
        assert "graceful shutdown" in comment

    def test_m4_http_500_generation_failure_is_not_swallowed(self):
        """
        M4: One section has an HTTP 500 generation failure from AI,
        while another section succeeds.
        The failure must NOT be swallowed or converted to missing.
        It must appear in the final PR comment under Generation Failures.
        """
        old_code = "def a(): pass\ndef b(): pass"
        new_code = "def a(x: int = 1): pass\ndef b(y: int = 2): pass"

        doc = "# API\n\n## Section A\nCall `a()`.\n\n## Section B\nCall `b()`.\n"

        call_count = 0

        def failing_ai(sys_p, user_p):
            nonlocal call_count
            call_count += 1
            if "Section A" in user_p:
                # Simulates HTTP 500 from LLM API
                raise RuntimeError("HTTP 500: Internal Server Error from AI Model Gateway")
            import json

            return json.dumps(
                {
                    "proposed_content": "Call `b(y)` with optional y (default 2).",
                    "explanation": "Updated b documentation.",
                    "requires_human_review": False,
                }
            )

        updater = DocumentationUpdater(ai_generator=failing_ai)
        report = updater.update_documentation(
            code_changes={"src/api.py": (old_code, new_code)},
            documents={"api.md": doc},
        )

        # Generation failure must be recorded in generation_failures and rejected_updates:
        assert len(report.generation_failures) == 1
        assert "HTTP 500" in report.generation_failures[0].explanation
        assert report.generation_failures[0].status == "generation_failed"

        comment = format_section_update_review_comment(report, review_mode=False)
        assert "### 💥 Generation Failures (1 item(s))" in comment
        assert "HTTP 500" in comment

    def test_x1_transaction_failure_in_update_mode_posts_comment(self):
        """
        X1: Transaction failure in update mode posts a visible PR comment
        clearly reporting the transaction failure.
        """
        tx_failed_rec = AppliedSectionUpdate(
            doc_path="guide.md",
            section_id="guide/install",
            section_heading="Installation",
            status="transaction_failed",
            original_section_content="Install instructions.",
            new_section_content="Updated install instructions.",
            explanation="Disk permission error: unable to commit transaction for guide.md",
            validation=ValidationResult(is_valid=True),
        )

        report = DocumentationUpdateReport(
            updated_documents={"guide.md": "Install instructions."},
            applied_updates=[],
            committed_updates=[],
            transaction_failures=[tx_failed_rec],
            total_sections_analyzed=1,
            total_sections_applied=0,
            total_sections_rejected=0,
        )

        comment = format_section_update_review_comment(report, review_mode=False)
        assert "### 🚨 Transaction Failures (1 item(s))" in comment
        assert "guide.md" in comment
        assert "Disk permission error" in comment
        # Must NOT claim any documentation was updated
        assert "### 📄 Documentation Updated" not in comment
