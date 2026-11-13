"""
Tests for isolating unrelated source analysis failures (Priority 5, M3).
Verifies that an unrelated Python syntax error in File A does not block
valid documentation updates for other successfully analyzed source files (File B).
"""

import json

from doc_updater import DocumentationUpdater


class TestIsolatedSourceAnalysis:
    def test_syntax_error_in_unrelated_file_does_not_block_valid_updates(self):
        # File A has a fatal syntax error
        broken_old = "def broken(): pass"
        broken_new = "def broken(::: broken syntax error !@#$"

        # File B has a valid change
        valid_old = "def calculate_total(subtotal: float): pass"
        valid_new = "def calculate_total(subtotal: float, tax_rate: float = 0.05): pass"

        doc_broken = "# Broken Module\n\n## Overview\nOverview of broken."
        doc_valid = "# Billing\n\n## Calculate\nCall `calculate_total(subtotal)` to get total."

        def mock_ai(_s, user_prompt):
            if "tax_rate" in user_prompt:
                return json.dumps(
                    {
                        "proposed_content": "Call `calculate_total(subtotal, tax_rate)` with optional tax_rate (default 0.05).",
                        "explanation": "Added tax_rate parameter documentation.",
                        "requires_human_review": False,
                    }
                )
            return json.dumps({"proposed_content": "Broken"})

        updater = DocumentationUpdater(ai_generator=mock_ai)
        report = updater.update_documentation(
            code_changes={
                "src/broken.py": (broken_old, broken_new),
                "src/billing.py": (valid_old, valid_new),
            },
            documents={
                "broken.md": doc_broken,
                "billing.md": doc_valid,
            },
        )

        # 1. Source analysis error for broken.py must be recorded
        assert len(report.source_analysis_errors) >= 1
        assert any("src/broken.py" in src for src, _err in report.source_analysis_errors)

        # 2. Valid documentation for billing.md MUST be applied successfully!
        assert report.total_sections_applied == 1
        assert "tax_rate" in report.updated_documents["billing.md"]

        # 3. Broken doc must remain unmodified
        assert report.updated_documents["broken.md"] == doc_broken
