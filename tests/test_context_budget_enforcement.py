"""
Tests for context budget enforcement and symbol extraction (Priority 6).
"""

import pytest

from doc_updater import (
    ContextBudgetExceededError,
    SectionUpdateRequest,
    build_update_prompt,
    extract_relevant_code_snippet,
)


class TestContextBudgetEnforcement:
    def test_extract_relevant_code_snippet_locates_symbol_at_end_of_file(self):
        # 1000 lines of dummy code, with target function at line 950
        lines = [
            "import os",
            "import sys",
            "CONFIG_A = 1",
            "CONFIG_B = 2",
        ]
        for i in range(940):
            lines.append(f"def dummy_{i}(): pass")

        lines.extend(
            [
                "def target_function(new_param: str = 'default'):",
                "    '''Documentation for target function.'''",
                "    return new_param",
            ]
        )
        for i in range(50):
            lines.append(f"def tail_{i}(): pass")

        full_code = "\n".join(lines)
        snippet = extract_relevant_code_snippet(
            full_code,
            changed_symbols=["target_function"],
            max_chars=2000,
        )

        assert len(snippet) <= 2500
        # Must retain top imports/constants
        assert "import os" in snippet
        assert "CONFIG_A = 1" in snippet
        # Must contain the target function even though it was near the end of the file!
        assert "target_function" in snippet
        assert "new_param" in snippet

    def test_target_section_exceeding_budget_flags_review_required(self):
        old_code = "def query(): pass"
        new_code = "def query(filter: str = ''): pass"

        # Create massive target section content
        huge_doc_section = "Detailed documentation paragraph. " * 5000

        # Run with small max_context_chars via environment or custom
        req = SectionUpdateRequest(
            doc_path="big.md",
            section_id="big-doc/section-1",
            section_heading="Section 1",
            current_section_content=huge_doc_section,
            source_files=["src/query.py"],
            changed_symbols=["query"],
            change_categories=["api"],
            impact_reasons=["Signature change"],
            impact_evidence=[],
            old_code_snippets={"src/query.py": old_code},
            new_code_snippets={"src/query.py": new_code},
            ast_deltas={},
            style_guidelines="",
        )

        with pytest.raises(ContextBudgetExceededError):
            build_update_prompt(req, max_context_chars=10000)

    def test_oversized_style_guide_is_trimmed(self):
        old_code = "def process(): pass"
        new_code = "def process(val: int): pass"

        huge_style = "Rule 1: Always be concise.\n" * 1000

        req = SectionUpdateRequest(
            doc_path="guide.md",
            section_id="guide/process",
            section_heading="Process",
            current_section_content="Call `process`.",
            source_files=["src/proc.py"],
            changed_symbols=["process"],
            change_categories=["api"],
            impact_reasons=["Param added"],
            impact_evidence=[],
            old_code_snippets={"src/proc.py": old_code},
            new_code_snippets={"src/proc.py": new_code},
            ast_deltas={},
            style_guidelines=huge_style,
        )

        prompt = build_update_prompt(req, max_context_chars=20000)
        assert "[TRUNCATED STYLE GUIDELINES]" in prompt
        # Target section and essential symbols are preserved
        assert "Call `process`." in prompt
        assert "process" in prompt
