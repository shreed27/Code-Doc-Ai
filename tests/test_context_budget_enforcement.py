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

    def test_serialized_http_body_budget_enforcement_60k(self):
        """
        Priority 7: 60,000 configured budget with potentially 61,668-byte HTTP body.
        Verifies that calculate_serialized_request_size accurately computes JSON escaping
        and envelope overhead, and that build_update_prompt never allows the outbound
        serialized JSON HTTP request to exceed the 60,000 budget.
        """
        from doc_updater import _SYSTEM_PROMPT, calculate_serialized_request_size

        # Construct code snippets that, when combined with target doc and JSON escaping,
        # would produce ~61,668 bytes if unconstrained.
        doc_content = "# Section\n" + (
            'Paragraph with quotes "and" newlines \\ and symbols.\n' * 300
        )
        old_code = "def worker():\n" + (
            "    x = 'some string with \"quotes\" and \\escapes\\'\n" * 400
        )
        new_code = "def worker(flag: bool = True):\n" + (
            "    x = 'some string with \"quotes\" and \\escapes\\'\n" * 400
        )

        req = SectionUpdateRequest(
            doc_path="doc.md",
            section_id="doc/worker",
            section_heading="Worker",
            current_section_content=doc_content,
            source_files=["src/worker.py"],
            changed_symbols=["worker"],
            change_categories=["api"],
            impact_reasons=["Signature change"],
            impact_evidence=[],
            old_code_snippets={"src/worker.py": old_code},
            new_code_snippets={"src/worker.py": new_code},
            ast_deltas={},
            style_guidelines="Be concise.",
        )

        budget = 60000
        prompt = build_update_prompt(req, max_context_chars=budget)
        serialized_size = calculate_serialized_request_size("gpt-4o", _SYSTEM_PROMPT, prompt)

        # The serialized HTTP payload MUST NOT exceed 60,000 bytes!
        assert serialized_size <= budget

    def test_oversized_diff_12k_budget_produces_visible_pr_comment_and_exits_1(self, monkeypatch):
        """
        Priority 7: 12,000 budget with oversized diff (>90%).
        Must NOT silently exit 0.
        Must post a visible PR comment reporting review_required and exit with code 1.
        """
        import sys

        import suggest_docs

        monkeypatch.setenv("MAX_CONTEXT_CHARS", "12000")
        monkeypatch.setenv("PR_NUMBER", "42")
        monkeypatch.setattr(suggest_docs, "get_diff", lambda *args, **kwargs: "+" * 11000)
        monkeypatch.setattr(
            suggest_docs,
            "get_commit_info",
            lambda *args, **kwargs: {
                "short_hash": "1234567",
                "repo_url": "https://github.com/org/repo",
                "pr_url": "https://github.com/org/repo/pull/42",
            },
        )

        posted_comments = []

        def mock_post_review_comment(files, pr_num, commit_info, **kwargs):
            sec_report = kwargs.get("section_report")
            posted_comments.append((pr_num, sec_report))

        monkeypatch.setattr(suggest_docs, "post_review_comment", mock_post_review_comment)

        monkeypatch.setenv("COMMENT_BODY", "[review-docs]")
        test_args = ["suggest_docs.py"]
        monkeypatch.setattr(sys, "argv", test_args)

        with pytest.raises(SystemExit) as exc_info:
            suggest_docs.main()

        # Must exit with non-zero code 1
        assert exc_info.value.code == 1

        # Must have posted a comment with review_required status
        assert len(posted_comments) == 1
        pr_num, sec_report = posted_comments[0]
        assert pr_num == "42"
        assert sec_report is not None
        assert len(sec_report.review_required_updates) == 1
        assert "context budget" in sec_report.review_required_updates[0].explanation.lower()
