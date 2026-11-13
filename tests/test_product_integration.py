"""
Product Integration and Final Safety Hardening Tests (Phase 3.4.2).

Real integration scenarios invoking suggest_docs.main() against temporary Git
repositories, testing:
1.  test_subfolder_docs_updated_and_committed
2.  test_nested_subfolder_docs_updated
3.  test_separate_docs_repo_updated
4.  test_fabricated_ai_output_rejected_and_reported
5.  test_syntax_error_reported_without_ai_call
6.  test_rst_and_adoc_not_corrupted
7.  test_renamed_python_file_handled
8.  test_multi_doc_transaction_all_or_nothing_success
9.  test_multi_doc_transaction_rollback_on_failure
10. test_deleted_symbol_remains_in_doc_requires_review
11. test_feature_flag_disabled_executes_legacy
12. test_generator_failure_midway_visible_report_and_no_partial_writes
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure openai is stubbed
sys.modules.setdefault("openai", MagicMock())

from security_utils import run_command_safe
import suggest_docs


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """
    Creates a real Git repository with initial commit on 'main',
    configures user, sets remote origin, and switches CWD to tmp_path.
    """
    repo = tmp_path.resolve()
    run_command_safe(["git", "init", "-b", "main"], cwd=str(repo), check=True)
    run_command_safe(["git", "config", "user.name", "Test User"], cwd=str(repo), check=True)
    run_command_safe(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    run_command_safe(
        ["git", "config", "remote.origin.url", "https://github.com/shreed27/Code-Doc-Ai.git"],
        cwd=str(repo),
        check=True,
    )

    # Initial commit
    init_file = repo / "README.md"
    init_file.write_text("# Project\nInitial readme.\n", encoding="utf-8")
    run_command_safe(["git", "add", "README.md"], cwd=str(repo), check=True)
    run_command_safe(["git", "commit", "-m", "Initial commit"], cwd=str(repo), check=True)

    monkeypatch.chdir(repo)
    monkeypatch.setenv("PR_BASE", "main")
    monkeypatch.setenv("PR_NUMBER", "42")
    monkeypatch.setenv("GH_TOKEN", "dummy_token_ghp_123456789012345678901234567890123456")
    monkeypatch.setattr("sys.argv", ["suggest_docs.py"])

    return repo


class TestProductIntegrationScenarios:
    def test_subfolder_docs_updated_and_committed(self, git_repo, monkeypatch):
        """Scenario 1: same-repo docs/ subfolder updated and committed via Git."""
        repo = git_repo
        docs_dir = repo / "docs"
        docs_dir.mkdir()
        src_dir = repo / "src"
        src_dir.mkdir()

        py_file = src_dir / "calculator.py"
        py_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
        doc_file = docs_dir / "api.md"
        doc_file.write_text("# API Reference\n\n## add\nAdds two numbers together.\n", encoding="utf-8")

        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Add calculator and doc"], cwd=str(repo), check=True)

        # Create branch and make code change
        run_command_safe(["git", "checkout", "-b", "feat/round-calc"], cwd=str(repo), check=True)
        py_file.write_text(
            "def add(a: int, b: int, round_result: bool = False) -> int:\n    res = a + b\n    return round(res) if round_result else res\n",
            encoding="utf-8",
        )
        run_command_safe(["git", "add", "src/calculator.py"], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Add round_result parameter"], cwd=str(repo), check=True)

        monkeypatch.setenv("DOCS_SUBFOLDER", "docs")
        monkeypatch.setenv("COMMENT_BODY", "[update-docs]")
        monkeypatch.setenv("ENABLE_SECTION_UPDATER", "true")

        candidate = {
            "proposed_content": "## add\nAdds two numbers together with optional round_result.",
            "explanation": "Documented round_result parameter",
            "requires_human_review": False,
        }

        with patch("suggest_docs.find_relevant_files_optimized", return_value=["api.md"]), \
             patch("suggest_docs._resolve_pr_push_target", return_value=("feat/round-calc", "https://github.com/shreed27/Code-Doc-Ai.git", False)), \
             patch("suggest_docs.run_command_safe") as mock_run:

            # Allow git commands except git push which we verify
            def side_effect(cmd, *args, **kwargs):
                if cmd[:2] == ["git", "push"]:
                    return MagicMock(returncode=0, stdout="", stderr="")
                return run_command_safe(cmd, *args, **kwargs)

            mock_run.side_effect = side_effect

            with patch("doc_updater.DocumentationUpdater._default_ai_generator", return_value=json.dumps(candidate)):
                suggest_docs.main()

        # Doc file on disk must be updated
        updated_content = (repo / "docs" / "api.md").read_text(encoding="utf-8")
        assert "round_result" in updated_content

        # Git commit was created
        log_res = subprocess.run(["git", "log", "-n", "1", "--oneline"], cwd=str(repo), capture_output=True, text=True)
        assert "docs: update documentation" in log_res.stdout

    def test_nested_subfolder_docs_updated(self, git_repo, monkeypatch):
        """Scenario 2: nested subfolder content/docs/ updated correctly."""
        repo = git_repo
        nested_docs = repo / "content" / "docs"
        nested_docs.mkdir(parents=True)
        src_dir = repo / "service"
        src_dir.mkdir()

        py_file = src_dir / "auth.py"
        py_file.write_text("def login(user: str) -> bool:\n    return True\n", encoding="utf-8")
        doc_file = nested_docs / "auth.md"
        doc_file.write_text("# Authentication\n\n## login\nAuthenticates user.\n", encoding="utf-8")

        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Add auth and doc"], cwd=str(repo), check=True)

        run_command_safe(["git", "checkout", "-b", "feat/auth-timeout"], cwd=str(repo), check=True)
        py_file.write_text("def login(user: str, timeout: int = 30) -> bool:\n    return True\n", encoding="utf-8")
        run_command_safe(["git", "add", "service/auth.py"], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Add timeout"], cwd=str(repo), check=True)

        monkeypatch.setenv("DOCS_SUBFOLDER", "content/docs")
        monkeypatch.setenv("COMMENT_BODY", "[update-docs]")
        monkeypatch.setenv("ENABLE_SECTION_UPDATER", "true")

        candidate = {
            "proposed_content": "## login\nAuthenticates user with configurable timeout.",
            "explanation": "Added timeout parameter",
            "requires_human_review": False,
        }

        with patch("suggest_docs.find_relevant_files_optimized", return_value=["auth.md"]), \
             patch("suggest_docs._resolve_pr_push_target", return_value=("feat/auth-timeout", "https://github.com/shreed27/Code-Doc-Ai.git", False)), \
             patch("suggest_docs.run_command_safe") as mock_run:

            def side_effect(cmd, *args, **kwargs):
                if cmd[:2] == ["git", "push"]:
                    return MagicMock(returncode=0, stdout="", stderr="")
                return run_command_safe(cmd, *args, **kwargs)

            mock_run.side_effect = side_effect

            with patch("doc_updater.DocumentationUpdater._default_ai_generator", return_value=json.dumps(candidate)):
                suggest_docs.main()

        assert "timeout" in (nested_docs / "auth.md").read_text(encoding="utf-8")

    def test_separate_docs_repo_updated(self, git_repo, monkeypatch):
        """Scenario 3: separate documentation repository mode."""
        repo = git_repo
        src_dir = repo / "src"
        src_dir.mkdir()
        py_file = src_dir / "api.py"
        py_file.write_text("def serve(port: int = 8080): pass\n", encoding="utf-8")

        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Add api"], cwd=str(repo), check=True)

        run_command_safe(["git", "checkout", "-b", "feat/api-port"], cwd=str(repo), check=True)
        py_file.write_text("def serve(port: int = 9000): pass\n", encoding="utf-8")
        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Change default port"], cwd=str(repo), check=True)

        # Create separate docs_repo folder inside repo
        docs_repo = repo / "docs_repo"
        docs_repo.mkdir()
        (docs_repo / "server.md").write_text("# Server\n\n## serve\nListens on port 8080.\n", encoding="utf-8")

        # Unset DOCS_SUBFOLDER to trigger separate-repo mode
        monkeypatch.delenv("DOCS_SUBFOLDER", raising=False)
        monkeypatch.setenv("COMMENT_BODY", "[update-docs]")
        monkeypatch.setenv("ENABLE_SECTION_UPDATER", "true")

        candidate = {
            "proposed_content": "## serve\nListens on port 9000 by default.",
            "explanation": "Updated default port",
            "requires_human_review": False,
        }

        with patch("suggest_docs.setup_docs_environment") as mock_setup, \
             patch("suggest_docs.find_relevant_files_optimized", return_value=["server.md"]), \
             patch("suggest_docs.push_and_open_pr") as mock_push_pr:

            # setup_docs_environment chdirs to docs_repo and returns True
            def setup_side_effect():
                os.chdir(str(docs_repo))
                return True

            mock_setup.side_effect = setup_side_effect

            with patch("doc_updater.DocumentationUpdater._default_ai_generator", return_value=json.dumps(candidate)):
                suggest_docs.main()

            mock_push_pr.assert_called_once()
            args, _ = mock_push_pr.call_args
            assert "server.md" in args[0]
            assert "9000" in (docs_repo / "server.md").read_text(encoding="utf-8")

    def test_fabricated_ai_output_rejected_and_reported(self, git_repo, monkeypatch):
        """Scenario 4: fabricated parameter rejected, reported in comment, no docs modified."""
        repo = git_repo
        docs_dir = repo / "docs"
        docs_dir.mkdir()
        src_dir = repo / "src"
        src_dir.mkdir()

        py_file = src_dir / "user.py"
        py_file.write_text("def create_user(name: str): pass\n", encoding="utf-8")
        doc_file = docs_dir / "user.md"
        original_doc = "# Users\n\n## create_user\nCreates a user.\n"
        doc_file.write_text(original_doc, encoding="utf-8")

        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Initial"], cwd=str(repo), check=True)

        run_command_safe(["git", "checkout", "-b", "feat/user-email"], cwd=str(repo), check=True)
        py_file.write_text("def create_user(name: str, email: str = ''): pass\n", encoding="utf-8")
        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Add email"], cwd=str(repo), check=True)

        monkeypatch.setenv("DOCS_SUBFOLDER", "docs")
        monkeypatch.setenv("COMMENT_BODY", "[review-docs]")
        monkeypatch.setenv("ENABLE_SECTION_UPDATER", "true")

        # Hallucinate admin=True which doesn't exist in source
        hallucinated_candidate = {
            "proposed_content": "## create_user\nCall create_user(name, admin=True) to grant admin rights.",
            "explanation": "Added admin parameter",
            "requires_human_review": False,
        }

        with patch("suggest_docs.find_relevant_files_optimized", return_value=["user.md"]), \
             patch("suggest_docs.post_review_comment") as mock_comment:

            with patch("doc_updater.DocumentationUpdater._default_ai_generator", return_value=json.dumps(hallucinated_candidate)):
                suggest_docs.main()

            mock_comment.assert_called_once()
            _, kwargs = mock_comment.call_args
            report = kwargs.get("section_report")
            assert report is not None
            assert report.total_sections_rejected > 0
            assert any("admin" in str(f) for f in report.rejected_updates[0].validation.failed_checks)

        # File on disk remains unchanged
        assert doc_file.read_text(encoding="utf-8") == original_doc

    def test_syntax_error_reported_without_ai_call(self, git_repo, monkeypatch):
        """Scenario 5: syntax error in source file halts generation with 0 AI calls and visible report."""
        repo = git_repo
        docs_dir = repo / "docs"
        docs_dir.mkdir()
        src_dir = repo / "src"
        src_dir.mkdir()

        py_file = src_dir / "service.py"
        py_file.write_text("def run(): pass\n", encoding="utf-8")
        doc_file = docs_dir / "service.md"
        doc_file.write_text("# Service\n\n## run\nRuns the service.\n", encoding="utf-8")

        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Initial"], cwd=str(repo), check=True)

        run_command_safe(["git", "checkout", "-b", "feat/broken-syntax"], cwd=str(repo), check=True)
        py_file.write_text("def run(broken syntax here:\n", encoding="utf-8")
        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Broken code"], cwd=str(repo), check=True)

        monkeypatch.setenv("DOCS_SUBFOLDER", "docs")
        monkeypatch.setenv("COMMENT_BODY", "[review-docs]")
        monkeypatch.setenv("ENABLE_SECTION_UPDATER", "true")

        ai_mock = MagicMock()

        with patch("suggest_docs.find_relevant_files_optimized", return_value=["service.md"]), \
             patch("suggest_docs.post_review_comment") as mock_comment, \
             patch("doc_updater.DocumentationUpdater._default_ai_generator", ai_mock):

            suggest_docs.main()

            assert ai_mock.call_count == 0
            mock_comment.assert_called_once()
            _, kwargs = mock_comment.call_args
            report = kwargs.get("section_report")
            assert report is not None
            assert report.total_sections_rejected > 0
            assert "syntax" in report.rejected_updates[0].explanation.lower()

    def test_rst_and_adoc_not_corrupted(self, git_repo, monkeypatch):
        """Scenario 6: .rst and .adoc files are bypassed by section updater and routed safely."""
        repo = git_repo
        docs_dir = repo / "docs"
        docs_dir.mkdir()
        src_dir = repo / "src"
        src_dir.mkdir()

        py_file = src_dir / "worker.py"
        py_file.write_text("def start(): pass\n", encoding="utf-8")
        doc_rst = docs_dir / "guide.rst"
        doc_rst.write_text("Guide\n=====\n\nWorker guide in RST.\n", encoding="utf-8")
        doc_adoc = docs_dir / "spec.adoc"
        doc_adoc.write_text("= Worker Spec\n\nAdoc content.\n", encoding="utf-8")

        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Initial"], cwd=str(repo), check=True)

        run_command_safe(["git", "checkout", "-b", "feat/worker-threads"], cwd=str(repo), check=True)
        py_file.write_text("def start(threads: int = 4): pass\n", encoding="utf-8")
        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Add threads"], cwd=str(repo), check=True)

        monkeypatch.setenv("DOCS_SUBFOLDER", "docs")
        monkeypatch.setenv("COMMENT_BODY", "[update-docs]")
        monkeypatch.setenv("ENABLE_SECTION_UPDATER", "true")

        with patch("suggest_docs.find_relevant_files_optimized", return_value=["guide.rst", "spec.adoc"]), \
             patch("suggest_docs.ask_ai_for_updated_content", return_value="NO_UPDATE_NEEDED") as mock_legacy_ai, \
             patch("suggest_docs._resolve_pr_push_target", return_value=("feat/worker-threads", "https://github.com/shreed27/Code-Doc-Ai.git", False)):

            suggest_docs.main()

            # Non-markdown files were handed over to legacy AI updater without patcher corruption
            assert mock_legacy_ai.call_count == 2
            # Contents preserved exactly
            assert "Worker guide in RST" in doc_rst.read_text(encoding="utf-8")
            assert "Adoc content" in doc_adoc.read_text(encoding="utf-8")

    def test_renamed_python_file_handled(self, git_repo, monkeypatch):
        """Scenario 7: git mv renamed source file is tracked with old and new paths, not treated as deletion."""
        repo = git_repo
        docs_dir = repo / "docs"
        docs_dir.mkdir()
        src_dir = repo / "src"
        src_dir.mkdir()

        old_file = src_dir / "old_helper.py"
        old_file.write_text("def helper(x: int) -> int:\n    return x * 2\n", encoding="utf-8")
        doc_file = docs_dir / "api.md"
        doc_file.write_text("# Helpers\n\n## helper\nDoubles the number.\n", encoding="utf-8")

        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Initial helper"], cwd=str(repo), check=True)

        run_command_safe(["git", "checkout", "-b", "feat/rename-helper"], cwd=str(repo), check=True)
        run_command_safe(["git", "mv", "src/old_helper.py", "src/new_helper.py"], cwd=str(repo), check=True)
        new_file = src_dir / "new_helper.py"
        new_file.write_text("def helper(x: int, factor: int = 2) -> int:\n    return x * factor\n", encoding="utf-8")
        run_command_safe(["git", "add", "src/new_helper.py"], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Rename helper and add factor"], cwd=str(repo), check=True)

        monkeypatch.setenv("DOCS_SUBFOLDER", "docs")
        monkeypatch.setenv("COMMENT_BODY", "[update-docs]")
        monkeypatch.setenv("ENABLE_SECTION_UPDATER", "true")

        candidate = {
            "proposed_content": "## helper\nMultiplies number by factor (default 2).",
            "explanation": "Documented factor parameter",
            "requires_human_review": False,
        }

        with patch("suggest_docs.find_relevant_files_optimized", return_value=["api.md"]), \
             patch("suggest_docs._resolve_pr_push_target", return_value=("feat/rename-helper", "https://github.com/shreed27/Code-Doc-Ai.git", False)), \
             patch("suggest_docs.run_command_safe") as mock_run:

            def side_effect(cmd, *args, **kwargs):
                if cmd[:2] == ["git", "push"]:
                    return MagicMock(returncode=0, stdout="", stderr="")
                return run_command_safe(cmd, *args, **kwargs)

            mock_run.side_effect = side_effect

            with patch("doc_updater.DocumentationUpdater._default_ai_generator", return_value=json.dumps(candidate)):
                suggest_docs.main()

        assert "factor (default 2)" in doc_file.read_text(encoding="utf-8")

    def test_multi_doc_transaction_all_or_nothing_success(self, git_repo, monkeypatch):
        """Scenario 8: multi-document update succeeds atomically across all files."""
        repo = git_repo
        docs_dir = repo / "docs"
        docs_dir.mkdir()
        src_dir = repo / "src"
        src_dir.mkdir()

        py_file = src_dir / "client.py"
        py_file.write_text("def connect(): pass\n", encoding="utf-8")
        doc_api = docs_dir / "api.md"
        doc_api.write_text("# Client API\n\n## connect\nConnects to server.\n", encoding="utf-8")
        doc_overview = docs_dir / "overview.md"
        doc_overview.write_text("# Overview\n\n## connect\nOverview of connect.\n", encoding="utf-8")

        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Initial client and docs"], cwd=str(repo), check=True)

        run_command_safe(["git", "checkout", "-b", "feat/client-ssl"], cwd=str(repo), check=True)
        py_file.write_text("def connect(ssl: bool = True): pass\n", encoding="utf-8")
        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Add ssl parameter"], cwd=str(repo), check=True)

        monkeypatch.setenv("DOCS_SUBFOLDER", "docs")
        monkeypatch.setenv("COMMENT_BODY", "[update-docs]")
        monkeypatch.setenv("ENABLE_SECTION_UPDATER", "true")

        candidate_api = {
            "proposed_content": "## connect\nConnects to server with ssl.",
            "explanation": "Added ssl",
            "requires_human_review": False,
        }
        candidate_overview = {
            "proposed_content": "## connect\nOverview: connects to server with ssl.",
            "explanation": "Added ssl overview",
            "requires_human_review": False,
        }

        with patch("suggest_docs.find_relevant_files_optimized", return_value=["api.md", "overview.md"]), \
             patch("suggest_docs._resolve_pr_push_target", return_value=("feat/client-ssl", "https://github.com/shreed27/Code-Doc-Ai.git", False)), \
             patch("suggest_docs.run_command_safe") as mock_run:

            def side_effect(cmd, *args, **kwargs):
                if cmd[:2] == ["git", "push"]:
                    return MagicMock(returncode=0, stdout="", stderr="")
                return run_command_safe(cmd, *args, **kwargs)

            mock_run.side_effect = side_effect

            # Return candidate based on doc
            def gen_candidate(sys_p, user_p):
                if "overview.md" in user_p:
                    return json.dumps(candidate_overview)
                return json.dumps(candidate_api)

            with patch("doc_updater.DocumentationUpdater._default_ai_generator", side_effect=gen_candidate):
                suggest_docs.main()

        assert "ssl" in doc_api.read_text(encoding="utf-8")
        assert "ssl" in doc_overview.read_text(encoding="utf-8")

    def test_multi_doc_transaction_rollback_on_failure(self, git_repo, monkeypatch):
        """Scenario 9: all-or-nothing policy rolls back doc1 when doc2 is rejected."""
        repo = git_repo
        docs_dir = repo / "docs"
        docs_dir.mkdir()
        src_dir = repo / "src"
        src_dir.mkdir()

        py_file = src_dir / "payment.py"
        py_file.write_text("def pay(amount: int): pass\n", encoding="utf-8")
        doc1 = docs_dir / "api.md"
        doc1_original = "# Payments\n\n## pay\nExecutes payment.\n"
        doc1.write_text(doc1_original, encoding="utf-8")
        doc2 = docs_dir / "config.md"
        doc2_original = "# Config\n\n## pay\nPayment configuration options.\n"
        doc2.write_text(doc2_original, encoding="utf-8")

        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Initial payments"], cwd=str(repo), check=True)

        run_command_safe(["git", "checkout", "-b", "feat/payment-currency"], cwd=str(repo), check=True)
        py_file.write_text("def pay(amount: int, currency: str = 'USD'): pass\n", encoding="utf-8")
        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Add currency"], cwd=str(repo), check=True)

        monkeypatch.setenv("DOCS_SUBFOLDER", "docs")
        monkeypatch.setenv("COMMENT_BODY", "[update-docs]")
        monkeypatch.setenv("ENABLE_SECTION_UPDATER", "true")

        # Valid candidate for doc1, invalid candidate with fake env var for doc2
        candidate_valid = {
            "proposed_content": "## pay\nExecutes payment in specified currency.",
            "explanation": "Added currency",
            "requires_human_review": False,
        }
        candidate_invalid = {
            "proposed_content": "## pay\nConfigure with UNSUPPORTED_FAKE_PAYMENT_ENV_VAR.",
            "explanation": "Invented env var",
            "requires_human_review": False,
        }

        def gen_candidate(sys_p, user_p):
            if "config.md" in user_p:
                return json.dumps(candidate_invalid)
            return json.dumps(candidate_valid)

        with patch("suggest_docs.find_relevant_files_optimized", return_value=["api.md", "config.md"]), \
             patch("suggest_docs._resolve_pr_push_target", return_value=("feat/payment-currency", "https://github.com/shreed27/Code-Doc-Ai.git", False)), \
             patch("suggest_docs.post_review_comment") as mock_comment:

            with patch("doc_updater.DocumentationUpdater._default_ai_generator", side_effect=gen_candidate):
                suggest_docs.main()

            mock_comment.assert_called_once()
            _, kwargs = mock_comment.call_args
            report = kwargs.get("section_report")
            assert report is not None
            assert report.total_sections_rejected > 0

        # Neither file should be updated on disk
        assert doc1.read_text(encoding="utf-8") == doc1_original
        assert doc2.read_text(encoding="utf-8") == doc2_original

    def test_deleted_symbol_remains_in_doc_requires_review(self, git_repo, monkeypatch):
        """Scenario 10: deleted symbol remaining in active doc instructions requires human review."""
        repo = git_repo
        docs_dir = repo / "docs"
        docs_dir.mkdir()
        src_dir = repo / "src"
        src_dir.mkdir()

        py_file = src_dir / "session.py"
        py_file.write_text("def start_session(): pass\ndef legacy_auth(): pass\n", encoding="utf-8")
        doc_file = docs_dir / "auth.md"
        doc_file.write_text("# Session\n\n## legacy_auth\nCall legacy_auth to start session.\n", encoding="utf-8")

        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Initial session"], cwd=str(repo), check=True)

        run_command_safe(["git", "checkout", "-b", "feat/remove-legacy"], cwd=str(repo), check=True)
        # Remove legacy_auth entirely
        py_file.write_text("def start_session(): pass\n", encoding="utf-8")
        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Remove legacy_auth"], cwd=str(repo), check=True)

        monkeypatch.setenv("DOCS_SUBFOLDER", "docs")
        monkeypatch.setenv("COMMENT_BODY", "[review-docs]")
        monkeypatch.setenv("ENABLE_SECTION_UPDATER", "true")

        # Propose keeping active instructions to the removed symbol
        active_removed_candidate = {
            "proposed_content": "## legacy_auth\nCall legacy_auth to begin migration.",
            "explanation": "Instructed use of legacy_auth",
            "requires_human_review": False,
        }

        with patch("suggest_docs.find_relevant_files_optimized", return_value=["auth.md"]), \
             patch("suggest_docs.post_review_comment") as mock_comment:

            with patch("doc_updater.DocumentationUpdater._default_ai_generator", return_value=json.dumps(active_removed_candidate)):
                suggest_docs.main()

            mock_comment.assert_called_once()
            _, kwargs = mock_comment.call_args
            report = kwargs.get("section_report")
            assert report is not None
            # Must be rejected because it still instructs active usage of removed symbol
            assert report.total_sections_rejected > 0
            assert "removed" in report.rejected_updates[0].explanation.lower()

    def test_feature_flag_disabled_executes_legacy(self, git_repo, monkeypatch):
        """Scenario 11: with ENABLE_SECTION_UPDATER=false, executes legacy pipeline."""
        repo = git_repo
        docs_dir = repo / "docs"
        docs_dir.mkdir()
        src_dir = repo / "src"
        src_dir.mkdir()

        py_file = src_dir / "mod.py"
        py_file.write_text("def compute(): pass\n", encoding="utf-8")
        doc_file = docs_dir / "mod.md"
        doc_file.write_text("# Mod\nInitial.\n", encoding="utf-8")

        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Initial"], cwd=str(repo), check=True)

        run_command_safe(["git", "checkout", "-b", "feat/legacy-run"], cwd=str(repo), check=True)
        py_file.write_text("def compute(x: int): pass\n", encoding="utf-8")
        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Change"], cwd=str(repo), check=True)

        monkeypatch.setenv("DOCS_SUBFOLDER", "docs")
        monkeypatch.setenv("COMMENT_BODY", "[update-docs]")
        monkeypatch.setenv("ENABLE_SECTION_UPDATER", "false")

        with patch("suggest_docs.find_relevant_files_optimized", return_value=["mod.md"]), \
             patch("suggest_docs.generate_updates_parallel", return_value=[("mod.md", "old", "new")]) as mock_legacy_gen, \
             patch("suggest_docs.overwrite_file", return_value=True), \
             patch("suggest_docs._resolve_pr_push_target", return_value=("feat/legacy-run", "https://github.com/shreed27/Code-Doc-Ai.git", False)):

            suggest_docs.main()

            mock_legacy_gen.assert_called_once()

    def test_generator_failure_midway_visible_report_and_no_partial_writes(self, git_repo, monkeypatch):
        """Scenario 12: generator failure midway rolls back transaction and surfaces visible error."""
        repo = git_repo
        docs_dir = repo / "docs"
        docs_dir.mkdir()
        src_dir = repo / "src"
        src_dir.mkdir()

        py_file = src_dir / "task.py"
        py_file.write_text("def run_task(): pass\n", encoding="utf-8")
        doc1 = docs_dir / "task.md"
        doc1_orig = "# Task\n\n## run_task\nRuns the task.\n"
        doc1.write_text(doc1_orig, encoding="utf-8")

        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Initial"], cwd=str(repo), check=True)

        run_command_safe(["git", "checkout", "-b", "feat/task-delay"], cwd=str(repo), check=True)
        py_file.write_text("def run_task(delay: int = 5): pass\n", encoding="utf-8")
        run_command_safe(["git", "add", "."], cwd=str(repo), check=True)
        run_command_safe(["git", "commit", "-m", "Add delay"], cwd=str(repo), check=True)

        monkeypatch.setenv("DOCS_SUBFOLDER", "docs")
        monkeypatch.setenv("COMMENT_BODY", "[update-docs]")
        monkeypatch.setenv("ENABLE_SECTION_UPDATER", "true")

        # Generator raises runtime error midway
        def buggy_generator(sys_p, user_p):
            raise RuntimeError("Midway AI connection timeout")

        with patch("suggest_docs.find_relevant_files_optimized", return_value=["task.md"]), \
             patch("suggest_docs._resolve_pr_push_target", return_value=("feat/task-delay", "https://github.com/shreed27/Code-Doc-Ai.git", False)), \
             patch("suggest_docs.post_review_comment") as mock_comment:

            with patch("doc_updater.DocumentationUpdater._default_ai_generator", side_effect=buggy_generator):
                suggest_docs.main()

            mock_comment.assert_called_once()
            _, kwargs = mock_comment.call_args
            report = kwargs.get("section_report")
            assert report is not None
            # Must require human review or be marked as failed
            assert len(report.review_required_updates) > 0 or len(report.rejected_updates) > 0

        # No partial write to disk
        assert doc1.read_text(encoding="utf-8") == doc1_orig
