"""
Tests for Documentation Impact Mapping Engine.

Verifies:
- Function changes mapped to README sections
- API changes mapped to API documentation
- Constant changes mapped to configuration documentation
- Deleted symbols mapped to stale documentation references
- Multiple code changes affecting one section
- One code change affecting multiple documents
- Unrelated documents not being selected
- Duplicate section titles
- Nested documentation sections
- Private implementation changes with documented behavioral impact
- No documentation impact found
- Missing or malformed documentation files
- Repository documentation discovery and consolidation
"""

import tempfile
from pathlib import Path

from impact_mapper import (
    DocumentationImpactMapper,
    consolidate_impacts,
    discover_documentation_files,
    map_repository_impacts,
    tokenize_name,
)
from section_patcher import parse_sections


class TestTokenize:
    """Test identifier splitting into semantic tokens."""

    def test_snake_and_camel_case(self):
        assert tokenize_name("extract_python_symbols") == {"extract", "python", "symbols"}
        assert tokenize_name("ASTDelta") == {"ast", "delta"}
        assert tokenize_name("compute_ast_delta") == {"compute", "ast", "delta"}
        assert tokenize_name("") == set()


class TestDocumentationDiscovery:
    """Test discovering documentation files and ignoring junk directories."""

    def test_discover_markdown_files_and_ignore_noise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Valid doc files
            (base / "README.md").write_text("# Readme")
            (base / "docs").mkdir()
            (base / "docs" / "api.md").write_text("# API")
            (base / "docs" / "guide.markdown").write_text("# Guide")

            # Ignored files / dirs
            (base / ".git").mkdir()
            (base / ".git" / "COMMIT_EDITMSG.md").write_text("commit")
            (base / "node_modules").mkdir()
            (base / "node_modules" / "pkg.md").write_text("pkg")
            (base / ".venv").mkdir()
            (base / ".venv" / "info.md").write_text("venv")
            (base / "app.py").write_text("print('hello')")

            discovered = discover_documentation_files(base)
            str_paths = [str(p) for p in discovered]

            assert "README.md" in str_paths
            assert "docs/api.md" in str_paths
            assert "docs/guide.markdown" in str_paths
            assert not any(".git" in p for p in str_paths)
            assert not any("node_modules" in p for p in str_paths)
            assert not any(".venv" in p for p in str_paths)

    def test_discover_nonexistent_directory(self):
        assert discover_documentation_files("/nonexistent/path/here") == []


class TestImpactMapping:
    """Test matching code changes to documentation sections."""

    def test_function_change_mapped_to_readme(self):
        old_code = """
def parse_markdown(text: str) -> dict:
    return {}
"""
        new_code = """
def parse_markdown(text: str, strict: bool = False) -> dict:
    return {}
"""
        readme = """# Tool Documentation

## Installation
Run pip install tool.

## Usage
Use `parse_markdown` to parse document structure.
"""
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/parser.py",
            {"README.md": readme},
        )

        assert len(impacts) == 1
        imp = impacts[0]
        assert imp.doc_path == "README.md"
        assert imp.section_id == "tool-documentation/usage"
        assert imp.changed_symbol == "parse_markdown"
        assert imp.change_category == "modified_parameters"
        assert imp.is_deterministic is True
        assert imp.confidence == 1.0
        assert "parse_markdown" in imp.reason

    def test_api_changes_mapped_to_api_doc(self):
        old_code = "class Client:\n    pass"
        new_code = """
class Client:
    def connect(self, host: str, port: int = 8080) -> bool:
        \"\"\"Establish connection to server.\"\"\"
        return True
"""
        api_doc = """# API Reference

## Client Class
Overview of Client.

### Methods
Details on client methods.
"""
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/client.py",
            {"docs/api.md": api_doc},
        )

        assert len(impacts) >= 1
        # Client added should map to Client Class
        client_impacts = [i for i in impacts if i.changed_symbol == "Client"]
        assert len(client_impacts) >= 1
        assert client_impacts[0].section_id == "api-reference/client-class"

    def test_constant_changes_mapped_to_configuration_doc(self):
        old_code = "DEFAULT_TIMEOUT = 30"
        new_code = "DEFAULT_TIMEOUT = 60"

        config_doc = """# Configuration Guide

## Network Settings
Timeout settings are controlled by `DEFAULT_TIMEOUT`. Default was 30 seconds.

## Database Settings
Database connection pool size.
"""
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/config.py",
            {"docs/config.md": config_doc},
        )

        assert len(impacts) == 1
        imp = impacts[0]
        assert imp.doc_path == "docs/config.md"
        assert imp.section_id == "configuration-guide/network-settings"
        assert imp.changed_symbol == "DEFAULT_TIMEOUT"
        assert imp.change_category == "modified_constant"
        assert imp.is_deterministic is True

    def test_deleted_symbol_mapped_to_stale_doc_references(self):
        old_code = """
def legacy_sync():
    pass

def modern_sync():
    pass
"""
        new_code = """
def modern_sync():
    pass
"""
        doc = """# Sync Guide

## Legacy Method
Call `legacy_sync()` to sync files.

## Modern Method
Call `modern_sync()` instead.
"""
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/sync.py",
            {"docs/sync.md": doc},
        )

        assert len(impacts) == 1
        imp = impacts[0]
        assert imp.changed_symbol == "legacy_sync"
        assert imp.change_category == "removed_symbol"
        assert imp.section_id == "sync-guide/legacy-method"
        assert imp.confidence == 1.0
        assert "stale documentation" in imp.reason

    def test_multiple_code_changes_affecting_one_section(self):
        old_code = """
def start_engine(): pass
def stop_engine(): pass
"""
        new_code = """
def start_engine(fast: bool = True): pass
def stop_engine(force: bool = False): pass
"""
        doc = """# Engine Control

## Lifecycle
Manage lifecycle using `start_engine` and `stop_engine`.
"""
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/engine.py",
            {"docs/lifecycle.md": doc},
        )

        assert len(impacts) == 2
        symbols = {i.changed_symbol for i in impacts}
        assert symbols == {"start_engine", "stop_engine"}
        assert {i.section_id for i in impacts} == {"engine-control/lifecycle"}

        # Test consolidation
        consolidated = consolidate_impacts(impacts)
        assert len(consolidated) == 1
        c = consolidated[0]
        assert c.doc_path == "docs/lifecycle.md"
        assert c.section_id == "engine-control/lifecycle"
        assert set(c.changed_symbols) == {"start_engine", "stop_engine"}

    def test_one_code_change_affecting_multiple_documents(self):
        old_code = "def export_data(fmt: str): pass"
        new_code = "def export_data(fmt: str, compress: bool = False): pass"

        doc1 = "# User Guide\n\n## Exporting\nRun `export_data`."
        doc2 = "# API Reference\n\n## Data Functions\nAPI `export_data` function details."

        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/export.py",
            {"docs/guide.md": doc1, "docs/api.md": doc2},
        )

        docs_affected = {i.doc_path for i in impacts}
        assert docs_affected == {"docs/guide.md", "docs/api.md"}

    def test_unrelated_documents_not_selected(self):
        old_code = "def calculate_tax(): pass"
        new_code = "def calculate_tax(rate: float): pass"

        unrelated_doc = """# Contributing Guide

## Code of Conduct
Be respectful and constructive.

## Opening PRs
Ensure tests pass.
"""
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/tax.py",
            {"CONTRIBUTING.md": unrelated_doc},
        )

        assert len(impacts) == 0

    def test_duplicate_section_titles_resolved_unambiguously(self):
        old_code = "def execute_task(): pass"
        new_code = "def execute_task(retry: int = 3): pass"

        doc = """# Worker A
## Tasks
Call `execute_task` on Worker A.

# Worker B
## Tasks
Do not call execute_task on Worker B.
"""
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/worker.py",
            {"docs/workers.md": doc},
        )

        # Worker A tasks has section_id worker-a/tasks, Worker B tasks has worker-b/tasks
        assert len(impacts) == 2
        sec_ids = {i.section_id for i in impacts}
        assert sec_ids == {"worker-a/tasks", "worker-b/tasks"}

    def test_nested_documentation_sections_selects_specific_leaf(self):
        old_code = "def stream_logs(): pass"
        new_code = "def stream_logs(follow: bool = True): pass"

        doc = """# Logging System
Overview of logging architecture.

## Streaming Interface
Streaming operations overview.

### Log Watcher
The `stream_logs` function streams events live.
"""
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/logging.py",
            {"docs/logs.md": doc},
        )

        assert len(impacts) == 1
        # Targets the exact leaf section containing the reference
        assert impacts[0].section_id == "logging-system/streaming-interface/log-watcher"

    def test_private_implementation_change_with_documented_behavior(self):
        old_code = """
def _classify_nested(node):
    # simple logic
    return "nested"
"""
        new_code = """
def _classify_nested(node):
    # improved nested classification algorithm
    return "nested_function"
"""
        doc = """# Architecture Internals

## Nested Classification
Explains `_classify_nested` internal behavior and nested classification rules.
"""
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/classifier.py",
            {"docs/internals.md": doc},
        )

        assert len(impacts) == 1
        assert impacts[0].changed_symbol == "_classify_nested"
        assert impacts[0].change_category == "implementation_change"
        assert impacts[0].section_id == "architecture-internals/nested-classification"

    def test_no_documentation_impact_found(self):
        old_code = "def helper(): pass"
        new_code = "def helper():\n    # harmless comment\n    pass"

        doc = "# Readme\nNo mention of helper."
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/helper.py",
            {"README.md": doc},
        )
        assert impacts == []

    def test_empty_or_malformed_documents_handled_safely(self):
        old_code = "def compute(): pass"
        new_code = "def compute(x: int): pass"

        docs = {
            "empty.md": "",
            "malformed.md": "###\n```python\n# broken code block with no end\n",
        }
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/compute.py",
            docs,
        )
        assert isinstance(impacts, list)
        assert len(impacts) == 0

    def test_import_change_mapped_to_dependencies_section(self):
        old_code = "import json"
        new_code = "import json\nimport requests"

        doc = """# Project Guide

## Prerequisites & Dependencies
Requires python 3.12 and requests library for HTTP networking.
"""
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/api.py",
            {"docs/guide.md": doc},
        )

        assert len(impacts) == 1
        imp = impacts[0]
        assert imp.changed_symbol == "requests"
        assert imp.change_category == "added_import"
        assert imp.section_id == "project-guide/prerequisites-dependencies"


class TestRepositoryImpactsConvenienceFunction:
    """Test map_repository_impacts across multiple files and disk discovery."""

    def test_map_repository_impacts_in_memory(self):
        code_changes = {
            "src/auth.py": (
                "def login(): pass",
                "def login(mfa_token: str = ''): pass",
            ),
            "src/db.py": (
                "MAX_CONNECTIONS = 10",
                "MAX_CONNECTIONS = 50",
            ),
        }
        doc_files = {
            "README.md": "# Readme\n\n## Authentication\nCall `login` with user credentials.",
            "docs/config.md": "# Config\n\n## Database\nConfigured with `MAX_CONNECTIONS`.",
        }

        impacts = map_repository_impacts(code_changes, doc_files=doc_files)
        assert len(impacts) == 2

        consolidated = consolidate_impacts(impacts)
        assert len(consolidated) == 2
        assert consolidated[0].doc_path in ("README.md", "docs/config.md")
        assert consolidated[1].doc_path in ("README.md", "docs/config.md")

    def test_map_repository_impacts_from_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "README.md").write_text("# App\n\n## Server\nUses `start_server` to run.")

            code_changes = {
                "src/server.py": (
                    "def start_server(): pass",
                    "def start_server(port: int = 8080): pass",
                )
            }

            impacts = map_repository_impacts(code_changes, repo_path=base)
            assert len(impacts) == 1
            assert impacts[0].doc_path == "README.md"
            assert impacts[0].section_id == "app/server"

    def test_added_symbol_matching_heading_and_file_path(self):
        old_code = "# empty"
        new_code = "def authenticate_user(): pass"

        doc = """# Security Module
Detailed in `src/auth.py`.

## authenticate_user
New authentication method.

## User Session
Session handling.
"""
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code,
            new_code,
            "src/auth.py",
            {"README.md": doc},
        )
        assert len(impacts) >= 1
        heading_matches = [
            i for i in impacts if i.section_id == "security-module/authenticate-user"
        ]
        assert len(heading_matches) >= 1

    def test_modified_symbol_parameter_removed_and_return_type_and_docstring(self):
        # 1. Parameter removed
        old_param = "def search(q: str, limit: int = 10): pass"
        new_param = "def search(q: str): pass"
        doc_param = "# Overview\n\n## Search Limits\nThe `limit` parameter controls max results. Call `search`."

        mapper = DocumentationImpactMapper()
        imp_param = mapper.map_file_impacts(
            old_param, new_param, "src/search.py", {"doc.md": doc_param}
        )
        assert len(imp_param) == 1
        assert any(ev.evidence_type == "signature_parameter" for ev in imp_param[0].evidence)
        assert imp_param[0].section_id == "overview/search-limits"

        # 2. Return type changed
        old_ret = "def get_status() -> bool: pass"
        new_ret = "def get_status() -> str: pass"
        doc_ret = "# API\n## Status\nReturns current state via `get_status`."
        imp_ret = mapper.map_file_impacts(old_ret, new_ret, "src/status.py", {"doc.md": doc_ret})
        assert len(imp_ret) == 1
        assert imp_ret[0].change_category == "modified_return_type"

        # 3. Docstring changed
        old_doc = 'def helper():\n    """Old docstring."""\n    pass'
        new_doc = 'def helper():\n    """New updated docstring."""\n    pass'
        doc_helper = "# Overview\n\n## Helper Function\nDescribes `helper` function."
        imp_doc = mapper.map_file_impacts(old_doc, new_doc, "src/helper.py", {"doc.md": doc_helper})
        assert len(imp_doc) == 1
        assert imp_doc[0].change_category == "docstring_change"

    def test_added_and_removed_constants(self):
        # Added constant
        old_c = "# no constants"
        new_c = "CACHE_TTL = 3600"
        doc_c = "# Caching\n## TTL\nConfigured via `CACHE_TTL`."

        mapper = DocumentationImpactMapper()
        imp_add = mapper.map_file_impacts(old_c, new_c, "src/cache.py", {"doc.md": doc_c})
        assert len(imp_add) == 1
        assert imp_add[0].change_category == "added_constant"

        # Removed constant
        imp_rem = mapper.map_file_impacts(new_c, old_c, "src/cache.py", {"doc.md": doc_c})
        assert len(imp_rem) == 1
        assert imp_rem[0].change_category == "removed_constant"

    def test_removed_import_mapped_to_dependencies(self):
        old_imp = "import redis"
        new_imp = "# redis removed"
        doc_imp = "# Requirements\n## Dependencies\nRequires redis server."

        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(old_imp, new_imp, "src/cache.py", {"doc.md": doc_imp})
        assert len(impacts) == 1
        assert impacts[0].change_category == "removed_import"

    def test_pre_parsed_document_input(self):
        old_code = "def ping(): pass"
        new_code = "def ping(timeout: int = 5): pass"
        parsed = parse_sections("# Health\n## Ping\nCall `ping`.")

        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(
            old_code, new_code, "src/health.py", {"health.md": parsed}
        )
        assert len(impacts) == 1
        assert impacts[0].section_id == "health/ping"

    def test_removed_symbol_in_heading(self):
        old_code = "def deprecated_fn(): pass"
        new_code = "# gone"
        doc = "# API\n## deprecated_fn\nDetails here."

        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(old_code, new_code, "src/api.py", {"api.md": doc})
        assert len(impacts) == 1
        assert any(ev.evidence_type == "heading_match" for ev in impacts[0].evidence)

    def test_modified_parameter_default_or_type_and_param_name_in_docs(self):
        old_code = "def render(theme: str = 'light'): pass"
        new_code = "def render(theme: str = 'dark', verbose: bool = False): pass"
        doc = """# UI Module
## Render Options
Configured in `ui.py`. Pass `verbose` to enable verbose output and `theme` for style.
"""
        mapper = DocumentationImpactMapper()
        impacts = mapper.map_file_impacts(old_code, new_code, "src/ui.py", {"ui.md": doc})
        assert len(impacts) >= 1
        imp = [i for i in impacts if i.section_id == "ui-module/render-options"][0]
        assert any(ev.evidence_type == "signature_parameter" for ev in imp.evidence)
        assert any(ev.evidence_type == "explicit_file_path" for ev in imp.evidence)

    def test_semantic_fallback_keyword_overlap(self):
        old_code = "# empty"
        new_code = "def generate_report(): pass"
        doc = """# Analytics
## Report Generation
Analytics reports.
"""
        mapper = DocumentationImpactMapper(semantic_threshold=0.3)
        impacts = mapper.map_file_impacts(
            old_code, new_code, "src/analytics.py", {"analytics.md": doc}
        )
        assert len(impacts) >= 1
        semantic_ev = [
            ev for i in impacts for ev in i.evidence if ev.evidence_type == "semantic_keyword"
        ]
        assert len(semantic_ev) >= 1

    def test_unreadable_file_handled_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            f = base / "broken.md"
            f.write_text("# Broken")
            # Make unreadable by removing permissions
            f.chmod(0o000)
            try:
                impacts = map_repository_impacts({"src/foo.py": ("x=1", "x=2")}, repo_path=base)
                assert isinstance(impacts, list)
            finally:
                f.chmod(0o644)
