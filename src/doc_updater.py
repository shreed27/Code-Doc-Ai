"""
AI Documentation Updating Engine.

Coordinates the end-to-end documentation update pipeline:
1. Detects code changes via AST analysis (`src/ast_intel.py`).
2. Maps affected documentation sections via impact mapping (`src/impact_mapper.py`).
3. Generates grounded section updates via an AI documentation model.
4. Validates proposed updates deterministically against source code truth.
5. Surgically updates affected Markdown sections via `src/section_patcher.py`.
6. Staged multi-document transaction with atomic rollback.
"""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ast_intel import (
    ASTDelta,
    ConstantInfo,
    ParameterInfo,
    SymbolInfo,
    compute_ast_delta,
    extract_python_constants,
    extract_python_imports,
    extract_python_symbols,
)
from impact_mapper import (
    DocumentationImpactMapper,
    ImpactEvidence,
    SectionImpact,
    consolidate_impacts,
)
from section_patcher import ParsedDocument, parse_sections, replace_section_content
from security_utils import redact_secrets

# =============================================================================
# DATA CONTRACTS
# =============================================================================


class ContextBudgetExceededError(Exception):
    """Raised when target section and essential context exceed configured MAX_CONTEXT_CHARS."""

    pass


@dataclass
class SectionUpdateRequest:
    """Context bundle provided to the AI documentation generator."""

    doc_path: str
    section_id: str
    section_heading: str
    current_section_content: str
    source_files: list[str]
    changed_symbols: list[str]
    change_categories: list[str]
    impact_reasons: list[str]
    impact_evidence: list[ImpactEvidence]
    old_code_snippets: dict[str, str]
    new_code_snippets: dict[str, str]
    ast_deltas: dict[str, ASTDelta] = field(default_factory=dict)
    surrounding_context: str = ""
    style_guidelines: str = ""


@dataclass
class AISectionUpdateResponse:
    """Structured response from the AI documentation generator."""

    doc_path: str
    section_id: str
    proposed_content: str
    explanation: str
    technical_evidence: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    review_reason: str = ""


@dataclass
class ValidationResult:
    """Results of deterministic safety and hallucination checks."""

    is_valid: bool
    passed_checks: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    review_reasons: list[str] = field(default_factory=list)


@dataclass
class AppliedSectionUpdate:
    """Status record of an evaluated and/or applied section update."""

    doc_path: str
    section_id: str
    section_heading: str
    status: str  # "applied", "updated", "rejected", "review_required", "no_change", "source_analysis_failed", "generation_failed", "transaction_failed"
    original_section_content: str
    new_section_content: str
    explanation: str
    validation: ValidationResult
    diff_snippet: str = ""


@dataclass
class DocumentationUpdateReport:
    """Comprehensive summary of documentation update run (Priority 1 & Priority 5)."""

    updated_documents: dict[str, str]  # doc_path -> updated markdown text
    applied_updates: list[AppliedSectionUpdate] = field(default_factory=list)
    candidate_updates: list[AppliedSectionUpdate] = field(default_factory=list)
    validated_updates: list[AppliedSectionUpdate] = field(default_factory=list)
    committed_updates: list[AppliedSectionUpdate] = field(default_factory=list)
    staged_updates: list[str] = field(default_factory=list)
    rejected_updates: list[AppliedSectionUpdate] = field(default_factory=list)
    review_required_updates: list[AppliedSectionUpdate] = field(default_factory=list)
    source_analysis_errors: list[tuple[str, str]] = field(default_factory=list)
    generation_failures: list[AppliedSectionUpdate] = field(default_factory=list)
    transaction_failures: list[AppliedSectionUpdate] = field(default_factory=list)
    total_sections_analyzed: int = 0
    total_sections_applied: int = 0
    total_sections_rejected: int = 0


# =============================================================================
# TRANSACTION & STAGED MULTI-DOCUMENT UPDATES
# =============================================================================


class DocumentTransactionError(Exception):
    """Raised when document staging, validation, or transaction commit fails."""

    def __init__(
        self,
        message: str,
        unrecovered_backups: list[tuple[Path, Path]] | None = None,
    ):
        super().__init__(message)
        self.unrecovered_backups = unrecovered_backups or []


class UnsupportedDocumentExtensionError(DocumentTransactionError):
    """Raised when an update targets a non-markdown file extension."""

    pass


class PathTraversalSecurityError(DocumentTransactionError):
    """Raised when a destination path attempts traversal outside docs_root or follows an escaping symlink."""

    pass


@dataclass
class TransactionJournalEntry:
    dest_path: Path
    temp_path: Path
    backup_path: Path | None = None
    orig_bytes: bytes | None = None
    orig_mode: int | None = None
    orig_times: tuple[float, float] | None = None
    existed: bool = False
    state: str = "staged"  # "staged", "prepared", "replacing", "replaced", "rolled_back", "rollback_failed"


class DocumentTransaction:
    """
    Manages multi-document updates with application-level transactional guarantees.

    Provides staged temporary writes, pre-replacement journal registration,
    permission and metadata preservation, rollback on commit failure,
    and separated cleanup outcome (T7 and T8).

    NOTE ON ATOMICITY LIMITATIONS:
    This class provides in-process application-level rollback against Python exceptions,
    OS errors, and BaseExceptions (e.g. KeyboardInterrupt). It does NOT guarantee durable
    multi-file atomicity across catastrophic hardware power loss or uncatchable SIGKILL,
    which would require an external write-ahead log (WAL) or filesystem-level snapshotting.
    """

    def __init__(self, docs_root: str | Path):
        self.docs_root = Path(docs_root).resolve()
        self.journal: list[TransactionJournalEntry] = []
        self.unrecovered_backups: list[tuple[Path, Path]] = []
        self.cleanup_warnings: list[tuple[Path, str]] = []
        self.committed = False

    @property
    def staged_files(
        self,
    ) -> list[tuple[Path, Path, bytes | None, int | None, tuple[float, float] | None, bool]]:
        return [
            (e.dest_path, e.temp_path, e.orig_bytes, e.orig_mode, e.orig_times, e.existed)
            for e in self.journal
        ]

    @property
    def backups(self) -> list[tuple[Path, Path, int | None, tuple[float, float] | None]]:
        return [
            (e.dest_path, e.backup_path, e.orig_mode, e.orig_times)
            for e in self.journal
            if e.backup_path is not None
        ]

    def validate_destination_path(self, rel_path: str | Path) -> Path:
        """Validate extension, symlinks, and prevent path traversal."""
        path_str = str(rel_path).strip()
        lower = path_str.lower()
        if not (lower.endswith(".md") or lower.endswith(".markdown")):
            raise UnsupportedDocumentExtensionError(
                f"Unsupported document extension: '{path_str}'. Only .md and .markdown are supported."
            )

        dest = (self.docs_root / path_str).resolve()
        if dest != self.docs_root and self.docs_root not in dest.parents:
            raise PathTraversalSecurityError(
                f"Path traversal detected: '{rel_path}' resolves outside docs root '{self.docs_root}'."
            )

        unresolved = self.docs_root / path_str
        if unresolved.is_symlink():
            target_resolved = unresolved.resolve()
            if target_resolved != self.docs_root and self.docs_root not in target_resolved.parents:
                raise PathTraversalSecurityError(
                    f"Symlink traversal detected: '{rel_path}' links outside docs root to '{target_resolved}'."
                )

        return dest

    def stage_updates(self, updates: dict[str, str]) -> None:
        """Stage all document updates to temporary files with matching permissions."""
        try:
            for rel_path, content in updates.items():
                dest_path = self.validate_destination_path(rel_path)
                try:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                except OSError as mkdir_err:
                    raise DocumentTransactionError(
                        f"Failed creating directory for '{rel_path}': {mkdir_err}"
                    ) from mkdir_err

                orig_bytes = None
                orig_mode = None
                orig_times = None
                existed = dest_path.is_file()
                if existed:
                    try:
                        orig_bytes = dest_path.read_bytes()
                        stat_res = dest_path.stat()
                        orig_mode = stat_res.st_mode & 0o777
                        orig_times = (stat_res.st_atime, stat_res.st_mtime)
                    except OSError:
                        orig_mode = 0o644
                else:
                    orig_mode = 0o644

                temp_path = None
                try:
                    temp_fd, temp_path_str = tempfile.mkstemp(
                        dir=dest_path.parent, prefix=".stage_doc_"
                    )
                    temp_path = Path(temp_path_str)
                    with os.fdopen(temp_fd, "wb") as f:
                        f.write(content.encode("utf-8"))
                        f.flush()
                        os.fsync(f.fileno())
                    if orig_mode is not None:
                        try:
                            os.chmod(temp_path, orig_mode)
                        except OSError:
                            pass
                except Exception as write_err:
                    if temp_path and temp_path.exists():
                        temp_path.unlink(missing_ok=True)
                    raise DocumentTransactionError(
                        f"Failed staging document update for '{rel_path}': {write_err}"
                    ) from write_err

                self.journal.append(
                    TransactionJournalEntry(
                        dest_path=dest_path,
                        temp_path=temp_path,
                        backup_path=None,
                        orig_bytes=orig_bytes,
                        orig_mode=orig_mode,
                        orig_times=orig_times,
                        existed=existed,
                        state="staged",
                    )
                )
        except Exception:
            self.cleanup()
            raise

    def commit(self) -> None:
        """
        Apply all staged replacements atomically; roll back all on any failure.
        Maintains pre-replacement journal registration (T7) and decoupled cleanup (T8).
        """
        if not self.journal:
            return

        try:
            # Stage 1: Pre-replacement backup preparation.
            # ALL backups are created, fsynced, and recorded in the journal BEFORE any replacement.
            for entry in self.journal:
                if entry.existed and entry.orig_bytes is not None and entry.backup_path is None:
                    try:
                        bak_fd, bak_path_str = tempfile.mkstemp(
                            dir=entry.dest_path.parent, prefix=".bak_doc_"
                        )
                        backup_path = Path(bak_path_str)
                        with os.fdopen(bak_fd, "wb") as dst:
                            dst.write(entry.orig_bytes)
                            dst.flush()
                            os.fsync(dst.fileno())
                        if entry.orig_mode is not None:
                            try:
                                os.chmod(backup_path, entry.orig_mode)
                            except OSError:
                                pass
                        entry.backup_path = backup_path
                        entry.state = "prepared"
                    except Exception as bak_err:
                        raise DocumentTransactionError(
                            f"Failed creating backup for '{entry.dest_path}': {bak_err}"
                        ) from bak_err

            # Stage 2: Replacement with pre-replacement state transition.
            # Entry state is marked 'replacing' BEFORE os.replace so interruption
            # immediately after os.replace retains knowledge of modified destination (T7).
            for entry in self.journal:
                entry.state = "replacing"
                os.replace(entry.temp_path, entry.dest_path)
                entry.state = "replaced"
                if entry.orig_mode is not None:
                    try:
                        os.chmod(entry.dest_path, entry.orig_mode)
                    except OSError:
                        pass

            self.committed = True
        except (Exception, BaseException) as commit_err:
            # ROLLBACK: Inspect journal in reverse. Any entry in "replacing" or "replaced"
            # must be restored from its pre-registered backup.
            rollback_failures: list[tuple[Path, Path | None, str]] = []
            for entry in reversed(self.journal):
                if entry.state in ("replacing", "replaced"):
                    try:
                        if entry.backup_path is not None and entry.backup_path.exists():
                            os.replace(entry.backup_path, entry.dest_path)
                            if entry.orig_mode is not None:
                                try:
                                    os.chmod(entry.dest_path, entry.orig_mode)
                                except OSError:
                                    pass
                            if entry.orig_times is not None:
                                try:
                                    os.utime(entry.dest_path, entry.orig_times)
                                except OSError:
                                    pass
                            entry.state = "rolled_back"
                        elif not entry.existed and entry.dest_path.exists():
                            entry.dest_path.unlink(missing_ok=True)
                            entry.state = "rolled_back"
                    except (Exception, BaseException) as rb_err:
                        entry.state = "rollback_failed"
                        if entry.backup_path is not None and entry.backup_path.exists():
                            self.unrecovered_backups.append((entry.dest_path, entry.backup_path))
                        rollback_failures.append((entry.dest_path, entry.backup_path, str(rb_err)))

            if rollback_failures:
                msg = (
                    f"Transaction commit failed and rollback could NOT restore all files: {commit_err}. "
                    f"Failures: {rollback_failures}. Retained backups for manual recovery: {self.unrecovered_backups}"
                )
                raise DocumentTransactionError(
                    msg, unrecovered_backups=self.unrecovered_backups
                ) from commit_err
            else:
                raise DocumentTransactionError(
                    f"Transaction commit failed; rolled back all replaced documents successfully: {commit_err}"
                ) from commit_err
        finally:
            # Stage 3: Cleanup. Separate outcome (T8).
            # Unlink errors do not mask commit success or abort transaction.
            try:
                self.cleanup()
            except Exception as cleanup_err:
                self.cleanup_warnings.append((Path("all"), str(cleanup_err)))

    def cleanup(self) -> None:
        """
        Clean up temporary staged files and backups, NEVER unlinking surviving backups from failed rollbacks.
        Captures unlink errors as non-fatal cleanup warnings (T8).
        """
        unrecovered_baks = {b for _d, b in self.unrecovered_backups}
        for entry in self.journal:
            if entry.temp_path and entry.temp_path.exists():
                try:
                    entry.temp_path.unlink(missing_ok=True)
                except OSError as e:
                    self.cleanup_warnings.append((entry.temp_path, str(e)))
            if (
                entry.backup_path
                and entry.backup_path not in unrecovered_baks
                and entry.backup_path.exists()
                and entry.state != "rollback_failed"
            ):
                try:
                    entry.backup_path.unlink(missing_ok=True)
                except OSError as e:
                    self.cleanup_warnings.append((entry.backup_path, str(e)))

    def retry_cleanup(self) -> list[tuple[Path, str]]:
        """Retry deleting any leftover temporary or backup files that failed cleanup."""
        remaining: list[tuple[Path, str]] = []
        for path, _prev_err in self.cleanup_warnings:
            if path.exists():
                try:
                    path.unlink(missing_ok=True)
                except OSError as e:
                    remaining.append((path, str(e)))
        self.cleanup_warnings = remaining
        return list(remaining)


def atomic_overwrite_file(
    file_path: str | Path, content: str, docs_root: str | Path | None = None
) -> bool:
    """
    Atomically overwrite a single documentation file with validation and rollback.
    Fails safely without falling back to unvalidated direct writes.
    """
    path = Path(file_path).resolve()
    root = Path(docs_root).resolve() if docs_root else path.parent
    try:
        tx = DocumentTransaction(docs_root=root)
        rel_path = path.relative_to(root)
        tx.stage_updates({str(rel_path): content})
        tx.commit()
        return True
    except Exception as e:
        print(f"Error: atomic_overwrite_file failed for {path}: {e}")
        return False


# =============================================================================
# SECRET REDACTION & PROMPT FORMATTER
# =============================================================================


def values_are_equivalent(val1: str, val2: str) -> bool:
    """Check if two values are semantically equivalent (bools, floats, ints, null)."""
    s1 = str(val1).strip().strip("'\"")
    s2 = str(val2).strip().strip("'\"")
    if s1 == s2:
        return True
    # Boolean equivalence
    if s1.lower() in ("true", "false") and s2.lower() in ("true", "false"):
        return s1.lower() == s2.lower()
    # None/null equivalence
    if s1.lower() in ("none", "null") and s2.lower() in ("none", "null"):
        return True
    # Numeric equivalence (e.g. 30 and 30.0)
    try:
        f1 = float(s1)
        f2 = float(s2)
        return f1 == f2
    except (ValueError, TypeError):
        pass
    return False


def is_already_documented(current_content: str, req: SectionUpdateRequest) -> bool:
    """
    Check if the current documentation section already accurately reflects
    all technical facts introduced by the AST deltas.
    Prevents repetitive rewrites from model wording variations (Priority 4).
    Uses strict fact-bound matching so numbers/names from unrelated entities
    do not cause stale documentation to be falsely marked as already documented (D3, D4, D5, D6, D11).
    """
    if not current_content or not current_content.strip():
        return False

    # 1. Removed symbols:
    # If any removed symbol is still actively mentioned (not deprecated/removed notice),
    # the documentation is NOT up to date.
    for delta in req.ast_deltas.values():
        for sym in delta.removed_symbols:
            short_name = sym.name.split(".")[-1]
            if re.search(rf"\b{re.escape(short_name)}\b", current_content):
                clauses = re.split(r"[.\n;]", current_content)
                has_active = False
                for cl in clauses:
                    if re.search(rf"\b{re.escape(short_name)}\b", cl):
                        cl_lower = cl.lower()
                        is_negated = any(
                            k in cl_lower
                            for k in [
                                "removed",
                                "deprecated",
                                "deleted",
                                "obsolete",
                                "do not use",
                                "do not call",
                                "never use",
                                "never call",
                                "replaced with",
                                "no longer",
                            ]
                        )
                        if not is_negated:
                            has_active = True
                            break
                if has_active:
                    return False

    has_deltas = False

    # 2. Check modified constants (D3, D4, D6, D11):
    for delta in req.ast_deltas.values():
        for old_c, new_c in delta.modified_constants:
            has_deltas = True
            c_name = new_c.name.split(".")[-1]
            old_val = old_c.value.strip().strip("'\"") if old_c.value else None
            new_val = new_c.value.strip().strip("'\"") if new_c.value else None

            # D3 / D4: If old value is bound to constant name in current doc -> NOT up to date!
            if old_val:
                bound_to_old = bool(
                    re.search(
                        rf"\b{re.escape(c_name)}\b[^\n\r|:=]*[:=\s|`]+[^\n\r|:=]*\b{re.escape(old_val)}\b",
                        current_content,
                        re.IGNORECASE,
                    )
                )
                if bound_to_old:
                    return False

            # The new value MUST be bound to constant name in current doc
            if new_val:
                bound_to_new = bool(
                    re.search(
                        rf"\b{re.escape(c_name)}\b[^\n\r|:=]*[:=\s|`]+[^\n\r|:=]*\b{re.escape(new_val)}\b",
                        current_content,
                        re.IGNORECASE,
                    )
                )
                if not bound_to_new:
                    return False

    # 3. Check added constants:
    for delta in req.ast_deltas.values():
        for c in delta.added_constants:
            has_deltas = True
            c_name = c.name.split(".")[-1]
            if c.value:
                val = c.value.strip().strip("'\"")
                bound = bool(
                    re.search(
                        rf"\b{re.escape(c_name)}\b[^\n\r|:=]*[:=\s|`]+[^\n\r|:=]*\b{re.escape(val)}\b",
                        current_content,
                        re.IGNORECASE,
                    )
                )
                if not bound:
                    return False
            else:
                if not re.search(rf"\b{re.escape(c_name)}\b", current_content):
                    return False

    # 4. Check modified parameters:
    for delta in req.ast_deltas.values():
        for _fname, old_p, new_p in delta.modified_parameters:
            has_deltas = True
            p_name = new_p.name
            old_def = old_p.default_value.strip().strip("'\"") if old_p.default_value else None
            new_def = new_p.default_value.strip().strip("'\"") if new_p.default_value else None

            # If default changed:
            if old_def and new_def and old_def != new_def:
                bound_to_old = bool(
                    re.search(
                        rf"\b{re.escape(p_name)}\b[^\.\n]*?(?:defaults?\s*(?:is|=|to|:)|default:|\(default\s+|=)\s*`?{re.escape(old_def)}\b",
                        current_content,
                        re.IGNORECASE,
                    )
                )
                if bound_to_old:
                    return False

                bound_to_new = bool(
                    re.search(
                        rf"\b{re.escape(p_name)}\b[^\.\n]*?(?:defaults?\s*(?:is|=|to|:)|default:|\(default\s+|=)\s*`?{re.escape(new_def)}\b",
                        current_content,
                        re.IGNORECASE,
                    )
                )
                if not bound_to_new:
                    return False

            # If type annotation changed:
            old_type = old_p.type_annotation.strip() if old_p.type_annotation else None
            new_type = new_p.type_annotation.strip() if new_p.type_annotation else None
            if old_type and new_type and old_type != new_type:
                bound_to_old_t = bool(
                    re.search(
                        rf"\b{re.escape(p_name)}\b\s*[:(]\s*{re.escape(old_type)}\b",
                        current_content,
                        re.IGNORECASE,
                    )
                )
                if bound_to_old_t:
                    return False

    # 5. Check modified function return types (D5):
    for delta in req.ast_deltas.values():
        for old_s, new_s in delta.modified_symbols:
            has_deltas = True
            f_name = new_s.name.split(".")[-1]
            old_ret = getattr(old_s, "return_type", None) or (
                old_s.signature.split("->")[-1].strip() if "->" in old_s.signature else None
            )
            new_ret = getattr(new_s, "return_type", None) or (
                new_s.signature.split("->")[-1].strip() if "->" in new_s.signature else None
            )
            if old_ret and new_ret and old_ret != new_ret:
                old_ret_clean = old_ret.strip().strip("'\"")
                new_ret_clean = new_ret.strip().strip("'\"")
                bound_to_old = bool(
                    re.search(
                        rf"\b{re.escape(f_name)}\b[^\.\n]*?(?:returns|returning|return value is)\s+(?:a\s+)?`?{re.escape(old_ret_clean)}\b",
                        current_content,
                        re.IGNORECASE,
                    )
                )
                if bound_to_old:
                    return False

                bound_to_new = bool(
                    re.search(
                        rf"\b{re.escape(f_name)}\b[^\.\n]*?(?:returns|returning|return value is)\s+(?:a\s+)?`?{re.escape(new_ret_clean)}\b",
                        current_content,
                        re.IGNORECASE,
                    )
                )
                if not bound_to_new:
                    return False

    # 6. Check added symbols:
    for delta in req.ast_deltas.values():
        for sym in delta.added_symbols:
            has_deltas = True
            short_name = sym.name.split(".")[-1]
            if not re.search(rf"\b{re.escape(short_name)}\b", current_content):
                return False

    return has_deltas


def extract_relevant_code_snippet(
    source_code: str,
    changed_symbols: list[str],
    max_chars: int,
) -> str:
    """
    Extract a targeted code window around the changed symbols, plus relevant
    top-of-file imports and constants, rather than naively slicing from line 0.
    """
    if not source_code or len(source_code) <= max_chars:
        return source_code

    lines = source_code.splitlines(keepends=True)
    total_lines = len(lines)

    # 1. Grab top-of-file imports and constants (up to 30 lines)
    header_lines: list[str] = []
    header_chars = 0
    for i in range(min(30, total_lines)):
        line = lines[i]
        if line.startswith(("import ", "from ", "#")) or re.match(r"^[A-Z0-9_]+\s*=", line):
            header_lines.append(line)
            header_chars += len(line)

    remaining_chars = max(300, max_chars - header_chars - 100)

    # 2. Locate changed symbols using AST or regex
    symbol_line_ranges: list[tuple[int, int]] = []
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if node.name in changed_symbols or any(
                    cs.endswith(f".{node.name}") for cs in changed_symbols
                ):
                    start = getattr(node, "lineno", 1) - 1
                    end = getattr(node, "end_lineno", start + 20)
                    symbol_line_ranges.append((start, end))
    except Exception:
        pass

    if not symbol_line_ranges:
        for idx, line in enumerate(lines):
            for sym in changed_symbols:
                short = sym.split(".")[-1]
                if short in line:
                    symbol_line_ranges.append((max(0, idx - 5), min(total_lines, idx + 25)))
                    break

    if not symbol_line_ranges:
        symbol_line_ranges = [(max(0, total_lines - 40), total_lines)]

    # Collect lines around the symbols
    selected_indices: set[int] = set()
    for s_start, s_end in symbol_line_ranges:
        for idx in range(max(0, s_start - 3), min(total_lines, s_end + 3)):
            selected_indices.add(idx)

    snippet_lines = []
    if header_lines:
        snippet_lines.extend(header_lines)
        snippet_lines.append("\n# ... [code omitted for context budget] ...\n\n")

    sorted_indices = sorted(selected_indices)
    curr_len = sum(len(line_item) for line_item in snippet_lines)
    for idx in sorted_indices:
        line_str = lines[idx]
        if curr_len + len(line_str) > remaining_chars:
            snippet_lines.append("# ... [truncated for context budget] ...\n")
            break
        snippet_lines.append(line_str)
        curr_len += len(line_str)

    return "".join(snippet_lines)


_SYSTEM_PROMPT = """You are an expert technical documentation assistant.
Your job is to update an existing Markdown documentation section so that it accurately reflects recent code changes.

CRITICAL INSTRUCTIONS:
1. ONLY update the provided section. Do NOT generate other sections or document-level headers.
2. Ground all claims STRICTLY in the provided code diff and AST changes.
3. NEVER invent, hallucinate, or assume functions, parameters, types, configuration values, or API endpoints.
4. If a symbol was removed from code, remove or update references to it in the documentation.
5. If the evidence is insufficient to confidently document the change, set "requires_human_review": true.
6. Preserve existing formatting style, tone, code fence languages, and structure.

You must respond ONLY with a JSON object in this exact format:
{
  "doc_path": "<target document path>",
  "section_id": "<target section id>",
  "proposed_content": "<the exact replacement text for this section's body>",
  "explanation": "<concise summary of what was updated and why>",
  "technical_evidence": ["<evidence 1 from code>", "<evidence 2 from code>"],
  "uncertainties": [],
  "requires_human_review": false,
  "review_reason": ""
}"""


def calculate_serialized_request_size(
    model: str, system_prompt: str, user_prompt: str
) -> int:
    """Calculate the exact byte size of the serialized outbound JSON HTTP request body (Priority 7)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    return len(json.dumps(payload).encode("utf-8"))


def build_update_prompt(
    req: SectionUpdateRequest,
    max_context_chars: int | None = None,
    model: str = "gpt-4o",
) -> str:
    """
    Construct an evidence-rich prompt for the AI documentation generator.

    Applies clear untrusted boundary delimiters, secret redaction,
    strictly enforces the context budget across the serialized HTTP request body
    (Priority 7), and prioritizes the target section and exact changed symbol snippets.
    """
    if max_context_chars is None:
        try:
            from config import get_max_context_chars

            max_context_chars = get_max_context_chars()
        except Exception:
            max_context_chars = 400000

    # 1. Critical: Target Section Content (MUST NOT BE TRUNCATED)
    safe_current_content = redact_secrets(req.current_section_content)
    target_sec_text = (
        f"\n### Current Section Content:\n"
        f'<untrusted_current_documentation path="{req.doc_path}" section="{req.section_id}">\n'
        f"{safe_current_content}\n"
        f"</untrusted_current_documentation>\n"
    )

    # 2. Header and metadata
    header_parts = [
        f"### Target Section: {req.section_heading} (`{req.section_id}`)",
        f"**Document**: `{req.doc_path}`",
        f"**Changed Symbols**: {', '.join(req.changed_symbols)}",
        f"**Change Categories**: {', '.join(req.change_categories)}",
        "",
        "### Impact Reasons:",
    ]
    for r in req.impact_reasons:
        header_parts.append(f"- {r}")

    header_parts.extend(["", "### Supporting Code Evidence:"])
    for ev in req.impact_evidence:
        safe_ctx = redact_secrets(ev.context)
        header_parts.append(f'- [{ev.evidence_type}] `{ev.matched_text}`: "{safe_ctx}"')

    header_text = "\n".join(header_parts)

    # Check if target section and metadata alone exceed serialized budget
    base_user_prompt = f"{header_text}\n{target_sec_text}"
    base_bytes = calculate_serialized_request_size(model, _SYSTEM_PROMPT, base_user_prompt)
    if base_bytes > max_context_chars:
        raise ContextBudgetExceededError(
            f"Target section and required metadata ({base_bytes:,} serialized bytes) exceed context budget ({max_context_chars:,} bytes)."
        )

    available_for_other = max(0, max_context_chars - base_bytes)
    if available_for_other < 200:
        raise ContextBudgetExceededError(
            "Essential metadata and target section leave insufficient context for code evidence."
        )

    # 3. Style guidelines: allocate at most 15% of remaining budget
    style_text = ""
    if req.style_guidelines:
        clean_style = redact_secrets(req.style_guidelines).strip()
        max_style_len = max(200, int(available_for_other * 0.15))
        if len(clean_style) > max_style_len:
            clean_style = clean_style[:max_style_len] + "\n# [TRUNCATED STYLE GUIDELINES]"
        style_text = f"\n### Style Guidelines:\n{clean_style}\n"

    available_for_code = max(100, available_for_other - len(style_text))

    # 4. Code changes: prioritize relevant source files and targeted symbol snippets
    relevant_files = []
    for src in req.source_files:
        old_snip = req.old_code_snippets.get(src, "")
        new_snip = req.new_code_snippets.get(src, "")
        is_relevant = any(sym in old_snip or sym in new_snip for sym in req.changed_symbols) or any(
            ev.matched_text in old_snip or ev.matched_text in new_snip for ev in req.impact_evidence
        )
        if is_relevant:
            relevant_files.append(src)

    if not relevant_files:
        relevant_files = req.source_files[:10]

    code_parts = ["\n### Code Changes:"]
    budget_per_file = max(300, (available_for_code - 100) // max(1, len(relevant_files)))
    for src in relevant_files:
        raw_old = req.old_code_snippets.get(src, "").strip()
        raw_new = req.new_code_snippets.get(src, "").strip()

        old_snip = extract_relevant_code_snippet(raw_old, req.changed_symbols, budget_per_file // 2)
        new_snip = extract_relevant_code_snippet(raw_new, req.changed_symbols, budget_per_file // 2)

        old_snip = redact_secrets(old_snip)
        new_snip = redact_secrets(new_snip)

        code_parts.append(f"#### File: `{src}`")
        if old_snip:
            code_parts.append(
                f'<untrusted_source_code file="{src}" version="previous">\n{old_snip}\n</untrusted_source_code>'
            )
        if new_snip:
            code_parts.append(
                f'<untrusted_source_code file="{src}" version="updated">\n{new_snip}\n</untrusted_source_code>'
            )

    code_text = "\n".join(code_parts)
    full_prompt = f"{header_text}\n{code_text}\n{target_sec_text}{style_text}"

    # Strict final enforcement of actual serialized HTTP request body size (Priority 7)
    current_req_size = calculate_serialized_request_size(model, _SYSTEM_PROMPT, full_prompt)
    if current_req_size > max_context_chars:
        # Trim code_text until serialized body fits within max_context_chars
        excess = current_req_size - max_context_chars
        trim_len = len(code_text) - excess - 150
        if trim_len > 100:
            code_text = code_text[:trim_len] + "\n# [TRUNCATED CODE EVIDENCE]"
            full_prompt = f"{header_text}\n{code_text}\n{target_sec_text}{style_text}"
        else:
            # Code cannot fit, omit code
            full_prompt = f"{header_text}\n{target_sec_text}{style_text}"

        # Final verification
        final_size = calculate_serialized_request_size(model, _SYSTEM_PROMPT, full_prompt)
        if final_size > max_context_chars:
            raise ContextBudgetExceededError(
                f"Request serialized size ({final_size:,} bytes) exceeds context budget ({max_context_chars:,} bytes)."
            )

    return full_prompt

    return full_prompt


def parse_ai_response(
    raw_response: str, expected_doc_path: str, expected_section_id: str
) -> AISectionUpdateResponse:
    """
    Parse and strictly validate JSON response from the AI model (C4).

    Rejects or flags for human review on null content, non-string types,
    or structural format discrepancies.
    """
    if not raw_response or not raw_response.strip():
        return AISectionUpdateResponse(
            doc_path=expected_doc_path,
            section_id=expected_section_id,
            proposed_content="",
            explanation="Empty response from AI model",
            requires_human_review=True,
            review_reason="AI returned an empty response.",
        )

    text = raw_response.strip()

    # Strip code block wrapping if model wrapped JSON in ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return AISectionUpdateResponse(
            doc_path=expected_doc_path,
            section_id=expected_section_id,
            proposed_content=raw_response.strip(),
            explanation=f"AI produced unstructured text: {exc}",
            requires_human_review=True,
            review_reason=f"AI response could not be parsed as structured JSON ({exc}); flagged for review.",
        )

    if not isinstance(data, dict):
        return AISectionUpdateResponse(
            doc_path=expected_doc_path,
            section_id=expected_section_id,
            proposed_content="",
            explanation="AI response root is not a JSON object",
            requires_human_review=True,
            review_reason=f"Expected JSON object, got {type(data).__name__}.",
        )

    # Strict type verification (C4)
    raw_content = data.get("proposed_content")
    if raw_content is None or not isinstance(raw_content, str):
        return AISectionUpdateResponse(
            doc_path=expected_doc_path,
            section_id=expected_section_id,
            proposed_content="",
            explanation="Invalid proposed_content type",
            requires_human_review=True,
            review_reason=(f"proposed_content must be a string, got {type(raw_content).__name__}."),
        )

    doc_path = data.get("doc_path", expected_doc_path)
    if not isinstance(doc_path, str):
        doc_path = expected_doc_path

    section_id = data.get("section_id", expected_section_id)
    if not isinstance(section_id, str):
        section_id = expected_section_id

    explanation = data.get("explanation", "")
    if not isinstance(explanation, str):
        explanation = str(explanation)

    raw_evidence = data.get("technical_evidence", [])
    if isinstance(raw_evidence, list):
        technical_evidence = [str(x) for x in raw_evidence if isinstance(x, str | int | float)]
    else:
        technical_evidence = []

    raw_uncertainties = data.get("uncertainties", [])
    if isinstance(raw_uncertainties, list):
        uncertainties = [str(x) for x in raw_uncertainties if isinstance(x, str | int | float)]
    else:
        uncertainties = []

    requires_human_review = bool(data.get("requires_human_review", False))
    review_reason = data.get("review_reason", "")
    if not isinstance(review_reason, str):
        review_reason = str(review_reason)

    if doc_path != expected_doc_path or section_id != expected_section_id:
        requires_human_review = True
        review_reason = (
            f"AI returned target mismatch: doc='{doc_path}', sec='{section_id}' "
            f"(expected doc='{expected_doc_path}', sec='{expected_section_id}')."
        )

    return AISectionUpdateResponse(
        doc_path=doc_path,
        section_id=section_id,
        proposed_content=raw_content,
        explanation=explanation,
        technical_evidence=technical_evidence,
        uncertainties=uncertainties,
        requires_human_review=requires_human_review,
        review_reason=review_reason,
    )


# =============================================================================
# DETERMINISTIC VALIDATOR
# =============================================================================


class _CodeFactExtractor(ast.NodeVisitor):
    """Walks Python AST to extract literals, function details, classes, type aliases, and environment variables."""

    def __init__(self):
        self.numeric_literals: set[str] = set()
        self.env_vars: set[str] = set()
        self.env_defaults: dict[str, str] = {}
        self.func_params: dict[str, set[str]] = {}
        self.func_param_details: dict[str, dict[str, ParameterInfo]] = {}
        self.func_returns: dict[str, str] = {}
        self.constants: dict[str, str] = {}
        self.type_aliases: dict[str, str] = {}
        self.classes: dict[str, list[str]] = {}
        self.class_methods: dict[str, set[str]] = {}
        self.current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)
        self.classes[node.name] = bases
        self.class_methods[node.name] = set()
        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, int | float):
            self.numeric_literals.add(str(node.value))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_func(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_func(node)
        self.generic_visit(node)

    def _record_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        all_args = node.args.args + node.args.kwonlyargs + getattr(node.args, "posonlyargs", [])
        self.func_params[node.name] = {a.arg for a in all_args}
        if self.current_class:
            self.class_methods.setdefault(self.current_class, set()).add(node.name)
            self.func_params[f"{self.current_class}.{node.name}"] = {a.arg for a in all_args}

        param_dict = {}
        defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + list(
            node.args.defaults
        )
        for a, d in zip(node.args.args, defaults, strict=False):
            type_ann = ast.unparse(a.annotation).strip() if a.annotation else None
            def_val = ast.unparse(d).strip().strip("'\"") if d else None
            param_dict[a.arg] = ParameterInfo(
                name=a.arg, type_annotation=type_ann, default_value=def_val
            )
        for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=False):
            type_ann = ast.unparse(a.annotation).strip() if a.annotation else None
            def_val = ast.unparse(d).strip().strip("'\"") if d else None
            param_dict[a.arg] = ParameterInfo(
                name=a.arg, type_annotation=type_ann, default_value=def_val
            )

        self.func_param_details[node.name] = param_dict
        if self.current_class:
            self.func_param_details[f"{self.current_class}.{node.name}"] = param_dict

        if node.returns:
            try:
                ret_str = ast.unparse(node.returns).strip()
                self.func_returns[node.name] = ret_str
                if self.current_class:
                    self.func_returns[f"{self.current_class}.{node.name}"] = ret_str
            except Exception:
                pass

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                try:
                    val_str = ast.unparse(node.value).strip().strip("'\"")
                    self.constants[target.id] = val_str
                    # Type alias detection (e.g. UserId = int, Payload = dict[str, Any])
                    if isinstance(node.value, ast.Name):
                        self.type_aliases[target.id] = node.value.id
                    elif isinstance(node.value, ast.Subscript):
                        self.type_aliases[target.id] = ast.unparse(node.value).strip()
                except Exception:
                    pass
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value:
            try:
                val_str = ast.unparse(node.value).strip().strip("'\"")
                self.constants[node.target.id] = val_str
                ann_str = ast.unparse(node.annotation).strip()
                if "TypeAlias" in ann_str or isinstance(node.value, ast.Name | ast.Subscript):
                    self.type_aliases[node.target.id] = ast.unparse(node.value).strip()
            except Exception:
                pass
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (isinstance(node.value, ast.Attribute) and node.value.attr == "environ") or (
            isinstance(node.value, ast.Name) and node.value.id == "environ"
        ):
            slice_node = getattr(node.slice, "value", node.slice)
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                self.env_vars.add(slice_node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("getenv", "get"):
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                var_name = node.args[0].value
                self.env_vars.add(var_name)
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    self.env_defaults[var_name] = str(node.args[1].value)
        self.generic_visit(node)


class DocumentationValidator:
    """Performs deterministic validation on AI-generated documentation updates."""

    _STANDARD_ALLOWED_TOKENS = {
        "str",
        "int",
        "bool",
        "float",
        "dict",
        "list",
        "tuple",
        "set",
        "bytes",
        "None",
        "Optional",
        "Union",
        "Any",
        "Callable",
        "Iterable",
        "Sequence",
        "Mapping",
        "true",
        "false",
        "null",
        "self",
        "cls",
        "args",
        "kwargs",
        "return",
        "yield",
        "raise",
        "try",
        "except",
        "finally",
        "with",
        "as",
        "import",
        "from",
        "def",
        "class",
        "pass",
        "lambda",
        "async",
        "await",
    }

    @staticmethod
    def validate(
        req: SectionUpdateRequest,
        resp: AISectionUpdateResponse,
        parsed_doc: ParsedDocument,
    ) -> ValidationResult:
        """Run strict validation checks on proposed section content."""
        passed: list[str] = []
        failed: list[str] = []
        warnings: list[str] = []
        review_reasons: list[str] = []

        # 1. Target existence check
        sec = parsed_doc.find_section(req.section_id)
        if sec is None:
            failed.append(f"target_section_not_found: '{req.section_id}'")
        else:
            passed.append("target_section_exists")

        # 2. Review flag check
        if resp.requires_human_review:
            review_reasons.append(resp.review_reason or "AI requested human review.")

        if resp.uncertainties:
            warnings.append(f"AI reported uncertainties: {', '.join(resp.uncertainties)}")
            review_reasons.append(f"Uncertainties noted: {', '.join(resp.uncertainties)}")

        # 3. Empty content check
        clean_prop = resp.proposed_content.strip()
        clean_orig = req.current_section_content.strip()
        if not clean_prop and clean_orig:
            warnings.append("proposed_content_is_empty")
            review_reasons.append("Proposed content is empty while original section had text.")
        else:
            passed.append("non_empty_content")

        # 4. Markdown Structure Safety (C2)
        backtick_fence_count = len(re.findall(r"(?m)^[ \t]*```", resp.proposed_content))
        tilde_fence_count = len(re.findall(r"(?m)^[ \t]*~~~", resp.proposed_content))
        if backtick_fence_count % 2 != 0 or tilde_fence_count % 2 != 0:
            failed.append("unbalanced_code_fences")
        else:
            passed.append("balanced_code_fences")

        orig_has_headings = bool(
            re.search(r"(?m)^[ \t]*#{1,6}\s+\S+", clean_orig)
            or re.search(r"(?m)^[^\n]+\n[ \t]*(=+|-+)[ \t]*$", clean_orig)
        )
        prop_has_atx = bool(re.search(r"(?m)^[ \t]*#{1,6}\s+\S+", resp.proposed_content))
        prop_has_setext = bool(
            re.search(
                r"(?m)^[A-Za-z0-9_`*][^\n]*\n[ \t]*(=+|-{2,})[ \t]*$",
                resp.proposed_content,
            )
        )
        if not orig_has_headings and (prop_has_atx or prop_has_setext):
            failed.append("headings_introduced_in_leaf_section")
            warnings.append("subheadings_introduced_in_leaf_section")
        else:
            passed.append("no_leaf_heading_injection")

        # 5. Extract Structured Code Facts from AST
        valid_symbols: set[str] = set()
        valid_params: dict[str, ParameterInfo] = {}
        valid_constants: dict[str, ConstantInfo] = {}
        valid_imports: set[str] = set()
        fact_extractor = _CodeFactExtractor()

        for src in req.source_files:
            code = req.new_code_snippets.get(src, "")
            if code:
                try:
                    tree = ast.parse(code)
                    fact_extractor.visit(tree)
                except SyntaxError:
                    pass

                sym_map = extract_python_symbols(code)
                for s in sym_map.values():
                    valid_symbols.add(s.name)
                    valid_symbols.add(s.name.split(".")[-1])
                    for p in s.parameters:
                        valid_params[p.name] = p

                const_map = extract_python_constants(code)
                for c in const_map.values():
                    valid_constants[c.name] = c
                    valid_constants[c.name.split(".")[-1]] = c

                imp_list = extract_python_imports(code)
                for imp in imp_list:
                    valid_imports.add(imp.alias if imp.alias else imp.name)

            delta = req.ast_deltas.get(src)
            if delta:
                for s in delta.added_symbols:
                    valid_symbols.add(s.name)
                    valid_symbols.add(s.name.split(".")[-1])
                    for p in s.parameters:
                        valid_params[p.name] = p
                for _old_s, new_s in delta.modified_symbols:
                    valid_symbols.add(new_s.name)
                    valid_symbols.add(new_s.name.split(".")[-1])
                    for p in new_s.parameters:
                        valid_params[p.name] = p
                for _fname, p in delta.added_parameters:
                    valid_params[p.name] = p
                for _fname, _old_p, new_p in delta.modified_parameters:
                    valid_params[new_p.name] = new_p
                for c in delta.added_constants:
                    valid_constants[c.name] = c
                    valid_constants[c.name.split(".")[-1]] = c
                for _old_c, new_c in delta.modified_constants:
                    valid_constants[new_c.name] = new_c
                    valid_constants[new_c.name.split(".")[-1]] = new_c

        # 6. Verify Technical Tokens & Factual Accuracy (C1)
        new_tokens = DocumentationValidator._extract_new_technical_tokens(clean_orig, clean_prop)
        hallucinated_tokens: list[str] = []

        allowed_all = (
            valid_symbols
            | set(valid_params.keys())
            | set(valid_constants.keys())
            | valid_imports
            | DocumentationValidator._STANDARD_ALLOWED_TOKENS
            | set(req.changed_symbols)
        )

        for token in new_tokens:
            is_valid = token in allowed_all or len(token) <= 2
            if not is_valid:
                in_evidence = any(token == ev.matched_text for ev in req.impact_evidence)
                if not in_evidence:
                    hallucinated_tokens.append(token)

        if hallucinated_tokens:
            failed.append(f"unsupported_technical_tokens: {sorted(hallucinated_tokens)}")
        else:
            passed.append("no_hallucinated_technical_symbols")

        # 7. Function Call Keyword Arguments Validation (C1 Case 3)
        calls_found = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", clean_prop)
        for func_name, args_text in calls_found:
            if func_name in fact_extractor.func_params:
                valid_kw_set = fact_extractor.func_params[func_name]
                kw_args = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=", args_text)
                for kw in kw_args:
                    if kw not in valid_kw_set:
                        failed.append(
                            f"unknown_parameter_in_call: function '{func_name}' has no parameter '{kw}' (valid parameters: {sorted(valid_kw_set)})"
                        )

        # 8. Prose Parameter / Option References (C1 Case 2)
        opt_matches = re.findall(
            r"(?:supports|accepts|with|has)\s+(?:the\s+)?\`?([A-Za-z_][A-Za-z0-9_]+)\`?\s+(?:option|parameter|argument|flag)\b",
            clean_prop,
            re.IGNORECASE,
        )
        opt_matches += re.findall(
            r"\b\`?([A-Za-z_][A-Za-z0-9_]+)\`?\s+(?:option|parameter|argument|flag)\b",
            clean_prop,
            re.IGNORECASE,
        )
        for opt in opt_matches:
            if (
                opt not in valid_params
                and opt not in valid_constants
                and opt not in fact_extractor.constants
                and opt not in req.changed_symbols
                and opt not in DocumentationValidator._STANDARD_ALLOWED_TOKENS
            ):
                failed.append(
                    f"unsupported_option_or_parameter: option or parameter '{opt}' does not exist in source code"
                )

        # 9. Parameter Default Value & Type Verification (C1 Case 1 & Priority 3)
        for p_name, p_info in valid_params.items():
            if p_info.default_value is not None:
                actual_val = p_info.default_value.strip().strip("'\"")

                eq_matches = re.findall(
                    rf"\b{re.escape(p_name)}\s*=\s*`?([0-9]+(?:\.[0-9]+)?|[a-zA-Z_][0-9a-zA-Z_]*)`?",
                    clean_prop,
                )
                for claimed in eq_matches:
                    clean_claimed = claimed.strip().rstrip(".").strip("'\"")
                    if clean_claimed != "" and not values_are_equivalent(clean_claimed, actual_val):
                        failed.append(
                            f"parameter_default_mismatch: parameter '{p_name}' default is '{actual_val}', but documentation claims '{clean_claimed}'"
                        )

                def_matches = re.findall(
                    rf"\b{re.escape(p_name)}\b[^\.\n]*?(?:defaults?\s*(?:is|=|to|:)|default:|\(default\s+)\s*`?([0-9]+(?:\.[0-9]+)?|[a-zA-Z_][0-9a-zA-Z_]*)`?",
                    clean_prop,
                    re.IGNORECASE,
                )
                for claimed in def_matches:
                    clean_claimed = claimed.strip().rstrip(".").strip("'\"")
                    if clean_claimed != "" and not values_are_equivalent(clean_claimed, actual_val):
                        failed.append(
                            f"parameter_default_mismatch: parameter '{p_name}' default is '{actual_val}', but documentation claims '{clean_claimed}'"
                        )

            if p_info.type_annotation is not None:
                actual_type = p_info.type_annotation.strip()
                known_types = {"str", "int", "bool", "float", "list", "dict", "bytes"}

                type_matches = re.findall(
                    rf"\b{re.escape(p_name)}\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", clean_prop
                )
                type_matches += re.findall(
                    rf"\b{re.escape(p_name)}\s*\(([A-Za-z_][A-Za-z0-9_]*)[,\)]",
                    clean_prop,
                )
                for claimed_t in type_matches:
                    c_type = claimed_t.strip()
                    # Type alias resolution (Priority 3)
                    alias_actual = fact_extractor.type_aliases.get(actual_type, actual_type)
                    alias_claimed = fact_extractor.type_aliases.get(c_type, c_type)
                    matches = (
                        c_type == actual_type
                        or alias_actual == c_type
                        or alias_claimed == actual_type
                        or alias_actual == alias_claimed
                    )
                    if not matches:
                        if (c_type in known_types or alias_claimed in known_types) and (
                            actual_type in known_types or alias_actual in known_types
                        ):
                            failed.append(
                                f"parameter_type_mismatch: parameter '{p_name}' type is '{actual_type}', but documentation claims '{c_type}'"
                            )
                        else:
                            review_reasons.append(
                                f"Unverifiable parameter type: '{p_name}' is documented as '{c_type}' (source: '{actual_type}'); requires human review."
                            )

        # 9b. Constant Value Verification (Priority 2 & Priority 3)
        for c_name, c_info in valid_constants.items():
            if c_info.value is not None:
                actual_val = c_info.value.strip().strip("'\"")
                c_matches = re.findall(
                    rf"\b{re.escape(c_name)}\s*=\s*([0-9a-zA-Z_.'\"]+)", clean_prop
                )
                c_matches += re.findall(
                    rf"\b{re.escape(c_name)}\b[^\.\n]*?(?:default\s*)?(?:is|=|to|:)\s*`?([0-9a-zA-Z_.'\"]+)`?",
                    clean_prop,
                    re.IGNORECASE,
                )
                for claimed in c_matches:
                    clean_claimed = claimed.strip().strip("'\"")
                    if clean_claimed != "" and not values_are_equivalent(clean_claimed, actual_val):
                        failed.append(
                            f"constant_value_mismatch: constant '{c_name}' value is '{actual_val}', but documentation claims '{clean_claimed}'"
                        )

        # 10. Entity-Bound Numeric Claims and Limits Verification (Priority 3)
        # Timeout claims: must strictly match a timeout parameter or constant
        timeout_claims = re.findall(
            r"\b(?:timeout|TIMEOUT)\b[^\.\n]*?(?:of|is|=|:)\s*`?([0-9]+(?:\.[0-9]+)?)`?",
            clean_prop,
            re.IGNORECASE,
        )
        for t_val in timeout_claims:
            matching_entity = False
            contradicting_val = None
            for p_name, p_info in valid_params.items():
                if "timeout" in p_name.lower() and p_info.default_value:
                    if values_are_equivalent(t_val, p_info.default_value):
                        matching_entity = True
                        break
                    else:
                        contradicting_val = p_info.default_value
            if not matching_entity:
                for c_name, c_info in valid_constants.items():
                    if "timeout" in c_name.lower() and c_info.value:
                        if values_are_equivalent(t_val, c_info.value):
                            matching_entity = True
                            break
                        else:
                            contradicting_val = c_info.value
            if not matching_entity:
                if contradicting_val is not None:
                    failed.append(
                        f"timeout_mismatch: source timeout is '{contradicting_val}', but documentation claims '{t_val}' (SLEEP or other constants cannot authorize timeout)"
                    )
                else:
                    review_reasons.append(
                        f"Unverifiable timeout claim: timeout '{t_val}' does not match any source timeout parameter or constant."
                    )

        # Rate limit claims: PAGE_SIZE or other constants must not authorize rate limit
        rate_claims = re.findall(
            r"(?:rate\s*limit(?:ed)?(?:\s*to)?|requests\s*per\s*(?:second|minute|hour)|rps|rpm)\s*(?:of|is|=|:|to)?\s*`?([0-9]+(?:\.[0-9]+)?)\b`?",
            clean_prop,
            re.IGNORECASE,
        )
        for r_val in rate_claims:
            matching_entity = False
            for p_name, p_info in valid_params.items():
                if (
                    any(
                        k in p_name.lower()
                        for k in ["rate", "rps", "rpm", "max_req", "request_limit"]
                    )
                    and p_info.default_value
                    and values_are_equivalent(r_val, p_info.default_value)
                ):
                    matching_entity = True
                    break
            if not matching_entity:
                for c_name, c_info in valid_constants.items():
                    if (
                        any(
                            k in c_name.lower()
                            for k in ["rate", "rps", "rpm", "max_req", "request_limit"]
                        )
                        and c_info.value
                        and values_are_equivalent(r_val, c_info.value)
                    ):
                        matching_entity = True
                        break
            if not matching_entity:
                failed.append(
                    f"unsupported_rate_limit: rate limit of '{r_val}' claimed in documentation, but no rate limit entity exists in source (PAGE_SIZE or other constants do not authorize rate limits)"
                )

        # Retry delay claims: SLEEP must not authorize RETRY_DELAY
        retry_delay_claims = re.findall(
            r"\b(?:retry_delay|backoff|retry\s*interval)\b[^\.\n]*?(?:of|is|=|:)\s*`?([0-9]+(?:\.[0-9]+)?)`?",
            clean_prop,
            re.IGNORECASE,
        )
        for rd_val in retry_delay_claims:
            matching_entity = False
            contradicting_val = None
            for p_name, p_info in valid_params.items():
                if (
                    any(k in p_name.lower() for k in ["retry_delay", "backoff", "retry_interval"])
                    and p_info.default_value
                ):
                    if values_are_equivalent(rd_val, p_info.default_value):
                        matching_entity = True
                        break
                    else:
                        contradicting_val = p_info.default_value
            if not matching_entity:
                for c_name, c_info in valid_constants.items():
                    if (
                        any(
                            k in c_name.lower() for k in ["retry_delay", "backoff", "retry_interval"]
                        )
                        and c_info.value
                    ):
                        if values_are_equivalent(rd_val, c_info.value):
                            matching_entity = True
                            break
                        else:
                            contradicting_val = c_info.value
            if not matching_entity:
                failed.append(
                    f"unsupported_retry_delay: retry delay '{rd_val}' claimed in documentation does not match retry delay parameters or constants (SLEEP does not authorize RETRY_DELAY)"
                )

        # Retry count claims
        retry_claims = re.findall(
            r"\b(?:retries|max_retries)\b[^\.\n]*?(?:of|is|=|:)\s*`?([0-9]+(?:\.[0-9]+)?)`?",
            clean_prop,
            re.IGNORECASE,
        )
        for r_val in retry_claims:
            matching_entity = False
            for p_name, p_info in valid_params.items():
                if (
                    "retr" in p_name.lower()
                    and p_info.default_value
                    and values_are_equivalent(r_val, p_info.default_value)
                ):
                    matching_entity = True
                    break
            if not matching_entity:
                for c_name, c_info in valid_constants.items():
                    if (
                        "retr" in c_name.lower()
                        and c_info.value
                        and values_are_equivalent(r_val, c_info.value)
                    ):
                        matching_entity = True
                        break
            if not matching_entity:
                failed.append(
                    f"unsupported_numeric_claim: retries value '{r_val}' does not match retry constants or parameters"
                )

        # 11. Return Type Verification (C1 Case 5 & Priority 3)
        for f_name, ret_type in fact_extractor.func_returns.items():
            if ret_type:
                ret_matches = re.findall(
                    rf"\b{re.escape(f_name)}\b[^\.\n]*?(?:returns|returning|return value is)\s+(?:a\s+)?([A-Za-z_][A-Za-z0-9_\[\], ]*)",
                    clean_prop,
                    re.IGNORECASE,
                )
                for claimed_ret in ret_matches:
                    c_ret = claimed_ret.strip().strip("'\"")
                    alias_actual = fact_extractor.type_aliases.get(ret_type, ret_type)
                    alias_claimed = fact_extractor.type_aliases.get(c_ret, c_ret)
                    matches = (
                        c_ret == ret_type
                        or alias_actual == c_ret
                        or alias_claimed == ret_type
                        or alias_actual == alias_claimed
                        or c_ret.lower() in ret_type.lower()
                        or ret_type.lower() in c_ret.lower()
                    )
                    if not matches:
                        failed.append(
                            f"return_type_mismatch: function '{f_name}' returns '{ret_type}', but documentation claims '{claimed_ret.strip()}'"
                        )

        # 12. Environment Variables Verification (C1 Case 6 & Priority 3)
        env_candidates = re.findall(
            r"\b([A-Z][A-Z0-9_]{2,}(?:_KEY|_TOKEN|_URL|_PORT|_TIMEOUT|_CONFIG|_ENV|_DIR|_PATH|_SECRET|_ID|_HOST))\b",
            clean_prop,
        )
        for env_name in env_candidates:
            if (
                env_name not in fact_extractor.env_vars
                and env_name not in valid_constants
                and env_name not in fact_extractor.constants
                and env_name not in req.changed_symbols
                and env_name not in clean_orig
            ):
                # Standard external environment variables route to human review
                if env_name in {"PATH", "HOME", "PYTHONPATH", "CI", "AWS_REGION", "GITHUB_TOKEN"}:
                    review_reasons.append(
                        f"External environment variable '{env_name}' referenced; requires human review."
                    )
                else:
                    failed.append(
                        f"unsupported_environment_variable: '{env_name}' is mentioned in documentation but is not defined or accessed in source code"
                    )

        # 13. Removed Symbols & Negated/Historical Notices (Priority 3 & Priority 5)
        removed_symbols: list[SymbolInfo] = []
        for delta in req.ast_deltas.values():
            removed_symbols.extend(delta.removed_symbols)

        for sym in removed_symbols:
            short_name = sym.name.split(".")[-1]
            if re.search(rf"\b{re.escape(short_name)}\b", clean_prop):
                clauses = re.split(r"[.\n;]", clean_prop)
                for clause in clauses:
                    if re.search(rf"\b{re.escape(short_name)}\b", clause):
                        cl = clause.strip()
                        cl_lower = cl.lower()

                        # Explicitly negated notices or deprecation statements are VALID (Priority 3)
                        negated_phrases = [
                            "do not call",
                            "do not use",
                            "do not invoke",
                            "never call",
                            "never use",
                            "never invoke",
                            "avoid calling",
                            "avoid using",
                            "must not call",
                            "must not use",
                            "should not call",
                            "should not use",
                            "no longer call",
                            "no longer use",
                            "no longer supported",
                            "removed",
                            "deleted",
                            "dropped",
                            "obsolete",
                            "deprecated",
                            "replace",
                            "replaced",
                        ]
                        is_negated_or_notice = any(
                            re.search(rf"\b{re.escape(np)}\b", cl_lower) for np in negated_phrases
                        )

                        if is_negated_or_notice:
                            # Valid historical / warning statement!
                            continue

                        # If actively called or instructed without negation -> fail!
                        is_active_call = bool(re.search(rf"\b{re.escape(short_name)}\s*\(", cl))
                        active_verbs = ["call", "use", "invoke", "run", "execute", "import"]
                        has_active_verb = any(
                            re.search(rf"\b{v}\b.*?\b{re.escape(short_name)}\b", cl_lower)
                            for v in active_verbs
                        )

                        if is_active_call or has_active_verb:
                            failed.append(
                                f"active_reference_to_removed_symbol: '{short_name}' was removed from source code but is actively instructed or used in '{cl}'"
                            )
                            break
                        else:
                            failed.append(
                                f"unsupported_reference_to_removed_symbol: '{short_name}' was removed; references must be explicit removal, deprecation, or replacement notices."
                            )
                            break

        is_valid = len(failed) == 0
        needs_review = len(review_reasons) > 0 or not is_valid

        return ValidationResult(
            is_valid=is_valid,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
            requires_human_review=needs_review,
            review_reasons=review_reasons,
        )

    @staticmethod
    def _extract_new_technical_tokens(original_text: str, proposed_text: str) -> set[str]:
        """Extract code identifiers present in proposed text but absent in original text."""
        orig_backticks = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", original_text))
        prop_backticks = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", proposed_text))

        orig_calls = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*\(", original_text))
        prop_calls = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*\(", proposed_text))

        new_backticks = prop_backticks - orig_backticks
        new_calls = prop_calls - orig_calls

        tokens = set()
        for t in new_backticks | new_calls:
            tokens.add(t)
            if "." in t:
                tokens.add(t.split(".")[-1])

        return tokens


# =============================================================================
# DOCUMENTATION UPDATING ENGINE
# =============================================================================


class DocumentationUpdater:
    """
    Main engine orchestrating code diff analysis, impact mapping,
    AI generation, validation, and surgical Markdown section updating.
    """

    def __init__(
        self,
        ai_generator: Callable[[str, str], str] | None = None,
        style_guidelines: str = "",
    ):
        """
        Initialize the DocumentationUpdater.
        """
        self.ai_generator = ai_generator or self._default_ai_generator
        self.style_guidelines = style_guidelines
        self.mapper = DocumentationImpactMapper()

    def update_documentation(
        self,
        code_changes: dict[str, tuple[str, str]],
        documents: dict[str, str],
        precomputed_impacts: list[SectionImpact] | None = None,
    ) -> DocumentationUpdateReport:
        """
        Run the end-to-end documentation update pipeline.

        - code_changes: mapping of source_file_path -> (old_code, new_code).
        - documents: mapping of doc_path -> original_markdown_text.
        - precomputed_impacts: optional pre-calculated impact records.
        """
        report = DocumentationUpdateReport(
            updated_documents=dict(documents),
        )

        # 1. Compute AST deltas & check syntax errors upfront (H7)
        deltas: dict[str, ASTDelta] = {}
        for src_path, (old_src, new_src) in code_changes.items():
            delta = compute_ast_delta(old_src, new_src, file_path=src_path)
            deltas[src_path] = delta
            if delta.syntax_error:
                report.source_analysis_errors.append((src_path, delta.syntax_error))
                applied_rec = AppliedSectionUpdate(
                    doc_path=src_path,
                    section_id="syntax_error",
                    section_heading="Syntax Error",
                    status="source_analysis_failed",
                    original_section_content="",
                    new_section_content="",
                    explanation=f"Syntax error in source file {src_path}: {delta.syntax_error}",
                    validation=ValidationResult(
                        is_valid=False,
                        failed_checks=[f"syntax_error_in_source: {delta.syntax_error}"],
                    ),
                )
                report.rejected_updates.append(applied_rec)
                report.total_sections_rejected += 1

        # 2. Impact Mapping: Map changes to documentation sections
        parsed_docs: dict[str, ParsedDocument] = {
            doc_path: parse_sections(text) for doc_path, text in documents.items()
        }

        if precomputed_impacts is not None:
            raw_impacts = precomputed_impacts
        else:
            raw_impacts = []
            for src_path, (old_src, new_src) in code_changes.items():
                if deltas.get(src_path) and deltas[src_path].syntax_error:
                    continue
                file_impacts = self.mapper.map_file_impacts(old_src, new_src, src_path, parsed_docs)
                raw_impacts.extend(file_impacts)

        # Filter weak, non-deterministic impacts (H5: Over-targeting)
        filtered_impacts = [
            imp for imp in raw_impacts if imp.is_deterministic or imp.confidence >= 0.8
        ]

        consolidated = consolidate_impacts(filtered_impacts)
        report.total_sections_analyzed = len(consolidated)

        if not consolidated:
            return report

        # Collect removed symbols for unchanged doc checks (Priority 5, C3)
        removed_symbols: list[SymbolInfo] = []
        for delta in deltas.values():
            removed_symbols.extend(delta.removed_symbols)

        # 3. Process each affected section
        pending_section_patches: list[tuple[str, str, str, str, str, ValidationResult]] = []
        # (doc_path, section_id, section_heading, orig_content, proposed_content, validation)

        for target in consolidated:
            doc_path = target.doc_path
            current_markdown = report.updated_documents.get(doc_path, "")
            parsed_doc = parse_sections(current_markdown)

            sec = parsed_doc.find_section(target.section_id)
            if sec is None:
                applied_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=target.section_id,
                    section_heading=target.section_heading,
                    status="rejected",
                    original_section_content="",
                    new_section_content="",
                    explanation="Target section not found in document.",
                    validation=ValidationResult(
                        is_valid=False,
                        failed_checks=[f"section_not_found: {target.section_id}"],
                    ),
                )
                report.rejected_updates.append(applied_rec)
                report.total_sections_rejected += 1
                continue

            current_section_content = parsed_doc.get_section_content(sec, include_children=False)

            # Check for source syntax errors upfront: do not call AI
            syntax_errors = [
                deltas[src].syntax_error
                for src in target.source_files
                if src in deltas and deltas[src].syntax_error
            ]
            if syntax_errors:
                applied_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=target.section_id,
                    section_heading=target.section_heading,
                    status="rejected",
                    original_section_content=current_section_content,
                    new_section_content="",
                    explanation=f"Source syntax error: {syntax_errors[0]}",
                    validation=ValidationResult(
                        is_valid=False,
                        failed_checks=[f"syntax_error: {syntax_errors[0]}"],
                    ),
                )
                report.rejected_updates.append(applied_rec)
                report.total_sections_rejected += 1
                continue

            # Check if unchanged doc has active removed symbols (Priority 5, C3)
            active_removed_in_orig = [
                sym.name
                for sym in removed_symbols
                if re.search(
                    rf"\b{re.escape(sym.name.split('.')[-1])}\b",
                    current_section_content,
                )
            ]

            # Build Request Context
            old_snips = {
                src: code_changes[src][0] for src in target.source_files if src in code_changes
            }
            new_snips = {
                src: code_changes[src][1] for src in target.source_files if src in code_changes
            }

            req = SectionUpdateRequest(
                doc_path=doc_path,
                section_id=target.section_id,
                section_heading=target.section_heading,
                current_section_content=current_section_content,
                source_files=target.source_files,
                changed_symbols=target.changed_symbols,
                change_categories=target.change_categories,
                impact_reasons=target.reasons,
                impact_evidence=target.evidence,
                old_code_snippets=old_snips,
                new_code_snippets=new_snips,
                ast_deltas=deltas,
                style_guidelines=self.style_guidelines,
            )

            # 4. Generate Update via AI
            try:
                user_prompt = build_update_prompt(req)
                raw_ai_out = self.ai_generator(_SYSTEM_PROMPT, user_prompt)
                ai_resp = parse_ai_response(raw_ai_out, doc_path, target.section_id)
                cand_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=target.section_id,
                    section_heading=target.section_heading,
                    status="candidate",
                    original_section_content=current_section_content,
                    new_section_content=ai_resp.proposed_content,
                    explanation=ai_resp.explanation or "Generated candidate update.",
                    validation=ValidationResult(is_valid=True),
                )
                report.candidate_updates.append(cand_rec)
            except ContextBudgetExceededError as budget_err:
                applied_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=target.section_id,
                    section_heading=target.section_heading,
                    status="review_required",
                    original_section_content=current_section_content,
                    new_section_content="",
                    explanation=f"Context budget exceeded: {budget_err}",
                    validation=ValidationResult(
                        is_valid=False,
                        requires_human_review=True,
                        review_reasons=[f"Context budget exceeded: {budget_err}"],
                    ),
                )
                report.review_required_updates.append(applied_rec)
                continue
            except Exception as gen_err:
                applied_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=target.section_id,
                    section_heading=target.section_heading,
                    status="generation_failed",
                    original_section_content=current_section_content,
                    new_section_content="",
                    explanation=f"AI generation failed: {gen_err}",
                    validation=ValidationResult(
                        is_valid=False,
                        failed_checks=[f"ai_generation_error: {gen_err}"],
                    ),
                )
                report.generation_failures.append(applied_rec)
                report.rejected_updates.append(applied_rec)
                report.total_sections_rejected += 1
                continue

            # 5. Deterministic Validation
            val_result = DocumentationValidator.validate(req, ai_resp, parsed_doc)

            # Idempotency / No-Change Check (Priority 5 & Priority 6, H4, Priority 7)
            already_documented = is_already_documented(current_section_content, req)
            if (
                ai_resp.proposed_content.strip() == current_section_content.strip()
                or already_documented
            ):
                if active_removed_in_orig:
                    # Stale removed symbol remains in unchanged doc! (C3)
                    applied_rec = AppliedSectionUpdate(
                        doc_path=doc_path,
                        section_id=target.section_id,
                        section_heading=target.section_heading,
                        status="review_required",
                        original_section_content=current_section_content,
                        new_section_content=ai_resp.proposed_content,
                        explanation=f"Removed symbol(s) {active_removed_in_orig} remain in active documentation instructions; manual review required.",
                        validation=ValidationResult(
                            is_valid=False,
                            requires_human_review=True,
                            review_reasons=[
                                f"Removed symbol(s) {active_removed_in_orig} remain in documentation."
                            ],
                        ),
                    )
                    report.review_required_updates.append(applied_rec)
                    continue

                applied_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=target.section_id,
                    section_heading=target.section_heading,
                    status="no_change",
                    original_section_content=current_section_content,
                    new_section_content=current_section_content,
                    explanation="Section content already factually reflects code changes.",
                    validation=val_result,
                )
                report.applied_updates.append(applied_rec)
                continue

            if ai_resp.requires_human_review:
                applied_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=target.section_id,
                    section_heading=target.section_heading,
                    status="review_required",
                    original_section_content=current_section_content,
                    new_section_content=ai_resp.proposed_content,
                    explanation=ai_resp.explanation or "Update requires human review.",
                    validation=val_result,
                )
                report.review_required_updates.append(applied_rec)
                continue

            if not val_result.is_valid:
                applied_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=target.section_id,
                    section_heading=target.section_heading,
                    status="rejected",
                    original_section_content=current_section_content,
                    new_section_content=ai_resp.proposed_content,
                    explanation=ai_resp.explanation or "Failed validation checks.",
                    validation=val_result,
                )
                report.rejected_updates.append(applied_rec)
                report.total_sections_rejected += 1
                continue

            if val_result.requires_human_review:
                applied_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=target.section_id,
                    section_heading=target.section_heading,
                    status="review_required",
                    original_section_content=current_section_content,
                    new_section_content=ai_resp.proposed_content,
                    explanation=ai_resp.explanation or "Update requires human review.",
                    validation=val_result,
                )
                report.review_required_updates.append(applied_rec)
                continue

            # Stage valid section patch for multi-document all-or-nothing commit
            validated_rec = AppliedSectionUpdate(
                doc_path=doc_path,
                section_id=target.section_id,
                section_heading=target.section_heading,
                status="validated",
                original_section_content=current_section_content,
                new_section_content=ai_resp.proposed_content,
                explanation=ai_resp.explanation or "Validated section update.",
                validation=val_result,
            )
            report.validated_updates.append(validated_rec)
            pending_section_patches.append(
                (
                    doc_path,
                    target.section_id,
                    target.section_heading,
                    current_section_content,
                    ai_resp.proposed_content,
                    val_result,
                )
            )

        # 6. Multi-Document All-or-Nothing Commit (Priority 3, N3, Priority 5)
        # If any generated section in the run was rejected by validation, abort the run!
        # Unrelated source analysis failures do NOT abort valid documentation updates for other files (Priority 5, M3).
        has_validation_rejection = any(
            u.status == "rejected"
            and not any("syntax_error" in f for f in u.validation.failed_checks)
            for u in report.rejected_updates
        )
        if has_validation_rejection or (
            report.total_sections_rejected > 0 and not pending_section_patches
        ):
            report.total_sections_applied = 0
            report.applied_updates = [u for u in report.applied_updates if u.status == "no_change"]
            report.updated_documents = dict(documents)
            return report

        # Apply staged section patches and verify document invariants
        for (
            doc_path,
            sec_id,
            sec_heading,
            orig_content,
            prop_content,
            val_res,
        ) in pending_section_patches:
            current_markdown = report.updated_documents[doc_path]
            parsed_doc = parse_sections(current_markdown)
            try:
                updated_markdown = replace_section_content(
                    current_markdown,
                    target=sec_id,
                    new_content=prop_content,
                    replace_children=False,
                )

                post_doc = parse_sections(updated_markdown)
                orig_sec_ids = {s.id for s in parsed_doc.sections}
                post_sec_ids = {s.id for s in post_doc.sections}
                if orig_sec_ids != post_sec_ids:
                    raise ValueError(
                        f"Document invariant violation: section IDs changed from {orig_sec_ids} to {post_sec_ids}"
                    )

                for s_orig in parsed_doc.sections:
                    if s_orig.id == sec_id:
                        continue
                    s_post = post_doc.find_section(s_orig.id)
                    if s_post is None:
                        raise ValueError(
                            f"Document invariant violation: section '{s_orig.id}' missing"
                        )
                    c_orig = parsed_doc.get_section_content(s_orig, include_children=False)
                    c_post = post_doc.get_section_content(s_post, include_children=False)
                    if c_orig != c_post:
                        raise ValueError(
                            f"Document invariant violation: non-target section '{s_orig.id}' was altered"
                        )

                report.updated_documents[doc_path] = updated_markdown
                if doc_path not in report.staged_updates:
                    report.staged_updates.append(doc_path)
                applied_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=sec_id,
                    section_heading=sec_heading,
                    status="applied",
                    original_section_content=orig_content,
                    new_section_content=prop_content,
                    explanation="Section updated successfully.",
                    validation=val_res,
                )
                report.applied_updates.append(applied_rec)
                report.total_sections_applied += 1

            except Exception as patch_err:
                # All-or-nothing rollback across all documents
                report.updated_documents = dict(documents)
                report.total_sections_applied = 0
                report.applied_updates = [
                    u for u in report.applied_updates if u.status == "no_change"
                ]
                applied_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=sec_id,
                    section_heading=sec_heading,
                    status="rejected",
                    original_section_content=orig_content,
                    new_section_content=prop_content,
                    explanation=f"Section patcher failed: {patch_err}",
                    validation=ValidationResult(
                        is_valid=False,
                        failed_checks=[f"patcher_exception: {patch_err}"],
                    ),
                )
                report.rejected_updates.append(applied_rec)
                report.total_sections_rejected += 1
                break

        return report

    def _default_ai_generator(self, system_prompt: str, user_prompt: str) -> str:
        """Default fallback completion caller using project's OpenAI-compatible client."""
        try:
            from config import get_client, get_model_name

            client = get_client()
            resp = client.chat.completions.create(
                model=get_model_name(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return json.dumps(
                {
                    "proposed_content": "",
                    "explanation": f"LLM call failed: {e}",
                    "requires_human_review": True,
                    "review_reason": f"AI completion unavailable: {e}",
                }
            )
