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
from security_utils import sanitize_output

# =============================================================================
# DATA CONTRACTS
# =============================================================================


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
    status: str  # "applied", "rejected", "review_required", "no_change"
    original_section_content: str
    new_section_content: str
    explanation: str
    validation: ValidationResult
    diff_snippet: str = ""


@dataclass
class DocumentationUpdateReport:
    """Comprehensive summary of documentation update run."""

    updated_documents: dict[str, str]  # doc_path -> updated markdown text
    applied_updates: list[AppliedSectionUpdate] = field(default_factory=list)
    rejected_updates: list[AppliedSectionUpdate] = field(default_factory=list)
    review_required_updates: list[AppliedSectionUpdate] = field(default_factory=list)
    total_sections_analyzed: int = 0
    total_sections_applied: int = 0
    total_sections_rejected: int = 0


# =============================================================================
# TRANSACTION & STAGED MULTI-DOCUMENT UPDATES
# =============================================================================


class DocumentTransactionError(Exception):
    """Raised when document staging, validation, or transaction commit fails."""

    pass


class UnsupportedDocumentExtensionError(DocumentTransactionError):
    """Raised when an update targets a non-markdown file extension."""

    pass


class PathTraversalSecurityError(DocumentTransactionError):
    """Raised when a destination path attempts traversal outside docs_root."""

    pass


class DocumentTransaction:
    """
    Manages multi-document updates with all-or-nothing transactional guarantees.

    Staged temporary writes, permission preservation, and atomic rollback on failure.
    """

    def __init__(self, docs_root: str | Path):
        self.docs_root = Path(docs_root).resolve()
        self.staged_files: list[tuple[Path, Path, int | None, bool]] = []
        # (dest_path, temp_staged_path, original_mode, originally_existed)
        self.backups: list[tuple[Path, Path]] = []
        # (dest_path, backup_path)
        self.committed = False

    def validate_destination_path(self, rel_path: str | Path) -> Path:
        """Validate extension and prevent path traversal."""
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
        return dest

    def stage_updates(self, updates: dict[str, str]) -> None:
        """Stage all document updates to temporary files with matching permissions."""
        try:
            for rel_path, content in updates.items():
                dest_path = self.validate_destination_path(rel_path)
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                orig_mode = None
                existed = dest_path.is_file()
                if existed:
                    try:
                        orig_mode = dest_path.stat().st_mode & 0o777
                    except OSError:
                        orig_mode = 0o644
                else:
                    orig_mode = 0o644

                # Create temp file in same directory
                temp_fd, temp_path_str = tempfile.mkstemp(
                    dir=dest_path.parent, prefix=".stage_doc_"
                )
                temp_path = Path(temp_path_str)
                try:
                    with open(temp_fd, "w", encoding="utf-8") as f:
                        f.write(content)
                        f.flush()
                        os.fsync(f.fileno())
                    if orig_mode is not None:
                        try:
                            os.chmod(temp_path, orig_mode)
                        except OSError:
                            pass
                except Exception as write_err:
                    if temp_path.exists():
                        temp_path.unlink(missing_ok=True)
                    raise DocumentTransactionError(
                        f"Failed writing staged document '{rel_path}': {write_err}"
                    ) from write_err

                self.staged_files.append((dest_path, temp_path, orig_mode, existed))
        except Exception:
            self.cleanup()
            raise

    def commit(self) -> None:
        """Apply all staged replacements atomically; roll back all on any failure."""
        if not self.staged_files:
            return

        replaced_backups: list[tuple[Path, Path | None]] = []
        try:
            for dest_path, temp_path, orig_mode, existed in self.staged_files:
                backup_path = None
                if existed:
                    bak_fd, bak_path_str = tempfile.mkstemp(
                        dir=dest_path.parent, prefix=".bak_doc_"
                    )
                    backup_path = Path(bak_path_str)
                    with open(dest_path, "rb") as src, open(bak_fd, "wb") as dst:
                        dst.write(src.read())
                        dst.flush()
                        os.fsync(dst.fileno())
                    self.backups.append((dest_path, backup_path))

                os.replace(temp_path, dest_path)
                if orig_mode is not None:
                    try:
                        os.chmod(dest_path, orig_mode)
                    except OSError:
                        pass
                replaced_backups.append((dest_path, backup_path))

            self.committed = True
        except Exception as commit_err:
            # ROLLBACK EVERYTHING ALREADY REPLACED
            for d_path, b_path in reversed(replaced_backups):
                try:
                    if b_path is not None and b_path.exists():
                        os.replace(b_path, d_path)
                    elif b_path is None and d_path.exists():
                        d_path.unlink(missing_ok=True)
                except OSError as rollback_err:
                    print(f"Critical error during rollback: {rollback_err}")
            self.cleanup()
            raise DocumentTransactionError(
                f"Transaction commit failed; rolled back all replaced documents: {commit_err}"
            ) from commit_err
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up temporary staged files and backup files."""
        for _dest, temp_path, _mode, _existed in self.staged_files:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        for _dest, backup_path in self.backups:
            if backup_path.exists():
                backup_path.unlink(missing_ok=True)
        self.staged_files.clear()
        self.backups.clear()


def atomic_overwrite_file(
    file_path: str | Path, content: str, docs_root: str | Path | None = None
) -> bool:
    """
    Atomically overwrite a single documentation file with validation and rollback.
    """
    path = Path(file_path).resolve()
    root = Path(docs_root).resolve() if docs_root else path.parent
    try:
        tx = DocumentTransaction(docs_root=root)
        rel_path = path.relative_to(root)
        tx.stage_updates({str(rel_path): content})
        tx.commit()
        return True
    except Exception:
        # Fallback direct atomic replace if outside provided root
        dir_path = path.parent
        dir_path.mkdir(parents=True, exist_ok=True)
        temp_fd = None
        temp_path = None
        try:
            temp_fd, temp_path_str = tempfile.mkstemp(dir=dir_path, prefix=".tmp_doc_")
            temp_path = Path(temp_path_str)
            with open(temp_fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)
            return True
        except Exception:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            return False


# =============================================================================
# SECRET REDACTION & PROMPT FORMATTER
# =============================================================================


def redact_secrets(text: str) -> str:
    """
    Redact secrets including AWS keys, GitHub PATs, private keys, auth headers, and tokens.
    """
    if not text:
        return text
    # 1. AWS access key ID
    text = re.sub(r"\bAKIA[0-9A-Z]{16}\b", "[REDACTED_AWS_KEY]", text)
    # 2. GitHub PATs / tokens
    text = re.sub(r"\bghp_[0-9a-zA-Z]{36}\b", "[REDACTED_GITHUB_TOKEN]", text)
    text = re.sub(r"\bgithub_pat_[0-9a-zA-Z_]{82}\b", "[REDACTED_GITHUB_TOKEN]", text)
    # 3. Private keys
    text = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        text,
    )
    # 4. Bearer / Authorization headers
    text = re.sub(
        r"(?i)(Authorization:\s*Bearer\s+)[A-Za-z0-9._~+/-]+", r"\1[REDACTED_TOKEN]", text
    )
    text = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]{16,}", r"\1[REDACTED_TOKEN]", text)
    # 5. Common secret variable assignments
    text = re.sub(
        r"(?i)\b((?:api_key|secret|password|private_token|access_token)\s*[:=]\s*['\"])[^'\"]+(['\"])",
        r"\1[REDACTED]\2",
        text,
    )
    # 6. Apply sanitize_output for environment variables
    return sanitize_output(text)


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


def build_update_prompt(req: SectionUpdateRequest, max_context_chars: int | None = None) -> str:
    """
    Construct an evidence-rich prompt for the AI documentation generator.

    Applies clear untrusted boundary delimiters, secret redaction,
    and prioritizes the target section so it is never truncated away.
    """
    if max_context_chars is None:
        try:
            from config import get_max_context_chars

            max_context_chars = get_max_context_chars()
        except Exception:
            max_context_chars = 400000

    # 1. High priority: Header and metadata
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

    # 2. Critical: Target Section Content (MUST NOT BE TRUNCATED)
    safe_current_content = redact_secrets(req.current_section_content)
    target_sec_text = (
        f"\n### Current Section Content:\n"
        f'<untrusted_current_documentation path="{req.doc_path}" section="{req.section_id}">\n'
        f"{safe_current_content}\n"
        f"</untrusted_current_documentation>\n"
    )

    style_text = ""
    if req.style_guidelines:
        style_text = f"\n### Style Guidelines:\n{redact_secrets(req.style_guidelines)}\n"

    base_len = sum(len(p) for p in header_parts) + len(target_sec_text) + len(style_text) + 200
    available_for_code = max_context_chars - base_len
    if available_for_code < 1000:
        available_for_code = 1000

    # 3. Code changes within available budget
    code_parts = ["\n### Code Changes:"]
    budget_per_file = max(500, available_for_code // max(1, len(req.source_files)))
    for src in req.source_files:
        old_snip = redact_secrets(req.old_code_snippets.get(src, "").strip())
        new_snip = redact_secrets(req.new_code_snippets.get(src, "").strip())
        code_parts.append(f"#### File: `{src}`")
        if old_snip:
            if len(old_snip) > budget_per_file:
                old_snip = old_snip[:budget_per_file] + "\n# [TRUNCATED FOR LENGTH]"
            code_parts.append(
                f'<untrusted_source_code file="{src}" version="previous">\n{old_snip}\n</untrusted_source_code>'
            )
        if new_snip:
            if len(new_snip) > budget_per_file:
                new_snip = new_snip[:budget_per_file] + "\n# [TRUNCATED FOR LENGTH]"
            code_parts.append(
                f'<untrusted_source_code file="{src}" version="updated">\n{new_snip}\n</untrusted_source_code>'
            )

    full_prompt = (
        "\n".join(header_parts) + "\n" + "\n".join(code_parts) + "\n" + target_sec_text + style_text
    )
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
    """Walks Python AST to extract literals, function details, and environment variables."""

    def __init__(self):
        self.numeric_literals: set[str] = set()
        self.env_vars: set[str] = set()
        self.func_params: dict[str, set[str]] = {}
        self.func_returns: dict[str, str] = {}
        self.constants: dict[str, str] = {}

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
        if node.returns:
            try:
                self.func_returns[node.name] = ast.unparse(node.returns).strip()
            except Exception:
                pass

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                try:
                    self.constants[target.id] = ast.unparse(node.value).strip().strip("'\"")
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
                self.env_vars.add(node.args[0].value)
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

        # 9. Parameter Default Value & Type Verification (C1 Case 1)
        for p_name, p_info in valid_params.items():
            if p_info.default_value is not None:
                actual_val = p_info.default_value.strip().strip("'\"")

                eq_matches = re.findall(
                    rf"\b{re.escape(p_name)}\s*=\s*([0-9a-zA-Z_'\"]+)", clean_prop
                )
                for claimed in eq_matches:
                    clean_claimed = claimed.strip().strip("'\"")
                    if clean_claimed != actual_val and clean_claimed != "":
                        failed.append(
                            f"parameter_default_mismatch: parameter '{p_name}' default is '{actual_val}', but documentation claims '{clean_claimed}'"
                        )

                def_matches = re.findall(
                    rf"\b{re.escape(p_name)}\b[^\.\n]*?default\s*(?:is|=|to|:)?\s*`?([0-9a-zA-Z_'\"]+)`?",
                    clean_prop,
                    re.IGNORECASE,
                )
                for claimed in def_matches:
                    clean_claimed = claimed.strip().strip("'\"")
                    if clean_claimed != actual_val and clean_claimed != "":
                        failed.append(
                            f"parameter_default_mismatch: parameter '{p_name}' default is '{actual_val}', but documentation claims '{clean_claimed}'"
                        )

            if p_info.type_annotation is not None:
                actual_type = p_info.type_annotation.strip()
                known_types = {"str", "int", "bool", "float", "list", "dict", "bytes"}

                type_matches = re.findall(
                    rf"\b{re.escape(p_name)}\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", clean_prop
                )
                for claimed_t in type_matches:
                    c_type = claimed_t.strip()
                    if c_type in known_types and c_type != actual_type:
                        failed.append(
                            f"parameter_type_mismatch: parameter '{p_name}' type is '{actual_type}', but documentation claims '{c_type}'"
                        )

                paren_matches = re.findall(
                    rf"\b{re.escape(p_name)}\s*\(([A-Za-z_][A-Za-z0-9_]*)[,\)]",
                    clean_prop,
                )
                for claimed_t in paren_matches:
                    c_type = claimed_t.strip()
                    if c_type in known_types and c_type != actual_type:
                        failed.append(
                            f"parameter_type_mismatch: parameter '{p_name}' type is '{actual_type}', but documentation claims '{c_type}'"
                        )

        # 10. Numeric Claims and Limits Verification (C1 Case 4)
        num_claims = re.findall(
            r"(?:rate\s*limit(?:ed)?(?:\s*to)?|timeout\s*(?:of|is|=|:)?|default\s*(?:of|is|=|:)?|retries\s*(?:of|is|=|:)?|limit(?:ed)?(?:\s*to)?)\s*`?([0-9]+(?:\.[0-9]+)?)\b`?",
            clean_prop,
            re.IGNORECASE,
        )
        num_claims += re.findall(
            r"\b([0-9]+)\s*(?:requests|seconds|minutes|retries|items|nodes|connections|workers)\b",
            clean_prop,
            re.IGNORECASE,
        )
        orig_numbers = set(re.findall(r"\b[0-9]+\b", clean_orig))
        for num_str in num_claims:
            if (
                num_str not in fact_extractor.numeric_literals
                and num_str not in fact_extractor.constants.values()
                and num_str not in orig_numbers
            ):
                failed.append(
                    f"unsupported_numeric_claim: numeric value '{num_str}' is claimed in documentation but does not exist in source constants or literals"
                )

        # 11. Return Type Verification (C1 Case 5)
        for f_name, ret_type in fact_extractor.func_returns.items():
            if ret_type:
                # If proposed doc claims return type for this function
                ret_matches = re.findall(
                    rf"\b{re.escape(f_name)}\b[^\.\n]*?(?:returns|returning|return value is)\s+(?:a\s+)?([A-Za-z_][A-Za-z0-9_\[\], ]*)",
                    clean_prop,
                    re.IGNORECASE,
                )
                for claimed_ret in ret_matches:
                    c_ret = claimed_ret.strip().lower()
                    if (("list" in c_ret or "str" in c_ret) and ret_type == "int") or (
                        ("int" in c_ret or "number" in c_ret) and "str" in ret_type
                    ):
                        failed.append(
                            f"return_type_mismatch: function '{f_name}' returns '{ret_type}', but documentation claims '{claimed_ret.strip()}'"
                        )

        # 12. Environment Variables Verification (C1 Case 6)
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
                failed.append(
                    f"unsupported_environment_variable: '{env_name}' is mentioned in documentation but is not defined or accessed in source code"
                )

        # 13. Removed Symbols Verification (Priority 5, C3)
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

                        active_verbs = [
                            "call",
                            "use",
                            "invoke",
                            "run",
                            "execute",
                            "import",
                            "call on",
                        ]
                        starts_with_active = any(
                            re.search(rf"\b{v}\b.*?\b{re.escape(short_name)}\b", cl_lower)
                            for v in active_verbs
                        )
                        is_not_deprecated = (
                            "not deprecated" in cl_lower or "isn't deprecated" in cl_lower
                        )
                        is_active_call = bool(re.search(rf"\b{re.escape(short_name)}\s*\(", cl))

                        if starts_with_active or is_not_deprecated or is_active_call:
                            failed.append(
                                f"active_reference_to_removed_symbol: '{short_name}' was removed from source code but is actively instructed or used in '{cl}'"
                            )
                            break

                        is_removal = any(
                            kw in cl_lower for kw in ["removed", "deleted", "dropped", "obsolete"]
                        )
                        is_deprecation = "deprecated" in cl_lower and not is_not_deprecated
                        is_replacement = bool(
                            re.search(
                                rf"\breplace\b.*?\b{re.escape(short_name)}\b.*?\bwith\b\s+`?([A-Za-z_][A-Za-z0-9_]*)`?",
                                cl_lower,
                            )
                        )

                        if not (is_removal or is_deprecation or is_replacement):
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
                applied_rec = AppliedSectionUpdate(
                    doc_path=src_path,
                    section_id="syntax_error",
                    section_heading="Syntax Error",
                    status="rejected",
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
            except Exception as gen_err:
                applied_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=target.section_id,
                    section_heading=target.section_heading,
                    status="rejected",
                    original_section_content=current_section_content,
                    new_section_content="",
                    explanation=f"AI generation failed: {gen_err}",
                    validation=ValidationResult(
                        is_valid=False,
                        failed_checks=[f"ai_generation_error: {gen_err}"],
                    ),
                )
                report.rejected_updates.append(applied_rec)
                report.total_sections_rejected += 1
                continue

            # 5. Deterministic Validation
            val_result = DocumentationValidator.validate(req, ai_resp, parsed_doc)

            # Idempotency / No-Change Check (Priority 5 & Priority 6, H4)
            if ai_resp.proposed_content.strip() == current_section_content.strip():
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
                    new_section_content=ai_resp.proposed_content,
                    explanation="Section content already up to date.",
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

        # 6. Multi-Document All-or-Nothing Commit (Priority 3, N3)
        # If any section in the run was rejected, abort the whole run!
        if report.total_sections_rejected > 0:
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
