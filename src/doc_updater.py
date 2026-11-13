"""
AI Documentation Updating Engine.

Coordinates the end-to-end documentation update pipeline:
1. Detects code changes via AST analysis (`src/ast_intel.py`).
2. Maps affected documentation sections via impact mapping (`src/impact_mapper.py`).
3. Generates grounded section updates via an AI documentation model.
4. Validates proposed updates deterministically against source code truth.
5. Surgically updates affected Markdown sections via `src/section_patcher.py`.
"""

from __future__ import annotations

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
    ImportInfo,
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
# ATOMIC FILE OPERATION
# =============================================================================


def atomic_overwrite_file(file_path: str | Path, content: str) -> bool:
    """
    Atomically write content to file_path using a temp file and os.replace.

    Guarantees no partial writes or corrupted files on crash.
    """
    path = Path(file_path).resolve()
    dir_path = path.parent
    dir_path.mkdir(parents=True, exist_ok=True)

    temp_fd, temp_path = tempfile.mkstemp(dir=dir_path, prefix=".tmp_doc_")
    try:
        with open(temp_fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        return True
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return False


# =============================================================================
# PROMPT FORMATTER & PARSER
# =============================================================================

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


def build_update_prompt(req: SectionUpdateRequest, max_context_chars: int = 60000) -> str:
    """
    Construct an evidence-rich prompt for the AI documentation generator.

    Applies clear untrusted boundary delimiters, credential sanitization,
    and context window limits (H6).
    """
    sections = [
        f"### Target Section: {req.section_heading} (`{req.section_id}`)",
        f"**Document**: `{req.doc_path}`",
        f"**Changed Symbols**: {', '.join(req.changed_symbols)}",
        f"**Change Categories**: {', '.join(req.change_categories)}",
        "",
        "### Impact Reasons:",
    ]
    for r in req.impact_reasons:
        sections.append(f"- {r}")

    sections.extend(["", "### Supporting Code Evidence:"])
    for ev in req.impact_evidence:
        safe_ctx = sanitize_output(ev.context)
        sections.append(f'- [{ev.evidence_type}] `{ev.matched_text}`: "{safe_ctx}"')

    sections.extend(["", "### Code Changes:"])
    for src in req.source_files:
        old_snip = sanitize_output(req.old_code_snippets.get(src, "").strip())
        new_snip = sanitize_output(req.new_code_snippets.get(src, "").strip())
        sections.append(f"#### File: `{src}`")
        if old_snip:
            sections.append(
                f'<untrusted_source_code file="{src}" version="previous">\n{old_snip}\n</untrusted_source_code>'
            )
        if new_snip:
            sections.append(
                f'<untrusted_source_code file="{src}" version="updated">\n{new_snip}\n</untrusted_source_code>'
            )

    safe_current_content = sanitize_output(req.current_section_content)
    sections.extend(
        [
            "",
            "### Current Section Content:",
            f'<untrusted_current_documentation path="{req.doc_path}" section="{req.section_id}">\n{safe_current_content}\n</untrusted_current_documentation>',
        ]
    )

    if req.style_guidelines:
        safe_style = sanitize_output(req.style_guidelines)
        sections.extend(["", "### Style Guidelines:", safe_style])

    prompt = "\n".join(sections)
    if len(prompt) > max_context_chars:
        # Enforce context limits cleanly
        prompt = prompt[:max_context_chars] + "\n\n[TRUNCATED FOR LENGTH]"
    return prompt


def parse_ai_response(
    raw_response: str, expected_doc_path: str, expected_section_id: str
) -> AISectionUpdateResponse:
    """
    Parse and strictly validate JSON response from the AI model (C4).

    Rejects or flags for human review on null content, non-string types,
    or structural format discrepancies. Never uses blind str() conversion.
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
            explanation=f"Malformed JSON: {exc}",
            requires_human_review=True,
            review_reason=f"AI response is not valid JSON ({exc}); flagged for review.",
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
            review_reason=(
                f"proposed_content must be a string, got {type(raw_content).__name__}."
            ),
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
        technical_evidence = [str(x) for x in raw_evidence if isinstance(x, (str, int, float))]
    else:
        technical_evidence = []

    raw_uncertainties = data.get("uncertainties", [])
    if isinstance(raw_uncertainties, list):
        uncertainties = [str(x) for x in raw_uncertainties if isinstance(x, (str, int, float))]
    else:
        uncertainties = []

    requires_human_review = bool(data.get("requires_human_review", False))
    review_reason = data.get("review_reason", "")
    if not isinstance(review_reason, str):
        review_reason = str(review_reason)

    # Flag mismatched targets for review
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


class DocumentationValidator:
    """Performs deterministic validation on AI-generated documentation updates."""

    # Words indicating a removed/deprecated symbol is being documented historically (C3)
    _REMOVAL_EXPLANATORY_KEYWORDS = {
        "removed",
        "deprecated",
        "deleted",
        "dropped",
        "replaced",
        "superseded",
        "obsolete",
        "no longer",
        "migration",
    }

    # Standard Python / Markdown tokens that are safe in documentation
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
        # 4a. Unbalanced code fences check (``` and ~~~)
        backtick_fence_count = len(re.findall(r"(?m)^[ \t]*```", resp.proposed_content))
        tilde_fence_count = len(re.findall(r"(?m)^[ \t]*~~~", resp.proposed_content))
        if backtick_fence_count % 2 != 0 or tilde_fence_count % 2 != 0:
            failed.append("unbalanced_code_fences")
        else:
            passed.append("balanced_code_fences")

        # 4b. Leaf-section heading injection check
        # A section is a leaf section if it has no child headings in original content
        orig_has_headings = bool(
            re.search(r"(?m)^[ \t]*#{1,6}\s+\S+", clean_orig)
            or re.search(r"(?m)^[^\n]+\n[ \t]*(=+|-+)[ \t]*$", clean_orig)
        )
        prop_has_atx = bool(re.search(r"(?m)^[ \t]*#{1,6}\s+\S+", resp.proposed_content))
        # Note: Setext check ignores lines starting with '---' that are horizontal rules
        prop_has_setext = bool(
            re.search(
                r"(?m)^[A-Za-z0-9_`*][^\n]*\n[ \t]*(=+|-{2,})[ \t]*$", resp.proposed_content
            )
        )
        if not orig_has_headings and (prop_has_atx or prop_has_setext):
            failed.append("headings_introduced_in_leaf_section")
        else:
            passed.append("no_leaf_heading_injection")

        # 5. Build Technical Evidence Registry Strictly from Structured AST (C1, H1)
        # NO regex extraction of arbitrary words from raw source files!
        valid_symbols: set[str] = set()
        valid_params: dict[str, ParameterInfo] = {}
        valid_constants: dict[str, ConstantInfo] = {}
        valid_imports: set[str] = set()

        for src in req.source_files:
            code = req.new_code_snippets.get(src, "")
            if code:
                # Structured symbols
                sym_map = extract_python_symbols(code)
                for s in sym_map.values():
                    valid_symbols.add(s.name)
                    valid_symbols.add(s.name.split(".")[-1])
                    for p in s.parameters:
                        valid_params[p.name] = p

                # Structured constants
                const_map = extract_python_constants(code)
                for c in const_map.values():
                    valid_constants[c.name] = c
                    valid_constants[c.name.split(".")[-1]] = c

                # Structured imports
                imp_list = extract_python_imports(code)
                for imp in imp_list:
                    valid_imports.add(imp.alias if imp.alias else imp.name)

            # Also incorporate structured changes from AST deltas
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
            # Case-sensitive exact match check (Python is case-sensitive)
            is_valid = token in allowed_all or len(token) <= 2
            if not is_valid:
                # Evidence check: was it matched in deterministic impact evidence?
                in_evidence = any(
                    token == ev.matched_text
                    for ev in req.impact_evidence
                    if ev.is_deterministic
                )
                if not in_evidence:
                    hallucinated_tokens.append(token)

        if hallucinated_tokens:
            failed.append(f"unsupported_technical_tokens: {sorted(hallucinated_tokens)}")
        else:
            passed.append("no_hallucinated_technical_symbols")

        # 7. Parameter Default Value & Type Verification (C1, P6, P7)
        for p_name, p_info in valid_params.items():
            if p_info.default_value is not None:
                actual_val = p_info.default_value.strip().strip("'\"")

                # Match param=val or param = val
                eq_matches = re.findall(
                    rf"\b{re.escape(p_name)}\s*=\s*([0-9a-zA-Z_'\"]+)", clean_prop
                )
                for claimed in eq_matches:
                    clean_claimed = claimed.strip().strip("'\"")
                    if clean_claimed != actual_val and clean_claimed != "":
                        failed.append(
                            f"parameter_default_mismatch: parameter '{p_name}' default is '{actual_val}', but documentation claims '{clean_claimed}'"
                        )

                # Match "default <val>", "default is <val>", "default: <val>", "(default <val>)"
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

                # Match param: type
                type_matches = re.findall(
                    rf"\b{re.escape(p_name)}\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", clean_prop
                )
                for claimed_t in type_matches:
                    c_type = claimed_t.strip()
                    if c_type in known_types and c_type != actual_type:
                        failed.append(
                            f"parameter_type_mismatch: parameter '{p_name}' type is '{actual_type}', but documentation claims '{c_type}'"
                        )

                # Match param (type)
                paren_matches = re.findall(
                    rf"\b{re.escape(p_name)}\s*\(([A-Za-z_][A-Za-z0-9_]*)[,\)]", clean_prop
                )
                for claimed_t in paren_matches:
                    c_type = claimed_t.strip()
                    if c_type in known_types and c_type != actual_type:
                        failed.append(
                            f"parameter_type_mismatch: parameter '{p_name}' type is '{actual_type}', but documentation claims '{c_type}'"
                        )

        # 8. Removed Symbols Verification (C3)
        removed_symbols: list[SymbolInfo] = []
        for delta in req.ast_deltas.values():
            removed_symbols.extend(delta.removed_symbols)

        for sym in removed_symbols:
            short_name = sym.name.split(".")[-1]
            # Check if removed symbol appears in proposed text
            if re.search(rf"\b{re.escape(short_name)}\b", clean_prop):
                # Check whether it is an explanatory note describing removal or an active call
                sentences = re.split(r"[.\n]", clean_prop)
                for s in sentences:
                    if re.search(rf"\b{re.escape(short_name)}\b", s):
                        s_lower = s.lower()
                        has_removal_context = any(
                            kw in s_lower
                            for kw in DocumentationValidator._REMOVAL_EXPLANATORY_KEYWORDS
                        )
                        if not has_removal_context:
                            failed.append(
                                f"active_reference_to_removed_symbol: '{short_name}' was removed from source code but is actively referenced"
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

        # Unpack dotted tokens if needed (e.g. "mod.func" -> "mod.func" and "func")
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

        ai_generator: callable taking (system_prompt, user_prompt) -> str.
                      If None, a default completion caller will be created.
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

        # 1. Compute AST deltas
        deltas: dict[str, ASTDelta] = {}
        for src_path, (old_src, new_src) in code_changes.items():
            deltas[src_path] = compute_ast_delta(old_src, new_src, file_path=src_path)

        # 2. Impact Mapping: Map changes to documentation sections
        parsed_docs: dict[str, ParsedDocument] = {
            doc_path: parse_sections(text) for doc_path, text in documents.items()
        }

        if precomputed_impacts is not None:
            raw_impacts = precomputed_impacts
        else:
            raw_impacts = []
            for src_path, (old_src, new_src) in code_changes.items():
                file_impacts = self.mapper.map_file_impacts(
                    old_src, new_src, src_path, parsed_docs
                )
                raw_impacts.extend(file_impacts)

        # Consolidate impacts to avoid duplicate section updates
        consolidated = consolidate_impacts(raw_impacts)
        report.total_sections_analyzed = len(consolidated)

        if not consolidated:
            return report

        # 3. Process each affected section
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

            current_section_content = parsed_doc.get_section_content(
                sec, include_children=False
            )

            # Check for source syntax errors upfront (H7): do not call AI
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

            # Build Request Context
            old_snips = {
                src: code_changes[src][0]
                for src in target.source_files
                if src in code_changes
            }
            new_snips = {
                src: code_changes[src][1]
                for src in target.source_files
                if src in code_changes
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

            # 4. Generate Update via AI (Isolated per section, H3)
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

            # Idempotency / No-Change Check (H4)
            if ai_resp.proposed_content.strip() == current_section_content.strip():
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

            # 6. Apply Surgical Section Update with Document Invariants & Rollback (C2)
            try:
                updated_markdown = replace_section_content(
                    current_markdown,
                    target=target.section_id,
                    new_content=ai_resp.proposed_content,
                    replace_children=False,
                )

                # Invariant Verification (C2):
                post_doc = parse_sections(updated_markdown)
                orig_sec_ids = {s.identifier for s in parsed_doc.sections}
                post_sec_ids = {s.identifier for s in post_doc.sections}
                if orig_sec_ids != post_sec_ids:
                    raise ValueError(
                        f"Document invariant violation: section IDs changed from {orig_sec_ids} to {post_sec_ids}"
                    )

                # Invariant: all non-target sections must be byte-identical
                for s_orig in parsed_doc.sections:
                    if s_orig.identifier == target.section_id:
                        continue
                    s_post = post_doc.find_section(s_orig.identifier)
                    if s_post is None:
                        raise ValueError(
                            f"Document invariant violation: section '{s_orig.identifier}' missing"
                        )
                    c_orig = parsed_doc.get_section_content(s_orig, include_children=False)
                    c_post = post_doc.get_section_content(s_post, include_children=False)
                    if c_orig != c_post:
                        raise ValueError(
                            f"Document invariant violation: non-target section '{s_orig.identifier}' was altered"
                        )

                # Invariants satisfied: commit to memory state
                report.updated_documents[doc_path] = updated_markdown

                applied_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=target.section_id,
                    section_heading=target.section_heading,
                    status="applied",
                    original_section_content=current_section_content,
                    new_section_content=ai_resp.proposed_content,
                    explanation=ai_resp.explanation,
                    validation=val_result,
                )
                report.applied_updates.append(applied_rec)
                report.total_sections_applied += 1

            except Exception as patch_err:
                # Rollback: updated_documents[doc_path] remains untouched!
                applied_rec = AppliedSectionUpdate(
                    doc_path=doc_path,
                    section_id=target.section_id,
                    section_heading=target.section_heading,
                    status="rejected",
                    original_section_content=current_section_content,
                    new_section_content=ai_resp.proposed_content,
                    explanation=f"Section patcher failed: {patch_err}",
                    validation=ValidationResult(
                        is_valid=False,
                        failed_checks=[f"patcher_exception: {patch_err}"],
                    ),
                )
                report.rejected_updates.append(applied_rec)
                report.total_sections_rejected += 1

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
