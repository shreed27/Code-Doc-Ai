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
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ast_intel import ASTDelta, compute_ast_delta
from impact_mapper import (
    DocumentationImpactMapper,
    ImpactEvidence,
    SectionImpact,
    consolidate_impacts,
)
from section_patcher import ParsedDocument, parse_sections, replace_section_content

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


def build_update_prompt(req: SectionUpdateRequest) -> str:
    """Construct an evidence-rich prompt for the AI documentation generator."""
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
        sections.append(f'- [{ev.evidence_type}] `{ev.matched_text}`: "{ev.context}"')

    sections.extend(["", "### Code Changes:"])
    for src in req.source_files:
        old_snip = req.old_code_snippets.get(src, "").strip()
        new_snip = req.new_code_snippets.get(src, "").strip()
        sections.append(f"#### File: `{src}`")
        if old_snip:
            sections.append(f"```python\n# Previous code\n{old_snip}\n```")
        if new_snip:
            sections.append(f"```python\n# Updated code\n{new_snip}\n```")

    sections.extend(["", "### Current Section Content:"])
    sections.append(f"```markdown\n{req.current_section_content}\n```")

    if req.style_guidelines:
        sections.extend(["", "### Style Guidelines:"])
        sections.append(req.style_guidelines)

    return "\n".join(sections)


def parse_ai_response(
    raw_response: str, expected_doc_path: str, expected_section_id: str
) -> AISectionUpdateResponse:
    """Parse and validate JSON response from the AI model with safe fallback handling."""
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
        if isinstance(data, dict):
            return AISectionUpdateResponse(
                doc_path=str(data.get("doc_path", expected_doc_path)),
                section_id=str(data.get("section_id", expected_section_id)),
                proposed_content=str(data.get("proposed_content", "")),
                explanation=str(data.get("explanation", "")),
                technical_evidence=list(data.get("technical_evidence", [])),
                uncertainties=list(data.get("uncertainties", [])),
                requires_human_review=bool(data.get("requires_human_review", False)),
                review_reason=str(data.get("review_reason", "")),
            )
    except json.JSONDecodeError:
        pass

    # If the model returned plain Markdown text instead of JSON
    return AISectionUpdateResponse(
        doc_path=expected_doc_path,
        section_id=expected_section_id,
        proposed_content=raw_response.strip(),
        explanation="AI produced unstructured text",
        requires_human_review=True,
        review_reason="AI response could not be parsed as structured JSON; flagged for review.",
    )


# =============================================================================
# DETERMINISTIC VALIDATOR
# =============================================================================


class DocumentationValidator:
    """Performs deterministic validation on AI-generated documentation updates."""

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
            # If the original had content and proposal is completely empty without explanation
            warnings.append("proposed_content_is_empty")
            review_reasons.append("Proposed content is empty while original section had text.")
        else:
            passed.append("non_empty_content")

        # 4. Markdown structure validity
        # Ensure proposed content does not introduce unclosed fences or broken headings
        test_markdown = f"# Test\n\n{resp.proposed_content}\n"
        try:
            test_doc = parse_sections(test_markdown)
            passed.append("valid_markdown_structure")
            # If the update introduced unintended subheadings when none were planned
            if len(test_doc.sections) > 1 and not any(
                "#" in line for line in clean_orig.splitlines()
            ):
                warnings.append("subheadings_introduced_in_leaf_section")
        except Exception as e:
            failed.append(f"invalid_markdown_structure: {e}")

        # 5. Hallucination & Evidence check:
        # Extract backticked symbols and function calls from proposed content
        new_tokens = DocumentationValidator._extract_new_technical_tokens(clean_orig, clean_prop)

        # Collect ground truth valid symbols from AST delta and source code
        valid_symbols = set()
        valid_params = set()
        valid_constants = set()

        for delta in req.ast_deltas.values():
            for s in delta.added_symbols:
                valid_symbols.add(s.name)
            for _old_s, new_s in delta.modified_symbols:
                valid_symbols.add(new_s.name)
            for _fname, p in delta.added_parameters:
                valid_params.add(p.name)
            for _fname, _old_p, new_p in delta.modified_parameters:
                valid_params.add(new_p.name)
            for c in delta.added_constants:
                valid_constants.add(c.name.split(".")[-1])
            for _old_c, new_c in delta.modified_constants:
                valid_constants.add(new_c.name.split(".")[-1])

        # Also include symbols from source snippets
        for code in req.new_code_snippets.values():
            for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]+\b", code):
                valid_symbols.add(token)

        # Inspect new technical tokens
        hallucinated_tokens: list[str] = []
        for token in new_tokens:
            # Check if token looks like a function or parameter identifier
            is_valid = (
                token in valid_symbols
                or token in valid_params
                or token in valid_constants
                or token.lower()
                in {s.lower() for s in valid_symbols | valid_params | valid_constants}
                or len(token) <= 2  # Skip tiny words
                or token
                in (
                    "str",
                    "int",
                    "bool",
                    "float",
                    "dict",
                    "list",
                    "tuple",
                    "set",
                    "None",
                    "Optional",
                )
            )
            if not is_valid:
                # Check if it was in impact evidence
                in_evidence = any(
                    token.lower() in ev.matched_text.lower() for ev in req.impact_evidence
                )
                if not in_evidence:
                    hallucinated_tokens.append(token)

        if hallucinated_tokens:
            failed.append(f"unsupported_technical_tokens: {hallucinated_tokens}")
        else:
            passed.append("no_hallucinated_technical_symbols")

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
        orig_backticks = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", original_text))
        prop_backticks = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", proposed_text))

        orig_calls = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", original_text))
        prop_calls = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", proposed_text))

        new_backticks = prop_backticks - orig_backticks
        new_calls = prop_calls - orig_calls

        return new_backticks | new_calls


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
                file_impacts = self.mapper.map_file_impacts(old_src, new_src, src_path, parsed_docs)
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

            current_section_content = parsed_doc.get_section_content(sec, include_children=False)

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
            user_prompt = build_update_prompt(req)
            raw_ai_out = self.ai_generator(_SYSTEM_PROMPT, user_prompt)
            ai_resp = parse_ai_response(raw_ai_out, doc_path, target.section_id)

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

            # 6. Apply Surgical Section Update via section_patcher
            try:
                updated_markdown = replace_section_content(
                    current_markdown,
                    target=target.section_id,
                    new_content=ai_resp.proposed_content,
                    replace_children=False,
                )

                # Update in-memory document state for subsequent updates to the same document
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
