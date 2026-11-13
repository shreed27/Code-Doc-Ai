"""
Documentation Impact Mapping Engine.

Connects source code AST change detection (src/ast_intel.py) with Markdown section
parsing (src/section_patcher.py) to accurately identify which documentation sections
are impacted by code changes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ast_intel import (
    ASTDelta,
    ConstantInfo,
    ImportInfo,
    SymbolInfo,
    compute_ast_delta,
)
from section_patcher import ParsedDocument, parse_sections

# Common directories to ignore during documentation discovery
IGNORED_DIRS = {
    ".git",
    ".github",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".idea",
    ".vscode",
}

# English stop words for semantic token filtering
STOP_WORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


@dataclass
class ImpactEvidence:
    """Individual piece of evidence linking a code change to a documentation section."""

    evidence_type: str  # "explicit_symbol", "explicit_file_path", "explicit_constant", "heading_match", "signature_parameter", "semantic_keyword", "implementation_behavior", "deleted_symbol_reference"
    matched_text: str  # The exact token or phrase matched
    context: str  # Excerpt of section content where match occurred
    confidence: float  # 0.0 to 1.0


@dataclass
class SectionImpact:
    """Detailed record of an affected documentation section."""

    source_file: str
    changed_symbol: str
    change_category: str  # "added_symbol", "removed_symbol", "modified_signature", "modified_parameters", "modified_return_type", "modified_constant", "modified_import", "implementation_change", "docstring_change"
    doc_path: str
    section_id: str
    section_heading: str
    is_deterministic: bool  # True if backed by exact symbol/path/constant/heading match
    confidence: float  # 0.0 to 1.0
    evidence: list[ImpactEvidence] = field(default_factory=list)
    reason: str = ""


@dataclass
class ConsolidatedSectionImpact:
    """Consolidated summary of impacts targeting a specific documentation section."""

    doc_path: str
    section_id: str
    section_heading: str
    changed_symbols: list[str]
    source_files: list[str]
    change_categories: list[str]
    is_deterministic: bool
    confidence: float
    evidence: list[ImpactEvidence]
    reasons: list[str]


def tokenize_name(name: str) -> set[str]:
    """
    Split a programming identifier (snake_case, camelCase, kebab-case) into individual tokens.

    Example: 'extract_python_symbols' -> {'extract', 'python', 'symbols'}
    """
    if not name:
        return set()

    # Split on CamelCase transitions: e.g. ASTDelta -> AST, Delta
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)

    # Split on underscores, hyphens, and whitespace
    parts = re.split(r"[-_\s/.]+", s2.lower())
    tokens = {p for p in parts if len(p) >= 2 and p not in STOP_WORDS}
    return tokens


def discover_documentation_files(
    repo_path: str | Path,
    include_github_dir: bool = False,
    extra_extensions: tuple[str, ...] = (".md", ".markdown"),
) -> list[Path]:
    """
    Discover all Markdown documentation files within a repository directory.

    Respects standard ignored directories (.git, node_modules, .venv, etc.).
    Returns deterministic, sorted paths relative to repo_path.
    """
    base = Path(repo_path).resolve()
    if not base.is_dir():
        return []

    doc_files: list[Path] = []

    for root, dirs, files in os.walk(base):
        # Prune ignored directories
        dirs[:] = [
            d for d in dirs if d not in IGNORED_DIRS and (include_github_dir or d != ".github")
        ]

        for file in files:
            p = Path(file)
            if p.suffix.lower() in extra_extensions:
                full_path = Path(root) / file
                doc_files.append(full_path.relative_to(base))

    return sorted(doc_files)


class DocumentationImpactMapper:
    """Core engine matching code changes (ASTDelta) to parsed documentation sections."""

    def __init__(
        self,
        semantic_threshold: float = 0.55,
    ):
        self.semantic_threshold = semantic_threshold

    def map_file_impacts(
        self,
        old_code: str,
        new_code: str,
        source_path: str,
        documents: dict[str, str | ParsedDocument],
    ) -> list[SectionImpact]:
        """
        Analyze code differences for a source file and map them to affected documentation sections.

        documents: mapping of doc_path -> (raw markdown string or already parsed ParsedDocument).
        """
        delta = compute_ast_delta(old_code, new_code, file_path=source_path)
        if not delta.has_semantic_changes:
            return []

        parsed_docs: dict[str, ParsedDocument] = {}
        for doc_path, doc_val in documents.items():
            if isinstance(doc_val, ParsedDocument):
                parsed_docs[doc_path] = doc_val
            elif isinstance(doc_val, str):
                parsed_docs[doc_path] = parse_sections(doc_val)

        impacts: list[SectionImpact] = []

        # 1. Map Added Symbols
        for sym in delta.added_symbols:
            impacts.extend(self._match_added_symbol(source_path, sym, parsed_docs))

        # 2. Map Removed Symbols (stale documentation detection)
        for sym in delta.removed_symbols:
            impacts.extend(self._match_removed_symbol(source_path, sym, parsed_docs))

        # 3. Map Modified Symbols (parameters, return types, docstrings, decorators)
        for old_sym, new_sym in delta.modified_symbols:
            impacts.extend(
                self._match_modified_symbol(source_path, old_sym, new_sym, delta, parsed_docs)
            )

        # 4. Map Modified Constants
        for const in delta.added_constants:
            impacts.extend(self._match_constant(source_path, const, "added_constant", parsed_docs))
        for const in delta.removed_constants:
            impacts.extend(
                self._match_constant(source_path, const, "removed_constant", parsed_docs)
            )
        for old_c, new_c in delta.modified_constants:
            impacts.extend(
                self._match_constant(
                    source_path, new_c, "modified_constant", parsed_docs, old_const=old_c
                )
            )

        # 5. Map Modified Imports
        for imp in delta.added_imports:
            impacts.extend(self._match_import(source_path, imp, "added_import", parsed_docs))
        for imp in delta.removed_imports:
            impacts.extend(self._match_import(source_path, imp, "removed_import", parsed_docs))

        # 6. Map Pure Implementation / Behavioral Changes
        if delta.implementation_only_changes:
            for symbol_name in delta.implementation_only_changes:
                impacts.extend(
                    self._match_implementation_change(source_path, symbol_name, parsed_docs)
                )

        return impacts

    def _match_added_symbol(
        self,
        source_path: str,
        sym: SymbolInfo,
        docs: dict[str, ParsedDocument],
    ) -> list[SectionImpact]:
        """Find doc sections that should document a newly added symbol."""
        impacts: list[SectionImpact] = []
        sym_tokens = tokenize_name(sym.name)
        source_name = Path(source_path).name
        module_name = Path(source_path).stem

        for doc_path, doc in docs.items():
            for sec in doc.sections:
                if sec.is_preamble:
                    continue

                content = doc.get_section_content(sec, include_children=False)
                sec_tokens = tokenize_name(sec.title) | tokenize_name(" ".join(sec.path))
                heading_lower = sec.title.lower()

                evidence: list[ImpactEvidence] = []
                is_det = False
                conf = 0.0

                # Check if new symbol name is already explicitly mentioned (e.g. planned/spec docs)
                if re.search(rf"\b{re.escape(sym.name)}\b", content):
                    evidence.append(
                        ImpactEvidence(
                            evidence_type="explicit_symbol",
                            matched_text=sym.name,
                            context=self._extract_snippet(content, sym.name),
                            confidence=1.0,
                        )
                    )
                    is_det = True
                    conf = 1.0

                # Check if section title explicitly refers to the symbol
                if sym.name.lower() in heading_lower:
                    evidence.append(
                        ImpactEvidence(
                            evidence_type="heading_match",
                            matched_text=sym.title if hasattr(sym, "title") else sym.name,
                            context=sec.title,
                            confidence=0.95,
                        )
                    )
                    is_det = True
                    conf = max(conf, 0.95)

                # Check if section explicitly documents this source file/module
                if (
                    source_path.lower() in content.lower()
                    or source_name.lower() in content.lower()
                    or f"`{module_name}`" in content
                ):
                    evidence.append(
                        ImpactEvidence(
                            evidence_type="explicit_file_path",
                            matched_text=source_name,
                            context=self._extract_snippet(content, source_name),
                            confidence=0.5,
                        )
                    )
                    # File-path-only match is weak and non-deterministic unless symbol/heading matches
                    conf = max(conf, 0.5)

                # Determine whether mapping is deterministic based on direct symbol/heading match
                is_det = any(
                    ev.evidence_type in ("explicit_symbol", "heading_match") for ev in evidence
                )

                # Semantic fallback: section heading has high token overlap with added symbol/module
                if not is_det:
                    overlap = sym_tokens & sec_tokens
                    if overlap and len(overlap) >= max(1, len(sym_tokens) // 2):
                        score = len(overlap) / max(len(sym_tokens), len(sec_tokens))
                        if score >= self.semantic_threshold:
                            evidence.append(
                                ImpactEvidence(
                                    evidence_type="semantic_keyword",
                                    matched_text=", ".join(sorted(overlap)),
                                    context=sec.title,
                                    confidence=round(score, 2),
                                )
                            )
                            conf = round(score, 2)

                if evidence:
                    reason = (
                        f"New symbol '{sym.name}' added in {source_path}. "
                        f"Section '{sec.title}' documents related capabilities."
                    )
                    impacts.append(
                        SectionImpact(
                            source_file=source_path,
                            changed_symbol=sym.name,
                            change_category="added_symbol",
                            doc_path=doc_path,
                            section_id=sec.id,
                            section_heading=sec.title,
                            is_deterministic=is_det,
                            confidence=conf,
                            evidence=evidence,
                            reason=reason,
                        )
                    )

        return impacts

    def _match_removed_symbol(
        self,
        source_path: str,
        sym: SymbolInfo,
        docs: dict[str, ParsedDocument],
    ) -> list[SectionImpact]:
        """Find doc sections containing stale references to a removed symbol."""
        impacts: list[SectionImpact] = []

        for doc_path, doc in docs.items():
            for sec in doc.sections:
                content = doc.get_section_content(sec, include_children=False)
                evidence: list[ImpactEvidence] = []

                # Exact symbol reference in section body
                if re.search(rf"\b{re.escape(sym.name)}\b", content):
                    evidence.append(
                        ImpactEvidence(
                            evidence_type="deleted_symbol_reference",
                            matched_text=sym.name,
                            context=self._extract_snippet(content, sym.name),
                            confidence=1.0,
                        )
                    )

                # Symbol in section heading
                if sym.name.lower() in sec.title.lower():
                    evidence.append(
                        ImpactEvidence(
                            evidence_type="heading_match",
                            matched_text=sym.name,
                            context=sec.title,
                            confidence=1.0,
                        )
                    )

                if evidence:
                    reason = (
                        f"Symbol '{sym.name}' was removed from {source_path}, but is "
                        f"still referenced in section '{sec.title}' (stale documentation)."
                    )
                    impacts.append(
                        SectionImpact(
                            source_file=source_path,
                            changed_symbol=sym.name,
                            change_category="removed_symbol",
                            doc_path=doc_path,
                            section_id=sec.id,
                            section_heading=sec.title,
                            is_deterministic=True,
                            confidence=1.0,
                            evidence=evidence,
                            reason=reason,
                        )
                    )

        return impacts

    def _match_modified_symbol(
        self,
        source_path: str,
        old_sym: SymbolInfo,
        new_sym: SymbolInfo,
        delta: ASTDelta,
        docs: dict[str, ParsedDocument],
    ) -> list[SectionImpact]:
        """Find doc sections affected by signature, parameter, return type, or docstring modifications."""
        impacts: list[SectionImpact] = []
        sym_name = new_sym.name
        source_name = Path(source_path).name

        # Identify specific change category
        category = "modified_signature"
        param_diffs: list[str] = []
        for name, p in delta.added_parameters:
            if name == sym_name:
                param_diffs.append(f"added param '{p.name}'")
        for name, p in delta.removed_parameters:
            if name == sym_name:
                param_diffs.append(f"removed param '{p.name}'")
        for name, _old_p, new_p in delta.modified_parameters:
            if name == sym_name:
                param_diffs.append(f"modified param '{new_p.name}'")

        if param_diffs:
            category = "modified_parameters"
        elif any(item[0] == sym_name for item in delta.modified_return_types):
            category = "modified_return_type"
        elif old_sym.docstring != new_sym.docstring:
            category = "docstring_change"

        for doc_path, doc in docs.items():
            for sec in doc.sections:
                content = doc.get_section_content(sec, include_children=False)
                evidence: list[ImpactEvidence] = []
                is_det = False
                conf = 0.0

                # 1. Exact symbol match in content
                if re.search(rf"\b{re.escape(sym_name)}\b", content):
                    evidence.append(
                        ImpactEvidence(
                            evidence_type="explicit_symbol",
                            matched_text=sym_name,
                            context=self._extract_snippet(content, sym_name),
                            confidence=1.0,
                        )
                    )
                    is_det = True
                    conf = 1.0

                # 2. Heading match
                if sym_name.lower() in sec.title.lower():
                    evidence.append(
                        ImpactEvidence(
                            evidence_type="heading_match",
                            matched_text=sym_name,
                            context=sec.title,
                            confidence=0.95,
                        )
                    )
                    is_det = True
                    conf = max(conf, 0.95)

                # 3. Check for parameter mentions
                for p_added in [p.name for n, p in delta.added_parameters if n == sym_name]:
                    if re.search(rf"\b{re.escape(p_added)}\b", content):
                        evidence.append(
                            ImpactEvidence(
                                evidence_type="signature_parameter",
                                matched_text=p_added,
                                context=self._extract_snippet(content, p_added),
                                confidence=0.9,
                            )
                        )
                        is_det = True
                        conf = max(conf, 0.9)

                for p_removed in [p.name for n, p in delta.removed_parameters if n == sym_name]:
                    if re.search(rf"\b{re.escape(p_removed)}\b", content):
                        evidence.append(
                            ImpactEvidence(
                                evidence_type="signature_parameter",
                                matched_text=p_removed,
                                context=self._extract_snippet(content, p_removed),
                                confidence=0.95,
                            )
                        )
                        is_det = True
                        conf = max(conf, 0.95)

                # 4. File path match
                if source_path.lower() in content.lower() or f"`{source_name}`" in content:
                    evidence.append(
                        ImpactEvidence(
                            evidence_type="explicit_file_path",
                            matched_text=source_name,
                            context=self._extract_snippet(content, source_name),
                            confidence=0.5,
                        )
                    )
                    conf = max(conf, 0.5)

                # Direct reference needed for deterministic status
                is_det = any(
                    ev.evidence_type in ("explicit_symbol", "heading_match", "signature_parameter")
                    for ev in evidence
                )

                if evidence:
                    diff_desc = f" ({', '.join(param_diffs)})" if param_diffs else ""
                    reason = (
                        f"Symbol '{sym_name}' in {source_path} was modified{diff_desc}. "
                        f"Section '{sec.title}' documents its interface or usage."
                    )
                    impacts.append(
                        SectionImpact(
                            source_file=source_path,
                            changed_symbol=sym_name,
                            change_category=category,
                            doc_path=doc_path,
                            section_id=sec.id,
                            section_heading=sec.title,
                            is_deterministic=is_det,
                            confidence=conf,
                            evidence=evidence,
                            reason=reason,
                        )
                    )

        return impacts

    def _match_constant(
        self,
        source_path: str,
        const: ConstantInfo,
        category: str,
        docs: dict[str, ParsedDocument],
        old_const: ConstantInfo | None = None,
    ) -> list[SectionImpact]:
        """Find doc sections mentioning changed configuration constants or their values."""
        impacts: list[SectionImpact] = []
        c_name = const.name.split(".")[-1]  # Strip class prefix if present

        for doc_path, doc in docs.items():
            for sec in doc.sections:
                content = doc.get_section_content(sec, include_children=False)
                evidence: list[ImpactEvidence] = []
                is_det = False
                conf = 0.0

                # Match constant identifier (e.g. DEFAULT_PORT or default-port)
                kebab_name = c_name.lower().replace("_", "-")
                if re.search(rf"\b{re.escape(c_name)}\b", content) or re.search(
                    rf"\b{re.escape(kebab_name)}\b", content
                ):
                    evidence.append(
                        ImpactEvidence(
                            evidence_type="explicit_constant",
                            matched_text=c_name,
                            context=self._extract_snippet(content, c_name),
                            confidence=1.0,
                        )
                    )
                    is_det = True
                    conf = 1.0

                # Match constant value if old_const value is mentioned in docs
                if old_const and old_const.value and len(old_const.value) >= 2:
                    if old_const.value in content:
                        evidence.append(
                            ImpactEvidence(
                                evidence_type="explicit_constant",
                                matched_text=old_const.value,
                                context=self._extract_snippet(content, old_const.value),
                                confidence=0.85,
                            )
                        )
                        is_det = True
                        conf = max(conf, 0.85)

                if evidence:
                    reason = (
                        f"Constant '{c_name}' was {category.replace('_', ' ')} in {source_path}. "
                        f"Section '{sec.title}' references this configuration value."
                    )
                    impacts.append(
                        SectionImpact(
                            source_file=source_path,
                            changed_symbol=c_name,
                            change_category=category,
                            doc_path=doc_path,
                            section_id=sec.id,
                            section_heading=sec.title,
                            is_deterministic=is_det,
                            confidence=conf,
                            evidence=evidence,
                            reason=reason,
                        )
                    )

        return impacts

    def _match_import(
        self,
        source_path: str,
        imp: ImportInfo,
        category: str,
        docs: dict[str, ParsedDocument],
    ) -> list[SectionImpact]:
        """Find doc sections affected by dependency changes (Prerequisites, Setup, etc.)."""
        impacts: list[SectionImpact] = []
        dep_name = imp.module or imp.name

        for doc_path, doc in docs.items():
            for sec in doc.sections:
                content = doc.get_section_content(sec, include_children=False)
                sec_title_lower = sec.title.lower()

                evidence: list[ImpactEvidence] = []
                is_det = False
                conf = 0.0

                # Direct reference to dependency package/module
                if re.search(rf"\b{re.escape(dep_name)}\b", content, re.IGNORECASE):
                    evidence.append(
                        ImpactEvidence(
                            evidence_type="explicit_symbol",
                            matched_text=dep_name,
                            context=self._extract_snippet(content, dep_name),
                            confidence=0.9,
                        )
                    )
                    is_det = True
                    conf = 0.9

                # Check if this is an installation / dependencies / requirements section
                is_dep_section = any(
                    kw in sec_title_lower
                    for kw in ("prerequisite", "requirement", "dependenc", "installation", "setup")
                )
                if is_dep_section and dep_name in content:
                    evidence.append(
                        ImpactEvidence(
                            evidence_type="heading_match",
                            matched_text=sec.title,
                            context=sec.title,
                            confidence=0.85,
                        )
                    )
                    is_det = True
                    conf = max(conf, 0.85)

                if evidence:
                    reason = (
                        f"Import '{dep_name}' was {category.replace('_', ' ')} in {source_path}. "
                        f"Section '{sec.title}' documents dependencies or requirements."
                    )
                    impacts.append(
                        SectionImpact(
                            source_file=source_path,
                            changed_symbol=dep_name,
                            change_category=category,
                            doc_path=doc_path,
                            section_id=sec.id,
                            section_heading=sec.title,
                            is_deterministic=is_det,
                            confidence=conf,
                            evidence=evidence,
                            reason=reason,
                        )
                    )

        return impacts

    def _match_implementation_change(
        self,
        source_path: str,
        symbol_name: str,
        docs: dict[str, ParsedDocument],
    ) -> list[SectionImpact]:
        """
        Map implementation-only changes (including private functions and module statements).

        Private changes are NOT ignored if documentation references the feature, behavior,
        or internal component.
        """
        impacts: list[SectionImpact] = []
        sym_tokens = tokenize_name(symbol_name)

        for doc_path, doc in docs.items():
            for sec in doc.sections:
                content = doc.get_section_content(sec, include_children=False)
                sec_tokens = tokenize_name(sec.title) | tokenize_name(" ".join(sec.path))
                heading_lower = sec.title.lower()

                evidence: list[ImpactEvidence] = []
                is_det = False
                conf = 0.0

                # 1. Direct mention of symbol name (even if private)
                if re.search(rf"\b{re.escape(symbol_name)}\b", content):
                    evidence.append(
                        ImpactEvidence(
                            evidence_type="explicit_symbol",
                            matched_text=symbol_name,
                            context=self._extract_snippet(content, symbol_name),
                            confidence=0.9,
                        )
                    )
                    is_det = True
                    conf = 0.9

                # 2. Heading mentions the symbol name
                if symbol_name.lower().strip("_") in heading_lower:
                    evidence.append(
                        ImpactEvidence(
                            evidence_type="heading_match",
                            matched_text=symbol_name,
                            context=sec.title,
                            confidence=0.85,
                        )
                    )
                    is_det = True
                    conf = max(conf, 0.85)

                # 3. Behavioral / Semantic affinity for private or module-level changes
                if not is_det and sym_tokens:
                    overlap = sym_tokens & sec_tokens
                    if overlap and len(overlap) >= max(1, len(sym_tokens) // 2):
                        score = len(overlap) / max(len(sym_tokens), len(sec_tokens))
                        if score >= self.semantic_threshold:
                            evidence.append(
                                ImpactEvidence(
                                    evidence_type="implementation_behavior",
                                    matched_text=", ".join(sorted(overlap)),
                                    context=sec.title,
                                    confidence=round(score, 2),
                                )
                            )
                            conf = round(score, 2)

                if evidence:
                    reason = (
                        f"Implementation change in '{symbol_name}' ({source_path}). "
                        f"Section '{sec.title}' describes the affected behavior or component."
                    )
                    impacts.append(
                        SectionImpact(
                            source_file=source_path,
                            changed_symbol=symbol_name,
                            change_category="implementation_change",
                            doc_path=doc_path,
                            section_id=sec.id,
                            section_heading=sec.title,
                            is_deterministic=is_det,
                            confidence=conf,
                            evidence=evidence,
                            reason=reason,
                        )
                    )

        return impacts

    @staticmethod
    def _extract_snippet(text: str, target: str, window: int = 60) -> str:
        """Extract a readable context window around a target term."""
        idx = text.lower().find(target.lower())
        if idx == -1:
            return text[:window].strip().replace("\n", " ") + "..."

        start = max(0, idx - window // 2)
        end = min(len(text), idx + len(target) + window // 2)
        snippet = text[start:end].replace("\n", " ").strip()
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{snippet}{suffix}"


def consolidate_impacts(impacts: list[SectionImpact]) -> list[ConsolidatedSectionImpact]:
    """
    Group and deduplicate section impacts by (doc_path, section_id).

    Combines multiple code changes affecting the same section into a single,
    prioritized consolidated result.
    """
    groups: dict[tuple[str, str], list[SectionImpact]] = {}
    for imp in impacts:
        key = (imp.doc_path, imp.section_id)
        groups.setdefault(key, []).append(imp)

    consolidated: list[ConsolidatedSectionImpact] = []

    for (doc_path, section_id), group in groups.items():
        first = group[0]
        symbols = sorted({imp.changed_symbol for imp in group})
        sources = sorted({imp.source_file for imp in group})
        categories = sorted({imp.change_category for imp in group})
        is_det = any(imp.is_deterministic for imp in group)
        max_conf = max(imp.confidence for imp in group)

        all_evidence: list[ImpactEvidence] = []
        seen_evidence = set()
        for imp in group:
            for ev in imp.evidence:
                ev_key = (ev.evidence_type, ev.matched_text)
                if ev_key not in seen_evidence:
                    seen_evidence.add(ev_key)
                    all_evidence.append(ev)

        reasons = [imp.reason for imp in group if imp.reason]

        consolidated.append(
            ConsolidatedSectionImpact(
                doc_path=doc_path,
                section_id=section_id,
                section_heading=first.section_heading,
                changed_symbols=symbols,
                source_files=sources,
                change_categories=categories,
                is_deterministic=is_det,
                confidence=round(max_conf, 2),
                evidence=all_evidence,
                reasons=reasons,
            )
        )

    # Sort deterministically: highest confidence first, then doc_path, then section_id
    consolidated.sort(key=lambda c: (-c.confidence, c.doc_path, c.section_id))
    return consolidated


def map_repository_impacts(
    code_changes: dict[str, tuple[str, str]],
    doc_files: dict[str, str] | None = None,
    repo_path: str | Path | None = None,
) -> list[SectionImpact]:
    """
    Convenience entrypoint to map changes across multiple source files to repository documentation.

    - code_changes: mapping of source_file_path -> (old_code, new_code).
    - doc_files: optional mapping of doc_path -> content (in-memory).
    - repo_path: optional base path on disk to auto-discover documentation if doc_files not given.
    """
    mapper = DocumentationImpactMapper()
    all_impacts: list[SectionImpact] = []

    # If doc_files not provided, load from disk
    docs: dict[str, str] = {}
    if doc_files is not None:
        docs = doc_files
    elif repo_path is not None:
        base = Path(repo_path)
        discovered = discover_documentation_files(base)
        for rel_p in discovered:
            full_p = base / rel_p
            try:
                docs[str(rel_p)] = full_p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

    for src_path, (old_src, new_src) in code_changes.items():
        file_impacts = mapper.map_file_impacts(old_src, new_src, src_path, docs)
        all_impacts.extend(file_impacts)

    return all_impacts
