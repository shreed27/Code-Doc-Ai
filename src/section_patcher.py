"""
Markdown Documentation Section Updater.

Provides robust, deterministic parsing, extraction, replacement, and insertion
of sections in Markdown documents without rewriting untouched content.
Preserves non-target sections, formatting, headings, code blocks, and whitespace byte-for-byte.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


class SectionPatcherError(Exception):
    """Base exception for all section patcher operations."""


class SectionNotFoundError(SectionPatcherError):
    """Raised when a requested section cannot be found in the document."""


class AmbiguousSectionError(SectionPatcherError):
    """Raised when a section query matches multiple sections ambiguously."""


class InvalidSectionOperationError(SectionPatcherError):
    """Raised when an invalid operation (e.g. invalid position or level) is attempted."""


def slugify(text: str) -> str:
    """
    Generate a normalized, URL-safe slug from a heading string.

    Converts to lowercase, removes accents/diacritics, and collapses
    non-alphanumeric characters into single hyphens.
    """
    if not text:
        return "section"
    # Normalize unicode (decompose accented characters)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    # Replace non-alphanumeric chars with hyphens
    slug = re.sub(r"[^\w\s-]", "", text)
    slug = re.sub(r"[-\s_]+", "-", slug).strip("-")
    return slug or "section"


@dataclass
class SectionInfo:
    """Metadata and boundary information for a parsed Markdown section."""

    id: str
    title: str
    level: int  # 1-6 for headings, 0 for document preamble
    heading_type: str  # "atx", "setext", or "preamble"
    path: list[str]  # Heading titles from root down to this section
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    occurrence: int = 1  # 1-based occurrence among identical hierarchical paths

    # 0-based character offsets in original text
    start_offset: int = 0  # Start of heading (or 0 for preamble)
    heading_end_offset: int = 0  # End of heading line(s)
    body_end_offset: int = (
        0  # End of immediate body (before first child heading, or total_end_offset)
    )
    total_end_offset: int = 0  # End of entire section including all descendants

    # 1-based line positions for inspection
    start_line: int = 1
    heading_line_count: int = 1
    body_start_line: int = 2
    body_end_line: int = 2
    total_end_line: int = 2

    @property
    def is_preamble(self) -> bool:
        """True if this section represents content preceding the first heading."""
        return self.level == 0


@dataclass
class _RawHeading:
    """Internal representation of a discovered heading before tree linking."""

    title: str
    level: int
    heading_type: str
    start_line: int
    heading_line_count: int
    start_offset: int
    heading_end_offset: int
    path: list[str] = field(default_factory=list)
    parent_index: int | None = None
    child_indices: list[int] = field(default_factory=list)


class ParsedDocument:
    """Represents a parsed Markdown document with indexed section hierarchy."""

    def __init__(self, raw_content: str, sections: list[SectionInfo]):
        self.raw_content = raw_content
        self.sections = sections
        self._id_map: dict[str, SectionInfo] = {s.id: s for s in sections}
        self.newline = "\r\n" if "\r\n" in raw_content else "\n"

    def __len__(self) -> int:
        return len(self.sections)

    def __iter__(self):
        return iter(self.sections)

    def __getitem__(self, item: int | str) -> SectionInfo:
        return self.get_section(item)

    @property
    def section_ids(self) -> list[str]:
        """List of all section IDs in document order."""
        return [s.id for s in self.sections]

    def get_section(
        self, query: str | int | SectionInfo | list[str] | tuple[str, ...]
    ) -> SectionInfo:
        """
        Retrieve a section by its ID, title, hierarchical path, or index.

        Raises SectionNotFoundError if not found.
        Raises AmbiguousSectionError if multiple sections match.
        """
        sec = self.find_section(query)
        if sec is None:
            raise SectionNotFoundError(f"Section '{query}' not found in document.")
        return sec

    def find_section(
        self, query: str | int | SectionInfo | list[str] | tuple[str, ...]
    ) -> SectionInfo | None:
        """
        Locate a section by query, returning None if not found.

        Raises AmbiguousSectionError if multiple sections match.
        """
        if isinstance(query, SectionInfo):
            if query.id in self._id_map:
                return self._id_map[query.id]
            raise SectionNotFoundError(f"Section with ID '{query.id}' not found in document.")

        if isinstance(query, int):
            if -len(self.sections) <= query < len(self.sections):
                return self.sections[query]
            return None

        if isinstance(query, list | tuple):
            return self._resolve_path_query(list(query))

        if not isinstance(query, str):
            raise InvalidSectionOperationError(f"Unsupported query type: {type(query).__name__}")

        query_str = query.strip()
        if not query_str:
            raise SectionNotFoundError("Empty section query provided.")

        # 1. Exact match by canonical ID
        if query_str in self._id_map:
            return self._id_map[query_str]

        # 2. Check for explicit path separator ' > ' or ' / '
        if " > " in query_str or " / " in query_str:
            parts = [p.strip() for p in re.split(r"\s+[>/]\s+", query_str) if p.strip()]
            return self._resolve_path_query(parts)

        # 3. Check for occurrence qualifier on ID or path (e.g. 'slug#2')
        if "#" in query_str:
            base_q, occ_str = query_str.rsplit("#", 1)
            if occ_str.isdigit():
                occ_target = int(occ_str)
                matching_occ = [
                    s
                    for s in self.sections
                    if (
                        s.id == base_q
                        or s.title.lower() == base_q.lower()
                        or s.id.startswith(base_q)
                    )
                    and s.occurrence == occ_target
                ]
                if len(matching_occ) == 1:
                    return matching_occ[0]
                if len(matching_occ) > 1:
                    raise AmbiguousSectionError(
                        f"Ambiguous query '{query_str}'. Matches: {[s.id for s in matching_occ]}"
                    )

        # 4. Match by path slug
        slug_target = slugify(query_str)
        slug_matches = [
            s
            for s in self.sections
            if s.id == query_str or s.id == slug_target or s.id.split("#")[0] == slug_target
        ]
        if len(slug_matches) == 1:
            return slug_matches[0]
        if len(slug_matches) > 1:
            raise AmbiguousSectionError(
                f"Multiple sections match slug '{query_str}': {[s.id for s in slug_matches]}. "
                f"Specify hierarchical path or occurrence number (e.g. '{slug_matches[0].id}')."
            )

        # 5. Case-insensitive exact title match
        title_matches = [s for s in self.sections if s.title.lower() == query_str.lower()]
        if len(title_matches) == 1:
            return title_matches[0]
        if len(title_matches) > 1:
            raise AmbiguousSectionError(
                f"Multiple sections share title '{query_str}': {[s.id for s in title_matches]}. "
                f"Specify parent hierarchy or exact ID."
            )

        # 6. Suffix match on hierarchical ID (e.g. 'prerequisites' matching 'setup/prerequisites')
        suffix_matches = [
            s
            for s in self.sections
            if s.id.endswith("/" + slug_target) or s.id.split("#")[0].endswith("/" + slug_target)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            raise AmbiguousSectionError(
                f"Multiple sections end with '{slug_target}': {[s.id for s in suffix_matches]}."
            )

        return None

    def _resolve_path_query(self, path_parts: list[str]) -> SectionInfo:
        """Resolve a sequence of nested titles/slugs."""
        if not path_parts:
            raise SectionNotFoundError("Empty path query.")

        norm_path = [slugify(p) for p in path_parts]
        candidates = []
        for s in self.sections:
            s_slug_path = [slugify(p) for p in s.path]
            if len(s_slug_path) >= len(norm_path):
                # Check if end of path matches
                if s_slug_path[-len(norm_path) :] == norm_path:
                    candidates.append(s)

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # Check for exact length match
            exact_depth = [s for s in candidates if len(s.path) == len(norm_path)]
            if len(exact_depth) == 1:
                return exact_depth[0]
            raise AmbiguousSectionError(
                f"Path query '{' > '.join(path_parts)}' matches multiple sections: "
                f"{[s.id for s in candidates]}."
            )

        raise SectionNotFoundError(f"Path query '{' > '.join(path_parts)}' not found in document.")

    def get_section_content(
        self,
        query: str | int | SectionInfo | list[str] | tuple[str, ...],
        include_children: bool = False,
    ) -> str:
        """
        Extract the text content of a section.

        If include_children is False, returns only the immediate section body.
        If include_children is True, returns the body plus all subsection contents.
        """
        sec = self.get_section(query)
        end = sec.total_end_offset if include_children else sec.body_end_offset
        return self.raw_content[sec.heading_end_offset : end]

    def get_section_heading(
        self, query: str | int | SectionInfo | list[str] | tuple[str, ...]
    ) -> str:
        """Extract the exact raw heading text of a section."""
        sec = self.get_section(query)
        return self.raw_content[sec.start_offset : sec.heading_end_offset]

    def get_parent(
        self, query: str | int | SectionInfo | list[str] | tuple[str, ...]
    ) -> SectionInfo | None:
        """Get the parent section of the specified section, or None if top-level."""
        sec = self.get_section(query)
        if sec.parent_id:
            return self._id_map.get(sec.parent_id)
        return None

    def get_children(
        self, query: str | int | SectionInfo | list[str] | tuple[str, ...]
    ) -> list[SectionInfo]:
        """Get immediate child sections of the specified section."""
        sec = self.get_section(query)
        return [self._id_map[cid] for cid in sec.child_ids if cid in self._id_map]

    def get_siblings(
        self, query: str | int | SectionInfo | list[str] | tuple[str, ...]
    ) -> list[SectionInfo]:
        """Get sibling sections sharing the same parent (excluding self)."""
        sec = self.get_section(query)
        if sec.parent_id:
            parent = self._id_map[sec.parent_id]
            return [self._id_map[cid] for cid in parent.child_ids if cid != sec.id]
        # Top-level siblings
        return [
            s for s in self.sections if s.parent_id is None and s.id != sec.id and not s.is_preamble
        ]


def parse_sections(markdown_text: str) -> ParsedDocument:
    """
    Parse a Markdown document into a structured hierarchy of sections.

    Supports ATX headings (# - ######), Setext headings (=== and ---),
    fenced code blocks (``` and ~~~), and preamble sections.
    """
    if not markdown_text:
        return ParsedDocument(markdown_text, [])

    lines = markdown_text.splitlines(keepends=True)
    line_offsets: list[int] = []
    curr_offset = 0
    for line in lines:
        line_offsets.append(curr_offset)
        curr_offset += len(line)
    line_offsets.append(curr_offset)  # End of text

    raw_headings: list[_RawHeading] = []
    in_code_block = False
    fence_char = ""
    fence_len = 0

    idx = 0
    total_lines = len(lines)

    while idx < total_lines:
        line = lines[idx]

        # 1. Handle fenced code blocks
        if in_code_block:
            close_pattern = rf"^( {{0,3}}){re.escape(fence_char)}{{{fence_len},}}[ \t]*$"
            if re.match(close_pattern, line):
                in_code_block = False
                fence_char = ""
                fence_len = 0
            idx += 1
            continue

        open_fence_match = re.match(r"^( {0,3})(`{3,}|~{3,})(.*)$", line)
        if open_fence_match:
            f_char = open_fence_match.group(2)[0]
            f_len = len(open_fence_match.group(2))
            info = open_fence_match.group(3).strip()
            # In CommonMark, backtick fence info cannot contain backtick
            if not (f_char == "`" and "`" in info):
                in_code_block = True
                fence_char = f_char
                fence_len = f_len
                idx += 1
                continue

        # 2. Check for ATX Heading (# through ######)
        atx_match = re.match(r"^( {0,3})(#{1,6})(?:[ \t]+(.*)|$)", line.rstrip("\r\n"))
        if atx_match:
            level = len(atx_match.group(2))
            raw_title = atx_match.group(3) or ""
            # Strip optional trailing sequence of hashes preceded by whitespace
            clean_title = re.sub(r"[ \t]+#+[ \t]*$", "", raw_title).strip()
            start_off = line_offsets[idx]
            end_off = line_offsets[idx + 1]
            raw_headings.append(
                _RawHeading(
                    title=clean_title,
                    level=level,
                    heading_type="atx",
                    start_line=idx + 1,
                    heading_line_count=1,
                    start_offset=start_off,
                    heading_end_offset=end_off,
                )
            )
            idx += 1
            continue

        # 3. Check for Setext Heading (line followed by === or ---)
        if idx + 1 < total_lines:
            next_line = lines[idx + 1]
            line_stripped = line.strip()
            # Disqualify empty lines, list items, blockquotes, or thematic breaks
            is_disqualified = (
                not line_stripped
                or re.match(r"^ {0,3}(?:[-*+]|\d+\.)[ \t]+", line) is not None
                or line.lstrip().startswith(">")
            )
            if not is_disqualified:
                setext_match = re.match(r"^( {0,3})(=+|-+)[ \t]*$", next_line.rstrip("\r\n"))
                if setext_match:
                    under_char = setext_match.group(2)[0]
                    level = 1 if under_char == "=" else 2
                    clean_title = line_stripped
                    start_off = line_offsets[idx]
                    end_off = line_offsets[idx + 2]
                    raw_headings.append(
                        _RawHeading(
                            title=clean_title,
                            level=level,
                            heading_type="setext",
                            start_line=idx + 1,
                            heading_line_count=2,
                            start_offset=start_off,
                            heading_end_offset=end_off,
                        )
                    )
                    idx += 2
                    continue

        idx += 1

    sections: list[SectionInfo] = []

    # 4. Handle Preamble (content before first heading)
    first_heading_offset = raw_headings[0].start_offset if raw_headings else len(markdown_text)
    if first_heading_offset > 0:
        first_heading_line = raw_headings[0].start_line if raw_headings else total_lines + 1
        preamble = SectionInfo(
            id="__preamble__",
            title="",
            level=0,
            heading_type="preamble",
            path=[],
            start_offset=0,
            heading_end_offset=0,
            body_end_offset=first_heading_offset,
            total_end_offset=first_heading_offset,
            start_line=1,
            heading_line_count=0,
            body_start_line=1,
            body_end_line=first_heading_line,
            total_end_line=first_heading_line,
        )
        sections.append(preamble)

    if not raw_headings:
        return ParsedDocument(markdown_text, sections)

    # 5. Build Parent-Child Hierarchy
    ancestor_stack: list[int] = []  # Stores indices in raw_headings
    for i, rh in enumerate(raw_headings):
        while ancestor_stack and raw_headings[ancestor_stack[-1]].level >= rh.level:
            ancestor_stack.pop()

        if ancestor_stack:
            parent_idx = ancestor_stack[-1]
            rh.parent_index = parent_idx
            raw_headings[parent_idx].child_indices.append(i)
            rh.path = raw_headings[parent_idx].path + [rh.title]
        else:
            rh.parent_index = None
            rh.path = [rh.title]

        ancestor_stack.append(i)

    # 6. Assign Canonical Identifiers and Compute Offsets
    slug_counts: dict[str, int] = {}
    sec_infos: list[SectionInfo] = []

    for i, rh in enumerate(raw_headings):
        base_slug = "/".join(slugify(t) for t in rh.path) if rh.path else "section"
        slug_counts[base_slug] = slug_counts.get(base_slug, 0) + 1
        occ = slug_counts[base_slug]

        canonical_id = base_slug if occ == 1 else f"{base_slug}#{occ}"

        # Total section ends at next heading with level <= rh.level, or end of text
        total_end_off = len(markdown_text)
        total_end_line = total_lines + 1
        for j in range(i + 1, len(raw_headings)):
            if raw_headings[j].level <= rh.level:
                total_end_off = raw_headings[j].start_offset
                total_end_line = raw_headings[j].start_line
                break

        # Immediate body ends at first child heading, or at total_end_off
        if rh.child_indices:
            first_child = raw_headings[rh.child_indices[0]]
            body_end_off = first_child.start_offset
            body_end_line = first_child.start_line
        else:
            body_end_off = total_end_off
            body_end_line = total_end_line

        sec_info = SectionInfo(
            id=canonical_id,
            title=rh.title,
            level=rh.level,
            heading_type=rh.heading_type,
            path=rh.path,
            occurrence=occ,
            start_offset=rh.start_offset,
            heading_end_offset=rh.heading_end_offset,
            body_end_offset=body_end_off,
            total_end_offset=total_end_off,
            start_line=rh.start_line,
            heading_line_count=rh.heading_line_count,
            body_start_line=rh.start_line + rh.heading_line_count,
            body_end_line=body_end_line,
            total_end_line=total_end_line,
        )
        sec_infos.append(sec_info)

    # 7. Link Parent and Child IDs
    for i, rh in enumerate(raw_headings):
        current_sec = sec_infos[i]
        if rh.parent_index is not None:
            current_sec.parent_id = sec_infos[rh.parent_index].id
        current_sec.child_ids = [sec_infos[ci].id for ci in rh.child_indices]

    sections.extend(sec_infos)
    return ParsedDocument(markdown_text, sections)


def replace_section_content(
    markdown_text: str,
    target: str | int | SectionInfo | list[str] | tuple[str, ...],
    new_content: str,
    replace_children: bool = False,
) -> str:
    """
    Replace the content of a target Markdown section.

    - If replace_children is False: only replaces the immediate body between
      the heading and the first child subsection. All child sections remain intact.
    - If replace_children is True: replaces from after the heading through
      the end of all descendant subsections.
    - Preserves all untouched content byte-for-byte.
    - Preserves the original heading.
    """
    doc = parse_sections(markdown_text)
    section = doc.get_section(target)

    # Determine replacement boundaries
    start_offset = section.heading_end_offset
    end_offset = section.total_end_offset if replace_children else section.body_end_offset

    nl = doc.newline

    # Format new content
    clean_body = new_content.strip("\r\n")

    if section.is_preamble:
        if clean_body:
            formatted_body = (
                f"{clean_body}{nl}{nl}" if end_offset < len(markdown_text) else f"{clean_body}{nl}"
            )
        else:
            formatted_body = ""
    else:
        if clean_body:
            if end_offset < len(markdown_text):
                formatted_body = f"{nl}{clean_body}{nl}{nl}"
            else:
                formatted_body = f"{nl}{clean_body}{nl}"
        else:
            if end_offset < len(markdown_text):
                formatted_body = f"{nl}"
            else:
                formatted_body = ""

    prefix = markdown_text[:start_offset]
    suffix = markdown_text[end_offset:]

    return prefix + formatted_body + suffix


def insert_section(
    markdown_text: str,
    title: str,
    content: str,
    level: int | None = None,
    target: str | int | SectionInfo | list[str] | tuple[str, ...] | None = None,
    position: str = "after",
) -> str:
    """
    Insert a new section into a Markdown document.

    Positions:
    - "after": insert after target section (and after all its child sections).
    - "before": insert immediately before target section.
    - "child": insert as the last child subsection of target section.
    - "prepend_child": insert as the first child subsection of target section.
    - "append": append at document end (when target is None or document).
    - "prepend": prepend at document start (after preamble if present).
    """
    clean_title = title.strip()
    if not clean_title:
        raise InvalidSectionOperationError("Section title cannot be empty.")

    clean_content = content.strip("\r\n")
    pos = position.lower().strip()

    if not markdown_text:
        lvl = level if level is not None else 1
        lvl = max(1, min(6, lvl))
        heading_line = f"{'#' * lvl} {clean_title}\n"
        body = f"\n{clean_content}\n" if clean_content else ""
        return heading_line + body

    doc = parse_sections(markdown_text)
    nl = doc.newline

    if target is None:
        if pos in ("append", "after"):
            lvl = level if level is not None else 1
            lvl = max(1, min(6, lvl))
            heading_str = f"{'#' * lvl} {clean_title}"
            body_str = f"{nl}{nl}{clean_content}" if clean_content else ""
            block = f"{nl}{nl}{heading_str}{body_str}{nl}"
            return markdown_text.rstrip("\r\n") + block
        if pos in ("prepend", "before"):
            lvl = level if level is not None else 1
            lvl = max(1, min(6, lvl))
            heading_str = f"{'#' * lvl} {clean_title}"
            body_str = f"{nl}{nl}{clean_content}" if clean_content else ""
            block = f"{heading_str}{body_str}{nl}{nl}"
            preamble = doc.find_section("__preamble__")
            if preamble and preamble.body_end_offset > 0:
                off = preamble.body_end_offset
                return markdown_text[:off] + block + markdown_text[off:]
            return block + markdown_text
        raise InvalidSectionOperationError(
            f"Position '{position}' requires a target section to be specified."
        )

    target_sec = doc.get_section(target)

    if pos == "after":
        insert_offset = target_sec.total_end_offset
        lvl = level if level is not None else target_sec.level
        lvl = max(1, min(6, lvl))
    elif pos == "before":
        insert_offset = target_sec.start_offset
        lvl = level if level is not None else target_sec.level
        lvl = max(1, min(6, lvl))
    elif pos in ("child", "inside"):
        insert_offset = target_sec.total_end_offset
        lvl = level if level is not None else min(6, target_sec.level + 1)
        lvl = max(1, min(6, lvl))
    elif pos == "prepend_child":
        insert_offset = target_sec.body_end_offset
        lvl = level if level is not None else min(6, target_sec.level + 1)
        lvl = max(1, min(6, lvl))
    else:
        raise InvalidSectionOperationError(
            f"Unsupported insert position '{position}'. "
            f"Choose from: 'after', 'before', 'child', 'prepend_child', 'append', 'prepend'."
        )

    heading_str = f"{'#' * lvl} {clean_title}"
    body_str = f"{nl}{nl}{clean_content}" if clean_content else ""

    if insert_offset == 0:
        section_block = f"{heading_str}{body_str}{nl}{nl}"
    elif insert_offset >= len(markdown_text):
        section_block = f"{nl}{nl}{heading_str}{body_str}{nl}"
    else:
        section_block = f"{nl}{nl}{heading_str}{body_str}"

    return markdown_text[:insert_offset] + section_block + markdown_text[insert_offset:]


def delete_section(
    markdown_text: str,
    target: str | int | SectionInfo | list[str] | tuple[str, ...],
    delete_children: bool = True,
) -> str:
    """
    Remove a section from the document.

    If delete_children is True, removes the section and all its descendants.
    If delete_children is False, removes the section heading and immediate body,
    promoting child sections.
    """
    doc = parse_sections(markdown_text)
    sec = doc.get_section(target)

    start_off = sec.start_offset
    end_off = sec.total_end_offset if delete_children else sec.body_end_offset

    prefix = markdown_text[:start_off]
    suffix = markdown_text[end_off:]

    return prefix + suffix


def get_section_content(
    markdown_text: str,
    target: str | int | SectionInfo | list[str] | tuple[str, ...],
    include_children: bool = False,
) -> str:
    """Convenience helper to extract section content directly from Markdown text."""
    doc = parse_sections(markdown_text)
    return doc.get_section_content(target, include_children=include_children)
