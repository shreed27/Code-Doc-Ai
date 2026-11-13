"""
Tests for Markdown Section Patcher.

Verifies section parsing, hierarchy building, ambiguous resolution,
surgical replacement, section insertion, code block fencing, Setext headings,
and byte-for-byte preservation of untouched content.
"""

import pytest

from section_patcher import (
    AmbiguousSectionError,
    InvalidSectionOperationError,
    SectionInfo,
    SectionNotFoundError,
    delete_section,
    get_section_content,
    insert_section,
    parse_sections,
    replace_section_content,
    slugify,
)


class TestSlugify:
    """Test heading slugification and normalization."""

    def test_basic_slugify(self):
        assert slugify("Getting Started") == "getting-started"
        assert slugify("API Reference (v2.0)") == "api-reference-v20"
        assert slugify("  Trim & Spaces  ") == "trim-spaces"

    def test_unicode_and_empty(self):
        assert slugify("Café & Résumé") == "cafe-resume"
        assert slugify("") == "section"
        assert slugify("---") == "section"


class TestMarkdownParser:
    """Test parsing headings, hierarchy, code blocks, and edge cases."""

    def test_empty_document(self):
        doc = parse_sections("")
        assert len(doc) == 0
        assert doc.sections == []

    def test_headingless_document(self):
        text = "This is a document with no headings at all.\nJust several lines of text.\n"
        doc = parse_sections(text)
        assert len(doc) == 1
        sec = doc[0]
        assert sec.id == "__preamble__"
        assert sec.is_preamble is True
        assert sec.level == 0
        assert doc.get_section_content(sec) == text

    def test_atx_headings_levels_and_trailing_hashes(self):
        text = """# Heading 1
Intro text.

## Heading 2 ##
Subtext.

### Heading 3 #######
Deep text.

#### Heading 4
More text.

##### Heading 5
Level 5.

###### Heading 6
Level 6.
"""
        doc = parse_sections(text)
        assert len(doc) == 6
        assert [s.level for s in doc] == [1, 2, 3, 4, 5, 6]
        assert doc[0].title == "Heading 1"
        assert doc[1].title == "Heading 2"
        assert doc[2].title == "Heading 3"
        assert doc[3].title == "Heading 4"
        assert doc[4].title == "Heading 5"
        assert doc[5].title == "Heading 6"

    def test_setext_headings(self):
        text = """First Title
===========
This is level 1 setext.

Second Title
------------
This is level 2 setext.

Normal paragraph
not a heading.
"""
        doc = parse_sections(text)
        assert len(doc) == 2
        assert doc[0].title == "First Title"
        assert doc[0].level == 1
        assert doc[0].heading_type == "setext"

        assert doc[1].title == "Second Title"
        assert doc[1].level == 2
        assert doc[1].heading_type == "setext"

    def test_fenced_code_blocks_ignore_headings(self):
        text = """# Real Title
Before code block.

```python
# This is a python comment, not a markdown heading
def foo():
    pass
## Still not a heading
```

~~~markdown
# Fenced with tildes
~~~

## Real Subsection
End content.
"""
        doc = parse_sections(text)
        assert len(doc) == 2
        assert doc[0].title == "Real Title"
        assert doc[1].title == "Real Subsection"
        assert doc[1].parent_id == "real-title"

    def test_nested_hierarchy_and_skipping_levels(self):
        text = """# Top
Root text.

### Deep Child (skipping h2)
Deep child text.

## Sibling H2
Sibling text.

#### Grandchild
Grandchild text.
"""
        doc = parse_sections(text)
        assert len(doc) == 4

        top = doc.get_section("top")
        deep = doc.get_section("top/deep-child-skipping-h2")
        h2 = doc.get_section("top/sibling-h2")
        grand = doc.get_section("top/sibling-h2/grandchild")

        assert deep.parent_id == top.id
        assert h2.parent_id == top.id
        assert grand.parent_id == h2.id

        assert top.child_ids == [deep.id, h2.id]
        assert h2.child_ids == [grand.id]

    def test_duplicate_heading_names_same_and_different_parents(self):
        text = """# Service A
Intro A.

## Methods
Methods of A.

# Service B
Intro B.

## Methods
Methods of B.

# Summary
Summary 1.

# Summary
Summary 2.
"""
        doc = parse_sections(text)
        assert len(doc) == 6

        # Different parents have unique hierarchical IDs
        assert "service-a/methods" in doc.section_ids
        assert "service-b/methods" in doc.section_ids

        # Same parent / top-level duplicates receive occurrence suffixes
        assert "summary" in doc.section_ids
        assert "summary#2" in doc.section_ids

    def test_preamble_with_headings(self):
        text = """This is a document preamble.
It has multiple lines.

# First Section
Section body.
"""
        doc = parse_sections(text)
        assert len(doc) == 2
        assert doc[0].id == "__preamble__"
        assert doc[0].is_preamble is True
        assert "This is a document preamble." in doc.get_section_content("__preamble__")
        assert doc[1].id == "first-section"


class TestSectionQueryResolution:
    """Test resolving sections by ID, title, path, index, and occurrence."""

    text = """# Overview
General overview.

## Installation
Setup guide.

### Prerequisites
Need Python 3.12.

# Advanced
Advanced topics.

## Configuration
Config options.

### Prerequisites
Need root access.

# Epilogue
First epilogue.

# Epilogue
Second epilogue.
"""

    def test_exact_id_resolution(self):
        doc = parse_sections(self.text)
        sec = doc.get_section("overview/installation/prerequisites")
        assert sec.title == "Prerequisites"
        assert sec.parent_id == "overview/installation"

    def test_index_resolution(self):
        doc = parse_sections(self.text)
        assert doc.get_section(0).title == "Overview"
        assert doc[1].title == "Installation"
        assert doc.find_section(999) is None

    def test_hierarchical_path_list_and_string(self):
        doc = parse_sections(self.text)
        sec1 = doc.get_section(["Overview", "Installation", "Prerequisites"])
        sec2 = doc.get_section("Overview > Installation > Prerequisites")
        assert sec1.id == sec2.id == "overview/installation/prerequisites"

    def test_ambiguous_title_raises_error(self):
        doc = parse_sections(self.text)
        # 'Prerequisites' exists in both Overview and Advanced
        with pytest.raises(AmbiguousSectionError) as exc_info:
            doc.get_section("Prerequisites")
        assert "Multiple sections share title" in str(exc_info.value)

    def test_ambiguous_duplicate_top_level_raises_error(self):
        doc = parse_sections(self.text)
        # 'Epilogue' appears twice at top level
        with pytest.raises(AmbiguousSectionError) as exc_info:
            doc.get_section("Epilogue")
        assert "Multiple sections" in str(exc_info.value)

    def test_disambiguated_occurrence_queries(self):
        doc = parse_sections(self.text)
        sec1 = doc.get_section("epilogue")
        sec2 = doc.get_section("epilogue#2")
        assert sec1.occurrence == 1
        assert sec2.occurrence == 2
        assert "First epilogue" in doc.get_section_content(sec1)
        assert "Second epilogue" in doc.get_section_content(sec2)

    def test_section_not_found(self):
        doc = parse_sections(self.text)
        with pytest.raises(SectionNotFoundError):
            doc.get_section("NonexistentSection")

    def test_section_hierarchy_navigation(self):
        doc = parse_sections(self.text)
        install = doc.get_section("overview/installation")
        parent = doc.get_parent(install)
        children = doc.get_children(install)
        siblings = doc.get_siblings(install)

        assert parent is not None and parent.id == "overview"
        assert len(children) == 1 and children[0].title == "Prerequisites"
        assert siblings == []  # No other h2 under overview


class TestSectionReplacement:
    """Test surgical section replacement and byte-for-byte preservation."""

    sample_doc = """# Introduction
Welcome to our project.
This is the intro body.

## Getting Started
First steps here.

### Installation
Run `pip install foo`.

### Quickstart
Run `foo start`.

## Troubleshooting
Common issues here.

# FAQ
Frequently asked questions.
"""

    def test_replace_middle_leaf_section_preserves_everything_else(self):
        new_install = "Run `uv pip install modern-pkg`.\nEnsure Python >= 3.12."
        updated = replace_section_content(self.sample_doc, "installation", new_install)

        doc_orig = parse_sections(self.sample_doc)
        install_sec = doc_orig.get_section("installation")

        # Untouched prefix up to end of heading line
        prefix = self.sample_doc[: install_sec.heading_end_offset]
        assert updated.startswith(prefix)

        # Untouched suffix after installation body
        suffix = self.sample_doc[install_sec.body_end_offset :]
        assert updated.endswith(suffix)

        # Verify parsed updated doc has the new content
        doc_updated = parse_sections(updated)
        assert "uv pip install modern-pkg" in doc_updated.get_section_content("installation")
        assert "Run `foo start`." in doc_updated.get_section_content("quickstart")

    def test_replace_first_section_preserves_tail(self):
        updated = replace_section_content(
            self.sample_doc, "introduction", "Brand new introduction text."
        )
        assert updated.startswith(
            "# Introduction\n\nBrand new introduction text.\n\n## Getting Started"
        )
        assert "Frequently asked questions." in updated

    def test_replace_last_section_preserves_head(self):
        updated = replace_section_content(self.sample_doc, "faq", "No questions yet.")
        assert "This is the intro body." in updated
        assert updated.endswith("# FAQ\n\nNo questions yet.\n")

    def test_replace_parent_without_children_preserves_subsections(self):
        # Replacing 'Getting Started' with replace_children=False must leave Installation and Quickstart intact
        updated = replace_section_content(
            self.sample_doc,
            "getting-started",
            "Updated getting started overview.",
            replace_children=False,
        )
        doc = parse_sections(updated)
        assert "Updated getting started overview." in doc.get_section_content(
            "getting-started", include_children=False
        )
        assert "Run `pip install foo`." in doc.get_section_content("installation")
        quick_content = doc.get_section_content("quickstart")
        assert "Run `foo start`." in quick_content

    def test_replace_parent_with_children_replaces_entire_subtree(self):
        # Replacing 'Getting Started' with replace_children=True wipes out Installation and Quickstart
        updated = replace_section_content(
            self.sample_doc,
            "getting-started",
            "Getting started is now a single unified step.",
            replace_children=True,
        )
        doc = parse_sections(updated)
        assert "installation" not in doc.section_ids
        assert "quickstart" not in doc.section_ids
        assert "introduction/troubleshooting" in doc.section_ids
        assert "Common issues here." in doc.get_section_content("troubleshooting")

    def test_replace_preamble(self):
        text = """Old preamble text.

# Main Title
Body.
"""
        updated = replace_section_content(text, "__preamble__", "New preamble text.")
        assert updated == "New preamble text.\n\n# Main Title\nBody.\n"

    def test_replacement_idempotency(self):
        new_content = "Consistently updated content."
        first_pass = replace_section_content(self.sample_doc, "troubleshooting", new_content)
        second_pass = replace_section_content(first_pass, "troubleshooting", new_content)
        assert first_pass == second_pass


class TestSectionInsertion:
    """Test inserting sections at various document locations."""

    sample_doc = """# Guide
Introductory guide.

## Setup
Setup info.

## Usage
Usage info.
"""

    def test_insert_after_section(self):
        updated = insert_section(
            self.sample_doc,
            title="Prerequisites",
            content="Install Docker.",
            target="setup",
            position="after",
        )
        doc = parse_sections(updated)
        assert "guide/prerequisites" in doc.section_ids
        prereq = doc.get_section("guide/prerequisites")
        assert prereq.level == 2
        assert "Install Docker." in doc.get_section_content(prereq)

    def test_insert_before_section(self):
        updated = insert_section(
            self.sample_doc,
            title="Before Setup",
            content="Check system requirements.",
            target="setup",
            position="before",
        )
        doc = parse_sections(updated)
        assert "guide/before-setup" in doc.section_ids
        sec = doc.get_section("guide/before-setup")
        assert sec.level == 2

    def test_insert_as_child_subsection(self):
        updated = insert_section(
            self.sample_doc,
            title="Configuration Details",
            content="Configure config.yaml.",
            target="usage",
            position="child",
        )
        doc = parse_sections(updated)
        sec = doc.get_section("guide/usage/configuration-details")
        assert sec.level == 3
        assert sec.parent_id == "guide/usage"

    def test_insert_prepend_child(self):
        updated = insert_section(
            self.sample_doc,
            title="Quick Notes",
            content="Read this first.",
            target="guide",
            position="prepend_child",
        )
        doc = parse_sections(updated)
        guide = doc.get_section("guide")
        first_child = doc.get_section(guide.child_ids[0])
        assert first_child.title == "Quick Notes"

    def test_insert_append_to_document_without_target(self):
        updated = insert_section(
            self.sample_doc,
            title="License",
            content="MIT License.",
            target=None,
            position="append",
        )
        doc = parse_sections(updated)
        assert doc[-1].title == "License"
        assert "MIT License." in doc.get_section_content("license")

    def test_insert_into_empty_document(self):
        updated = insert_section(
            "",
            title="Brand New Doc",
            content="Welcome.",
            level=1,
        )
        assert updated == "# Brand New Doc\n\nWelcome.\n"

    def test_insert_prepend_to_document_with_and_without_preamble(self):
        doc_with_pre = "Introductory text.\n\n# First\nBody."
        updated1 = insert_section(
            doc_with_pre,
            title="Pre-inserted",
            content="Top note.",
            target=None,
            position="prepend",
        )
        assert "Introductory text." in updated1
        assert updated1.startswith("Introductory text.\n\n# Pre-inserted\n\nTop note.\n\n# First")

        doc_no_pre = "# First\nBody."
        updated2 = insert_section(
            doc_no_pre,
            title="Pre-inserted",
            content="Top note.",
            target=None,
            position="prepend",
        )
        assert updated2.startswith("# Pre-inserted\n\nTop note.\n\n# First")

    def test_insert_before_first_section(self):
        doc = "# First\nBody."
        updated = insert_section(
            doc,
            title="Zero",
            content="Pre-first content.",
            target="first",
            position="before",
        )
        assert updated.startswith("# Zero\n\nPre-first content.\n\n# First")

    def test_insert_missing_target_for_child_raises_error(self):
        with pytest.raises(InvalidSectionOperationError):
            insert_section("# Doc\n", "Title", "Body", target=None, position="child")


class TestSectionDeletion:
    """Test section deletion with and without child removal."""

    sample_doc = """# Chapter 1
Text 1.

## Section 1.1
Text 1.1.

# Chapter 2
Text 2.
"""

    def test_delete_section_with_children(self):
        updated = delete_section(self.sample_doc, "chapter-1", delete_children=True)
        doc = parse_sections(updated)
        assert "chapter-1" not in doc.section_ids
        assert "chapter-1/section-11" not in doc.section_ids
        assert "chapter-2" in doc.section_ids

    def test_delete_leaf_section(self):
        updated = delete_section(self.sample_doc, "section-11")
        doc = parse_sections(updated)
        assert "chapter-1" in doc.section_ids
        assert "chapter-1/section-11" not in doc.section_ids
        assert "chapter-2" in doc.section_ids


class TestErrorHandlingAndValidation:
    """Test safe failure behaviors on invalid or ambiguous actions."""

    def test_empty_query_raises_not_found(self):
        doc = parse_sections("# Title\nBody")
        with pytest.raises(SectionNotFoundError):
            doc.get_section("")

    def test_invalid_position_raises_error(self):
        with pytest.raises(InvalidSectionOperationError):
            insert_section("# Doc\n", "Title", "Body", target="doc", position="invalid_pos")

    def test_empty_insert_title_raises_error(self):
        with pytest.raises(InvalidSectionOperationError):
            insert_section("# Doc\n", "", "Body")

    def test_get_section_content_standalone_helper(self):
        text = "# Section\nContent here."
        content = get_section_content(text, "section")
        assert "Content here." in content

    def test_get_section_heading_and_parent_siblings(self):
        text = "# Top 1\nBody 1.\n\n## Sub\nSub body.\n\n# Top 2\nBody 2."
        doc = parse_sections(text)
        assert doc.get_section_heading("top-1") == "# Top 1\n"
        assert doc.get_parent("top-1") is None
        top_siblings = doc.get_siblings("top-1")
        assert len(top_siblings) == 1
        assert top_siblings[0].title == "Top 2"

    def test_replace_with_empty_content_clears_section(self):
        text = "# First\nBody 1.\n\n# Second\nBody 2.\n"
        updated = replace_section_content(text, "first", "")
        assert updated.startswith("# First\n\n# Second")

        updated_last = replace_section_content(text, "second", "")
        assert updated_last.endswith("# Second\n")

        preamble_text = "Preamble.\n\n# First\nBody."
        updated_pre = replace_section_content(preamble_text, "__preamble__", "")
        assert updated_pre == "# First\nBody."

    def test_query_error_conditions(self):
        doc = parse_sections("# Title\nBody")
        with pytest.raises(SectionNotFoundError):
            doc.get_section(
                SectionInfo(id="fake", title="Fake", level=1, heading_type="atx", path=[])
            )

        with pytest.raises(InvalidSectionOperationError):
            doc.get_section(123.45)  # type: ignore

        with pytest.raises(SectionNotFoundError):
            doc.get_section([])

        with pytest.raises(SectionNotFoundError):
            doc.get_section(["Nonexistent", "Path"])

    def test_ambiguous_suffix_query(self):
        text = "# A\n## Item (v1)\n# B\n## Item (v1)"
        doc = parse_sections(text)
        with pytest.raises(AmbiguousSectionError):
            doc.get_section("item-v1")

    def test_path_query_disambiguation_by_depth_and_ambiguity(self):
        # 1. Disambiguated by exact depth
        text1 = "# Target\n# Parent\n## Target"
        doc1 = parse_sections(text1)
        resolved = doc1.get_section(["Target"])
        assert resolved.level == 1

        # 2. Ambiguous at same depth
        text2 = "# Target\n# Target"
        doc2 = parse_sections(text2)
        with pytest.raises(AmbiguousSectionError):
            doc2.get_section(["Target"])

    def test_query_occurrence_by_title(self):
        text = "# Title\n# Title"
        doc = parse_sections(text)
        sec = doc.get_section("Title#2")
        assert sec.occurrence == 2

    def test_query_occurrence_ambiguous(self):
        text = "# P1\n## Child\n# P2\n## Child"
        doc = parse_sections(text)
        with pytest.raises(AmbiguousSectionError):
            doc.get_section("Child#1")

    def test_query_by_uppercase_slug(self):
        text = "# Getting Started\nBody"
        doc = parse_sections(text)
        assert doc.get_section("GETTING_STARTED").id == "getting-started"

    def test_query_occurrence_non_digit_and_non_matching(self):
        text = "# Title\nBody"
        doc = parse_sections(text)
        # Non-digit after #
        assert doc.find_section("Title#abc") is None
        # Digit occurrence not found
        assert doc.find_section("Title#99") is None

    def test_invalid_fence_with_backtick_in_info(self):
        text = "```foo`bar\n# Heading Inside\n```"
        doc = parse_sections(text)
        assert len(doc) == 2
        assert doc[1].title == "Heading Inside"
