"""Keep published entry-point documentation linked to real local resources."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
CORE_DOCS = (
    "README.md",
    "docs/README.md",
    "docs/COMPILER_GUIDE.md",
    "docs/ARCHITECTURE.md",
    "docs/VERIFICATION.md",
    "docs/SECURITY_AND_BOUNDARIES.md",
    "docs/CONTRIBUTING.md",
    "docs/CLIENT_README.md",
    "docs/V27_ACTIVE_MODEL_LIFETIME.md",
    "docs/ROADMAP.md",
    "docs/V28_NATIVE_CPU_COMPILER.md",
)
INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")


@pytest.mark.parametrize("filename", CORE_DOCS)
def test_core_docs_have_no_broken_local_links(filename):
    document = REPOSITORY / filename
    assert document.is_file(), filename
    text = document.read_text(encoding="utf-8")
    checked = 0
    for match in INLINE_LINK.finditer(text):
        destination = match.group(1).strip().split(" ", 1)[0]
        if (
            not destination or destination.startswith("#")
            or urlsplit(destination).scheme in ("https", "http", "mailto")
        ):
            continue
        local = unquote(destination.split("#", 1)[0])
        if not local:
            continue
        candidate = (document.parent / local).resolve()
        assert candidate.is_relative_to(REPOSITORY), (
            f"{filename} escapes the repository: {destination}"
        )
        assert candidate.exists(), (
            f"{filename} references a missing local file: {destination}"
        )
        checked += 1
    assert checked > 0, f"No local documentation links checked in {filename}"
