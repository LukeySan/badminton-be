"""Thai line breaks in exported text overlays == the browser's: the shared fixture
(tests/fixtures/line_breaks/thai.json, made by the frontend with Intl.Segmenter)
against worker/caption_image.py's ICU line breaking (PyICU)."""

import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "worker"))
import caption_image  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "line_breaks" / "thai.json"


def cases() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))[
        "cases"
    ]
    assert found, f"no cases in {FIXTURE}"
    return found


def width(text: str) -> int:
    """The fixture's font-free width: characters that aren't combining marks
    (\\p{Mn}: Thai vowels above/below, tone marks)."""
    return sum(1 for ch in text if unicodedata.category(ch) != "Mn")


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["name"])
def test_line_breaks_match_the_browser(case: dict[str, Any]) -> None:
    """Same break opportunities (Thai dictionary words, spaces and punctuation kept
    with the word before) and the same lines at the case's width."""
    assert caption_image.line_pieces(case["text"]) == case["pieces"]
    assert (
        caption_image.wrap_lines(case["text"], width, case["max_width"])
        == case["lines"]
    )


def test_icu_is_the_one_pyicu_was_built_with() -> None:
    """A dictionary-based break (not one per character, which is what ICU without
    its Thai dictionary data would give)."""
    assert caption_image.line_pieces("ลูกนี้สุดยอด") == ["ลูก", "นี้", "สุด", "ยอด"]


def test_clusters_keep_marks_and_sara_am_with_their_letter() -> None:
    """SARA AM (ำ) isn't a combining mark in Unicode but belongs to its letter
    (a grapheme extender), like the tone mark before it."""
    assert caption_image.clusters("ทำน้ำ") == ["ทำ", "น้ำ"]
    assert caption_image.clusters("ok 🏸") == ["o", "k", " ", "🏸"]


def test_characters_outside_the_bmp_are_cut_correctly() -> None:
    """ICU counts UTF-16 code units; the pieces are still whole Python strings."""
    assert caption_image.line_pieces("𝐀𝐁 สุดยอด") == ["𝐀𝐁 ", "สุด", "ยอด"]
    assert caption_image.line_pieces("") == []
