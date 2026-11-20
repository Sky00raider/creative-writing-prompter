"""Build the per-genre story corpus from public-domain Project Gutenberg collections.

Usage (from the repo root):
    python scripts/build_corpus.py

Raw downloads are cached in corpus/raw/ (git-ignored, downloaded once, 1 s apart).
Cleaned text is written to corpus/{genre}.txt (UTF-8, paragraphs separated by a blank line).

`clean(text)` is reusable: it takes a raw Gutenberg plain-text file and returns clean prose.
"""
from __future__ import annotations

import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# genre -> [(gutenberg_id, title), ...]   (every ID verified against the "Title:" line)
GENRES: dict[str, list[tuple[int, str]]] = {
    "horror": [
        (2147, "The Works of Edgar Allan Poe, Volume 1"),
        (2148, "The Works of Edgar Allan Poe, Volume 2"),
        (8486, "Ghost Stories of an Antiquary"),
    ],
    "mystery": [
        (1661, "The Adventures of Sherlock Holmes"),
        (834, "The Memoirs of Sherlock Holmes"),
        (108, "The Return of Sherlock Holmes"),
    ],
    "fairy_tale": [
        (2591, "Grimms' Fairy Tales"),
        (1597, "Andersen's Fairy Tales"),
    ],
    "adventure": [
        (236, "The Jungle Book"),
        (710, "Love of Life and Other Stories"),
        (2429, "Lost Face"),
        (2781, "Just So Stories"),
    ],
    "humour": [
        (269, "Beasts and Super-Beasts"),
        (3688, "The Chronicles of Clovis"),
        (6340, "Literary Lapses"),
    ],
}

# Books whose front/back matter (memoirs, biographies, endnotes) is too irregular for the generic
# cleaner: id -> (regex for the first line of the first story, regex for the first line after the
# last story or None). Both are matched against whole lines (re.M).
BODY_MARKERS: dict[int, tuple[str | None, str | None]] = {
    2147: (r"^\s*THE UNPARALLELED ADVENTURES OF ONE HANS PFAAL \(", None),
    2148: (r"^THE PURLOINED LETTER\s*$", None),
    2591: (r"^THE GOLDEN BIRD\s*$", r"^The Brothers Grimm, Jacob"),
    3688: (r"^THE MATCH-MAKER\s*$", None),
    710: (r"^They limped painfully down the bank", None),
    8486: (r"^Canon Alberic.s Scrap-book\s*$", None),
    6340: (None, r"^Many of the sketches which form the present volume"),
}

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "corpus" / "raw"
OUT_DIR = ROOT / "corpus"
USER_AGENT = "creative-writing-prompter-corpus-builder/1.0 (educational project)"

# --------------------------------------------------------------------------- download


def _fetch(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _looks_like_book(data: bytes | None) -> bool:
    return bool(data) and b"*** START OF" in data[:20000].upper().replace(b"***START", b"*** START")


def _candidate_urls(book_id: int) -> list[str]:
    urls = [f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"]
    # Official mirror layout (used if the main site is down for maintenance).
    path = "/".join(str(book_id)[:-1]) if book_id >= 10 else "0"
    for suffix in ("-0.txt", ".txt", "-8.txt"):
        urls.append(f"https://gutenberg.pglaf.org/{path}/{book_id}/{book_id}{suffix}")
    return urls


def download(book_id: int) -> Path:
    """Download one book into corpus/raw/ unless already cached. Returns the cached path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"pg{book_id}.txt"
    if path.exists() and _looks_like_book(path.read_bytes()):
        return path
    for url in _candidate_urls(book_id):
        data = _fetch(url)
        time.sleep(1)  # be polite to Gutenberg
        if _looks_like_book(data):
            path.write_bytes(data)
            return path
    raise RuntimeError(f"could not download Gutenberg book {book_id}")


def read_title(raw: str) -> str:
    m = re.search(r"^Title:\s*(.+(?:\n[ \t]+\S.*)*)", raw, re.M)
    return " ".join(m.group(1).split()) if m else "(no Title: line)"


# --------------------------------------------------------------------------- cleaning

_START_RE = re.compile(r"^\*\*\*\s*START OF .*$", re.M | re.I)
_END_RE = re.compile(
    r"^\*\*\*\s*END OF .*$|^\s*End of (?:the )?Project Gutenberg\b.*$", re.M | re.I
)

_ROMAN = r"[IVXLCDM]+"
_NUMWORDS = r"(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|FIRST|SECOND|THIRD|FINAL|LAST)"
_ROMAN_ONLY = re.compile(rf"^\(?{_ROMAN}[.,)]?$")
_NUMBER_ONLY = re.compile(r"^\(?\d+[.,)]?$")
_CHAPTER_RE = re.compile(
    rf"^(?:CHAPTER|BOOK|PART|SECTION|ADVENTURE|STORY|TALE|STAVE|ACT|SCENE|LETTER|EPISODE)\b\s*"
    rf"(?:{_ROMAN}|\d+|{_NUMWORDS})?\b[.:,]?(?:\s.*)?$",
    re.I,
)
_ROMAN_TITLE_RE = re.compile(rf"^{_ROMAN}\.\s+\S.*$")
_END_MARK_RE = re.compile(r"^(?:THE\s+)?END\.?$", re.I)
_DECOR_RE = re.compile(r"^[\s*\-_=~.#+•]+$")
# Sections introduced by these headings are removed until the next real story heading.
_SKIP_HEADING_RE = re.compile(
    r"^(?:table of )?contents?$|^list of (?:illustrations|stories|plates)$|^illustrations?$|"
    r"^footnotes?\b.*$|^life of\b.*$|^death of\b.*$|^(?:an? |the |editor'?s |author'?s |translator'?s |publisher'?s |prefatory |transcriber'?s )?"
    r"(?:preface|acknowledge?ments?|introduction|foreword|prefatory note|note|notes|dedication|afterword|appendix|"
    r"bibliography|index|memoir|biographical (?:note|sketch)|advertisement|errata)\b.*$|"
    r"^to (?:my|the|those|one|a)\b.*$|^dedicated\b.*$|^by (?:the )?author.*$",
    re.I,
)
_BRACKET_DROP_RE = re.compile(
    r"^\[\s*(?:illustration|transcriber|note|footnote|sidenote|side note|pg|page|editor|"
    r"decoration|frontispiece|image|map|blank page)",
    re.I,
)
_TERMINAL_PUNCT = tuple('.!?,;:"\'”’*') + ("--",)

_QUOTE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"',
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "﻿": "", "­": "", "…": "...",
    "–": " -- ", "—": " -- ", "―": " -- ", "‒": " -- ",
}


def _is_verse_or_list(lines: list[str]) -> bool:
    """Short-lined blocks (verse, tables of contents, signature lists), not wrapped prose."""
    if len(lines) < 2:
        return False
    lengths = sorted(len(ln) for ln in lines)
    median = lengths[len(lengths) // 2]
    upper = sum(1 for ln in lines if ln[:1].isupper() or ln[:1].isdigit() or ln[:1] in "\"'(")
    return median < 52 and upper >= 0.7 * len(lines)


def strip_gutenberg(text: str) -> str:
    """Drop everything up to the START line and from the END line onwards."""
    m = _START_RE.search(text)
    if m:
        text = text[m.end():]
    m = _END_RE.search(text)
    if m:
        text = text[: m.start()]
    return text


def _is_caps(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 2 and not any(c.islower() for c in letters)


def _is_heading_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _ROMAN_ONLY.match(s) or _NUMBER_ONLY.match(s) or _END_MARK_RE.match(s):
        return True
    if len(s) <= 100 and _is_caps(s):
        return True
    if len(s) <= 80 and _CHAPTER_RE.match(s) and _is_caps(s.split()[0]):
        return True
    if len(s) <= 80 and _ROMAN_TITLE_RE.match(s) and not s.endswith(_TERMINAL_PUNCT[:6]):
        return True
    return False


def _is_short_title_line(line: str) -> bool:
    """A short single line with no sentence punctuation: a title, byline or ToC entry."""
    s = line.strip()
    return 0 < len(s) <= 80 and not s.endswith(_TERMINAL_PUNCT) and len(s.split()) <= 12


def _normalise_chars(s: str) -> str:
    out = []
    for ch in s:
        out.append(_QUOTE_MAP.get(ch, ch))
    s = "".join(out)
    s = s.replace("--", " -- ")
    s = re.sub(r"(?:\s*-{2,}\s*)", " -- ", s)  # any run of 2+ hyphens
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"(?<=\S) -- (?=\S)", " -- ", s)
    return s.strip()


def _inline_clean(s: str) -> str:
    s = re.sub(r"\[\s*Illustration[^\]]*\]", " ", s, flags=re.I)
    s = re.sub(r"\[\s*(?:Pg|Page|pg|page)\s*[\w.]*\s*\]", " ", s)
    s = re.sub(r"\[\s*(?:Sidenote|Side note)[^\]]*\]", " ", s, flags=re.I)
    s = re.sub(r"\[\s*(?:Footnote|Note)\s*\w*:[^\]]*\]", " ", s, flags=re.I)
    s = re.sub(r"\[\s*\d{1,3}\s*\]", "", s)          # footnote markers [1]
    s = re.sub(r"\(\s*\*\s*\d{0,3}\s*\)", "", s)      # footnote markers (*1), (*)
    s = re.sub(r"(?<=\s)\*+(?=\s)", "", s)                # lone asterisk footnote marks
    s = re.sub(r"\[\s*[A-Za-z*]\s*\]", "", s)         # [A], [*]
    s = re.sub(r"\{\s*\d{1,3}\s*\}", "", s)           # {12}
    s = re.sub(r"(?<=[\w.,;:!?'\"”’-])\*+", "", s)  # asterisk footnote marks
    s = re.sub(r"(?<![\w])_([^_\n]+?)_(?![\w])", r"\1", s)  # _italics_
    s = s.replace("_", "")
    s = re.sub(r"=([^=\n]{1,80})=", r"\1", s)         # =bold=
    s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)           # stray html tags
    return s


def clean(text: str, start: str | None = None, end: str | None = None) -> str:
    """Turn a raw Gutenberg plain-text book into clean prose paragraphs.

    Removes the Gutenberg header/footer, tables of contents, prefaces/introductions/notes,
    chapter and story headings, illustration and footnote markup, transcriber notes and
    underscore italics; rejoins hard-wrapped lines; normalises quotes and dashes.
    Paragraphs are separated by a single blank line.

    `start` / `end` are optional regexes (matched per line) for the first line of the first story
    and for the first line after the last one; they are for books whose front/back matter
    (memoirs, biographies, endnotes) is too irregular for the generic rules.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("﻿", "")
    text = strip_gutenberg(text)
    if start:
        m = re.search(start, text, re.M)
        if m:
            text = text[m.start():]
    if end:
        m = re.search(end, text, re.M)
        if m:
            text = text[: m.start()]
    text = re.sub(r"\{[^{}\n]*\.(?:jpg|jpeg|png|gif)\s*\}", " ", text, flags=re.I)
    # Multi-line [Illustration ...] / [Footnote ...] / [Transcriber's Note ...] spans.
    text = re.sub(
        r"\[\s*(?:Illustration|Transcriber|Footnote|Sidenote|Side note|Editor)[^\[\]]*\]",
        " ", text, flags=re.I,
    )
    text = re.sub(r"\[Illustration[^\n]*(?:\n(?!\s*\n)[^\n]*)*", " ", text, flags=re.I)

    blocks = re.split(r"\n[ \t]*\n+", text)
    paragraphs: list[str] = []
    skipping = False

    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        joined = " ".join(lines)

        # Bracketed editorial blocks and footnote text.
        if _BRACKET_DROP_RE.match(joined) or re.match(r"^\[\s*\d{1,3}\s*\]\s", joined):
            continue
        if joined.startswith("[") and joined.endswith("]") and len(joined) < 600:
            continue
        # Decoration such as "* * * * *" or "-----".
        if all(_DECOR_RE.match(ln) for ln in lines):
            continue
        # Footnote text introduced by an asterisk ("* Herostratus, or ...", "*The watchmen ...").
        if re.match(r"^(?:\*+|\(\*\d*\))\s*\S", joined):
            continue
        if re.match(r"^(?:Transcriber'?s? note|Note:|Produced by|Printed by|Etext|E-text|Scanned by|"
                    r"This (?:e-?book|etext) was|(?:This|These) (?:book|stories|volume|tales) "
                    r"(?:is|are) (?:\w+ )?dedicated)", joined.lstrip("_*= "), re.I):
            continue

        if _is_verse_or_list(lines):
            continue

        # Heading block: every line is a heading or the block is a short unpunctuated title / ToC.
        all_heading = all(_is_heading_line(ln) for ln in lines)
        titleish = all(_is_short_title_line(ln) for ln in lines)
        if all_heading or titleish:
            plain = re.sub(r"[^A-Za-z' ]", " ", joined.replace("’", "'")).strip()
            plain = re.sub(r"\s+", " ", plain)
            if _SKIP_HEADING_RE.match(plain):
                skipping = True
            elif not (_ROMAN_ONLY.match(joined) or _NUMBER_ONLY.match(joined)):
                # A genuine new story/chapter heading ends a skipped front-matter section.
                words = re.findall(r"[A-Za-z]{3,}", plain)
                if skipping and words and not re.search(r"\d", joined) and (all_heading or len(lines) == 1):
                    skipping = False
            continue
        # A heading glued to the front of a paragraph: strip leading heading lines.
        while len(lines) > 1 and _is_heading_line(lines[0]):
            plain = re.sub(r"[^A-Za-z' ]", " ", lines[0].replace("’", "'")).strip()
            if _SKIP_HEADING_RE.match(plain):
                skipping = True
            elif not (_ROMAN_ONLY.match(lines[0]) or _NUMBER_ONLY.match(lines[0])):
                skipping = False
            lines = lines[1:]
        if skipping:
            continue

        para = _inline_clean(" ".join(lines))
        para = _normalise_chars(para)
        if not re.search(r"[A-Za-z]", para):
            continue
        paragraphs.append(para)

    out = "\n\n".join(paragraphs)
    out = re.sub(r"[ \t]+", " ", out)
    return out.strip() + "\n"


# --------------------------------------------------------------------------- main


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for genre, books in GENRES.items():
        parts: list[str] = []
        for book_id, title in books:
            path = download(book_id)
            raw = path.read_text(encoding="utf-8-sig", errors="replace")
            found = read_title(raw)
            norm = lambda x: re.sub(r"[^a-z]", "", x.lower())
            head = norm(raw[:3000])
            if norm(title.split(",")[0])[:14] not in norm(found) + head:
                print(f"WARNING: id {book_id} expected '{title}' but Title: line is '{found}'")
            cleaned = clean(raw, *BODY_MARKERS.get(book_id, (None, None)))
            print(f"  {genre:10s} {book_id:>5} {found[:55]:55s} {len(cleaned.split()):>8,} words")
            parts.append(cleaned.strip())
        out = OUT_DIR / f"{genre}.txt"
        out.write_text("\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(build())
