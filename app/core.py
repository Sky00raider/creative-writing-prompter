"""The n-gram engine, built from scratch (standard library only).

Milestone 1  corpus -> sentences -> tokens            normalize, split_sentences, tokenize, prepare
Milestone 2  n-gram counts + maximum likelihood       NgramModel.fit, NgramModel.mle
Milestone 3  add-k smoothing (plus Witten-Bell)        NgramModel.prob, NgramModel.witten_bell
Milestone 4  sampling with temperature / top-k / top-p sample_next, generate
Milestone 5  what the CLI shows                        suggest, continue_text, compare, load_model
"""
from __future__ import annotations

import gzip
import math
import pickle
import random
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
CACHE_DIR = ROOT / ".cache"

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
PUNCT = set(".,!?;:")
ENDS = {".", "!", "?"}

# ---------------------------------------------------------------------------
# Milestone 1: Prepare a text corpus of short stories.
# ---------------------------------------------------------------------------
_ABBREV = re.compile(r"\b(mr|mrs|dr|st|mme|messrs|jr|sr|capt|col|gen|prof|rev)\.", re.I)
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n\s*\n")
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*|[.,!?;:]")


def normalize(text: str) -> str:
    """One consistent format: straight quotes, dashes as spaces, no abbreviation dots."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"—|–|--+", " ", text)
    return _ABBREV.sub(r"\1", text)  # "Mr. Holmes" must not end a sentence


def split_sentences(text: str) -> list[str]:
    """Split after . ! ? (keeping the mark) and on blank lines (paragraph breaks)."""
    parts = _SENT_END.split(normalize(text))
    return [" ".join(p.split()) for p in parts if p and p.strip()]


def tokenize(sentence: str) -> list[str]:
    """Lower-case words (apostrophes kept: "don't") and punctuation as its own token.
    Quotes, numbers and other symbols are dropped."""
    return _TOKEN.findall(normalize(sentence).lower())


def prepare(text: str) -> list[list[str]]:
    """Raw text -> list of token sentences, ready for n-gram counting."""
    return [t for t in (tokenize(s) for s in split_sentences(text)) if any(w not in PUNCT for w in t)]


def list_genres() -> list[str]:
    return sorted(p.stem for p in CORPUS_DIR.glob("*.txt"))


def load_corpus(genre: str) -> str:
    """Cleaned text for one genre, or every genre joined for "all"."""
    genres = list_genres() if genre == "all" else [genre]
    if not genres or any(not (CORPUS_DIR / f"{g}.txt").exists() for g in genres):
        raise ValueError(f"unknown genre {genre!r}; choose from {list_genres() + ['all']}")
    return "\n\n".join((CORPUS_DIR / f"{g}.txt").read_text(encoding="utf-8") for g in genres)


def train_test_split(sents: list, test_frac: float = 0.1, seed: int = 0) -> tuple[list, list]:
    """Shuffle sentences with a fixed seed and hold out test_frac of them."""
    sents = list(sents)
    random.Random(seed).shuffle(sents)
    cut = int(len(sents) * (1 - test_frac))
    return sents[:cut], sents[cut:]


def _ended(text: str) -> bool:
    return not text.strip() or text.rstrip().rstrip("\"')").endswith(tuple(ENDS))
