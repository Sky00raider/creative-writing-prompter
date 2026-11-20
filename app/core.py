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


# ---------------------------------------------------------------------------
# Milestone 2: n-gram counting and maximum-likelihood probabilities.
# Milestone 3: add-k smoothing for unseen n-grams.
# ---------------------------------------------------------------------------
class NgramModel:
    """An order-n language model over word + punctuation tokens.

    counts[m][context] is a Counter of the tokens that followed `context`
    (a tuple of m-1 tokens), for every order m = 1..n, so the model can back
    off to a shorter context when the full one was never seen.
    """

    def __init__(self, n: int = 3, k: float = 0.01, min_count: int = 2):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n, self.k, self.min_count = n, k, min_count
        self.sentences: list[list[str]] = []

    # ---- training -------------------------------------------------------
    def fit(self, sentences: list[list[str]]) -> "NgramModel":
        freq = Counter(w for s in sentences for w in s)
        self.vocab = {w for w, c in freq.items() if c >= self.min_count} | {EOS, UNK}
        self.sentences = sentences
        counts = {m: defaultdict(Counter) for m in range(1, self.n + 1)}
        for sent in sentences:
            toks = [BOS] * (self.n - 1) + [self._map(w) for w in sent] + [EOS]
            for i in range(self.n - 1, len(toks)):
                for m in range(1, self.n + 1):
                    counts[m][tuple(toks[i - m + 1 : i])][toks[i]] += 1
        self.counts = {m: dict(d) for m, d in counts.items()}
        self.totals = {m: {c: sum(f.values()) for c, f in d.items()} for m, d in self.counts.items()}
        return self

    @property
    def V(self) -> int:
        return len(self.vocab)

    def _map(self, w: str) -> str:
        return w if w in self.vocab else UNK

    def context(self, prefix, order: int | None = None) -> tuple:
        """The last (order-1) tokens of a prefix (text or token list), padded with <s>.
        Only the sentence currently being written counts as context."""
        order = self.n if order is None else order
        if isinstance(prefix, str):
            sents = split_sentences(prefix)
            prefix = [] if _ended(prefix) or not sents else tokenize(sents[-1])
        toks = [BOS] * (self.n - 1) + [self._map(w) for w in prefix]
        return tuple(toks[len(toks) - (order - 1) :]) if order > 1 else ()

    def mle(self, prefix, word: str) -> float:
        """Maximum likelihood: P(w | ctx) = count(ctx, w) / count(ctx). 0 for unseen."""
        ctx = self.context(prefix)
        total = self.totals[self.n].get(ctx, 0)
        return self.counts[self.n][ctx][self._map(word)] / total if total else 0.0

    def prob(self, prefix, word: str, order: int | None = None) -> float:
        """Add-k: P(w | ctx) = (count(ctx, w) + k) / (count(ctx) + k * V).
        Never zero, and sums to 1 over the vocabulary for any context."""
        order = self.n if order is None else order
        ctx = self.context(prefix, order)
        c = self.counts[order].get(ctx)
        count = c[self._map(word)] if c else 0
        return (count + self.k) / (self.totals[order].get(ctx, 0) + self.k * self.V)

    def witten_bell(self, prefix, word: str, order: int | None = None) -> float:
        """Interpolated Witten-Bell, for comparison: mixes order m with order m-1,
        trusting the longer context more when few kinds of word have followed it."""
        order = self.n if order is None else order
        w = self._map(word)
        if order == 1:
            return (self.counts[1][()][w] + self.k) / (self.totals[1][()] + self.k * self.V)
        lower = self.witten_bell(prefix, word, order - 1)
        ctx = self.context(prefix, order)
        c = self.counts[order].get(ctx)
        if not c:
            return lower
        types = len(c)
        return (c[w] + types * lower) / (self.totals[order][ctx] + types)

    def best_order(self, prefix) -> int:
        """Longest order whose context appears in training (n, else n-1, ... 1).
        Add-k on an unseen context is uniform noise, so suggestions back off instead."""
        for m in range(self.n, 0, -1):
            if self.context(prefix, m) in self.counts[m]:
                return m
        return 1

    def distribution(self, prefix, smoothing: str = "addk") -> tuple[dict[str, float], int]:
        """Smoothed P(w | prefix) for every vocabulary word, and the order used."""
        if smoothing == "witten-bell":
            return {w: self.witten_bell(prefix, w) for w in self.vocab}, self.n
        order = self.best_order(prefix)
        ctx = self.context(prefix, order)
        c = self.counts[order].get(ctx, Counter())
        total = self.totals[order].get(ctx, 0)
        if smoothing == "addk":
            denom = total + self.k * self.V
            return {w: (c[w] + self.k) / denom for w in self.vocab}, order
        if smoothing == "mle":
            return {w: c[w] / (total or 1) for w in self.vocab}, order
        raise ValueError("smoothing must be addk, witten-bell or mle")

    def perplexity(self, sentences: list[list[str]], smoothing: str = "addk") -> float:
        """exp(-mean log P) over every token + end of sentence in held-out sentences.
        Uses the full order-n context (no back-off) so the smoothing itself is measured."""
        f = self.prob if smoothing == "addk" else self.witten_bell
        log_sum, N = 0.0, 0
        for sent in sentences:
            seq = list(sent) + [EOS]
            for i, w in enumerate(seq):
                log_sum += math.log(f(seq[:i], w))
                N += 1
        return math.exp(-log_sum / max(N, 1))
