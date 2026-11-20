"""One group of tests per milestone. A tiny inline corpus keeps them fast; the
real genre corpora are checked separately (they are committed under corpus/)."""
import math
import random

import pytest

from app import cli, core
from app.storage import Store

TOY = """The cat sat on the mat. The cat ran away!

The dog sat on the rug. Mr. Brown didn't see the dog, did he?"""


@pytest.fixture
def toy():
    return core.NgramModel(n=2, k=0.5, min_count=1).fit(core.prepare(TOY))


# ---- Milestone 1: corpus -> sentences -> tokens ---------------------------
def test_tokenize_keeps_apostrophes_and_splits_punctuation():
    assert core.tokenize("“Don’t go,” she said—quietly.") == ["don't", "go", ",", "she", "said", "quietly", "."]


def test_sentences_split_on_marks_and_paragraphs_but_not_mr():
    sents = core.split_sentences(TOY)
    assert sents == ["The cat sat on the mat.", "The cat ran away!", "The dog sat on the rug.",
                     "Mr Brown didn't see the dog, did he?"]
    assert core.prepare(TOY)[0] == ["the", "cat", "sat", "on", "the", "mat", "."]


def test_train_test_split_is_reproducible_and_disjoint():
    data = list(range(100))
    a, b = core.train_test_split(data, 0.1, seed=0)
    assert (a, b) == core.train_test_split(data, 0.1, seed=0)
    assert len(b) == 10 and sorted(a + b) == data


def test_genre_corpora_are_clean():
    genres = core.list_genres()
    if not genres:
        pytest.skip("corpus/ not built (python scripts/build_corpus.py)")
    assert len(genres) >= 4
    for g in genres:
        text = core.load_corpus(g)
        assert len(text.split()) > 50_000, g
        assert "gutenberg" not in text.lower(), g
        assert "CHAPTER" not in text, g
    with pytest.raises(ValueError):
        core.load_corpus("no-such-genre")
