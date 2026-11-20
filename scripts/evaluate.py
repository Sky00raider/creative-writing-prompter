"""Held-out perplexity for every genre: add-k at several k, Witten-Bell, bigram vs trigram.
Writes results/results.md.  Run: python scripts/evaluate.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import core  # noqa: E402

KS = [1.0, 0.1, 0.01, 0.001]
rows = ["| genre | n | " + " | ".join(f"add-k k={k}" for k in KS) + " | Witten-Bell | MLE: test tokens with P=0 |",
        "|---|---|" + "---|" * (len(KS) + 2)]
for g in core.list_genres():
    train, test = core.train_test_split(core.prepare(core.load_corpus(g)), 0.1, 0)
    for n in (2, 3):
        m = core.NgramModel(n, min_count=2).fit(train)
        cells = []
        for k in KS:
            m.k = k
            cells.append(f"{m.perplexity(test):,.1f}")
        m.k = 0.01
        zero = sum(m.mle(s[:i], w) == 0 for s in test for i, w in enumerate(s)) / sum(map(len, test))
        rows.append(f"| {g} | {n} | " + " | ".join(cells) + f" | {m.perplexity(test, 'witten-bell'):,.1f} | {zero:.0%} |")
        print(rows[-1], flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/results.md").write_text("Held-out perplexity (10% of sentences, seed 0; lower is better)\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
