# Creative Writing Prompter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/g96b45322g/badge.svg)](https://ask.thili.ai/b/g96b45322g)

A command-line next-word engine for writers, built from scratch in pure Python on an n-gram language model (trigram by
default). It ships with five genre corpora of public-domain short stories (horror, mystery, fairy tale, adventure,
humour). It suggests the next words for your text, continues it, compares genres side by side, and explains each
suggestion with a real sentence from the corpus. An interactive writing session lets you build a draft word by word
and save it.

## Milestones

- [x] **1. Prepare a text corpus of short stories.**
      Done: `normalize`, `split_sentences`, `tokenize`, `prepare`, `load_corpus` in `app/core.py`; corpus built by `scripts/build_corpus.py`.
- [x] **2. Implement n-gram counting and maximum-likelihood probabilities.**
      Done: `NgramModel.fit` (counts for every order 1..n) and `NgramModel.mle` in `app/core.py`.
- [x] **3. Add add-k smoothing to handle unseen n-grams.**
      Done: `NgramModel.prob` (add-k), `best_order` (back-off to the longest seen context), `perplexity`; `witten_bell` for comparison.
- [x] **4. Develop a function to generate text by sampling from the model.**
      Done: `sample_next` (temperature / top-k / top-p), `generate`, `detokenize` in `app/core.py`.
- [x] **5. Create a CLI interface for users to input text and get suggestions.**
      Done: `suggest`, `continue_text`, `compare`, `load_model` in `app/core.py` and the `suggest`, `continue`, `compare`, `write`, `eval`, `genres` commands in `app/cli.py`.

## Quick start

Python 3.12+. The only dependencies in `requirements.txt` are pytest and httpx; the engine itself uses the standard
library only.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.cli genres
python -m app.cli suggest "my dear" --genre mystery
python -m app.cli suggest "the old house at the end of the" -g horror --explain
python -m app.cli continue "once upon a time there was a" -g fairy_tale --length 20 --temperature 0.7
python -m app.cli compare "the door opened slowly and"
python -m app.cli write --genre horror
python -m app.cli eval
pytest
```

The corpus is committed (`corpus/*.txt`, 4.5 MB), so everything works right after cloning;
`python scripts/build_corpus.py` rebuilds it from Project Gutenberg. The first run of a genre trains in a few seconds
and caches the counts in `.cache/`.

## Commands

| command | what it does | options |
|---|---|---|
| `suggest TEXT` | top next-word suggestions with probabilities | `--count/-c`, `--explain/-x`, `--genre/-g`, `--n`, `--k` |
| `continue TEXT` | sample a continuation | `--samples`, `--length`, `--temperature/-t`, `--top-k`, `--top-p`, `--seed` |
| `compare TEXT` | the same prompt continued in every genre | `--genres a,b`; seed 0 by default, so output is reproducible |
| `write` | interactive writing session | see below |
| `eval` | held-out perplexity for every genre and smoothing method | |
| `genres` | list the available genres | |

The genre `all` (the default) merges all five corpora.

Inside `write`, type a number to accept that suggestion, or use a command:

| command | effect |
|---|---|
| `/more [N]` | show more suggestions |
| `/genre NAME` | switch genre |
| `/temp T` | set sampling temperature |
| `/undo` | remove the last word |
| `/show` | print the draft so far |
| `/why` | explain the current suggestions |
| `/save [name]` | save the draft |
| `/drafts` | list saved drafts |
| `/quit` | leave the session |

Drafts are stored in `data/app.json` by `app/storage.py`.

## Corpus

| genre | books (Gutenberg id) | words |
|---|---|---|
| horror | Poe, Works vol 1 (2147) and vol 2 (2148); M. R. James, Ghost Stories of an Antiquary (8486) | 218,811 |
| mystery | Conan Doyle: Adventures (1661), Memoirs (834), Return of Sherlock Holmes (108) | 313,144 |
| fairy_tale | Grimms' Fairy Tales (2591), Andersen's Fairy Tales (1597) | 153,855 |
| adventure | Kipling, The Jungle Book (236) and Just So Stories (2781); Jack London, Love of Life (710) and Lost Face (2429) | 167,376 |
| humour | Saki, Beasts and Super-Beasts (269) and The Chronicles of Clovis (3688); Leacock, Literary Lapses (6340) | 152,031 |

Cleaning: Gutenberg header and footer stripped; per-book start/end markers cut prefaces, memoirs and notes;
headings, roman numerals, illustrations, footnote markers and italic underscores removed; hard-wrapped lines rejoined
into paragraphs; quotes and dashes normalised; verse dropped.

## How it works

- **Tokens** are lowercase words (apostrophes kept) plus the punctuation marks `. , ! ? ; :`, so generated text has
  punctuation.
- **Sentences** are split on `.`, `!`, `?` and blank lines, with "Mr.", "Dr." and similar abbreviations protected.
  Each sentence is padded with `<s>` and closed with `</s>`; words seen fewer than 2 times become `<unk>`.
- **Counts** are kept for every order 1..n.
- **MLE:** `count(ctx, w) / count(ctx)`.
- **Add-k:** `(count + k) / (count(ctx) + k * V)`.
- **Context choice:** for suggestions and sampling, the model uses the longest context it has actually seen (trigram,
  else bigram, else unigram) and applies add-k there. Add-k on an unseen context is just a uniform guess.
- **Sampling:** temperature (0 = greedy, computed in log space), top-k, and top-p (nucleus). An end-of-sentence token
  starts a new sentence, so `--length` can span several sentences. The detokenizer capitalises sentence starts and "I".
- **`--explain`** shows the probability, the count, the n-gram order used, and the shortest corpus sentence that
  contains the phrase.

## Results

Held-out perplexity (10% of sentences, seed 0; lower is better). Reproduce with `python -m app.cli eval`.

| genre | n | add-k k=1.0 | add-k k=0.1 | add-k k=0.01 | add-k k=0.001 | Witten-Bell | MLE: test tokens with P=0 |
|---|---|---|---|---|---|---|---|
| adventure | 2 | 500.3 | 219.1 | 158.0 | 177.9 | 106.0 | 21% |
| adventure | 3 | 2,255.7 | 1,100.8 | 646.4 | 600.9 | 104.5 | 57% |
| fairy_tale | 2 | 368.6 | 160.9 | 122.0 | 140.4 | 90.3 | 18% |
| fairy_tale | 3 | 1,651.2 | 764.3 | 444.5 | 440.4 | 85.8 | 54% |
| horror | 2 | 771.2 | 341.0 | 242.8 | 266.5 | 146.3 | 24% |
| horror | 3 | 3,708.2 | 1,919.4 | 1,182.3 | 1,107.6 | 150.6 | 61% |
| humour | 2 | 772.4 | 341.7 | 247.3 | 278.9 | 144.4 | 26% |
| humour | 3 | 3,452.3 | 1,879.4 | 1,210.2 | 1,166.8 | 153.6 | 65% |
| mystery | 2 | 462.0 | 197.4 | 140.3 | 148.6 | 98.6 | 17% |
| mystery | 3 | 2,407.8 | 1,054.2 | 576.8 | 514.0 | 94.7 | 52% |

Findings:

- **Add-k needs a small k.** k=1 is about 3x worse than k=0.01 for bigrams (mystery bigram: 462.0 vs 140.3).
- **With add-k the trigram is about 4x worse than the bigram in every genre** (mystery: 576.8 vs 140.3). With
  150-320k training tokens, 52-65% of held-out trigrams never occur in training (MLE gives them probability 0),
  against 17-26% for bigrams. Add-k then hands each of them a near-uniform 1/V slice.
- **Witten-Bell interpolation (included for comparison) fixes this.** Trigram perplexity on mystery drops to 94.7, and
  the trigram beats the bigram in 3 of 5 genres (adventure, fairy_tale, mystery). Horror and humour, the corpora with
  the largest vocabularies (8.3k and 7.4k words) and the most unseen trigrams, still slightly prefer the bigram.
- **This is why the CLI backs off to the longest seen context** instead of using add-k at a fixed order.
- **Fairy tales are the most predictable genre** (lowest perplexity, 85.8); humour and horror are the least.

## Sample outputs

```
$ python -m app.cli suggest "my dear" --genre mystery
  1. watson   24.5%
  2. fellow   13.4%
  3. sir       5.6%
  4. holmes    3.2%
  5. young     2.8%
```

```
$ python -m app.cli suggest "the old house at the end of the" -g horror -x -c 3
  1. house    0.9%   seen 29x after "of the" [3-gram]
       e.g. Of course it was in the rear of the house.
  2. most     0.9%   seen 28x after "of the" [3-gram]
       e.g. His family was one of the most illustrious of the empire.
  3. room     0.8%   seen 26x after "of the" [3-gram]
       e.g. Let us now revert to the interior of the room.
```

```
$ python -m app.cli compare "the door opened slowly and" --length 18
"the door opened slowly and" ...

adventure   ponderingly. The stranger man for a while the whale, and that night, shouting abroad in
            such a large
fairy_tale  so bitterly, but they could get in? He walked nearer than a flower pot. Asked he, I can
horror      steadily down, and that, having caught a glimpse, I am detailing any mystery. If any man
            desires to
humour      wearily into the gooseberry garden, for instance, I really. I can ask the hall the next
            stop was
mystery     solemnly he was a good deal in this matter is the best of his brother, your suspicions
            would
```

```
$ python -m app.cli continue "once upon a time there was a" -g fairy_tale --seed 1 --temperature 0.7
1. once upon a time there was a >> little for you. I am quite sorry at his dinner. It was not right.
   How
2. once upon a time there was a >> chandelier in it, said the father, said the raven. And then, too,
   how beautifully you
3. once upon a time there was a >> little piece of his mouth. The wolf; and while he summoned the
   little boy was
```

A trigram model is locally fluent but loses the thread after a few words. That is the expected ceiling of n-grams, and
a reason to read suggestions as prompts, not prose.

## Project layout

```
app/
  core.py        n-gram engine: tokenizing, counting, smoothing, sampling, suggestions
  cli.py         command-line interface and the interactive writing session
  storage.py     saved drafts (data/app.json)
  llm.py         starter file, unused
corpus/          one cleaned .txt per genre (committed)
scripts/
  build_corpus.py  download and clean the books from Project Gutenberg
  evaluate.py      held-out perplexity experiments
results/
  results.md     perplexity table
tests/
  test_core.py   one test group per milestone
  test_smoke.py  plumbing checks
```

## Limitations

- No Kneser-Ney smoothing; Witten-Bell is the only interpolated method.
- The sentence splitter is regex-based, so unusual abbreviations can cause bad splits.
- Each genre comes from a handful of authors, so the model mimics them (for example, "watson" after "my dear").
- Trigram context is short; text drifts after a few words.

## Settings and secrets

The starter includes `.env` and LLM settings. They are not needed here: the engine is pure standard library and makes
no network calls at run time. If you do create a `.env`, it is ignored by git; never commit it or paste a key into code.
