# Creative Writing Prompter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/g96b45322g/badge.svg)](https://ask.thili.ai/b/g96b45322g)

This Creative Writing Prompter is a CLI tool designed for aspiring writers. It helps overcome writer's block by providing context-aware next-word suggestions, leveraging an n-gram prediction engine built from scratch to analyze a genre corpus.

## What you are building

**Goal.** Aspiring writers will use this CLI tool to get context-aware next-word suggestions, helping them develop stories and overcome writer's block by sampling from a genre corpus.

**Done when.** The project is done when a CLI tool can take user input, process it with an n-gram next-word prediction engine, and provide context-aware word suggestions by sampling from the model.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a text corpus of short stories.**  
      Hint: Tokenize the corpus into words and sentences, ensuring consistent formatting for n-gram extraction.
- [ ] **2. Implement n-gram counting and maximum-likelihood probabilities.**  
      Hint: Count n-gram frequencies and calculate conditional probabilities P(word_n | word_1...word_{n-1}).
- [ ] **3. Add add-k smoothing to handle unseen n-grams.**  
      Hint: Modify your probability calculation to include a small constant 'k' and vocabulary size 'V' for unseen n-grams.
- [ ] **4. Develop a function to generate text by sampling from the model.**  
      Hint: Implement a sampling function that takes a prefix and randomly selects the next word based on its smoothed probability.
- [ ] **5. Create a CLI interface for users to input text and get suggestions.**  
      Hint: Design a command-line interface that accepts user input and displays generated word suggestions.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify the length of the generated suggestion.
- Implement different sampling strategies (e.g., top-k, temperature sampling).
