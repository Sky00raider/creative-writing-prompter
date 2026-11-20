"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import datetime

from app import core
from app.storage import Store

DESCRIPTION = ("Aspiring writers use this CLI tool to get context-aware next-word suggestions, helping them "
               "develop stories and overcome writer's block by sampling from a genre corpus.")


def _sampling_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--length", type=int, default=15, help="words to generate (default 15)")
    p.add_argument("--temperature", "-t", type=float, default=0.8, help="<1 safer, >1 wilder, 0 = greedy (default 0.8)")
    p.add_argument("--top-k", type=int, default=40, help="sample only from the k most likely words (0 = all; default 40)")
    p.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling mass (1 = off; default 0.95)")
    p.add_argument("--seed", type=int, default=None, help="make the output reproducible")


def _model_args(p: argparse.ArgumentParser, genre: bool = True) -> None:
    if genre:
        p.add_argument("--genre", "-g", default="all", help="horror, mystery, fairy_tale, ... or all (see `genres`)")
    p.add_argument("--n", type=int, default=3, help="n-gram order (default 3 = trigram)")
    p.add_argument("--k", type=float, default=0.01, help="add-k smoothing constant (default 0.01)")


def _sampling(a) -> dict:
    return {"temperature": a.temperature, "top_k": a.top_k, "top_p": a.top_p}


def _text(words: list[str]) -> str:
    text = " ".join(words).strip()
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    return text


def _print_suggestions(items: list[dict], explain: bool) -> None:
    if not items:
        print("  (no suggestions)")
        return
    width = max(len(i["word"]) for i in items)
    for n, i in enumerate(items, 1):
        line = f"  {n}. {i['word']:<{width}}  {i['prob']:6.1%}"
        if explain:
            ctx = " ".join(i["context"]) or "(start of sentence)"
            line += f"   seen {i['count']}x after \"{ctx}\" [{i['order']}-gram]"
        print(line)
        if explain and i.get("example"):
            print(textwrap.indent(textwrap.fill(f"e.g. {i['example']}", 88), "       "))


def cmd_genres(a) -> None:
    for g in core.list_genres():
        words = len(core.load_corpus(g).split())
        print(f"  {g:<12} {words:>9,} words")


def cmd_suggest(a) -> None:
    model = core.load_model(a.genre, a.n, a.k)
    _print_suggestions(core.suggest(model, _text(a.text), a.count, a.explain), a.explain)


def cmd_continue(a) -> None:
    prefix = _text(a.text)
    model = core.load_model(a.genre, a.n, a.k)
    for i, s in enumerate(core.continue_text(model, prefix, a.length, a.samples, a.seed, **_sampling(a)), 1):
        print(textwrap.fill(f"{i}. {prefix} >> {s}", 100, subsequent_indent="   "))


def cmd_compare(a) -> None:
    prefix = _text(a.text)
    genres = a.genres.split(",") if a.genres else None
    print(f'"{prefix}" ...\n')
    for g, s in core.compare(prefix, genres, a.length, a.seed, a.n, a.k, **_sampling(a)).items():
        print(textwrap.fill(f"{g:<11} {s}", 100, subsequent_indent=" " * 12))


def cmd_eval(a) -> None:
    print(f"{'genre':<12}{'tokens':>10}{'V':>8}{'MLE zero %':>12}{'add-k':>10}{'Witten-Bell':>13}")
    for g in ([a.genre] if a.genre != "each" else core.list_genres()):
        train, test = core.train_test_split(core.prepare(core.load_corpus(g)), 0.1, 0)
        m = core.NgramModel(a.n, a.k).fit(train)
        unseen = sum(m.mle(s[:i], w) == 0 for s in test for i, w in enumerate(s))
        total = sum(len(s) for s in test)
        print(f"{g:<12}{sum(map(len, train)):>10,}{m.V:>8,}{unseen / total:>12.1%}"
              f"{m.perplexity(test):>10.1f}{m.perplexity(test, 'witten-bell'):>13.1f}")


HELP = """Type to add to your story. Commands:
  1-9          accept that suggestion        /more [N]   sample N more words (default 12)
  /genre NAME  switch genre                  /temp T     set temperature
  /undo        remove the last word          /show       print the draft
  /save [name] save the draft                /drafts     list saved drafts
  /why         explain the suggestions       /quit       leave"""


def run_session(genre: str = "all", n: int = 3, k: float = 0.01, input_fn=input, out=print, store: Store | None = None,
                temperature: float = 0.8, seed: int | None = None) -> str:
    """The interactive writing session. Returns the final draft (handy for tests)."""
    store = store or Store()
    model, draft = core.load_model(genre, n, k), ""
    out(f"Writing with the {genre} model ({n}-gram). /help for commands, /quit to stop.")
    while True:
        sugg = core.suggest(model, draft, 5)
        if draft:
            out("\n" + textwrap.fill(draft, 100))
        out("   " + "   ".join(f"[{i}] {s['word']}" for i, s in enumerate(sugg, 1)))
        try:
            line = input_fn("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        cmd, _, arg = line.partition(" ")
        if not line:
            continue
        if line.isdigit() and 1 <= int(line) <= len(sugg):
            draft = f"{draft} {sugg[int(line) - 1]['word']}".strip()
        elif cmd == "/quit":
            break
        elif cmd == "/help":
            out(HELP)
        elif cmd == "/more":
            more = core.continue_text(model, draft, int(arg or 12), 1, seed, temperature=temperature,
                                      top_k=40, top_p=0.95)[0]
            seed = None if seed is None else seed + 1
            draft = f"{draft} {more}".strip()
        elif cmd == "/genre":
            try:
                model, genre = core.load_model(arg, n, k), arg
                out(f"switched to {genre}")
            except ValueError as e:
                out(str(e))
        elif cmd == "/temp":
            temperature = float(arg or 0.8)
            out(f"temperature = {temperature}")
        elif cmd == "/undo":
            draft = draft.rsplit(" ", 1)[0] if " " in draft else ""
        elif cmd == "/show":
            out(draft or "(empty)")
        elif cmd == "/why":
            _print_suggestions(core.suggest(model, draft, 5, explain=True), True)
        elif cmd == "/save":
            name = arg or datetime.now().strftime("draft-%Y%m%d-%H%M%S")
            store.put(name, {"genre": genre, "text": draft})
            out(f"saved as {name}")
        elif cmd == "/drafts":
            for name, d in store.all().items():
                out(f"  {name} [{d.get('genre')}]: {d.get('text', '')[:60]}")
        elif cmd.startswith("/"):
            out("unknown command; /help lists them")
        else:
            draft = f"{draft} {line}".strip()
    return draft


def cmd_write(a) -> None:
    run_session(a.genre, a.n, a.k, temperature=a.temperature, seed=a.seed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="creative-writing-prompter", description=DESCRIPTION)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("genres", help="list the genre corpora")

    p = sub.add_parser("suggest", help="the most likely next words for your text")
    p.add_argument("text", nargs="*")
    p.add_argument("--count", "-c", type=int, default=5)
    p.add_argument("--explain", "-x", action="store_true", help="show counts, n-gram order and a corpus example")
    _model_args(p)

    p = sub.add_parser("continue", help="sample a few continuations of your text")
    p.add_argument("text", nargs="*")
    p.add_argument("--samples", "-s", type=int, default=3)
    _model_args(p)
    _sampling_args(p)

    p = sub.add_parser("compare", help="continue the same opening in every genre")
    p.add_argument("text", nargs="*")
    p.add_argument("--genres", help="comma-separated subset (default: all genres)")
    _model_args(p, genre=False)
    _sampling_args(p)
    p.set_defaults(seed=0)

    p = sub.add_parser("write", help="interactive writing session with live suggestions")
    _model_args(p)
    p.add_argument("--temperature", "-t", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("eval", help="held-out perplexity: add-k vs Witten-Bell")
    p.add_argument("--genre", "-g", default="each", help="a genre, all, or each (default)")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--k", type=float, default=0.01)

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Creative Writing Prompter is ready. Open README.md for your milestones.")
        return
    try:
        {"genres": cmd_genres, "suggest": cmd_suggest, "continue": cmd_continue, "compare": cmd_compare,
         "write": cmd_write, "eval": cmd_eval}[args.command](args)
    except ValueError as e:
        parser.exit(2, f"error: {e}\n")


if __name__ == "__main__":
    main()
