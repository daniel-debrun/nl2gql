"""Command line: ``nl2gql translate | serve | train | eval | check | schema``."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path


def _today(value: str | None) -> dt.date:
    return dt.date.fromisoformat(value) if value else dt.date.today()


def _model(path: str | None):
    from nl2gql.experiments import MODEL_PATH
    from nl2gql.model import Translator

    target = Path(path) if path else MODEL_PATH
    if not target.exists():
        sys.exit(f"no trained model at {target}; run `nl2gql train` first")
    return Translator.load(target)


def cmd_translate(args) -> int:
    from nl2gql.translate import translate

    model = _model(args.model)
    model.config.strict = args.strict
    text = " ".join(args.text) if args.text else sys.stdin.read()
    result = translate(model, text.strip(), _today(args.today))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif result.kind == "query":
        print(result.graphql)
        for note in result.assumptions:
            print(f"# {note}")
        if result.defaulted:
            print(f"# defaulted: {', '.join(result.defaulted)}")
        print(f"# confidence {result.confidence['joint']:.3f}, {result.millis:.1f} ms")
    else:
        print(f"{result.kind} ({result.reason}): {result.message}")
    return 0 if result.kind == "query" else 2


def cmd_serve(args) -> int:
    from nl2gql.server import serve

    serve(_model(args.model), args.host, args.port)
    return 0


def cmd_train(args) -> int:
    from nl2gql.experiments import run, to_markdown

    out = run(Path(args.out), tune_grid=not args.no_tune)
    print(to_markdown(out))
    return 0


def cmd_eval(args) -> int:
    from nl2gql.evaluate import load_jsonl, score
    from nl2gql.translate import translate

    model = _model(args.model)
    model.config.strict = args.strict
    examples, report = load_jsonl(args.file)
    if report.invalid:
        print(f"skipped {len(report.invalid)} lines that failed the checker", file=sys.stderr)
    s = score(lambda u, d: translate(model, u, d), examples, _today(args.today))
    print(json.dumps(s.summary(), indent=2))
    if args.errors:
        for e in s.errors:
            print(json.dumps(e))
    return 0


def cmd_check(args) -> int:
    from nl2gql.checker import main

    return main(args.files)


def cmd_schema(args) -> int:
    from nl2gql.schema import sdl

    print(sdl())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nl2gql", description="Natural language to GraphQL, no LLM.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("translate", help="translate one request")
    p.add_argument("text", nargs="*", help="the request (or read from stdin)")
    p.add_argument("--today", help="reference date YYYY-MM-DD for relative time phrases (default: today)")
    p.add_argument("--model", help="model file (default: the bundled model)")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("--strict", action="store_true", help="ask instead of filling in unrecognised parameters")
    p.set_defaults(func=cmd_translate)

    p = sub.add_parser("serve", help="HTTP service: POST /translate")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("train", help="tune, run the ablation, and write the bundled model")
    p.add_argument("--out", default="results")
    p.add_argument("--no-tune", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="score the model on an annotated JSONL file")
    p.add_argument("file")
    p.add_argument("--today", default="2026-11-18")
    p.add_argument("--model")
    p.add_argument("--errors", action="store_true")
    p.add_argument("--strict", action="store_true", help="ask instead of filling in unrecognised parameters")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("check", help="validate the graphql field of JSONL files")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("schema", help="print the composed SDL")
    p.set_defaults(func=cmd_schema)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
