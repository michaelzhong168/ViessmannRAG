"""本地说明书问答入口。"""

import argparse
import json
from dataclasses import asdict

from rag_core import ask
from schema import RequestCtx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ask_command = commands.add_parser("ask")
    ask_command.add_argument("query")
    ask_command.add_argument("--model")
    args = parser.parse_args(argv)
    answer = ask(args.query, args.model, RequestCtx(platform="cli"))
    print(json.dumps(asdict(answer), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
