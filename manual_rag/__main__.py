"""菲斯曼说明书索引与问答命令。"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pymupdf
from openai import OpenAIError

from manual_rag.answer import DEFAULT_TOP_K, ask
from manual_rag.evaluation import DEFAULT_QUESTIONS, DEFAULT_RUNS, replay_question, run_evaluation
from manual_rag.index import DEFAULT_INDEX, build_index, load_index
from manual_rag.source import CONTROL_TYPES


def main(argv: list[str] | None = None) -> int:
    """运行 CLI，输出建库信息或带原页引用的回答。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="从核定的 PDF 构建并发布本地索引")
    index.add_argument("pdf", type=Path)
    index.add_argument("--output", type=Path, default=DEFAULT_INDEX, help="索引文件路径")
    scope = index.add_mutually_exclusive_group(required=True)
    scope.add_argument("--scopes", type=Path, help="按物理页段声明操作类型")
    scope.add_argument("--control-type", choices=CONTROL_TYPES, help="声明整本说明书的操作类型")
    query = commands.add_parser("ask", help="检索说明书并由模型回答")
    query.add_argument("question")
    query.add_argument("--control-type", help="将已知的设备操作类型提供给模型")
    query.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="已发布的索引文件")
    query.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="送给模型的候选片段数量")
    output = query.add_mutually_exclusive_group()
    output.add_argument("--stream", action="store_true", help="逐块输出正文，结束后显示引用")
    output.add_argument("--json", action="store_true", help="一次性输出 JSON")
    evaluation = commands.add_parser("eval", help="运行题集并保存逐题评测和 Trace")
    evaluation.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    evaluation.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    evaluation.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    evaluation.add_argument("--split", choices=("debug", "holdout"), default="debug")
    evaluation.add_argument("--question-id", help="仅运行所选分集中的指定题目")
    evaluation.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    replay = commands.add_parser("replay", help="按运行与题目 ID 离线回放")
    replay.add_argument("run_id")
    replay.add_argument("question_id")
    replay.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args(argv)

    pymupdf.no_recommend_layout()
    if args.command in ("eval", "replay"):
        try:
            if args.command == "eval":
                outcome = run_evaluation(
                    args.questions, args.index, args.runs,
                    split=args.split, question_id=args.question_id, top_k=args.top_k,
                )
            else:
                outcome = replay_question(args.runs, args.run_id, args.question_id)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            print(f"评测或回放失败：{type(exc).__name__}", file=sys.stderr)
            return 1
        print(json.dumps(outcome, ensure_ascii=False))
        return int(args.command == "replay" and not outcome["matches_original"])

    if args.command == "ask":
        streaming = args.stream or (not args.json and sys.stdout.isatty())
        emitted = False

        def write_delta(chunk: str) -> None:
            nonlocal emitted
            print(chunk, end="", flush=True)
            emitted = True

        try:
            result = ask(
                load_index(args.index), args.question, args.control_type,
                top_k=args.top_k, on_delta=write_delta if streaming else None,
            )
        except KeyboardInterrupt:
            print("\n问答已取消。" if emitted else "问答已取消。", file=sys.stderr)
            return 130
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"问答失败：{exc}", file=sys.stderr)
            return 1
        failed = result.error_code in {"model_timeout", "model_service_error", "invalid_model_response"}
        if streaming:
            if emitted:
                print()
            if failed or not emitted:
                print(result.answer, file=sys.stderr if failed else sys.stdout)
            if result.citations:
                print("引用")
                for citation in result.citations:
                    print(json.dumps(asdict(citation), ensure_ascii=False))
        else:
            print(json.dumps(asdict(result), ensure_ascii=False))
        return int(failed)

    try:
        count = build_index(
            args.pdf, args.output, scopes_path=args.scopes, control_type=args.control_type,
        )
    except OpenAIError as exc:
        status = getattr(exc, "status_code", None)
        print(f"建库失败：百炼 Embedding 请求失败（{status or type(exc).__name__}）", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, pymupdf.FileDataError) as exc:
        print(f"建库失败：{exc}", file=sys.stderr)
        return 1

    print(json.dumps({"index": str(args.output), "node_count": count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
