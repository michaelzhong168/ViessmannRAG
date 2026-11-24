"""菲斯曼说明书索引与问答命令。"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pymupdf
from openai import OpenAIError

from manual_rag.answer import DEFAULT_TOP_K, ask
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
    args = parser.parse_args(argv)

    pymupdf.no_recommend_layout()
    if args.command == "ask":
        try:
            result = ask(load_index(args.index), args.question, args.control_type, top_k=args.top_k)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"问答失败：{exc}", file=sys.stderr)
            return 1
        print(json.dumps(asdict(result), ensure_ascii=False))
        return int(result.error_code in {"model_timeout", "model_service_error", "invalid_model_response"})

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
