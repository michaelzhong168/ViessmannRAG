"""菲斯曼说明书向量建库命令。"""

import argparse
import json
import sys
from pathlib import Path

import pymupdf
from openai import OpenAIError

from manual_rag.index import DEFAULT_INDEX, build_index
from manual_rag.source import CONTROL_TYPES


def main(argv: list[str] | None = None) -> int:
    """运行 CLI，成功时输出索引路径和节点数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="从核定的 PDF 构建并发布本地索引")
    index.add_argument("pdf", type=Path)
    index.add_argument("--output", type=Path, default=DEFAULT_INDEX, help="索引文件路径")
    scope = index.add_mutually_exclusive_group(required=True)
    scope.add_argument("--scopes", type=Path, help="按物理页段声明操作类型")
    scope.add_argument("--control-type", choices=CONTROL_TYPES, help="声明整本说明书的操作类型")
    args = parser.parse_args(argv)

    pymupdf.no_recommend_layout()
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
