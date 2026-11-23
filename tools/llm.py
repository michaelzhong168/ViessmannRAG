"""G01 · 全项目唯一的模型调用入口。

业务层只允许经由本模块调用模型；换厂商只改 ``LLM_BASE_URL`` / ``LLM_MODEL``
两行环境变量，业务代码不动。

注意：本模块**导入期不读环境变量、不构造 client**——缺少 ``DASHSCOPE_API_KEY``
时也必须能 import 成功，否则 mock 测试无法加载模块（规格 v1.2 FR-004）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

#: system prompt 
SYSTEM_PROMPT = (
    "你是菲斯曼壁挂炉说明书的问答助手。只依据用户提供的原文回答问题，"
    "不编造原文中没有的内容。"
)

#: 非模型常量：允许作为缺省值，不属于「写死模型名」
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TIMEOUT_S = "10"
TEMPERATURE = 0.1
MAX_TOKENS = 400

#: 项目根目录 .env（不存在时 load_dotenv 是 no-op）
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

MISSING_KEY_HINT = "缺少 DASHSCOPE_API_KEY，请检查 DASHSCOPE_API_KEY 是否已写入项目根目录 .env"


def _load_env() -> None:
    """加载 .env；``override=False`` 保证已存在的环境变量优先。"""
    load_dotenv(ENV_FILE, override=False)


def _get_client() -> OpenAI:
    """延迟构造 client：只有真正调用模型时才校验密钥。"""
    _load_env()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(MISSING_KEY_HINT)
    base_url = os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL
    return OpenAI(api_key=api_key, base_url=base_url)


def _resolve_model(model: str | None) -> str:
    """显式入参优先，其次 ``LLM_MODEL``；两者都空则报错（不写死模型名）。"""
    name = (model or os.environ.get("LLM_MODEL") or "").strip()
    if not name:
        raise RuntimeError("缺少 LLM_MODEL，请检查 .env 中的 LLM_MODEL")
    return name


def _resolve_timeout() -> float:
    raw = os.environ.get("LLM_TIMEOUT_S") or DEFAULT_TIMEOUT_S
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"LLM_TIMEOUT_S 不是合法数字：{raw!r}") from exc


def generate(prompt: str, model: str | None = None) -> str:
    """唯一模型入口：固定 temperature / max_tokens / timeout，返回纯文本。"""
    client = _get_client()
    model_name = _resolve_model(model)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=_resolve_timeout(),
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"模型返回空内容（model={model_name}）")
    return content


def main(argv: list[str] | None = None) -> int:
    """``python -m tools.llm "你好"`` 手工冒烟入口。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print('用法：python -m tools.llm "你好"', file=sys.stderr)
        return 2

    try:
        print(generate(" ".join(args)))
    except (RuntimeError, OpenAIError) as exc:
        # 缺密钥 / 网络 / 鉴权失败：一行提示 + 非零退出，不抛 traceback
        print(f"调用失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
