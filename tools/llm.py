"""百炼 OpenAI 兼容客户端与文本生成入口。"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

SYSTEM_PROMPT = (
    "你是菲斯曼壁挂炉说明书的问答助手。只依据用户提供的原文回答问题，"
    "不编造原文中没有的内容。"
)

#: 非模型常量：允许作为缺省值，不属于「写死模型名」
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TEMPERATURE = 0.1
MAX_TOKENS = 400

#: 项目根目录 .env（不存在时 load_dotenv 是 no-op）
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

MISSING_KEY_HINT = "缺少 DASHSCOPE_API_KEY，请检查 DASHSCOPE_API_KEY 是否已写入项目根目录 .env"


def _load_env() -> None:
    """加载 .env；``override=False`` 保证已存在的环境变量优先。"""
    load_dotenv(ENV_FILE, override=False)


def _get_client(*, timeout: float | None = None) -> OpenAI:
    """延迟构造 client：只有真正调用模型时才校验密钥。"""
    _load_env()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(MISSING_KEY_HINT)
    base_url = os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL
    if timeout is None:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=None)
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)


def _resolve_model(model: str | None) -> str:
    """显式入参优先，其次 ``LLM_MODEL``；两者都空则报错（不写死模型名）。"""
    name = (model or os.environ.get("LLM_MODEL") or "").strip()
    if not name:
        raise RuntimeError("缺少 LLM_MODEL，请检查 .env 中的 LLM_MODEL")
    return name


def _messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def generate(
    prompt: str, model: str | None = None, *, system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """请求完整回复并返回纯文本。"""
    client = _get_client()
    model_name = _resolve_model(model)

    response = client.chat.completions.create(
        model=model_name,
        messages=_messages(prompt, system_prompt),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"模型返回空内容（model={model_name}）")
    return content


def stream(
    prompt: str, *, system_prompt: str = SYSTEM_PROMPT, timeout: float | None = None,
) -> Iterator[str]:
    """逐块返回回复正文，断流或截断时抛出服务异常。"""
    client = _get_client(timeout=timeout)
    completed = False
    with client.chat.completions.create(
        model=_resolve_model(None),
        messages=_messages(prompt, system_prompt),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
    ) as chunks:
        for chunk in chunks:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta.content:
                yield choice.delta.content
            if choice.finish_reason == "stop":
                completed = True
    if not completed:
        raise OpenAIError("模型回复流未完整结束")


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
