"""用原文证据构造提示，并经统一模型入口生成候选文本。"""

from collections.abc import Sequence

from schema import EU
from tools import llm

RULES = (
    "只依据下列说明书原文回答问题。每个事实句都要带对应的 [n] 角标。"
    "数字和单位必须逐字保留。没有依据时说不知道。仅输出答案文本。"
)


def build_prompt(query: str, eus: Sequence[EU]) -> str:
    sources = "\n\n".join(f"[{number}]\n{eu.text_raw}" for number, eu in enumerate(eus, 1))
    return f"{RULES}\n\n问题：{query}\n\n说明书原文：\n{sources}"


def generate(prompt: str) -> str:
    return llm.generate(prompt)
