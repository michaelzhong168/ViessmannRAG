"""为说明书全文检索生成词项和受控查询扩展。"""

import logging
import re

import jieba

jieba.setLogLevel(logging.WARNING)

WORD = re.compile(r"[^\W_]+")
_HAN = re.compile(r"[\u3400-\u9fff]")
_QUESTION_WORDS = frozenset({
    "怎么", "如何", "怎样", "哪里", "在哪", "什么", "哪个", "哪些", "是否", "请问",
    "一下", "帮忙", "步骤", "流程", "方法",
})
_TERM_EXPANSIONS = (
    (
        re.compile(
            r"(?:请问)?"
            r"(?:(?:怎么|如何|怎样)?(?:打开|开启|启动|开)暖气"
            r"|暖气(?:怎么|如何|怎样)(?:打开|开启|启动|开))"
            r"[呢吗啊呀]?"
        ),
        (("chapter", "设置供暖和热水的运行程序"), ("text", "常规供暖运行")),
    ),
    (
        re.compile(r"(?:请问)?(?:怎么|如何|怎样)排除故障[呢吗啊呀]?"),
        (("text", "故障信息"), ("text", "故障显示"), ("text", "排除故障")),
    ),
)


def word_tokens(text: str) -> list[str]:
    """切词后只保留 unicode61 可索引的文字和数字。"""
    return [word for token in jieba.cut(text, HMM=False) for word in WORD.findall(token)]


def word_query(query: str, parts: list[str]) -> str:
    """生成用于分词索引的 MATCH 表达式，只在普通中文检索中使用。"""
    if not _HAN.search(query):
        return ""
    normalized = "".join(parts)
    for pattern, expansion in _TERM_EXPANSIONS:
        if pattern.fullmatch(normalized):
            return " OR ".join(
                f'{column} : "{" ".join(word_tokens(term))}"'
                for column, term in expansion
            )

    terms = dict.fromkeys(
        term for term in word_tokens(query)
        if len(term) >= 2 and term not in _QUESTION_WORDS
    )
    # 合取有效词可以避免仅凭「故障」等一个宽泛词就召回无关段落。
    return " AND ".join(f'text : "{term}"' for term in terms)
