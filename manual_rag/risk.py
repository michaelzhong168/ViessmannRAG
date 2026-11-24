"""问句和故障处置共用的安全路由规则。"""

import re

# 单独出现就需要转人工的现象和作业。
_HAZARD_SIGNALS = (
    "漏气", "气体泄漏", "燃气泄漏", "煤气泄漏", "天然气泄漏",
    "燃气味", "煤气味", "天然气味", "焦糊", "异味", "臭鸡蛋味",
    "冒烟", "起火", "一氧化碳", "触电", "漏电",
    "接气", "改气", "换气源", "拆机", "拆卸", "拆开", "拆除", "打开外壳",
    "接线", "布线", "短接", "旁路",
)

# 查询这些部件的状态是只读操作。出现处置动作时转人工。
_RESTRICTED_PARTS = (
    "燃气阀", "燃气管", "燃气压力", "燃烧器", "燃烧室", "喷嘴", "点火电极",
    "烟道", "烟管", "电源线", "保护", "联锁", "安全装置",
)
_SERVICE_ACTIONS = (
    "调", "维修", "检修", "清理", "清洗", "换", "拆装",
    "关闭", "禁用", "取消", "停用", "屏蔽", "绕过",
)
_MANUAL_IGNITION = ("手动", "强制", "打火机", "火柴")
_LPG_FUELS = ("液化气", "液化石油气")
_BOILER_NAMES = ("壁挂炉", "锅炉")
_INSTALL_ACTIONS = ("安装", "改装", "移机")
_REQUEST_CUES = ("怎么", "如何", "怎样", "自行", "自己")

_ENGLISH_SIGNALS = ("gas leak", "gas wiring", "disassemble", "dismantle", "rewire")
_ENGLISH_ACTIONS = ("install", "adjust", "replace", "repair", "clean")
_ENGLISH_TARGETS = ("boiler", "gas valve", "burner", "combustion chamber")
_ENGLISH_WORD = re.compile(r"[a-z]+")
_SENTENCE_END = re.compile(r"[。！？.!?；;]+")


def _contains(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def requires_human_review(text: str) -> bool:
    """判断规范化后的文字是否出现明确危险现象或危险处置。"""
    if _contains(text, _HAZARD_SIGNALS):
        return True

    # 只组合一句话里的动作和对象。PDF 表格里的换行可能只是排版折行。
    for sentence in _SENTENCE_END.split(text):
        if "泄漏" in sentence and _contains(sentence, _BOILER_NAMES):
            return True
        if _contains(sentence, _RESTRICTED_PARTS) and _contains(sentence, _SERVICE_ACTIONS):
            return True
        if "点火" in sentence and _contains(sentence, _MANUAL_IGNITION):
            return True
        if _contains(sentence, _LPG_FUELS) and _contains(sentence, ("改", "换")):
            return True
        if _contains(sentence, _BOILER_NAMES):
            # 例如“刚安装的锅炉怎么开暖气”描述的是已完成的安装。
            for cue in _REQUEST_CUES:
                if _contains(sentence.partition(cue)[2], _INSTALL_ACTIONS):
                    return True

        # 英文按完整词匹配，避免把普通单词中的片段认成风险词。
        words = " " + " ".join(_ENGLISH_WORD.findall(sentence)) + " "
        if any(f" {signal} " in words for signal in _ENGLISH_SIGNALS):
            return True
        if " boiler " in words and (" leak " in words or " leaking " in words):
            return True
        if (
            any(f" {action} " in words for action in _ENGLISH_ACTIONS)
            and any(f" {target} " in words for target in _ENGLISH_TARGETS)
        ):
            return True
    return False
