"""平台无关的检索、生成与引用拼装入口。"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from time import perf_counter
from uuid import uuid4

from openai import APITimeoutError

from config import DANGER_WORDS
from generator import build_prompt, generate
from lookup import lookup_code, verify_quotes
from retrieval import load_eu, retrieve
from schema import Answer, Citation, EU, RequestCtx, _finite_number
from storage import Storage

_ERROR_TEXT = {
    "DANGER_REFUSAL": "请联系菲斯曼售后或具备资质的专业人员处理。",
    "MODEL_REQUIRED": "请提供设备型号，以便查询对应的故障代码。",
    "NO_EVIDENCE": "当前说明书中没有足够的依据回答这个问题。",
    "LLM_TIMEOUT": "回答生成超时，请稍后重试。",
    "INTERNAL": "暂时无法完成查询，请稍后重试。",
}
_STAGES = ("router", "retrieve", "generate", "validate")
_FAULT_CODE = re.compile(r"(?<![A-Za-z0-9])[FfEe][0-9]{1,2}(?![A-Za-z0-9])")
_MARKER = re.compile(r"\[([0-9]+)\]")
_SENTENCE_END = re.compile(
    r"\n+|(?:[。！？!?；;]+|\.(?=[ \t]|$|\[))(?:[ \t]*\[[0-9]+\])*"
)
_JSON_START = re.compile(r'\{|\[\s*(?=\{|"|\[)')


def _citation(
    n: int, eu: EU, quote: str, page_url_of: Callable[[int], str] | None,
) -> Citation:
    return Citation(
        n, eu.eu_id, eu.page_no, list(eu.bbox), quote,
        page_url_of(eu.page_no) if page_url_of else "",
    )


def _without_model_objects(text: str) -> tuple[str, list[str]]:
    """隔离候选文本中的 JSON 对象，保留其余文本供逐句校验。"""
    dropped = []
    parts = []
    offset = 0
    decoder = json.JSONDecoder()
    for match in _JSON_START.finditer(text):
        if match.start() < offset:
            continue
        try:
            _, end = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        parts.append(text[offset:match.start()])
        parts.append("\n")
        dropped.append(text[match.start():end])
        offset = end
    parts.append(text[offset:])
    return "".join(parts), dropped


def _sentences(text: str):
    start = 0
    for end in _SENTENCE_END.finditer(text):
        yield text[start:end.end()].strip()
        start = end.end()
    yield text[start:].strip()


def assemble(
    answer_text: str, eus: Sequence[EU], page_url_of: Callable[[int], str] | None = None,
) -> tuple[dict, list[str]]:
    """删除无有效角标的句子，按首次引用顺序重编号。"""
    candidate, dropped = _without_model_objects(answer_text)
    kept = []
    citations = {}
    source_numbers = {str(n): n for n in range(1, len(eus) + 1)}
    for sentence in _sentences(candidate):
        if not sentence:
            continue
        markers = _MARKER.findall(sentence)
        if (
            not markers
            or not any(char.isalnum() for char in _MARKER.sub("", sentence))
            or any(marker not in source_numbers for marker in markers)
            or any(not eus[source_numbers[marker] - 1].text_raw.strip() for marker in markers)
        ):
            dropped.append(sentence)
            continue
        for marker in markers:
            if marker not in citations:
                eu = eus[source_numbers[marker] - 1]
                citations[marker] = _citation(len(citations) + 1, eu, eu.text_raw, page_url_of)
        kept.append(_MARKER.sub(lambda match: f"[{citations[match[1]].n}]", sentence))
    return {"answer": "\n".join(kept), "citations": list(citations.values())}, dropped


def validate(
    resp: Answer | Mapping, eus: Sequence[EU] | None = None, model_code: str | None = None,
) -> tuple[bool, list[str]]:
    """核对响应形状、角标及引用。提供 EU 时再核对来源。"""
    data = asdict(resp) if isinstance(resp, Answer) else resp
    if not isinstance(data, Mapping) or data.keys() != Answer.__dataclass_fields__.keys():
        return False, ["answer_structure"]
    citations = data["citations"]
    timing = data["timing_ms"]
    error = data["error_code"]
    if (
        not isinstance(data["trace_id"], str) or not data["trace_id"].strip()
        or not isinstance(data["answer"], str) or not data["answer"].strip()
        or not isinstance(citations, list)
        or (error is not None and (not isinstance(error, str) or error not in _ERROR_TEXT))
        or not isinstance(timing, Mapping) or timing.keys() != set(_STAGES)
        or any(type(value) is not int or value < 0 for value in timing.values())
    ):
        return False, ["answer_structure"]

    markers = set(_MARKER.findall(data["answer"]))
    expected = {str(n) for n in range(1, len(citations) + 1)}
    if (error is None and not citations) or (
        error not in (None, "LLM_TIMEOUT") and citations
    ) or markers != expected:
        return False, ["citation_markers"]

    evidence = None if eus is None else {eu.eu_id: eu for eu in eus}
    for n, item in enumerate(citations, 1):
        citation = asdict(item) if isinstance(item, Citation) else item
        if not isinstance(citation, Mapping) or citation.keys() != Citation.__dataclass_fields__.keys():
            return False, ["citation_structure"]
        bbox = citation["bbox"]
        if (
            type(citation["n"]) is not int or citation["n"] != n
            or type(citation["page_no"]) is not int or citation["page_no"] < 1
            or not isinstance(bbox, list) or len(bbox) != 4
            or not all(_finite_number(coord) for coord in bbox)
            or not (bbox[0] < bbox[2] and bbox[1] < bbox[3])
            or not isinstance(citation["eu_id"], str) or not citation["eu_id"].strip()
            or not isinstance(citation["quote"], str) or not citation["quote"].strip()
            or not isinstance(citation["page_url"], str)
        ):
            return False, ["citation_structure"]
        if evidence is not None:
            eu = evidence.get(citation["eu_id"])
            if (
                eu is None or citation["page_no"] != eu.page_no or bbox != eu.bbox
                or citation["quote"] not in eu.text_raw
                or (model_code and eu.model_codes and model_code.upper() not in eu.model_codes)
            ):
                return False, ["citation_evidence"]
    return True, []


def ask(
    query: str, model_code: str | None, ctx: RequestCtx | dict, storage: Storage | None = None,
) -> Answer:
    trace_id = ctx.get("trace_id") if isinstance(ctx, dict) else ctx.trace_id
    trace_id = trace_id or f"tr_{uuid4().hex}"
    timing = dict.fromkeys(_STAGES, 0)
    stage = "router"
    started = perf_counter()

    def finish_stage() -> None:
        timing[stage] = int((perf_counter() - started) * 1_000)

    def refuse(error: str) -> Answer:
        return Answer(trace_id, _ERROR_TEXT[error], [], error, timing)

    page_url_of = storage.get_page_url if storage is not None else None
    model = (model_code or "").strip().upper() or None
    try:
        folded_query = query.casefold()
        codes = {match.upper() for match in _FAULT_CODE.findall(query)}
        if any(word in folded_query for word in DANGER_WORDS):
            finish_stage()
            return refuse("DANGER_REFUSAL")
        if codes:
            if model is None:
                finish_stage()
                return refuse("MODEL_REQUIRED")
            rows = [lookup_code(code, model) for code in sorted(codes)]
            if any(row and row["action_level"] == "P0_DANGER" for row in rows):
                finish_stage()
                return refuse("DANGER_REFUSAL")
            if len(rows) != 1 or rows[0] is None or rows[0]["action_level"] not in {"SAFE", "CAUTION"}:
                finish_stage()
                return refuse("NO_EVIDENCE")
            row = rows[0]
            finish_stage()

            stage, started = "validate", perf_counter()
            eus = [eu for eu in load_eu() if eu.eu_id == row["eu_id"]]
            if verify_quotes([row], eus)["mismatched"]:
                finish_stage()
                return refuse("NO_EVIDENCE")
            quote = row["quote"]
            body = {"answer": f"{quote}[1]", "citations": [_citation(1, eus[0], quote, page_url_of)]}
        else:
            finish_stage()
            stage, started = "retrieve", perf_counter()
            eus = retrieve(query, model=model)
            finish_stage()
            if not eus:
                return refuse("NO_EVIDENCE")

            stage, started = "generate", perf_counter()
            candidate = generate(build_prompt(query, eus))
            finish_stage()
            stage, started = "validate", perf_counter()
            body, _ = assemble(candidate, eus, page_url_of)

        response = Answer(trace_id, **body, error_code=None, timing_ms=timing)
        passed, _ = validate(response, eus, model)
        finish_stage()
        return response if passed else refuse("NO_EVIDENCE")
    except (APITimeoutError, TimeoutError):
        finish_stage()
        return refuse("LLM_TIMEOUT" if stage == "generate" else "INTERNAL")
    except Exception:
        finish_stage()
        return refuse("INTERNAL")
