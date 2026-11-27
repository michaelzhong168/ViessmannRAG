"""用问答入口执行逐题评测，并从本地运行材料离线回放。"""

import json
import os
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import pymupdf
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from manual_rag.answer import (
    DEFAULT_TOP_K, AnswerResult, AnswerTrace, ask, parse_answer_stream, service_error_result,
)
from manual_rag.evaluation_summary import render_attribution_chart, summarize_report
from manual_rag.fact_judgment import STRATEGY, decode_judgment, request_judgment
from manual_rag.index import DEFAULT_INDEX, restore_index
from manual_rag.question_set import is_identifier, load_question_set, normalize_evidence_text
from tools import llm

DEFAULT_QUESTIONS = Path("data/eval/golden_manual.jsonl")
DEFAULT_RUNS = Path("data/eval/runs")
_PHRASE_FORMAT_VERSION = 1
_FORMAT_VERSION = 2
_SCORING_VERSION = 1


def _evaluation_code_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    sources = sorted([*root.glob("manual_rag/*.py"), *root.glob("tools/*.py")])
    manifest = {path.relative_to(root).as_posix(): _sha256(path.read_bytes()) for path in sources}
    return _sha256(_json(manifest).encode("utf-8"))


def _evidence(node: TextNode) -> dict:
    return {
        "node_id": node.node_id,
        "doc_id": node.ref_doc_id,
        "pdf_version": node.source_node.metadata["pdf_version"],
        "page_no": node.metadata["page_no"],
        "bbox": list(node.metadata["bbox"]),
        "control_type": node.metadata["control_type"],
        "section": node.metadata["section"],
        "text": node.text,
    }


def _restore_node(evidence: dict) -> TextNode:
    return TextNode(
        id_=evidence["node_id"],
        text=evidence["text"],
        metadata={key: evidence[key] for key in ("page_no", "bbox", "control_type", "section")},
        relationships={
            NodeRelationship.SOURCE: RelatedNodeInfo(
                node_id=evidence["doc_id"], metadata={"pdf_version": evidence["pdf_version"]},
            ),
        },
    )


def _match_groups(question: dict, index_nodes: dict[str, dict]) -> list[set[str]]:
    locations: dict[tuple, list[dict]] = {}
    for node in index_nodes.values():
        if question["control_type"] in (None, node["control_type"]):
            key = (node["doc_id"], node["pdf_version"], node["page_no"])
            locations.setdefault(key, []).append(node)

    groups = []
    for group in question["evidence_groups"]:
        matches = set()
        for anchor in group["anchors"]:
            key = (anchor["doc_id"], anchor["pdf_version"], anchor["page_no"])
            quote = normalize_evidence_text(anchor["quote"])
            for node in locations.get(key, []):
                if quote not in normalize_evidence_text(node["text"]):
                    continue
                if "bbox" in anchor and not pymupdf.Rect(anchor["bbox"]).intersects(pymupdf.Rect(node["bbox"])):
                    continue
                matches.add(node["node_id"])
        groups.append(matches)
    return groups


def _score(
    question: dict, index_nodes: dict[str, dict], trace: dict, result: AnswerResult,
    citation_issues: list[str], facts: list[bool | None], fact_judgment_error: str | None,
) -> dict:
    failure = trace["service_failure"]
    behavior_ok = result.error_code == question["expected_error_code"]
    if failure is not None:
        return {
            "passed": False, "behavior_ok": False, "groups": [], "citations_ok": None,
            "attributions": ["service_error"],
        }

    recalled = {match["node_id"] for match in trace["retrieval"]}
    context = {node["node_id"] for node in trace["context"]}
    cited = {citation.node_id for citation in result.citations}
    groups = [
        {
            "required_fact": group["required_fact"],
            "indexed_node_ids": sorted(matches),
            "recalled": bool(matches & recalled),
            "in_context": bool(matches & context),
            "cited": bool(matches & cited),
            "fact_present": present,
        }
        for group, matches, present in zip(
            question["evidence_groups"], _match_groups(question, index_nodes), facts, strict=True,
        )
    ]
    answered = behavior_ok and question["expected_error_code"] is None
    citations_ok = not citation_issues and (not answered or all(group["cited"] for group in groups))
    attributions = []
    if any(not group["indexed_node_ids"] for group in groups):
        attributions.append("missing_material")
    if any(group["indexed_node_ids"] and not group["in_context"] for group in groups):
        attributions.append("retrieval_miss")
    if not behavior_ok and not citation_issues:
        attributions.append("answer_strategy_error")
    if citation_issues or (answered and any(group["in_context"] and not group["cited"] for group in groups)):
        attributions.append("citation_error")
    if fact_judgment_error is not None:
        attributions.append("service_error")
    if answered and any(group["in_context"] and group["fact_present"] is False for group in groups):
        attributions.append("missing_fact")
    return {
        "passed": behavior_ok and citations_ok and all(group["fact_present"] for group in groups),
        "behavior_ok": behavior_ok,
        "groups": groups,
        "citations_ok": citations_ok,
        "attributions": attributions,
    }


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _write_json(path: Path, value: object) -> None:
    path.write_text(_json(value) + "\n", encoding="utf-8")


def _sha256(contents: bytes) -> str:
    return sha256(contents).hexdigest()


def run_evaluation(
    questions_path: Path = DEFAULT_QUESTIONS, index_path: Path = DEFAULT_INDEX, runs: Path = DEFAULT_RUNS,
    *, split: str = "debug", question_id: str | None = None, top_k: int = DEFAULT_TOP_K,
) -> dict:
    """评测指定分集，完整发布冻结材料和汇总后返回运行 ID、路径和通过数。"""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 须为正整数")
    rows = [row for row in load_question_set(questions_path)
            if row["split"] == split and (question_id is None or row["id"] == question_id)]
    if not rows:
        raise ValueError("所选分集没有匹配的题目")

    index_bytes = index_path.read_bytes()
    snapshot = json.loads(index_bytes)
    index = restore_index(snapshot)
    nodes = {
        node.node_id: _evidence(node)
        for node in index.docstore.get_nodes(list(index.index_struct.nodes_dict.values()))
        if isinstance(node, TextNode) and node.source_node is not None
    }
    run_id = uuid4().hex
    run_dir = runs / run_id
    runs.mkdir(parents=True, exist_ok=True, mode=0o700)
    with TemporaryDirectory(dir=runs, prefix=".pending-") as directory:
        staged = Path(directory) / run_id
        staged.mkdir(mode=0o700)
        (staged / "traces").mkdir()
        frozen_questions = "".join(_json(row) + "\n" for row in rows).encode()
        frozen_nodes = (_json(nodes) + "\n").encode()
        (staged / "index.json").write_bytes(index_bytes)
        (staged / "questions.jsonl").write_bytes(frozen_questions)
        (staged / "evidence.json").write_bytes(frozen_nodes)
        generation = {
            "model": (os.environ.get("LLM_MODEL") or "").strip() or None,
            "temperature": llm.TEMPERATURE,
            "max_tokens": llm.MAX_TOKENS,
        }
        config = {
            "format_version": _FORMAT_VERSION,
            "run_id": run_id,
            "dataset_version": rows[0]["dataset_version"],
            "questions_sha256": _sha256(frozen_questions),
            "split": split,
            "question_id": question_id,
            "question_count": len(rows),
            "evaluation_code_sha256": _evaluation_code_sha256(),
            "scoring_version": _SCORING_VERSION,
            "index_sha256": _sha256(index_bytes),
            "evidence_sha256": _sha256(frozen_nodes),
            "index": {key: snapshot[key] for key in ("format_version", "layout_version", "source", "embedding_model")},
            "top_k": top_k,
            "generation": generation,
            "fact_judgment": {"strategy": STRATEGY, **generation},
        }
        _write_json(staged / "run.json", config)

        _record_questions(staged, rows, index, nodes, run_id, top_k)
        summary = summarize_report(staged / "report.jsonl", config, rows)
        _write_json(staged / "summary.json", summary)
        (staged / "attributions.html").write_text(render_attribution_chart(summary), encoding="utf-8")
        staged.rename(run_dir)
    return {"run_id": run_id, "run_dir": str(run_dir), "question_count": len(rows), "passed": summary["passed"]}


def _record_questions(
    run_dir: Path, rows: list[dict], index: VectorStoreIndex, nodes: dict[str, dict], run_id: str, top_k: int,
) -> None:
    with (run_dir / "report.jsonl").open("w", encoding="utf-8") as report:
        for row in rows:
            capture = AnswerTrace()
            result = ask(index, row["question"], row["control_type"], top_k=top_k, trace=capture)
            facts = [group["required_fact"] for group in row["evidence_groups"]]
            judgment = None
            if result.error_code is None and facts:
                started = perf_counter()
                judgment = request_judgment(result.answer, facts)
                elapsed = perf_counter() - started
                capture.timings_s["fact_judgment"] = elapsed
                capture.timings_s["total"] += elapsed
            record = {
                "format_version": _FORMAT_VERSION,
                "run_id": run_id,
                "question_id": row["id"],
                "question": row["question"],
                "control_type": row["control_type"],
                "retrieval": capture.retrieval,
                "context": [_evidence(node) for node in capture.context],
                "system_prompt": capture.system_prompt,
                "prompt": capture.prompt,
                "chunks": capture.chunks,
                "service_failure": capture.service_failure,
                "fact_judgment": judgment,
            }
            outcome = _outcome(row, nodes, record, result, capture.citation_issues)
            _write_json(run_dir / "traces" / f"{row['id']}.json", {**record, **outcome, "timings_s": capture.timings_s})
            report.write(_json({**outcome, "timings_s": capture.timings_s}) + "\n")


def _outcome(
    row: dict, nodes: dict[str, dict], record: dict, result: AnswerResult, citation_issues: list[str],
) -> dict:
    required_facts = [group["required_fact"] for group in row["evidence_groups"]]
    legacy = record["format_version"] == _PHRASE_FORMAT_VERSION
    facts: list[bool | None] = [False] * len(required_facts)
    judgment_error = None
    if result.error_code is None and required_facts:
        if legacy:
            answer = normalize_evidence_text(result.answer)
            facts = [normalize_evidence_text(fact) in answer for fact in required_facts]
        else:
            facts, judgment_error = decode_judgment(record["fact_judgment"], result.answer, required_facts)
    outcome = {
        "question_id": row["id"],
        "expected_error_code": row["expected_error_code"],
        "result": asdict(result),
        "service_failure": record["service_failure"],
        "citation_issues": citation_issues,
        "score": _score(row, nodes, record, result, citation_issues, facts, judgment_error),
    }
    if not legacy:
        outcome["fact_judgment_error"] = judgment_error
    return outcome


def replay_question(runs: Path, run_id: str, question_id: str) -> dict:
    """按运行和题目 ID 重新解析、校验引用和评分，完全使用冻结的本地材料。"""
    if not is_identifier(run_id) or not is_identifier(question_id):
        raise ValueError("运行和题目 ID 须为小写 ASCII 标识符")
    run_dir = runs / run_id
    config = json.loads((run_dir / "run.json").read_bytes())
    record = json.loads((run_dir / "traces" / f"{question_id}.json").read_bytes())
    version = config["format_version"]
    if (
        version not in (_PHRASE_FORMAT_VERSION, _FORMAT_VERSION) or record["format_version"] != version
        or config["run_id"] != run_id or record["run_id"] != run_id or record["question_id"] != question_id
    ):
        raise ValueError("运行或 Trace 格式及标识不匹配")
    if version == _FORMAT_VERSION and config["fact_judgment"]["strategy"] != STRATEGY:
        raise ValueError("事实判定版本不匹配")

    questions = (run_dir / "questions.jsonl").read_bytes()
    evidence = (run_dir / "evidence.json").read_bytes()
    if _sha256(questions) != config["questions_sha256"] or _sha256(evidence) != config["evidence_sha256"]:
        raise ValueError("冻结的题集或索引证据摘要不匹配")
    row = next((row for row in map(json.loads, questions.splitlines()) if row["id"] == question_id), None)
    nodes = json.loads(evidence)
    if row is None or any(record[key] != row[key] for key in ("question", "control_type")):
        raise ValueError("Trace 与冻结题目不匹配")
    if any(nodes.get(node["node_id"]) != node for node in record["context"]):
        raise ValueError("Trace 上下文与冻结索引证据不匹配")

    failure = record["service_failure"]
    citation_issues = []
    if failure is not None:
        result = service_error_result(failure["error_code"])
    else:
        context = [_restore_node(node) for node in record["context"]]
        result = parse_answer_stream(iter(record["chunks"]), context, None, citation_issues)
    outcome = _outcome(row, nodes, record, result, citation_issues)
    # 比较 JSON 结构，确保引用坐标与落盘后的表示相同。
    return {
        "run_id": run_id,
        **outcome,
        "matches_original": all(outcome[key] == record[key] for key in outcome),
    }
