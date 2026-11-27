"""用合成 PDF 和外部 HTTP 替身验证评测、归因和离线回放。"""

import json
from hashlib import sha256

import httpx
import pymupdf
import pytest
from openai import OpenAI

from manual_rag import __main__, index, nodes
from tools import llm


class _ModelStream(httpx.SyncByteStream):
    """将模型正文按原顺序送入 SSE，允许在中途模拟断流。"""

    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            choice = {"delta": {"content": chunk}, "finish_reason": "stop" if chunk is None else None}
            yield f"data: {json.dumps({'choices': [choice]})}\n\n".encode()


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    pdf_path = tmp_path / "manual.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        for y, line in [(60, "Heating"), (85, "1. Select mode."), (110, "2. Wait for display.\nDisplay")]:
            page.insert_text((50, y), line)
        pdf.set_toc([[1, "Heating", 1]])
        pdf.new_page().insert_text((50, 70), "Unrelated instructions.")
        pdf.save(pdf_path)
    version = sha256(pdf_path.read_bytes()).hexdigest()
    calls = []
    state = {"response": [], "offline": False, "decoy": False, "embedding_error": False}

    def respond(request):
        calls.append(request.url.path)
        assert not state["offline"], "离线回放访问了模型服务"
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            if state["embedding_error"]:
                return httpx.Response(400, json={"error": {"message": "secret-error-body"}})
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return httpx.Response(200, json={"data": [
                {"index": i, "embedding": [1, 0] if (
                    text.endswith("1. Select mode.") or text == "How to heat?" and not state["decoy"]
                ) else [0, 1] if (
                    text == "How to heat?" or text.endswith("Unrelated instructions.")
                ) else [-1, 0]}
                for i, text in enumerate(inputs)
            ]})
        assert request.url.path.endswith("/chat/completions")
        if not body.get("stream"):
            state["judge_request"] = body
            judgment = state.get("judgment", "[true]")
            if isinstance(judgment, Exception):
                raise judgment
            return httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": judgment}, "finish_reason": "stop"},
            ]})
        state["request"] = body
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=_ModelStream(state["response"]))

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-test-key")
    monkeypatch.setenv("EMBED_MODEL", "offline-embedding")
    monkeypatch.setenv("LLM_MODEL", "offline-chat")
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        def client(**kwargs):
            kwargs.setdefault("max_retries", 0)
            return OpenAI(**kwargs, http_client=http)

        monkeypatch.setattr(llm, "OpenAI", client)
        index_path = tmp_path / "index.json"
        index.build_index(pdf_path, index_path, control_type="按键式")
        question = {
            "dataset_version": "fixture-v1", "id": "q1", "intent_id": "heating", "split": "debug",
            "question": "How to heat?", "control_type": "按键式", "expected_error_code": None,
            "evidence_groups": [{
                "required_fact": "Wait for display.",
                "anchors": [{
                    "doc_id": "manual", "pdf_version": version, "page_no": 1,
                    "quote": "2. Wait for display.", "bbox": [49, 96, 165, 115],
                }],
            }],
        }
        yield pdf_path, index_path, question, calls, state


def _run(tmp_path, capsys, index_path, question):
    questions = tmp_path / "questions.jsonl"
    holdout = {**question, "id": "q2", "intent_id": "holdout-heating", "split": "holdout"}
    questions.write_text(json.dumps(question) + "\n" + json.dumps(holdout) + "\n", encoding="utf-8")
    assert __main__.main([
        "eval", "--questions", str(questions), "--index", str(index_path),
        "--runs", str(tmp_path / "runs"), "--top-k", "1",
    ]) == 0
    run = json.loads(capsys.readouterr().out)
    assert run["question_count"] == 1
    run_dir = tmp_path / "runs" / run["run_id"]
    contents = (run_dir / "traces" / "q1.json").read_text(encoding="utf-8")
    report, = map(json.loads, (run_dir / "report.jsonl").read_text(encoding="utf-8").splitlines())
    assert "secret-test-key" not in contents and "secret-error-body" not in contents
    assert "secret-test-key" not in (run_dir / "run.json").read_text(encoding="utf-8")
    trace = json.loads(contents)
    assert report["score"] == trace["score"]
    assert run["passed"] == trace["score"]["passed"]
    return run, trace


@pytest.mark.parametrize(("scenario", "attribution"), [
    ("pass", []),
    ("wrong-page", ["missing_material"]),
    ("wrong-version", ["missing_material"]),
    ("wrong-box", ["missing_material"]),
    ("retrieval", ["retrieval_miss"]),
    ("strategy", ["answer_strategy_error"]),
    ("wrong-anchor", ["citation_error"]),
    ("invalid-citation", ["citation_error"]),
    ("absent-citation", ["citation_error"]),
    ("omit-fact", ["missing_fact"]),
    ("service-generation", ["service_error"]),
    ("service-retrieval", ["service_error"]),
])
def test_run_and_replay(evaluation, tmp_path, capsys, monkeypatch, scenario, attribution):
    """用现场问答及回放区分六种失败，同时保留同节补入与部分流。"""
    _, index_path, question, calls, state = evaluation
    code, citations, answer = None, [2], "Wait for display."
    if scenario.startswith("wrong-") and scenario != "wrong-anchor":
        field, value = {
            "wrong-page": ("page_no", 2),
            "wrong-version": ("pdf_version", "f" * 64),
            "wrong-box": ("bbox", [100, 130, 130, 140]),
        }[scenario]
        question["evidence_groups"][0]["anchors"][0][field] = value
    if scenario == "retrieval":
        state["decoy"] = True
        code, citations = "insufficient_evidence", []
    if scenario == "strategy":
        code, citations = "missing_control_type", []
    if scenario == "wrong-anchor":
        citations = [1]
    if scenario == "invalid-citation":
        citations = [99]
    if scenario == "absent-citation":
        citations = []
    if scenario == "omit-fact":
        answer = "Select heating mode."
        state["judgment"] = "[false]"
    state["embedding_error"] = scenario == "service-retrieval"
    header = json.dumps({"citations": citations, "error_code": code})
    chunks = [header[:8], header[8:] + "\n", answer, None]
    if scenario == "service-generation":
        chunks[-1] = httpx.ReadTimeout("secret-error-body")
    state["response"] = chunks
    run, trace = _run(tmp_path, capsys, index_path, question)
    expected = attribution + (["answer_strategy_error"] if scenario == "retrieval" else [])
    assert trace["score"]["attributions"] == expected
    assert trace["score"]["passed"] == (scenario == "pass")
    if trace["result"]["error_code"] is not None:
        assert trace["fact_judgment"] is None and "judge_request" not in state
    assert trace["timings_s"]["total"] >= sum(value for stage, value in trace["timings_s"].items()
                                               if stage != "total")
    if scenario == "pass":
        first, added = trace["context"]
        assert trace["retrieval"] == [{"node_id": first["node_id"], "rank": 1, "score": 1.0}]
        assert trace["score"]["groups"][0] == {
            "required_fact": "Wait for display.", "indexed_node_ids": [added["node_id"]],
            "recalled": False, "in_context": True, "cited": True, "fact_present": True,
        }
        system, user = state["request"]["messages"]
        assert trace["system_prompt"] == system["content"]
        assert trace["prompt"] == user["content"]
        assert trace["chunks"] == chunks[:-1]
    if scenario == "service-generation":
        assert trace["chunks"] == chunks[:-1]
        assert trace["service_failure"] == {
            "stage": "generation", "error_code": "model_timeout", "category": "APITimeoutError",
        }
    if scenario == "service-retrieval":
        assert trace["service_failure"]["stage"] == "retrieval"
        assert trace["retrieval"] == trace["context"] == trace["chunks"] == []
        assert trace["prompt"] is None and "generation" not in trace["timings_s"]

    call_count = len(calls)
    state["offline"] = True
    for key in ("DASHSCOPE_API_KEY", "EMBED_MODEL", "LLM_MODEL"):
        monkeypatch.delenv(key)
    index_path.unlink()
    (tmp_path / "questions.jsonl").unlink()
    assert __main__.main(["replay", run["run_id"], "q1", "--runs", str(tmp_path / "runs")]) == 0
    replayed = json.loads(capsys.readouterr().out)
    assert replayed["matches_original"]
    for key in ("score", "result", "service_failure", "citation_issues"):
        assert replayed[key] == trace[key]
    assert len(calls) == call_count


def test_anchor_survives_rechunk(evaluation, tmp_path, capsys, monkeypatch):
    """更改版面规则与节点 ID 后，从同一物理页匹配同一引文。"""
    pdf_path, index_path, question, _, state = evaluation
    state["response"] = ['{"citations":[2],"error_code":null}\n', "Wait for display.", None]
    _, original = _run(tmp_path, capsys, index_path, question)
    monkeypatch.setattr(nodes, "LAYOUT_VERSION", 2)
    monkeypatch.setattr(index, "LAYOUT_VERSION", 2)
    monkeypatch.setattr(nodes, "_HEADING_SIZE_RATIO", 1.0)
    index.build_index(pdf_path, index_path, control_type="按键式")
    _, rechunked = _run(tmp_path, capsys, index_path, question)
    assert original["score"]["passed"] and rechunked["score"]["passed"]
    assert original["result"]["citations"][0]["node_id"] != rechunked["result"]["citations"][0]["node_id"]
    assert original["result"]["citations"][0]["quote"] == "2. Wait for display.\nDisplay"
    assert rechunked["result"]["citations"][0]["quote"] == "2. Wait for display."


def test_replay_recomputes_saved_answer_and_detects_snapshot_change(evaluation, tmp_path, capsys):
    """回放从流重算判定，拒绝与运行摘要不符的冻结标注。"""
    _, index_path, question, _, state = evaluation
    state["response"] = ['{"citations":[2],"error_code":null}\n', "Wait for display.", None]
    run, trace = _run(tmp_path, capsys, index_path, question)
    run_dir = tmp_path / "runs" / run["run_id"]
    trace["result"]["answer"] = "保存结果被篡改"
    trace["score"]["passed"] = False
    (run_dir / "traces" / "q1.json").write_text(json.dumps(trace), encoding="utf-8")
    argv = ["replay", run["run_id"], "q1", "--runs", str(tmp_path / "runs")]

    state["offline"] = True
    assert __main__.main(argv) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["result"]["answer"] == "Wait for display."
    assert result["score"]["passed"] and not result["matches_original"]

    trace["fact_judgment"]["response"] = "[false]"
    (run_dir / "traces" / "q1.json").write_text(json.dumps(trace), encoding="utf-8")
    assert __main__.main(argv) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["score"]["attributions"] == ["missing_fact"]

    with (run_dir / "questions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    assert __main__.main(argv) == 1
    assert "ValueError" in capsys.readouterr().err


@pytest.mark.parametrize(("answer", "judgment", "expected", "error"), [
    ("When both lights flash together, wait for the screen.", "[true,true]", [True, True], None),
    ("Wait for the screen.", "[true,false]", [True, False], None),
    ("Wait for the screen.", "[true]", [None, None], "invalid_fact_judgment"),
    ("Wait for the screen.", '[true,"false"]', [None, None], "invalid_fact_judgment"),
    ("Wait for the screen.", httpx.ReadTimeout("secret-error-body"), [None, None], "APITimeoutError"),
])
def test_semantic_fact_judgment(evaluation, tmp_path, capsys, answer, judgment, expected, error):
    """用同义答文、漏条件和判定异常验证逐事实应用模型结果。"""
    _, index_path, question, calls, state = evaluation
    original, = question["evidence_groups"]
    required_condition = "The two indicators must flash simultaneously."
    question["evidence_groups"].append({**original, "required_fact": required_condition})
    state["response"] = ['{"citations":[2],"error_code":null}\n', answer, None]
    state["judgment"] = judgment
    call_count = len(calls)
    run, trace = _run(tmp_path, capsys, index_path, question)

    score = trace["score"]
    assert [group["fact_present"] for group in score["groups"]] == expected
    assert score["passed"] == all(expected)
    assert trace["fact_judgment_error"] == error
    assert score["attributions"] == (
        ["service_error"] if error else [] if all(expected) else ["missing_fact"]
    )
    assert trace["result"]["answer"] == answer
    assert trace["timings_s"]["fact_judgment"] >= 0
    assert [call.split("/")[-1] for call in calls[call_count:]] == ["embeddings", "completions", "completions"]
    system, user = state["judge_request"]["messages"]
    assert trace["fact_judgment"]["prompt"] == user["content"]
    assert trace["fact_judgment"]["system_prompt"] == system["content"]
    assert json.loads(user["content"]) == {
        "answer": answer, "required_facts": [original["required_fact"], required_condition],
    }
    assert trace["fact_judgment"]["response"] == (None if isinstance(judgment, Exception) else judgment)

    state["offline"] = True
    call_count = len(calls)
    assert __main__.main(["replay", run["run_id"], "q1", "--runs", str(tmp_path / "runs")]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["matches_original"] and replay["score"] == score
    assert replay["fact_judgment_error"] == error
    assert len(calls) == call_count


def test_replay_legacy_phrase_trace(evaluation, tmp_path, capsys):
    """保留旧版评分历史，即使语义模型会接受同义答文。"""
    _, index_path, question, calls, state = evaluation
    state["response"] = ['{"citations":[2],"error_code":null}\n', "Wait for the screen.", None]
    run, trace = _run(tmp_path, capsys, index_path, question)
    assert trace["score"]["passed"]

    run_dir = tmp_path / "runs" / run["run_id"]
    config_path = run_dir / "run.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["format_version"] = trace["format_version"] = 1
    config.pop("fact_judgment")
    trace.pop("fact_judgment")
    trace.pop("fact_judgment_error")
    trace["score"]["groups"][0]["fact_present"] = False
    trace["score"]["passed"] = False
    trace["score"]["attributions"] = ["missing_fact"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "traces" / "q1.json").write_text(json.dumps(trace), encoding="utf-8")

    state["offline"] = True
    call_count = len(calls)
    assert __main__.main(["replay", run["run_id"], "q1", "--runs", str(tmp_path / "runs")]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["matches_original"] and replay["score"] == trace["score"]
    assert len(calls) == call_count
