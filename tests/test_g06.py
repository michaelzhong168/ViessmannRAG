"""G06 离线验收，使用真实 SQLite 索引与模型调用替身。"""

import inspect
import json
import re
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest

import cli
from pipeline.build_index import build_index
from rag_core import ask, assemble, build_prompt, validate
from retrieval import load_eu, retrieve
from schema import EU, RequestCtx, eu_hash
from tools import llm

FIXTURES = Path(__file__).with_name("fixtures")
BBOX = [12.1234, 21.98765, 32.456789, 42.3456]


def make_eu(page, text, models=()):
    return EU(
        f"manual:p{page:03d}:01", "manual", page, BBOX.copy(), "运行参数", text,
        list(models), eu_hash("manual", page, BBOX, text), False,
    )


def write_eus(path, eus):
    path.write_text(
        "".join(json.dumps(asdict(eu), ensure_ascii=False) + "\n" for eu in eus),
        encoding="utf-8",
    )


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    eus = [
        make_eu(7, "冷态水压为 0.15 MPa。", ["B1KA"]),
        make_eu(8, "F4：检查显示屏上的故障信息。", ["B1KA"]),
        make_eu(9, "E5：联系专业人员处理燃气故障。", ["B1KA"]),
    ]
    codes = [
        dict(code=code, model_code="B1KA", quote=eu.text_raw, page_no=eu.page_no,
             eu_id=eu.eu_id, action_level=level)
        for code, eu, level in [("F4", eus[1], "SAFE"), ("E5", eus[2], "P0_DANGER")]
    ]
    write_eus(Path("data/eu.jsonl"), eus)
    Path("data/codes.json").write_text(json.dumps(codes), encoding="utf-8")
    build_index()
    return eus


@pytest.fixture
def generation(monkeypatch):
    calls = []

    def fake_generate(prompt):
        calls.append(prompt)
        return (FIXTURES / "mock_answer.txt").read_text(encoding="utf-8")

    monkeypatch.setattr(llm, "generate", fake_generate)
    return calls


def test_trigram_index_is_repeatable(tmp_path):
    source, index = tmp_path / "eu.jsonl", tmp_path / "fts5.db"
    eu = make_eu(2, "请记录点火失败提示。")
    write_eus(source, [eu, make_eu(1, "点火失败。")])
    assert load_eu(source)[0] == eu
    assert build_index(source, index) == 2
    first = retrieve("点火失败", db_path=index)
    assert len(first) == 2 and eu in first
    with sqlite3.connect(index) as db:
        sql, = db.execute("SELECT sql FROM sqlite_master WHERE name='eu_fts'").fetchone()
    assert "tokenize='trigram'" in sql
    assert build_index(source, index) == 2
    assert retrieve("点火失败", db_path=index) == first


def test_model_filter_precedes_limit(tmp_path):
    source, index = tmp_path / "eu.jsonl", tmp_path / "fts5.db"
    excluded = [make_eu(page, "点火失败", ["B1HA"]) for page in range(1, 7)]
    allowed = [make_eu(page, "运行期间显示点火失败，请记录提示。", ["b1ka"]) for page in range(7, 10)]
    write_eus(source, excluded + allowed)
    build_index(source, index)
    assert inspect.signature(retrieve).parameters["k"].default == 6
    assert [eu.eu_id for eu in retrieve("点火失败", db_path=index)] == [eu.eu_id for eu in excluded]
    found = retrieve("点火失败", model="b1ka", k=6, db_path=index)
    assert [eu.eu_id for eu in found] == [eu.eu_id for eu in allowed]
    assert all(eu.model_codes == ["B1KA"] for eu in found)
    assert len(retrieve("点火失败", model=None, k=20, db_path=index)) == 9

    general = make_eu(10, "运行期间出现点火失败。")
    write_eus(source, excluded + allowed + [general])
    build_index(source, index)
    assert retrieve("点火失败", model="OTHER", db_path=index) == [general]


def test_query_is_cleaned_and_uses_or(runtime):
    assert retrieve('水压 0.15 MPa 正常吗', 'B1KA') == runtime[:1]
    assert retrieve('未知词:"0.15" *(-MPa.)', 'B1KA') == runtime[:1]
    assert retrieve('"*():.-，？') == []


@pytest.mark.parametrize(("query", "model", "error"), [
    ("f4 怎么办", " b1ka ", None),
    ("E5 怎么办", "B1KA", "DANGER_REFUSAL"),
    ("F4 怎么办", "B1HA", "NO_EVIDENCE"),
    ("F4 怎么办", None, "MODEL_REQUIRED"),
    ("F4 怎么接线", None, "DANGER_REFUSAL"),
    ("F4 和 E5", "B1KA", "DANGER_REFUSAL"),
    ("显示 F4-E5 出现报警", "B1KA", "DANGER_REFUSAL"),
    ("显示 F4/E5 出现报警", "B1KA", "DANGER_REFUSAL"),
    ("显示故障码：F4-E5", "B1KA", "DANGER_REFUSAL"),
])
def test_l0_paths(runtime, generation, query, model, error):
    # 移除索引。任何错误进入 FTS 的路由都会暴露为 INTERNAL。
    Path("data/fts5.db").unlink()
    if query == "F4 怎么接线":
        Path("data/codes.json").unlink()
    result = ask(query, model, {"trace_id": "tr_fixture"})
    assert result.trace_id == "tr_fixture"
    assert result.error_code == error
    assert result.timing_ms["retrieve"] == result.timing_ms["generate"] == 0
    assert generation == []
    if error is None:
        citation, = result.citations
        assert citation.eu_id == runtime[1].eu_id
        assert citation.page_no == runtime[1].page_no
        assert citation.bbox == runtime[1].bbox
        assert citation.quote == runtime[1].text_raw
        assert citation.n == 1 and result.answer.endswith("[1]")
        assert citation.page_url == ""
    else:
        assert result.citations == []


@pytest.mark.parametrize(("field", "value"), [("page_no", 99), ("quote", "改写的原文")])
def test_l0_rejects_invalid_evidence(runtime, generation, field, value):
    path = Path("data/codes.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0][field] = value
    rows[0]["action_level"] = "CAUTION"
    path.write_text(json.dumps(rows), encoding="utf-8")
    Path("data/fts5.db").unlink()
    result = ask("F4 怎么办", "B1KA", {})
    assert result.error_code == "NO_EVIDENCE"
    assert result.citations == [] and generation == []


@pytest.mark.parametrize("short_code", ["06", "0C", "A0", "CC"])
def test_additional_g05_short_codes_take_l0(runtime, generation, short_code):
    eus = [make_eu(10, f"{short_code}：记录故障显示。", ["B1KA"])]
    write_eus(Path("data/eu.jsonl"), runtime + eus)
    path = Path("data/codes.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows.append(dict(code=short_code, model_code="B1KA", quote=eus[0].text_raw,
                     page_no=10, eu_id=eus[0].eu_id, action_level="CAUTION"))
    path.write_text(json.dumps(rows), encoding="utf-8")
    Path("data/fts5.db").unlink()
    result = ask(f"{short_code.lower()} 怎么办", "b1ka", {})
    assert result.error_code is None and generation == []
    assert [citation.eu_id for citation in result.citations] == [eus[0].eu_id]
    assert ask(short_code, "B1HA", {}).error_code == "NO_EVIDENCE"
    assert ask(short_code, None, {}).error_code == "MODEL_REQUIRED"


@pytest.mark.parametrize("first_code", ["F4", "06"])
def test_dangerous_letter_code_after_other_code(runtime, generation, first_code):
    path = Path("data/codes.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows.extend([dict(rows[0], code="06"), dict(rows[1], code="CC")])
    path.write_text(json.dumps(rows), encoding="utf-8")
    result = ask(f"{first_code} CC 怎么办", "B1KA", {})
    assert result.error_code == "DANGER_REFUSAL"
    assert result.citations == [] and generation == []


def test_decimal_and_unit_are_not_fault_codes(runtime, generation):
    path = Path("data/codes.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows.extend(dict(rows[1], code=code) for code in ("06", "CC"))
    path.write_text(json.dumps(rows), encoding="utf-8")
    result = ask("水压 0.06 MPa，20 cc 正常吗", "B1KA", {})
    assert result.error_code is None
    assert len(generation) == 1


def test_prompt_uses_verbatim_evidence(runtime):
    prompt = build_prompt("水压是多少", runtime[:1])
    eu = runtime[0]
    assert eu.text_raw in prompt
    assert eu.eu_id not in prompt and eu.chapter not in prompt and eu.content_hash not in prompt
    assert "[n]" in prompt and "数字和单位必须逐字保留" in prompt and "不知道" in prompt


def test_assemble_uses_only_current_evidence(runtime):
    eus = retrieve("显示屏 OR MPa")
    assert {eu.eu_id for eu in eus} == {runtime[0].eu_id, runtime[1].eu_id}
    body, dropped = assemble(
        '第一句[2]。越界句[2][9]。\n{"eu_id":"fake","quote":"伪造内容[1]。"}\n'
        '第二句。[1] 无角标句。再次引用[2]。',
        eus, lambda page: f"https://example.test/{page}",
    )
    assert re.findall(r"\[([0-9]+)\]", body["answer"]) == ["1", "2", "1"]
    assert "越界句[2][9]。" in dropped and "无角标句。" in dropped
    assert "fake" not in body["answer"] and "伪造内容" not in body["answer"]
    for number, (citation, eu) in enumerate(zip(body["citations"], eus[::-1], strict=True), 1):
        assert asdict(citation) == {
            "n": number, "eu_id": eu.eu_id, "page_no": eu.page_no, "bbox": BBOX,
            "quote": eu.text_raw, "page_url": f"https://example.test/{eu.page_no}",
        }
    resp = dict(body, trace_id="tr_test", error_code=None,
                timing_ms=dict.fromkeys(("router", "retrieve", "generate", "validate"), 0))
    assert validate(resp, eus) == (True, [])
    resp["citations"][0].page_no += 1
    assert validate(resp)[0]
    assert not validate(resp, eus)[0]


def test_assemble_ignores_only_out_of_range_sentence(runtime):
    body, dropped = assemble("水压为 0.15 MPa[1].虚构内容[9].“查阅说明书。”[1]", runtime[:1])
    assert body["answer"].splitlines() == ["水压为 0.15 MPa[1].", "“查阅说明书。”[1]"]
    assert [citation.eu_id for citation in body["citations"]] == [runtime[0].eu_id]
    assert dropped == ["虚构内容[9]."]


def test_invalid_candidate_refuses(runtime, monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda prompt: '{"answer":"伪造[1]。"}\n无出处。错编号[9]。')
    result = ask("水压 0.15 MPa", "B1KA", RequestCtx())
    assert result.error_code == "NO_EVIDENCE"
    assert result.citations == []


def behavior_cases():
    path = FIXTURES / "g06_behavior.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("case", behavior_cases(), ids=lambda case: case["id"])
def test_offline_behavior_seed(runtime, generation, case):
    result = ask(case["query"], case["model_code"], {})
    assert result.error_code == case["expected_error_code"]
    assert [citation.eu_id for citation in result.citations] == case["expected_eu_ids"]
    assert [citation.page_no for citation in result.citations] == case["expected_pages"]
    assert len(generation) == (case["expected_error_code"] is None)


@pytest.mark.parametrize("case", behavior_cases(), ids=lambda case: case["id"])
def test_cli_contract(runtime, generation, case, capsys):
    if case["expected_error_code"] == "DANGER_REFUSAL":
        Path("data/fts5.db").unlink()
        Path("data/eu.jsonl").unlink()
    assert cli.main(["ask", case["query"], "--model", case["model_code"]]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    response = json.loads(captured.out)
    assert response["error_code"] == case["expected_error_code"]
    assert re.fullmatch(r"tr_[0-9a-f]{32}", response["trace_id"])
    assert validate(response) == (True, [])
    if case["expected_error_code"] is None:
        assert len(generation) == 1
        assert response["citations"][0]["n"] == 1 and "[1]" in response["answer"]
    else:
        assert generation == [] and response["citations"] == []
        assert response["timing_ms"]["generate"] == response["timing_ms"]["validate"] == 0
    if case["expected_error_code"] == "DANGER_REFUSAL":
        assert response["timing_ms"]["retrieve"] == 0
