"""用合成说明书和百炼 HTTP 替身验证问答入口。"""

import json

import httpx
import pymupdf
import pytest
from openai import OpenAI

from manual_rag.__main__ import main
from manual_rag.index import build_index
from tools import llm


@pytest.fixture
def query_service(tmp_path, monkeypatch):
    requests = []
    responses = []

    def respond(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return httpx.Response(200, json={"data": [
                {"index": i, "embedding": [1, 0] if (
                    text in {"Dial instructions", "Step question"}
                    or text.endswith("1. Select heating mode.")
                ) else [0, 1]}
                for i, text in enumerate(inputs)
            ]})
        assert request.url.path.endswith("/chat/completions")
        requests.append(body)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(response)}}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        def make_client(**kwargs):
            return OpenAI(**kwargs, max_retries=0, http_client=http_client)

        monkeypatch.setattr(llm, "OpenAI", make_client)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test")
        monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
        monkeypatch.setenv("EMBED_MODEL", "offline-embedding")
        monkeypatch.setenv("LLM_MODEL", "offline-chat")
        monkeypatch.setenv("LLM_BASE_URL", llm.DEFAULT_BASE_URL)
        pdf_path = tmp_path / "manual.pdf"
        scope_path = tmp_path / "scope.json"
        index_path = tmp_path / "index.json"
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((50, 70), "Button instructions")
            pdf.new_page().insert_text((50, 70), "Dial instructions")
            pdf.save(pdf_path)
        scope_path.write_text(json.dumps({
            "doc_id": "manual",
            "scopes": [
                {"first_page": 1, "last_page": 1, "control_type": "按键式"},
                {"first_page": 2, "last_page": 2, "control_type": "旋钮式"},
            ],
        }), encoding="utf-8")
        assert build_index(pdf_path, index_path, scopes_path=scope_path) == 2
        yield index_path, requests, responses


@pytest.mark.parametrize(("question", "device_type", "code", "citations"), [
    ("  闻到燃气味，还能打开暖气吗？  ", None, "safety_notice", []),
    ("怎么打开暖气", None, "missing_control_type", []),
    ("怎么打开暖气", "触控式", "control_type_mismatch", []),
    ("屏幕显示 F02", "按键式", None, [1]),
    ("我家锅炉好像出故障了", "按键式", "clarify_fault", []),
    ("设备支持哪个 Wi-Fi 协议版本？", "按键式", "insufficient_evidence", []),
])
def test_model_decides_all_question_outcomes(query_service, capsys, question, device_type, code, citations):
    """危险、类型不符、缺类型和故障码都经同一模型调用。"""
    path, requests, responses = query_service
    responses.append({"answer": "模型给出的内容", "citations": citations, "error_code": code, "page_no": 999})
    argv = ["ask", question, "--index", str(path)]
    if device_type is not None:
        argv += ["--control-type", device_type]

    assert main(argv) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["answer"] == "模型给出的内容"
    assert output["error_code"] == code
    request, = requests
    assert "有危险迹象" in request["messages"][0]["content"]
    payload = json.loads(request["messages"][1]["content"])
    assert payload["question"] == question
    assert payload["control_type"] == device_type
    assert {item["control_type"] for item in payload["excerpts"]} == {"按键式", "旋钮式"}
    if citations:
        citation, = output["citations"]
        assert citation["doc_id"] == "manual"
        assert citation["page_no"] == 1
        assert citation["quote"] == "Button instructions"
        assert citation["bbox"] and citation["node_id"]
    else:
        assert output["citations"] == []


@pytest.mark.parametrize(("response", "expected_exit", "code"), [
    ({"answer": "无证据的步骤", "citations": [], "error_code": None}, 0, "insufficient_evidence"),
    ({"answer": "无证据的步骤", "citations": [999], "error_code": None}, 1, "invalid_model_response"),
    ({"answer": "无证据的步骤", "citations": [1]}, 1, "invalid_model_response"),
    (httpx.ReadTimeout("provider timed out"), 1, "model_timeout"),
])
def test_model_response_keeps_only_retrieved_citations(query_service, capsys, response, expected_exit, code):
    """无引用或虚构编号的步骤不会返回，超时单独报告。"""
    path, _, responses = query_service
    responses.append(response)
    assert main(["ask", "怎么打开暖气", "--index", str(path)]) == expected_exit
    output = json.loads(capsys.readouterr().out)
    assert output["answer"] != "无证据的步骤"
    assert output["citations"] == []
    assert output["error_code"] == code


def test_model_receives_remaining_steps_from_same_page_and_section(query_service, tmp_path, capsys):
    """命中的步骤补齐同页同节，其他页和章节的步骤保持独立。"""
    _, requests, responses = query_service
    pdf_path = tmp_path / "steps.pdf"
    index_path = tmp_path / "steps.json"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((55, 50), "Heating", fontsize=20)
        page.insert_text((55, 80), "1. Select heating mode.")
        page.insert_text((55, 110), "2. Confirm mode.")
        page.insert_text((55, 160), "Other settings", fontsize=20)
        page.insert_text((55, 190), "1. Adjust unrelated settings.")
        page = pdf.new_page()
        page.insert_text((55, 50), "Heating", fontsize=20)
        page.insert_text((55, 80), "1. Other page instructions.")
        pdf.save(pdf_path)
    build_index(pdf_path, index_path, control_type="按键式")
    responses.append({"answer": "模型给出的内容", "citations": [2], "error_code": None})

    assert main(["ask", "Step question", "--index", str(index_path), "--top-k", "1"]) == 0
    request, = requests
    payload = json.loads(request["messages"][1]["content"])
    assert [item["text"] for item in payload["excerpts"]] == [
        "1. Select heating mode.", "2. Confirm mode.",
    ]
    output = json.loads(capsys.readouterr().out)
    assert output["citations"][0]["quote"] == "2. Confirm mode."
    assert output["citations"][0]["page_no"] == 1
