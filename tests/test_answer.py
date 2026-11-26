"""用合成说明书和百炼 HTTP 替身验证问答入口。"""

import json
import sys
from http.client import HTTPConnection
from threading import Event, Thread

import httpx
import pymupdf
import pytest
from openai import OpenAI

from manual_rag.__main__ import main
from manual_rag.api import AskServer
from manual_rag.index import build_index
from tools import llm


class _Events(httpx.SyncByteStream):
    """用可逐块推进的 SSE 替换百炼网络响应。"""

    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            if callable(chunk):
                chunk()
                continue
            choice = {
                "index": 0,
                "delta": {"content": chunk},
                "finish_reason": "stop" if chunk is None else None,
            }
            yield f"data: {json.dumps({'choices': [choice]})}\n\n".encode()


@pytest.fixture
def query_service(tmp_path, monkeypatch, request):
    requests = []
    responses = []
    network_timeouts = []

    def respond(request):
        network_timeouts.append(request.extensions["timeout"])
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
        assert body["stream"] is True
        requests.append(body)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            metadata = {key: value for key, value in response.items() if key != "answer"}
            response = [json.dumps(metadata) + "\n", response["answer"], None]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_Events(response))

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        def make_client(**kwargs):
            kwargs.setdefault("max_retries", 0)
            return OpenAI(**kwargs, http_client=http_client)

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
        for timeout in network_timeouts:
            assert timeout == dict.fromkeys(("connect", "read", "write", "pool"), timeout["read"])
        expected = getattr(request.node, "api_timeout", None)
        assert all(timeout["read"] == expected for timeout in network_timeouts[1:])


@pytest.mark.parametrize(("question", "device_type", "code", "citations"), [
    ("  闻到燃气味，还能打开暖气吗？  ", None, "safety_notice", [1, 999]),
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
    if citations and code != "safety_notice":
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


@pytest.mark.parametrize("explicit", [True, False])
def test_stream_displays_body_before_next_event_and_resolves_citations(
    query_service, capsys, monkeypatch, explicit,
):
    """校验跨块的头部，首段正文在下一事件到达前已输出。"""
    path, requests, responses = query_service

    def check_visible():
        assert capsys.readouterr().out == "先选择"

    responses.append(['{"citations":[1],', '"error_code":null}\n先选择', check_visible, "模式。", None])
    argv = ["ask", "怎么打开暖气", "--index", str(path)]
    if explicit:
        argv.append("--stream")
    else:
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(argv) == 0
    assert requests[0]["stream"] is True
    body, label, citation = capsys.readouterr().out.splitlines()
    assert body == "模式。"
    assert label == "引用"
    assert json.loads(citation)["page_no"] == 1
    assert json.loads(citation)["quote"] == "Button instructions"


def test_json_option_keeps_machine_output_on_terminal(query_service, capsys, monkeypatch):
    """显式 JSON 等待同一个模型流结束，再输出完整机器可读结果。"""
    path, requests, responses = query_service

    def check_buffered():
        assert capsys.readouterr().out == ""

    responses.append(['{"citations":[1],"error_code":null}\n回', check_buffered, "答", None])
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["ask", "怎么打开暖气", "--index", str(path), "--json"]) == 0
    assert requests[0]["stream"] is True
    assert json.loads(capsys.readouterr().out)["answer"] == "回答"


@pytest.mark.parametrize(("indices", "expected_exit"), [([], 0), ([999], 1)])
def test_stream_rejects_missing_or_fabricated_citations(query_service, capsys, indices, expected_exit):
    """无有效引用时，拒答发生在显示正文之前。"""
    path, _, responses = query_service
    header = json.dumps({"citations": indices, "error_code": None})
    responses.append([header + "\n未核实步骤。", None])
    assert main(["ask", "怎么打开暖气", "--index", str(path), "--stream"]) == expected_exit
    out, err = capsys.readouterr()
    assert "未核实步骤" not in out + err
    assert "引用\n" not in out


@pytest.mark.parametrize("tail", [[], [httpx.ReadError("stream disconnected")]])
def test_stream_interruption_exits_nonzero_without_citations(query_service, capsys, tail):
    """连接中断或缺少结束标记时，提示失败且不输出引用。"""
    path, _, responses = query_service
    responses.append(['{"citations":[1],"error_code":null}\n部分正文', *tail])
    assert main(["ask", "怎么打开暖气", "--index", str(path), "--stream"]) == 1
    out, err = capsys.readouterr()
    assert out == "部分正文\n"
    assert "模型服务请求失败" in err


def _sse_events(response):
    """逐帧读取真实 HTTP 响应，保留正文中的 Unicode 和换行。"""
    while line := response.readline():
        if line.startswith(b":"):
            assert response.readline() == b"\n"
            continue
        event = line.removeprefix(b"event: ").strip().decode()
        data = json.loads(response.readline().removeprefix(b"data: "))
        assert response.readline() == b"\n"
        yield event, data


@pytest.fixture
def ask_http(query_service, request):
    """通过真实 HTTP 契约测试，外部模型沿用合成说明书的替身。"""
    path, requests, responses = query_service
    origin = "http://localhost:5173"
    request.node.api_timeout = 2.0
    with AskServer(("127.0.0.1", 0), path, origins=(origin,), timeout=2) as server:
        worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        worker.start()

        def send(method, body=None, *, path="/api/ask", request_origin=origin, read_events=list):
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                headers = {"Content-Type": "application/json"}
                if request_origin is not None:
                    headers["Origin"] = request_origin
                connection.request(method, path, body, headers)
                response = connection.getresponse()
                if response.headers.get_content_type() == "text/event-stream":
                    data = read_events(_sse_events(response))
                else:
                    raw = response.read()
                    data = json.loads(raw) if raw else None
                return response.status, dict(response.getheaders()), data
            finally:
                connection.close()

        try:
            yield send, requests, responses
        finally:
            server.shutdown()
            worker.join()


def test_h5_preflight_and_control_type_followup_return_original_citations(ask_http):
    send, requests, responses = ask_http
    responses.extend([
        {"answer": "请先选操作面板。", "citations": [], "error_code": "missing_control_type"},
        {"answer": "请查看手册。", "citations": [1], "error_code": None},
    ])
    status, headers, body = send("OPTIONS")
    assert (status, body) == (204, None)
    assert headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
    assert headers["Access-Control-Allow-Methods"] == "POST"
    assert headers["Access-Control-Allow-Headers"] == "Content-Type"

    question = {"question": "怎么打开暖气"}
    status, headers, events = send("POST", json.dumps(question))
    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
    assert headers["Content-Type"] == "text/event-stream; charset=utf-8"
    assert headers["X-Accel-Buffering"] == "no"
    assert events == [
        ("delta", {"text": "请先选操作面板。"}),
        ("result", {"answer": "请先选操作面板。", "citations": [], "error_code": "missing_control_type"}),
    ]

    status, _, events = send("POST", json.dumps({**question, "control_type": "按键式"}))
    assert status == 200
    (delta, fragment), (terminal, result) = events
    assert (delta, fragment, terminal) == ("delta", {"text": "请查看手册。"}, "result")
    assert result["answer"] == "请查看手册。"
    assert result["error_code"] is None
    citation, = result["citations"]
    assert set(citation) == {"node_id", "doc_id", "page_no", "bbox", "quote"}
    assert citation["doc_id"] == "manual"
    assert citation["page_no"] == 1
    assert citation["quote"] == "Button instructions"
    assert len(citation["bbox"]) == 4
    prompts = [json.loads(request["messages"][-1]["content"]) for request in requests]
    assert [prompt["question"] for prompt in prompts] == [question["question"]] * 2
    assert [prompt["control_type"] for prompt in prompts] == [None, "按键式"]


def test_h5_timeout_is_retryable_and_service_failure_is_redacted(ask_http):
    send, requests, responses = ask_http
    responses.extend([
        [httpx.ReadTimeout("upstream details")],
        {"answer": "查阅说明书。", "citations": [1], "error_code": None},
        ['{"citations":[1],"error_code":null}\n', "部分正文", httpx.ReadError("private upstream detail")],
        ['{"citations":[999],"error_code":null}\n', "无效正文", None],
    ])
    body = json.dumps({"question": "怎么看水压", "control_type": "按键式"})
    status, headers, events = send("POST", body)
    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
    assert events == [("error", {
        "answer": "模型服务请求超时。", "citations": [], "error_code": "model_timeout",
    })]
    status, _, events = send("POST", body, request_origin=None)
    assert status == 200 and events[-1][0] == "result" and events[-1][1]["citations"]
    status, _, events = send("POST", body)
    assert status == 200
    assert events == [
        ("delta", {"text": "部分正文"}),
        ("error", {"answer": "模型服务请求失败。", "citations": [], "error_code": "model_service_error"}),
    ]
    status, _, events = send("POST", body)
    assert status == 200
    assert events == [("error", {
        "answer": "模型返回的回答或引用格式无效。", "citations": [], "error_code": "invalid_model_response",
    })]
    assert len(requests) == 4


def test_h5_flushes_delta_before_upstream_finishes(ask_http):
    send, requests, responses = ask_http
    release = Event()

    def wait_for_client():
        assert release.wait(timeout=3), "客户端未在上游完成前收到 SSE 片段"

    responses.append([
        '{"citations":[1],"error_code":null}\n', "第一行\n", wait_for_client, "第二行", None,
    ])
    body = json.dumps({"question": "怎么看水压", "control_type": "按键式"})

    def check_live(events):
        try:
            first = next(events)
            assert first == ("delta", {"text": "第一行\n"})
            status, _, busy = send("POST", body)
            assert status == 503 and busy["error_code"] == "service_busy"
        finally:
            release.set()
        return list(events)

    status, _, rest = send("POST", body, read_events=check_live)
    assert status == 200
    (delta, fragment), (terminal, result) = rest
    assert (delta, fragment, terminal) == ("delta", {"text": "第二行"}, "result")
    assert result["answer"] == "第一行\n第二行" and result["citations"]
    assert len(requests) == 1


@pytest.mark.parametrize("body", [
    "null", "{", " " * (16 * 1024 + 1), '{"question": " "}', '{"question": true}',
    '{"question": "test", "control_type": []}', '{"question": "\\ud800"}',
])
def test_h5_rejects_bad_payloads_without_model_calls(ask_http, body):
    send, requests, _ = ask_http
    status, _, result = send("POST", body)
    assert status == 400 and result["error_code"] == "invalid_request"
    assert requests == []


def test_h5_disallows_unapproved_origins_and_has_no_file_routes(ask_http):
    send, requests, _ = ask_http
    for method in ("OPTIONS", "POST"):
        status, headers, result = send(method, '{"question": "test"}', request_origin="http://other:5173")
        assert status == 403 and result["error_code"] == "origin_not_allowed"
        assert "Access-Control-Allow-Origin" not in headers
    status, _, result = send("GET", path="/data/semantic/index.json")
    assert status == 404 and result["error_code"] == "not_found"
    status, headers, result = send("GET")
    assert status == 405 and result["error_code"] == "method_not_allowed"
    assert headers["Allow"] == "POST, OPTIONS"
    assert requests == []
