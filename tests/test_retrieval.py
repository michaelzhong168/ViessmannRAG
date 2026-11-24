"""验证已发布快照上的类型内检索。"""

import json

import httpx
import pymupdf
import pytest
from openai import OpenAI

from manual_rag.index import build_index, load_index
from manual_rag.retrieval import retrieve_nodes
from tools import llm


def test_retrieve_filters_before_top_k(tmp_path, monkeypatch):
    """其他类型全局第一时，仍返回指定类型内分数最高的原生节点。"""
    question = "  怎么打开暖气？  "
    vectors = {
        "Button setup": [0.0, 1.0],
        "Button heating": [3.0, 4.0],
        "Dial heating": [1.0, 0.0],
        question: [1.0, 0.0],
    }
    requests = []

    def respond(request):
        assert request.url.path == "/compatible-mode/v1/embeddings"
        inputs = json.loads(request.content)["input"]
        requests.append(inputs)
        return httpx.Response(200, json={
            "data": [
                {"object": "embedding", "index": ordinal, "embedding": vectors[value]}
                for ordinal, value in enumerate(inputs)
            ],
        })

    pdf_path = tmp_path / "manual.pdf"
    with pymupdf.open() as pdf:
        for title in ("Button setup", "Button heating", "Dial heating"):
            pdf.new_page().insert_text((60, 70), title)
        pdf.save(pdf_path)
    scopes_path = tmp_path / "scopes.json"
    scopes_path.write_text(json.dumps({
        "doc_id": "manual",
        "scopes": [
            {"first_page": 1, "last_page": 2, "control_type": "按键式"},
            {"first_page": 3, "last_page": 3, "control_type": "旋钮式"},
        ],
    }), encoding="utf-8")

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        def make_client(**kwargs):
            return OpenAI(**kwargs, http_client=http_client, max_retries=0)

        monkeypatch.setattr(llm, "OpenAI", make_client)
        monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test")
        monkeypatch.setenv("LLM_BASE_URL", llm.DEFAULT_BASE_URL)
        monkeypatch.setenv("EMBED_MODEL", "offline-text")
        index_path = tmp_path / "index.json"
        assert build_index(pdf_path, index_path, scopes_path=scopes_path) == 3
        index = load_index(index_path)

        results = retrieve_nodes(index, question, "按键式", top_k=1)
        assert len(results) == 1
        result = results[0]
        assert result.node.text == "Button heating"
        assert result.node.metadata["control_type"] == "按键式"
        assert result.node.metadata["page_no"] == 2
        assert result.node == index.docstore.get_node(result.node.node_id)
        assert result.score == pytest.approx(0.6)

        results = retrieve_nodes(index, question, "按键式")
        assert [result.node.text for result in results] == ["Button heating", "Button setup"]
        assert [result.score for result in results] == pytest.approx([0.6, 0.0])
        dial = retrieve_nodes(index, question, "旋钮式", top_k=1)
        assert [result.node.text for result in dial] == ["Dial heating"]
        assert dial[0].score > results[0].score
        assert retrieve_nodes(index, question, "触控式") == []

        with pytest.raises(ValueError, match="问句不能为空"):
            retrieve_nodes(index, "  ", "按键式")
        with pytest.raises(ValueError, match="操作类型"):
            retrieve_nodes(index, question, "未知")
        for top_k in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match="top_k"):
                retrieve_nodes(index, question, "按键式", top_k=top_k)

    assert requests == [["Button setup", "Button heating", "Dial heating"]] + [[question]] * 4
