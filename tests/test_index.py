"""使用合成 PDF 和百炼 HTTP 替身验证向量快照。"""

import json
import os
from hashlib import sha256

import httpx
import pymupdf
import pytest
from openai import OpenAI

from manual_rag.__main__ import main
from manual_rag.index import load_index
from manual_rag.nodes import LAYOUT_VERSION, build_nodes
from manual_rag.source import load_document
from tools import llm


@pytest.fixture
def index_inputs(tmp_path):
    pdf_path = tmp_path / "manual.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((60, 70), "Heating", fontsize=18)
        page.insert_text((60, 115), [f"{i}. Check the display." for i in range(1, 12)])
        pdf.new_page().insert_text((60, 70), "Unapproved")
        pdf.save(pdf_path)
    scopes = tmp_path / "scopes.json"
    scopes.write_text(json.dumps({
        "doc_id": "manual",
        "scopes": [{"first_page": 1, "last_page": 1, "control_type": "按键式"}],
    }), encoding="utf-8")
    output = tmp_path / "output" / "index.json"
    return pdf_path, scopes, output


@pytest.fixture
def embedding_service(monkeypatch):
    """替换 HTTP 传输，保留 OpenAI 客户端与全部建库路径。"""
    requests = []

    def respond(request):
        assert request.url.path == "/compatible-mode/v1/embeddings"
        body = json.loads(request.content)
        requests.append(body)
        if body["model"] == "broken" and len(body["input"]) < 10:
            return httpx.Response(500, json={"error": {"message": "private upstream detail"}})
        return httpx.Response(200, json={
            "object": "list",
            "model": body["model"],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
            "data": [
                {"object": "embedding", "index": i, "embedding": [float(i + 1), 0.5]}
                for i in reversed(range(len(body["input"])))
            ],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        def make_client(**kwargs):
            return OpenAI(**kwargs, http_client=http_client, max_retries=0)

        monkeypatch.setattr(llm, "OpenAI", make_client)
        monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test")
        monkeypatch.setenv("LLM_BASE_URL", llm.DEFAULT_BASE_URL)
        monkeypatch.setenv("EMBED_MODEL", "offline-text")
        yield requests


def _build(inputs):
    pdf, scopes, output = inputs
    return main(["index", str(pdf), "--scopes", str(scopes), "--output", str(output)])


def test_cli_build_and_reload(index_inputs, embedding_service, monkeypatch, capsys):
    """核对核定节点的输入、向量顺序、快照元数据与无密钥重载。"""
    pdf, scopes, output = index_inputs
    assert _build(index_inputs) == 0
    assert json.loads(capsys.readouterr().out) == {"index": str(output), "node_count": 12}
    requests = embedding_service
    assert [len(request["input"]) for request in requests] == [10, 2]
    assert {request["model"] for request in requests} == {"offline-text"}
    assert requests[0]["input"][1] == "section: Heating\n\n1. Check the display."

    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["format_version"] == 1
    assert snapshot["layout_version"] == LAYOUT_VERSION
    assert snapshot["embedding_model"] == "offline-text"
    assert snapshot["source"] == {
        "doc_id": "manual", "source": "manual.pdf", "pdf_version": sha256(pdf.read_bytes()).hexdigest(),
    }
    assert snapshot["node_count"] == 12

    monkeypatch.delenv("DASHSCOPE_API_KEY")
    index = load_index(output)
    expected_nodes = build_nodes(load_document(pdf, scopes), pdf)
    assert set(index.index_struct.nodes_dict.values()) == {node.node_id for node in expected_nodes}
    for ordinal, expected in enumerate(expected_nodes):
        assert index.docstore.get_node(expected.node_id) == expected
        assert index.vector_store.get(expected.node_id) == [ordinal % 10 + 1, 0.5]
    assert len(requests) == 2


def test_service_failure_preserves_snapshot(index_inputs, embedding_service, monkeypatch, capsys):
    """第二批百炼请求失败后，旧快照仍能无损重载。"""
    assert _build(index_inputs) == 0
    output = index_inputs[2]
    previous = output.read_bytes()
    monkeypatch.setenv("EMBED_MODEL", "broken")
    assert _build(index_inputs) == 1
    assert "500" in capsys.readouterr().err
    assert [len(request["input"]) for request in embedding_service] == [10, 2, 10, 2]
    assert output.read_bytes() == previous
    monkeypatch.setenv("EMBED_MODEL", "offline-text")
    assert len(load_index(output).index_struct.nodes_dict) == 12


def test_publish_failure_preserves_snapshot(index_inputs, embedding_service, monkeypatch, capsys):
    """文件系统拒绝最后的原子替换时，旧索引与输出目录保持完整。"""
    assert _build(index_inputs) == 0
    output = index_inputs[2]
    previous = output.read_bytes()

    def refuse_replace(source, target):
        assert target == output
        assert json.loads(source.read_text(encoding="utf-8"))["node_count"] == 12
        raise PermissionError("filesystem refused replacement")

    monkeypatch.setattr(os, "replace", refuse_replace)
    assert _build(index_inputs) == 1
    assert "filesystem refused replacement" in capsys.readouterr().err
    assert output.read_bytes() == previous
    assert list(output.parent.iterdir()) == [output]
    assert len(load_index(output).index_struct.nodes_dict) == 12


def test_incompatible_snapshot_requires_rebuild(index_inputs, embedding_service, monkeypatch):
    """格式或模型不兼容时，在访问百炼之前给出重建提示。"""
    assert _build(index_inputs) == 0
    output = index_inputs[2]
    monkeypatch.setenv("EMBED_MODEL", "different-model")
    with pytest.raises(ValueError, match="Embedding 模型.*重新建库"):
        load_index(output)
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    snapshot["format_version"] += 1
    output.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(ValueError, match="索引格式.*重新建库"):
        load_index(output)
    assert len(embedding_service) == 2
