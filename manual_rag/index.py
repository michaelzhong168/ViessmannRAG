"""用核定的 PDF 生成、发布和重载 LlamaIndex 本地向量快照。"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage

from manual_rag.nodes import LAYOUT_VERSION, build_nodes
from manual_rag.source import load_document
from tools.embedding import BailianEmbedding

DEFAULT_INDEX = Path("data/semantic/index.json")
_FORMAT_VERSION = 1


def build_index(
    pdf_path: str | Path,
    index_path: str | Path = DEFAULT_INDEX,
    *,
    scopes_path: str | Path | None = None,
    control_type: str | None = None,
) -> int:
    """构建并原子发布索引，返回节点数。输入、建库或写入失败时保留已有索引。"""
    target = Path(index_path)
    if any(target.resolve() == Path(path).resolve() for path in (pdf_path, scopes_path) if path):
        raise ValueError("索引路径不能覆盖说明书或核定页段文件")
    document = load_document(pdf_path, scopes_path, control_type=control_type)
    nodes = build_nodes(document, pdf_path)
    if not nodes:
        raise ValueError("核定页段没有可索引的文本节点")

    embed = BailianEmbedding()
    index = VectorStoreIndex(nodes, embed_model=embed)
    snapshot = {
        "format_version": _FORMAT_VERSION,
        "embedding_model": embed.model_name,
        "layout_version": LAYOUT_VERSION,
        "source": {
            "doc_id": document.id_,
            "source": document.metadata["source"],
            "pdf_version": document.metadata["pdf_version"],
        },
        "node_count": len(nodes),
        "storage": index.storage_context.to_dict(),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as directory:
        staged = Path(directory) / "index.json"
        with staged.open("w", encoding="utf-8") as stream:
            json.dump(snapshot, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        staged.replace(target)
    return len(nodes)


def load_index(
    index_path: str | Path = DEFAULT_INDEX, *, timeout: float | None = None,
) -> VectorStoreIndex:
    """从已发布快照重载索引。格式或 EMBED_MODEL 不兼容时提示重新建库。"""
    with Path(index_path).open(encoding="utf-8") as stream:
        snapshot = json.load(stream)
    if snapshot.get("format_version") != _FORMAT_VERSION:
        raise ValueError("索引格式不匹配，请运行 index 重新建库")
    embed = BailianEmbedding(timeout=timeout)
    if snapshot.get("embedding_model") != embed.model_name:
        raise ValueError("索引的 Embedding 模型与 EMBED_MODEL 不匹配，请运行 index 重新建库")
    storage = StorageContext.from_dict(snapshot["storage"])
    return load_index_from_storage(storage, embed_model=embed)
