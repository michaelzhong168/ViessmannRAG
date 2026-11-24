"""百炼 OpenAI 兼容文本向量接口。"""

import asyncio
import os
from math import isfinite

from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI
from pydantic import PrivateAttr

from tools import llm

_BATCH_SIZE = 10  # text-embedding-v4 每次请求最多接受 10 条文本。


class BailianEmbedding(BaseEmbedding):
    """读取 EMBED_MODEL，首次生成向量时再连接百炼。"""

    _client: OpenAI | None = PrivateAttr(default=None)
    _dimension: int | None = PrivateAttr(default=None)

    def __init__(self) -> None:
        llm._load_env()
        model = os.environ.get("EMBED_MODEL", "").strip()
        if not model:
            raise ValueError("缺少 EMBED_MODEL，请检查 .env 中的 EMBED_MODEL")
        super().__init__(model_name=model, embed_batch_size=_BATCH_SIZE)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            self._client = llm._get_client()
        response = self._client.embeddings.create(
            model=self.model_name, input=texts, encoding_format="float",
            timeout=llm._resolve_timeout(),
        )
        rows = sorted(response.data, key=lambda row: row.index)
        if [row.index for row in rows] != list(range(len(texts))):
            raise ValueError("百炼返回的向量数量或序号有误")

        vectors = [row.embedding for row in rows]
        dimension = self._dimension or len(vectors[0])
        for vector in vectors:
            if len(vector) != dimension or not all(map(isfinite, vector)) or not any(vector):
                raise ValueError("百炼返回的向量维度或内容有误")
        self._dimension = dimension
        return vectors

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)
