"""在核定操作类型内检索原生说明书节点。"""

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from manual_rag.source import CONTROL_TYPES


def retrieve_nodes(
    index: VectorStoreIndex,
    question: str,
    control_type: str,
    *,
    top_k: int = 5,
) -> list[NodeWithScore]:
    """用原始问句检索指定操作类型，按 LlamaIndex 相似度降序返回节点和分数。

    操作类型没有匹配节点时返回空列表。空白问句、未知操作类型或非正整数
    top_k 抛出 ValueError。
    """
    if not question.strip():
        raise ValueError("问句不能为空")
    if control_type not in CONTROL_TYPES:
        raise ValueError("操作类型须为旋钮式、按键式或触控式")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 须为正整数")

    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=MetadataFilters(filters=[
            MetadataFilter(key="control_type", value=control_type, operator=FilterOperator.EQ),
        ]),
    )
    return retriever.retrieve(question)
