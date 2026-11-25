# ViessmannRAG

菲斯曼壁挂炉说明书问答项目。开发按[需求规格总纲](specs/semantic-rag.md)逐单元推进。R01-03 将核定说明书的版面节点写入本地 LlamaIndex 向量索引。

## 安装与离线验证

使用 `.python-version` 指定的 Python。在仓库根目录运行。

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
env -u DASHSCOPE_API_KEY PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest -q -rs
```

## R01-01 加载说明书

将 PDF 放在 `data/raw/`。整本只有一种操作类型时直接指定类型，加载器从 PDF 获取实际页数。

```bash
.venv/bin/python -m manual_rag.source data/raw/manual.pdf --control-type 按键式
```

操作类型为 `旋钮式`、`按键式` 或 `触控式`。合订本或只纳入部分页时，改用 `--scopes data/raw/control-scopes.json`，每段指定 `control_type`。物理页从 1 开始，含两端。

```json
{
  "doc_id": "manual",
  "scopes": [
    {"first_page": 1, "last_page": 12, "control_type": "按键式"}
  ]
}
```

```bash
.venv/bin/python -m manual_rag.source data/raw/manual.pdf --scopes data/raw/control-scopes.json
```

命令输出文件指纹、总页数、核定物理页、操作类型和无文本的核定页。加载器在内存中保留原文及每页字符范围，供下一单元创建可追溯的 `TextNode`。本单元全程在本地执行，无模型调用。

原始资料、配置密钥和评测记录留在本地。

## R01-02 生成版面节点

```python
from manual_rag.nodes import build_nodes
from manual_rag.source import load_document

path = "data/raw/manual.pdf"
document = load_document(path, control_type="按键式")
nodes = build_nodes(document, path)
```

合订本沿用 `load_document(path, scopes_path=...)` 的页段配置。生成前会核对 PDF 指纹和来源文件名。每个节点的 `SOURCE` 关系含来源 Document ID、文件名和版本。节点元数据记录操作类型、从 1 开始的物理页码、版面单元类型、同页章节和 `bbox`。坐标使用 PyMuPDF 未旋转的页坐标，原点在左上，单位为 pt。对同一来源版本和版面规则重新生成时，节点 ID 可复现。

规则使用 PDF 文本层识别标题、段落、编号步骤和线框表格。节点保留页内原文，表格行用制表符分隔单元格。未识别为线框表格的文字按正文提取。仅含图像的页面需要另行 OCR。

## R01-03 向量建库

在项目根目录的 `.env` 中配置 `DASHSCOPE_API_KEY`、`LLM_BASE_URL` 和 `EMBED_MODEL`。建库命令将核定节点的章节和原文发给百炼 Embedding。用法如下。

```bash
.venv/bin/python -m manual_rag index data/raw/manual.pdf --control-type 按键式
```

只纳入核定页段时，使用 `--scopes data/raw/control-scopes.json`。命令输出节点数量，默认发布到 `data/semantic/index.json`，也可用 `--output` 指定本地索引文件。将自定义索引放在受 Git 忽略规则保护的 `data/` 下。

本地快照包含原生 LlamaIndex 存储、PDF 内容指纹、版面分块版本、Embedding 模型和索引格式版本。构建或写入失败时，已发布索引仍然可用。

```python
from manual_rag.index import load_index

index = load_index()
print(len(index.index_struct.nodes_dict))
```

重载读取本地快照，无需联网或提供 API 密钥。若配置的 `EMBED_MODEL` 与快照不同，使用目标模型重新运行建库命令。

## R01-04 类型内检索

使用 R01-03 保存的快照，并沿用建库时的 `EMBED_MODEL`。将原始问句和操作类型传给检索入口。问句通过百炼生成向量，LlamaIndex 在指定操作类型内选取最相似的节点。

```python
from manual_rag.index import load_index
from manual_rag.retrieval import retrieve_nodes

index = load_index()
for result in retrieve_nodes(index, "查看水压", "按键式", top_k=5):
    print(result.score, result.node.metadata["page_no"], result.node.text)
```

默认返回至多 5 个 `NodeWithScore`，保留原始节点和相似度分数。指定类型没有匹配节点时返回空列表。

## R01-06 说明书问答

在 `.env` 配置 `LLM_MODEL`，并沿用建库时配置的 `EMBED_MODEL`。
客户端不设百炼请求超时。等待期间可按 Ctrl+C 取消。

```bash
.venv/bin/python -m manual_rag ask "菲斯曼壁挂炉怎么打开暖气" --control-type 按键式
.venv/bin/python -m manual_rag ask "屏幕显示 F02"
```

在终端运行时，正文逐段显示，结束后列出引用。重定向或管道默认输出完整 JSON。可用 `--stream` 或 `--json` 显式选择输出方式。

操作类型可选。问答检索将语义候选、原问句和用户提供的类型一起交给模型。召回操作步骤时，补充同一说明书同页同节的其他步骤。模型回答危险、类型适配和报码问题，程序从模型选用的片段回填来源、物理页、坐标和逐字引用。可用 `--index` 选择已发布的索引、`--top-k` 调整初次检索的候选数。
