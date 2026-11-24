# ViessmannRAG

菲斯曼壁挂炉说明书问答项目。开发按[需求规格总纲](specs/semantic-rag.md)逐单元推进。当前单元 R01-01 负责从本地 PDF 加载 LlamaIndex `Document`。

## 安装与离线验证

使用 `.python-version` 指定的 Python。在仓库根目录运行。

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
env -u DASHSCOPE_API_KEY PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest -q -rs
```

## R01-01 加载说明书

将 PDF 放在 `data/raw/`。对照原件核定型号和物理页段，在本地写入范围文件。物理页从 1 开始，含两端。下例对应 `manual.pdf`。

```json
{
  "doc_id": "manual",
  "scopes": [
    {"first_page": 1, "last_page": 48, "model_codes": ["L1P20-A1JF", "L1P24-A1JF"]}
  ]
}
```

```bash
.venv/bin/python -m manual_rag.source data/raw/manual.pdf --scopes data/raw/model-scopes.json
```

命令输出文件指纹、总页数、核定物理页、型号和无文本的核定页。加载器在内存中保留原文及每页字符范围，供下一单元创建可追溯的 `TextNode`。本单元全程在本地执行，无模型调用。

原始资料、配置密钥和评测记录留在本地。
