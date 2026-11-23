# ViessmannRAG

菲斯曼壁挂炉说明书问答项目。目标是让答案能回到原文那一页，并附页码、`bbox` 和逐字原文。

G01 提供唯一模型调用入口 `tools.llm.generate()`。包含 G02 的版本还可以将说明书 PDF 转为逐页原文、逐页 PNG 和语料审计表。

需求、验收规格和进度见[腾讯文档中的「项目主档」与规格](https://docs.qq.com/desktop/mydoc/folder/AdkPbToQjhNO)。仓库开发流程见 [AGENTS.md](AGENTS.md)。

## 安装与无密钥测试

使用 `.python-version` 指定的 Python 3.11。在仓库根目录运行：

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q -rs
```

## PDF 夹具演示（G02）

检出包含 `pipeline/parse.py` 和 `tests/fixtures/mini.pdf` 的版本后，按上节安装依赖，在仓库根目录运行：

```bash
demo_dir="$(mktemp -d)"
.venv/bin/python -m pipeline.parse tests/fixtures/mini.pdf \
  --out-dir "$demo_dir/pages" --audit "$demo_dir/语料审计.md"
```

命令输出 `doc_id=mini` 和 `page_count=2`，并在临时目录生成两张 PNG 和一份审计表。

## 可选的模型冒烟

首次配置时运行 `cp .env.example .env`。在 `.env` 中填写有效的 `DASHSCOPE_API_KEY` 和百炼控制台可用的 `LLM_MODEL`，然后运行：

```bash
.venv/bin/python -m tools.llm "你好"
```

模型入口使用 OpenAI 兼容接口。切换到项目允许的服务商时，配置对应的 `LLM_BASE_URL` 和 `LLM_MODEL`，并将该服务商的有效凭证填入 `DASHSCOPE_API_KEY`。
