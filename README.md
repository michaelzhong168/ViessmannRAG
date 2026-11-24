# ViessmannRAG

菲斯曼壁挂炉说明书问答项目。当前阶段清理旧问答实现，为后续开发保留 PDF 版面提取、百炼客户端和风险词规则。

## 安装与离线验证

使用 `.python-version` 指定的 Python。从仓库根目录运行。

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
env -u DASHSCOPE_API_KEY PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest -q -rs
```

## 通用组件

- `pipeline.layout.analyze_layout` 接收已打开的 PyMuPDF 文件和可选物理页段，提取带坐标的正文、章节和表格。
- `tools.llm.generate` 使用百炼 OpenAI 兼容接口。联通性检查可复制 `.env.example` 为 `.env`，填入有效凭证和模型后执行 `.venv/bin/python -m tools.llm "你好"`。
- `config.DANGER_WORDS` 提供壁挂炉危险操作词规则。

原十四关的交付版本可在 Git 历史查看。真实资料、密钥和评测记录保存在本地。
