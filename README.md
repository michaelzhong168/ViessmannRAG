# ViessmannRAG

> 菲斯曼壁挂炉说明书问答 —— 每个答案都能回到原文那一页（页码 + `bbox` + 逐字原文）。

**当前状态**：G01 完成 —— Python 3.11 环境 + 唯一模型入口 `tools/llm.py`，尚无业务代码。

## 仓库内容

| 文件 | 用途 |
|---|---|
| `README.md` | 本文件 |
| `AGENTS.md` | 开发约定：Reader → Coder → Reviewer，一关一分支一 PR |
| `.github/PULL_REQUEST_TEMPLATE.md` | PR 模板：描述 / 关联 / 改动 / 测试 / 检查清单 |
| `.python-version` | Python 3.11 |
| `.gitignore` | 密钥、可重跑产物、本地文档副本均不进仓 |

## 文档

需求规格、十四关验收断言、项目主档、计划与审计记录都在在线文档：
<https://docs.qq.com/desktop/mydoc/folder/AdkPbToQjhNO>

本仓库不存放这些文档。本地 `docs/` 只是 agent 读取用的工作副本，已被 git 忽略。

## 运行

```bash
# 1. 环境（Python 3.11，见 .python-version）
uv venv --python 3.11 .venv
# uv 建的 venv 默认不带 pip，下面这条不要写成 `.venv/bin/python -m pip install`
uv pip install --python .venv/bin/python -r requirements.txt

# 2. 配置（.env 不进仓）
cp .env.example .env    # 然后填入真实 DASHSCOPE_API_KEY

# 3. 测试（全离线 mock，不需要真实 KEY）
.venv/bin/python -m pytest -q

# 4. 手工冒烟（需要真实 KEY；模型名以百炼控制台为准）
.venv/bin/python -m tools.llm "你好"
```

模型调用只有 `tools/llm.py:generate()` 一个入口；换厂商只改 `.env` 里的
`LLM_BASE_URL` + `LLM_MODEL`，业务代码不动。