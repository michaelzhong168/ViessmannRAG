# ViessmannRAG

> 菲斯曼壁挂炉说明书问答 —— 每个答案都能回到原文那一页（页码 + `bbox` + 逐字原文）。

**当前状态**：M0 工程骨架，尚无业务代码。

## 仓库内容

| 文件 | 用途 |
|---|---|
| `README.md` | 本文件 |
| `AGENTS.md` | 开发约定：Reader → Coder → Reviewer，一关一分支一 PR |
| `.github/pull_request_template.md` | 每关 PR 模板（承载计划 / 交付 / 评审三段产物） |
| `.python-version` | Python 3.11 |
| `.gitignore` | 密钥、可重跑产物、本地文档副本均不进仓 |

## 文档

需求规格、十四关验收断言、项目主档、计划与审计记录都在在线文档：
<https://docs.qq.com/desktop/mydoc/folder/AdkPbToQjhNO>

本仓库不存放这些文档。本地 `docs/` 只是 agent 读取用的工作副本，已被 git 忽略。

## 运行

暂时没有可运行的代码。G01 落地后这里会补上安装与测试命令。