# ViessmannRAG

> 菲斯曼壁挂炉说明书 · 可溯源检索问答。**每个答案都能回到原文那一页** —— 页码 + `bbox` + 逐字原文。

## 状态

| 项 | 值 |
|---|---|
| 阶段 | M0 规格冻结，**代码未开工**（0 / 14 关） |
| 交付形态 | 微信小程序 4 页 + 云上 FastAPI（微信云托管） |
| Python | 3.11（`.python-version` 已固定，SQLite 需支持 FTS5 `trigram`） |

需求规格、验收断言、审计报告、背景计划**都不在本仓库**，在在线文档（见文末「文档索引」）。
本仓库只放：**项目说明 + 代码 + 测试 + 工程配置**。

## 它解决什么

说明书 PDF 里的答案有三个硬问题，普通问答机器人一个都答不好：

1. **型号不同答案不同** —— `F4` 在 B1KA 与 B1HA 上的含义与处置不一样，必须按型号隔离检索。
2. **短代码检索失配** —— FTS5 的 `trigram` 分词要求查询词 ≥ 3 字符，`F4` 这类 2 字符代码在全文检索里**永远 0 命中**，所以短码必须走精确表（L0）。
3. **答对了但没证据** —— 用户要的是「第 69 页，这一段」，不是一段看似合理的解释。

对应的三条设计：型号槽位硬过滤、L0 故障码精确表短路、citation 由代码拼装而非模型生成。

## 架构

```text
query ──► router ──► [DANGER] ──────────────────────────► 拒答（不检索、不生成）
             │
             ├──► [L0 命中] ──► codes.json 精确行 ──────► 答案 + citation
             │
             └──► retrieve ──► eu_fts(trigram) + eu_meta(型号过滤) ──► top_k=6
                                  │
                                  ▼
                            build_prompt（只塞原文，不塞分数/id）
                                  │
                                  ▼
                            generate（temperature 0.1 / max_tokens 400）
                                  │
                                  ▼
                    assemble（★ 按 [n] 拼 citation，模型碰不到）
                                  │
                                  ▼
                    validate（数字逐字 / 引用归属 / 危险词 / 拒答）
```

**三条不变量**（改代码时不要破）：

1. **模型永远不产出 citation 对象**，只输出 `[n]` 角标；`eu_id` / `page_no` / `bbox` / `quote` 只能由 `assemble()` 从本次召回集拼装。
2. **`bbox` 是 PDF 的 pt 坐标**，不是像素；渲染缩放只在页面层发生。
3. **`rag_core.py` 平台无关**：不 import 任何 HTTP / 微信 SDK，多端靠 `adapters/` 适配。

## 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 生成 / 路由 | 百炼 OpenAI 兼容（`qwen` 系列） | 换厂商只改 `LLM_BASE_URL` + `LLM_MODEL` |
| 解析 | PyMuPDF | 直接拿 `page_no + bbox`；托管解析拿不到坐标 |
| 检索 | SQLite FTS5 `tokenize=trigram` + `codes.json` 精确表 | 短码不进 FTS |
| 编排 | 手写内核（8 个函数） | 不用 LangChain / LlamaIndex 作主实现 |
| 服务 | FastAPI + 微信云托管 | 同一镜像可迁通用云 |
| 前端 | 原生微信小程序 4 页 | 先 CLI → 本地单页 → 云上 API → 小程序 |

## 目录（随进度点亮，`○` 未产出）

```text
ViessmannRAG/
├── README.md                 项目说明（本文件）
├── AGENTS.md                 agent 开发流程约定
├── rag_core.py               内核 8 函数：load_eu / lookup_code / retrieve / build_prompt
│                             / generate / assemble / validate / ask
├── schema.py  lookup.py      EU 结构 / L0 精确表
├── retrieval.py  generator.py 召回与生成封装
├── validator.py  router.py  校验器 / 意图路由
├── config.py  refusal.py    常量与话术 / 拒答
├── trace.py  observe.py     trace 记录 / Langfuse 埋点
├── cli.py                   命令行入口
├── storage/                 Storage 协议 + 本地/云端实现
├── adapters/                wx / web / im 适配层
├── renderers/               按端渲染 citation
├── server/main.py           FastAPI，只做 adapter 分发
├── pipeline/                PDF → 版面 → EU → 索引（离线，可重跑）
├── tools/llm.py             唯一模型调用入口
├── eval/                    评测规则、金标、报告
├── data/                    eu.jsonl / codes.json / fts5.db（页图与索引不进仓）
├── web/                     本地单页（先于小程序）
├── miniprogram/             小程序 4 页
├── tests/                   每关一个 test_gNN.py + fixtures
├── deploy/                  Dockerfile 与云托管说明
└── notes/                   实验笔记（对照实验）
```

## 开发方式

一关一个增量、一个分支、一个 PR；每关固定走 **Reader → Coder → Reviewer** 三段，Reviewer 必须换上下文自己重跑（详见 [AGENTS.md](AGENTS.md)）。

```bash
pytest -q tests/test_gNN.py   # 单关验收（断言在在线文档的对应关卡规格里）
pytest -q                     # 全量回归，必须一直保持绿
```

## 文档索引（在线文档）

| 文档 | 内容 |
|---|---|
| 项目主档 | 进度看板、文件树、演进日志 |
| 需求规格总纲 | 契约基线、全局禁止项、验收框架 |
| 十四关需求规格 | 每关目标、文件清单、函数签名、可执行断言 |
| 规格审计与复核 | 三轮审计发现与修复记录 |
| 背景计划 | 14 关主线、模型与部署细节、阶段规划 |

在线文档位置：<https://docs.qq.com/desktop/mydoc/folder/AdkPbToQjhNO>

> 仓库内不提交上述文档；本地 `docs/` 是工作副本（已在 `.gitignore` 中），供 agent 读取。