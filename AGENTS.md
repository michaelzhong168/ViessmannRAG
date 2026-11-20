# AGENTS.md · 开发约定

> 任何进入本仓库的 agent 先读这一页。项目说明见 [README.md](README.md)。

## 0. 文档在哪（先看这条）

**需求规格与计划文档不在本仓库**，在在线文档：<https://docs.qq.com/desktop/mydoc/folder/AdkPbToQjhNO>

| 内容 | 位置 |
|---|---|
| 14 关需求规格（含可执行断言） | 在线文档「十四关需求规格」 |
| 契约基线（EU schema / HTTP / citation / trace / 内核签名） | 在线文档「需求规格总纲」 |
| 项目主档（进度看板、演进日志） | 在线文档「项目主档」 |
| 审计与复核记录 | 在线文档「规格审计与复核」 |
| 本地工作副本 | `docs/`（**已 gitignore，不入仓**） |

**仓库只放：项目说明 + 代码 + 测试 + 工程配置。** 禁止把规格、计划、审计、笔记提交进 git；
`docs/` 下的一切对 agent 可读，但不进版本库。

## 1. 铁律

1. **规格是验收依据，agent 不得修改。** 规格有问题 → 在 PR 描述里写 `CHANGE_REQUEST`（改哪个字段 / 为什么 / 影响哪些关），由维护者改在线文档与本地副本。
2. **一次只做一关。** 不合并多关，不顺手改别的关的产出；已验收关卡的代码只在规格明确要求时动。
3. **断言不可放宽。** 不删断言、不 `skip`、不 `xfail`、不 mock 掉被测函数本身、不为过测试改测试。
4. **证据优先。** 结论必须带原始命令输出；「测试通过」四个字不算证据。
5. **禁止项**：不用 LangChain / LlamaIndex / Dify / RAGFlow 作主实现；不上向量库 / rerank / Agent 自主规划；模型不得生成 citation 对象、`eu_id`、页码；不改写原文、不做单位换算；不用境外模型 API；密钥零进仓、零进镜像、零进前端。
6. **mock-first。** 门禁必须在无 `DASHSCOPE_API_KEY`、无真实 PDF、无 Docker 的环境跑通；真实联调是人工确认项，不阻塞合并。

## 2. 三角色流水线

```text
规格 GNN ──► Reader ──► [门禁] ──► Coder ──► [自测] ──► Reviewer ──► PASS ──► PR
              │                     ▲                    │
          实现计划              返工（≤2 轮）◄──── CHANGES_REQUESTED
```

| 角色 | 产出（写进 PR 正文） | 禁止 | 通过条件 |
|---|---|---|---|
| **Reader** | 前置核验 + §7 断言逐条映射到测试函数 + 歧义清单 + 结论（READY / BLOCKED / NEED_DECISION） | 写实现代码、改规格、装依赖 | 前置清楚、断言无遗漏、结论 READY |
| **Coder** | 文件清单 + **原始** pytest 输出 + 断言逐条状态 + 未解决项 + 是否需 CHANGE_REQUEST | 改测试放宽、编造前置数据 | 单关 + 全量回归全绿 |
| **Reviewer** | 结论（PASS / CHANGES_REQUESTED）+ 自己重跑的原始输出 + **变异探针**结果 + 问题清单（P0/P1/P2） | 改实现、放行无证据项 | 自己跑绿、断言全覆盖、变异能被抓住、禁止项无违例 |

### 门禁规则

- **Reader 不是 READY 就停**：`BLOCKED` 报告缺什么，不得编造；`NEED_DECISION` 交维护者定。
- **Coder 首次交付必须全绿**才进评审；带失败进评审的直接 `CHANGES_REQUESTED`。
- **Reviewer 必须换上下文**：新 agent 实例，只给规格 + 仓库，不给 Coder 的推理过程；同一会话自评不算评审。
- **变异探针**：在临时副本（`cp -R . /tmp/mut_$$`）里破坏 2–4 处实现（改阈值、删过滤、伪造 citation），确认对应断言**会红**；抓不住的断言记为无效。**绝不在工作区做变异。**
- **P0 一票否决**：有 P0 不得合并，不得「先合后修」。
- **返工最多 2 轮**：第 2 轮仍不通过 → 暂停，多半是规格歧义，走 CHANGE_REQUEST。

## 3. 分支模型 / 提交 / PR

| 分支 | 角色 | 谁能写 |
|---|---|---|
| `main` | **发布线**，只接受来自 `dev` 的合并 | 任何人都不直接推送（GitHub 分支保护 + 本地 pre-push hook 双重拦截） |
| `dev` | **开发基线**，所有功能在这里汇聚 | 只接受 PR 合并 |
| `feat/gNN-<slug>` | 每关一个特性分支 | agent 推送，之后提 PR |
| `fix/<slug>`、`chore/<slug>` | 修 bug / 工程改动 | 同上 |

```bash
git switch dev && git pull                 # 从 dev 起
git switch -c feat/gNN-<slug>              # 例：feat/g06-rag-core
git commit -m "feat(gNN): <一句话>"
git push -u origin feat/gNN-<slug>         # 推特性分支，不要推 main
# → 在 GitHub 上对 `dev` 开 PR
```

- **提交信息**：`feat(g06): 第一个 RAG 闭环`；修 bug `fix(gNN): …`；工程改动 `chore: …`。
- **PR 标题**：`[GNN] <关卡主题>`；**base 分支必须是 `dev`**，不是 `main`。
- **PR 正文必须自带三段产物**（计划 / 交付 / 评审），因为仓库不存这些文件 —— PR 是唯一留档。
- **合并**：squash 到 `dev`，一关一条干净提交。
- **发布**：`dev` → `main` 也走 PR（发布批次），由维护者操作。
- **红线**：任何情况下不得 `git push origin main`；本地 pre-push hook 会直接拒绝。

## 4. 每关收工

1. PR 合并、链接记录到在线文档「项目主档」的进度看板。
2. 在线文档回填四件事：当前状态 → 进度与工时 → 文件树 `○`→`●` → 演进日志追加一行。
3. 若发生契约变更：维护者改在线文档并升版本号，受影响的关卡规格同步改。
4. §8 人工确认项（真机 / 真密钥 / 真 PDF）写进 PR 正文，留给真人；不阻塞合并，但要标注未验证。

## 5. 命令速查

```bash
pytest -q tests/test_gNN.py    # 单关验收
pytest -q                      # 全量回归
pytest -q -rs                  # 看 skipped（应为 0）
python -m cli ask "F4 怎么办" --model B1KA   # G06 起的手工冒烟
git log --oneline main..HEAD   # 本关提交
git diff --stat main...HEAD    # 本关改动面
```

## 6. 环境

- Python **3.11**（`.python-version` 已固定；本机用 uv 装的 3.11.16，SQLite 3.53.1，`trigram` 已实测）。
- Node ≥ 18 用于个别前端纯函数断言（G10 / G14）。
- 依赖只加当关需要的，逐关追加进 `requirements.txt`。
- **文档类产物一律不进仓**：`语料审计.md`、`验收报告.md`、`eval/report.md`、`eval/golden_manual.jsonl`（人写金标）、`notes/` 全部已在 `.gitignore`；需要留档的贴进 PR 正文或在线文档。
- 模型名以百炼控制台当前可用为准（人工确认项）。