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

所有问答都以流式方式接收模型答复。终端逐段显示正文，结束后列出引用。重定向或管道默认收完答复后输出单个 JSON。可用 `--stream` 或 `--json` 显式选择显示方式。

操作类型可选。问答检索将语义候选、原问句和用户提供的类型一起交给模型。召回操作步骤时，补充同一说明书同页同节的其他步骤。模型回答危险、类型适配和报码问题，程序从模型选用的片段回填来源、物理页、坐标和逐字引用。可用 `--index` 选择已发布的索引、`--top-k` 调整初次检索的候选数。

## 2A · H5 问答 API

在后端工作目录配置 `.env`（`DASHSCOPE_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`、`EMBED_MODEL`），并准备与向量模型匹配的已发布索引。启动开发服务。

```bash
.venv/bin/python -m manual_rag.api --port 8000 --allow-origin http://localhost:5173
```

默认地址为 `127.0.0.1:8000`，默认索引为 `data/semantic/index.json`。可用 `--index` 选择核定的索引。H5 可将 `/api/ask` 代理到后端，或者用完整后端地址发起跨源请求。用 `--allow-origin` 指定手机浏览器实际打开的 H5 来源（协议、主机和端口），多来源可重复指定。授权局域网联调时可用 `--host` 绑定开发机的局域网 IP，手机通过该 IP 或 H5 开发代理访问。模型密钥、PDF 和索引保存在后端。

```bash
curl -NsS http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' \
  --data '{"question":"怎么查看当前水压","control_type":"按键式"}'
```

请求体为 UTF-8 JSON，最大 16 KiB。`question` 为必填的非空字符串。`control_type` 可以省略、填 `null`，或使用 `旋钮式`、`按键式`、`触控式`。

请求受理后返回 HTTP 200 和 `Content-Type: text/event-stream; charset=utf-8`。H5 用 `fetch()` 发送 POST JSON，并从 `response.body` 逐段解析 SSE。每个事件由空行分隔，`data:` 后是 JSON。忽略以 `:` 开头的注释行。使用开发代理时关闭响应缓冲。

```text
event: delta
data: {"text":"请先选择"}

event: delta
data: {"text":"设备操作面板。"}

event: result
data: {"answer":"请先选择设备操作面板。","citations":[],"error_code":"missing_control_type"}

```

| 事件 | 数据与处理 |
| --- | --- |
| `delta` | `{"text":"正文增量"}`。按收到的顺序追加到临时正文，可能收到零个或多个。 |
| `result` | `{answer, citations, error_code}`。用 `answer` 定稿并展示引用。收到 `missing_control_type` 时展示面板选项，用原问题和所选类型重新发送。 |
| `error` | `{answer, citations, error_code}`。以 `answer` 提示失败并替换已经显示的临时正文。`error_code` 可能为 `model_timeout`、`model_service_error`、`invalid_model_response` 或 `internal_error`，可保留请求以供重试。 |

每条引用包含 `node_id`、`doc_id`、`page_no`、`bbox` 和 `quote`。`page_no` 是从 1 开始的 PDF 物理页，`bbox` 是页坐标中的矩形 `[x0, y0, x1, y1]`。前端以文本渲染答案和引用。

服务端在 `result` 或 `error` 后关闭响应。若连接在终止事件之前断开，H5 应将此次回答视为未完成并允许重试。上游超时发生在流开启后，通过 `event: error` 和 `error_code: "model_timeout"` 返回，HTTP 状态保持 200。流开启前的拒绝仍使用 JSON，例如请求有误（400）、来源未获允许（403）、接收超时（408）或服务正忙（503）。

`--upstream-timeout` 默认 30 秒，分别用于向量与生成的网络操作等待。完整问答耗时还包括两次调用及流式生成各段时间。

## R02-01 · 标注与分集

将正式 UTF-8 JSONL 题集保存到本机 `data/eval/golden_manual.jsonl`。每行一题。下面是合成澄清题。

```json
{"dataset_version":"manual-v1","id":"q-0001","intent_id":"pressure-read","split":"debug","control_type":null,"question":"这个面板怎么看水压？","expected_error_code":"missing_control_type","evidence_groups":[]}
```

| 字段 | 标注规则 |
| --- | --- |
| `dataset_version` | 每行使用同一版本号。修改问句、期望、证据或分集后递增版本，在本地保留冻结 JSONL、摘要和修订记录。 |
| `id` | 固定且唯一。重排或修订同一道题时保留 ID，新增问法分配新 ID。 |
| `intent_id`、`split` | 先按用户目标归并同义问法和补选面板前后的变体，再整组选择 `debug` 或 `holdout`。留出题供固定版本复测。 |
| `control_type`、`question` | 保存实际发送的面板类型和原问句。未提供面板时填 `null`。 |
| `expected_error_code` | `null` 表示带依据应答。`missing_control_type` 表示补选面板，`clarify_fault` 表示补充故障信息，`control_type_mismatch` 表示资料不适用，`insufficient_evidence` 表示资料不足，`safety_notice` 表示安全提示。 |
| `evidence_groups` | 应答题每项用 `required_fact` 记录一条必要事实，用 `anchors` 记录支持该事实的可替代原页锚点。全部事实组均需覆盖。其他题填空数组。 |

每个锚点填写 `doc_id`（来源 PDF 文件名去扩展名）、`pdf_version`（文件 SHA-256）、`page_no`（从 1 开始的物理页）和 `quote`（逐字引文）。同页原文重复时提供 `bbox`，坐标沿用 R01-02 的未旋转页坐标。引文检查忽略排版空白，保留其余字符。维护者在启用留出集前逐题按原页核对操作适用性、原句、事实和预期行为，并记录所复核的题集版本与摘要。

使用题集及核定页段配置做本地校验。省略 PDF 与配置参数可仅检查题集结构及分集。

```bash
.venv/bin/python -m manual_rag.question_set data/eval/golden_manual.jsonl \
  --pdf 'data/raw/按键式官方中文操作指南.pdf' \
  --scopes data/eval/control-scopes.json
```

## R02-02 · 逐题评测与离线回放

评测沿用问答的索引和模型配置。默认跑调试集，可用 `--question-id` 运行一题、`--top-k` 调整召回深度。维护者核定留出集后用 `--split holdout` 评测。

```bash
.venv/bin/python -m manual_rag eval --questions data/eval/golden_manual.jsonl --split debug
.venv/bin/python -m manual_rag replay <运行ID> <题目ID>
```

每次运行在 `data/eval/runs/<运行ID>/` 下生成 `report.jsonl`、`traces/<题目ID>.json`、运行配置、该次使用的标注和索引证据快照。两条命令均可用 `--runs` 指定其他本地目录。原始内容保存在该目录，报告和 Trace 含手册片段和题集标注，须留在本机。

报告逐事实组列出是否入库、原始召回、进入模型上下文、被引用和出现在答案中。证据按文档、PDF 版本、物理页、引文与可选坐标匹配当前节点，含补入的同节步骤。新运行对有必要事实的正常答复使用同一配置模型额外做一次语义判定。判定允许同义改写，并要求保留数值、单位、顺序和适用条件。人工复核仍负责确认忠实性与操作完整性。

`eval` 完成记录时退出 0，题目结果见报告的 `score`。归因包括资料缺失 `missing_material`、召回遗漏 `retrieval_miss`、应答策略错误 `answer_strategy_error`、引用错误 `citation_error`、事实遗漏 `missing_fact` 和服务异常 `service_error`，同一题可能有多个归因。事实判定请求、原始响应或失败类别保存在 Trace。判定请求异常或响应无效时，`fact_present` 为 `null`，报告保留 `fact_judgment_error`，归因为 `service_error`。回放用冻结材料重算，无需模型配置及索引原文件。格式 1 继续按当时的短语规则回放。回放结果与现场逐题判定一致时 `matches_original` 为 `true`，并退出 0。

## R02-03 · 汇总与基线冻结

分别对工程调试集和已核定的正式留出集运行 `eval`，各自生成报告。使用 `--question-id` 时，报告和图会注明单题范围。

```bash
.venv/bin/python -m manual_rag eval --split debug
.venv/bin/python -m manual_rag eval --split holdout
```

每个新运行目录完整生成后才发布。在运行目录中打开 `attributions.html` 查看失败归因条形图。图从 `report.jsonl` 生成，同题同类只计一次，一题可出现在多类中。汇总 `summary.json` 记录样本量、通过数、以下比例的分子与分母，以及逐题端到端耗时的合计、均值、中位数和最近秩 P95（秒）。空分母的比例值为 `null`。

| 指标 | 分子 | 分母 |
| --- | --- | --- |
| 证据覆盖率 | 逐题评分确认进入模型上下文的证据组数 | 本次全部标注证据组数 |
| 引用检查通过率 | 完成预期应答且通过引用检查的题数 | 本次所有预期应答题数 |
| 必答事实通过率 | 判定通过的必答事实数 | 本次所有必答事实数 |
| 应答行为准确率 | 行为与预期一致的题数 | 本次评测题数 |

服务异常计入适用指标的分母，缺少判定的条目不计入分子。

`run.json` 保存题集版本与选题范围、冻结文件的摘要、PDF 指纹、分块规则版本、Embedding 与生成模型、检索深度、事实判定策略、评分规则版本和评测源码指纹。源码指纹按路径和内容覆盖 `manual_rag/*.py` 及 `tools/*.py`。每次运行保留本次选题的 `questions.jsonl`、完整向量 `index.json`、`evidence.json` 与逐题 Trace。所有材料留在本机。

在线复跑时使用相同源码及模型配置，从 `run.json` 取分集和 `top_k`，把输入路径指向冻结副本，结果写入新的运行目录。下例使用调试集和默认检索深度 12。

```bash
run_dir="data/eval/runs/运行ID"
.venv/bin/python -m manual_rag eval --questions "$run_dir/questions.jsonl" \
  --index "$run_dir/index.json" --split debug --top-k 12
```

要复现当次判定，使用 `replay` 从冻结材料离线重算。较早运行仍可用原有回放入口。
