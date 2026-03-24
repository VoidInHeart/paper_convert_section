啊啊啊换行处理和小标题编号怎么就弄不好呢

# Paper Review System

一个面向论文评审场景的 Python 项目，核心目标是把 PDF 论文解析成可回溯的 Markdown 与中间表示，并进一步产出符合协议的评审报告 JSON。

## 能力范围

- PDF -> `Document IR`
- `Document IR` -> `Evidence IR`
- 清洗噪音块并构建章节树
- 导出 Markdown
- 基于启发式规则生成 `logic_analysis`
- 基于规则扫描生成 `rule_violations`
- 基于前两者生成 `improvement_plan`
- 装配并校验最终报告 JSON

## 架构

```text
PDF
 ↓
[parser-service]
 ↓
Document IR
 ↓
[anchor-service + noise-cleaner + section-builder]
 ↓
Evidence IR / Clean Blocks / Section Tree
 ├─→ [logic-engine] → logic_analysis
 ├─→ [rule-engine]  → rule_violations
 └─→ [retrieval-service] → improvement_plan
 ↓
[report-assembler + validator]
 ↓
Final Review Report JSON
```

## 安装

```bash
pip install -e .
```

## 使用

导出 Markdown 与中间表示：

```bash
paper-review convert path/to/paper.pdf --output-dir outputs
```

生成 Markdown 与评审报告：

```bash
paper-review review path/to/paper.pdf --output-dir outputs
```

输出目录下会包含：

- `document_ir.json`
- `evidence_ir.json`
- `paper.md`
- `review_report.json`

## 设计说明

- 解析层与协议层解耦，便于未来替换 parser 或升级协议版本。
- 所有数组字段在最终 JSON 中都强制输出 `[]`，不会输出 `null`。
- `rule_violations.id` 在单次报告内保证唯一。
- Markdown 导出面向“结构化复原”，不是像素级排版复刻。

## 后续扩展点

- 接入 LLM 做更强的 `logic_analysis`
- 接入向量库做 `expert_samples` 检索
- 增强图表、公式、参考文献解析
- 增加 Web API 与前端高亮联动
