# Abliterix vs Heretic：默认稳定性与独立验证审计

日期：2026-08-06

对照基线：Heretic `9069d3c754fcb3d20e62c7fd57addd1635eee054`（2026-08-05）

结论置信度：高（代码级逐项对照 + 本地全量测试 + 真实 tiny-model 端到端复现）

## 结论

审计前，Abliterix 不是“整体落后”：它在语义 LLM judge、未知标签处理、固定 benchmark contract、多语言与 MoE/vLLM 支持上更强。但在 Heretic 最成熟的两条工程链——稳定默认值和可证明的独立复现——确实存在五个实质差距：默认保护性变换较弱、远程输入未钉死、`reproducible` 标签过宽、`--reproduce` 实际重新搜索而非重放获胜 trial、CI 没有完整 tiny-model 权重哈希门禁。

本轮已补齐上述链路。现在 HF 默认启用正交投影与 full row-norm preservation；vLLM 未显式配置时采用其可实现的 `pre`。所有 Hub 模型和四份数据集在加载前解析为 immutable commit。复现清单升级为 schema v2，包含精确 steering recipe 和自身完整性哈希；`--reproduce` 不再搜索，而是重放获胜 trial 并重新测量 KL/拒绝数。CI 增加 Python 3.13、构建产物验证，以及两次独立 tiny-model 导出权重 SHA256 相等和清单重放验证。

## 对照证据

Heretic 的稳定默认值是 `orthogonalize_direction=true`、`row_normalization="full"`、200 trials、60 startup trials、批量自动探测和 100 token 响应上限；这些是本次默认值对齐的基准。[citation:Heretic default configuration](https://github.com/p-e-w/heretic/blob/9069d3c754fcb3d20e62c7fd57addd1635eee054/config.default.toml)

Heretic 只有在模型与全部数据集都来自 Hub、commit 已解析、所有 scorer 明确声明 reproducible 时才提供复现发布；插件的默认资格为 false，内置 KeywordRate 与 KL scorer 才主动声明 true。[citation:Heretic reproducibility eligibility](https://github.com/p-e-w/heretic/blob/9069d3c754fcb3d20e62c7fd57addd1635eee054/src/heretic/main.py) [citation:Heretic plugin reproducibility contract](https://github.com/p-e-w/heretic/blob/9069d3c754fcb3d20e62c7fd57addd1635eee054/src/heretic/plugin.py)

Heretic 的复现路径读取发布参数、恢复指定 trial，并执行环境/权重验证，而不是重新运行参数搜索。[citation:Heretic reproduction implementation](https://github.com/p-e-w/heretic/blob/9069d3c754fcb3d20e62c7fd57addd1635eee054/src/heretic/reproduce.py)

Heretic 的端到端测试使用多种 tiny 模型、commit-pinned 数据集和允许的跨平台 SHA256 列表验证完整 CLI 输出。[citation:Heretic E2E hash runner](https://github.com/p-e-w/heretic/blob/9069d3c754fcb3d20e62c7fd57addd1635eee054/tests/run_tests.py) [citation:Heretic pinned Qwen2.5 fixture](https://github.com/p-e-w/heretic/blob/9069d3c754fcb3d20e62c7fd57addd1635eee054/tests/qwen2.5/config.toml)

## 差距与修复

| 维度 | 审计前 Abliterix | Heretic | 本轮结果 |
| --- | --- | --- | --- |
| 保护性默认值 | `orthogonal=false`, `weight_normalization=none` | 正交 + full | HF 改为正交 + full；vLLM 隐式采用 pre，显式不支持组合仍拒绝 |
| 默认评估稳定性 | 外部 judge 默认开启，缺 key 到晚期才失败 | 内置确定性 scorer | 默认改为离线确定性；外部 judge 显式启用并在模型加载前校验 key |
| 输入身份 | 清单可记录 revision，但模型/数据未真正按 revision 加载 | 模型与数据 commit pin | 模型、tokenizer、HF/vLLM/SGLang 路径和四份数据集统一使用解析后的 commit |
| 复现资格 | 上传时总加 `reproducible` tag | 严格资格门控 | 本地输入、未 pin、外部 judge、脏源码、不支持精确 materialization、缺 trial recipe 均拒绝标签 |
| 清单完整性 | schema v1，无自校验，缺完整 recipe | 完整 reproduce 资料 | schema v2，含完整 trial recipe、数据 revisions、环境、指标、权重 SHA、自身 canonical SHA256 |
| `--reproduce` | 只还原 config，再跑完整搜索 | 精确 trial 恢复 | 无搜索精确重放；KL 使用容差复核，拒绝数严格相等，漂移硬失败 |
| Headless 输出 | 搜索结束即退出 | headless 选择/保存 | 可自动选择最佳完成 trial、重放、导出、哈希并生成 reproduce 目录 |
| CI Python | 3.10–3.12 | 3.10–3.13 | 增加 3.13 |
| 构建验证 | 仅执行 build | 检查构建结果 | wheel 与 sdist 必须同时存在 |
| 独立 E2E | 无完整权重哈希门禁 | 多模型固定 SHA | pinned tiny Qwen2.5 两次完整运行必须权重 SHA 相同，并执行 schema v2 精确重放 |

## 验证结果

- 单元/集成测试：在最新 `master` 上 `863 passed`。
- 真实端到端：`tiny-random/qwen2.5` 与四个 dataset split 全部 commit-pinned；两次独立两-trial 优化得到相同导出权重 SHA256。
- 清单复现：从第一次导出的 `reproduce.json` 恢复，不运行 Optuna；重新测得 `KL=0.000601151`、`refusals=0`，与发布指标一致。
- 格式与 lint：本轮涉及文件通过 Ruff 检查。

## 仍需持续扩展的边界

这轮补齐的是默认与证明链路，不代表所有 backend 都能获得 bit-for-bit 标签。vLLM/SGLang 的运行时或原位编辑状态目前仍会被资格门控拒绝精确标签，必须通过 HF twin materialization 导出；这是明确拒绝而不是伪装成功。Heretic 仍有更多模型和预先登记的跨平台 allowed-hash fixture，Abliterix 当前的 CI gold path 是一个真实 tiny 模型、同一 runner 上的双独立运行。后续每获得一种稳定 runner 输出，可继续加入 Linux/CUDA/MPS 的长期 golden hash，而无需改变清单协议。

## 最终判断

就“默认稳定性”和“独立验证是否诚实”而言，Abliterix 已不再落后于 Heretic：关键默认值已对齐，且资格判定更保守；清单还有 canonical integrity 和指标再验证。Heretic 暂时仍领先的是 fixture 的模型/平台覆盖数量，不是协议或默认行为。Abliterix 保持领先的部分是语义评测、benchmark contract、未知标签处理、多语言和大模型多 backend 支持。
