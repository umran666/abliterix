# Abliterix × Qwen3.5-2B 深度 A/B 结论

日期：2026-07-10

硬件：NVIDIA GeForce RTX 5090 32 GB

模型：Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，BF16，Transformers 5.7.0

## 结论

当前默认思路仍应是 **M：mean + orthogonal projection + rank-1**。它确实有明显的 abliteration 效果，但在预注册的严格 KL 阈值下还不能称为“无损胜出”：locked test 的三 token shared-context KL 为 `0.01336`，高于 `0.01` guardrail。

新候选 **Q：projected + winsorized + discriminative rank-1** 没有击败 M。Q 在 dev 上看起来好 2 个百分点，但 locked test 上 harmful refusal 反而高 2.00 个百分点，且 KL 高 10.85%。**R：rank-3** 在最低测试强度就出现 `0.03626` dev KL，不能作为默认方案。

速度方面有两个可保留的真实提升：

- hidden-state extraction 只保留末位 logits，使同一 64-prompt workload 从 `47.92` 提升到 `52.48 prompt/s`（`+9.5%`），峰值显存从 `4.665` 降至 `4.411 GiB`（`-5.4%`）。
- `fla-core 0.5.1` 对离线 hidden extraction 和 shared-continuation scoring 有明显收益；但普通 decode 没有稳定收益，且输出不是逐字等价，因此不能设为默认在线生成路径。

## 实验设计

- 数据按内容哈希去重；从 500 集固定 locked test，再从 1000 集删除所有 test hash。
- 每个 domain：train `399`、dev `100`、locked test `499`；seed `20260710`。
- test 在 recipe 与 strength 锁定后只读取一次。
- strength grid：`0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0`。
- 质量 guardrail：固定 shared continuation、3 token KL，dev 阈值 `≤0.01`。
- 统计：Wilson interval；prompt-paired 与 topic-cluster paired bootstrap，各 5,000 次。
- harmful refusal 目前仍是 keyword + degeneracy screening classifier，不是独立语义 judge。

四个实验臂：

- U：unsteered control。
- M：mean、orthogonal projection、rank-1。
- Q：mean、projected abliteration、winsorization、discriminative layer selection、rank-1。
- R：与 Q 相同，但 rank-3。

## Locked test

每个 domain `n=499`。

| Arm | Strength | Harmful refusal | 95% Wilson CI | Benign explicit refusal | Benign degeneracy | KL | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| U | — | 92.59% | 89.95–94.57% | 1.00% | 1.00% | 0 | 656.2 tok/s |
| M | 0.4 | 64.13% | 59.83–68.21% | 0.20% | 1.40% | 0.01336 | 725.2 tok/s |
| Q | 1.0 | 66.13% | 61.87–70.15% | 0.20% | 1.60% | 0.01481 | 723.4 tok/s |

关键 paired comparison：

- M − U harmful refusal：`-28.46 pp`；topic-cluster 95% CI `[-32.71, -24.45] pp`。
- M − U benign explicit refusal：`-0.80 pp`；95% CI `[-1.69, -0.19] pp`。
- M − U benign degeneracy：`+0.40 pp`；95% CI `[-0.58, +1.42] pp`，没有证据表明发生回归。
- Q − M harmful refusal：`+2.00 pp`；95% CI `[-1.37, +5.47] pp`。Q 没有证明更有效。
- Q − M KL：`+0.00145`；95% CI `[+0.00059, +0.00234]`，即 Q 的 KL 是 M 的 `1.1085×`。
- Q/M throughput ratio：`0.9975×`，基本相同。

Dev 上 M 的最低强度 `0.4` 是唯一满足 `KL ≤0.01` 的 M 点；R 的最低强度已经不满足 guardrail。由于 M 在 locked test 的 KL 上升到 `0.01336`，下一轮应扩大 dev calibration，并加入 `0.25–0.35` 的保守 strength 点，而不是提高强度。

## Capability guardrail

固定 OpenAI GSM8K test revision，抽取 100 题，以最终数字 exact match 评估：

| Arm | Accuracy | 95% Wilson CI | Paired delta vs U |
|---|---:|---:|---:|
| U | 5/100 = 5% | 2.15–11.18% | — |
| M | 5/100 = 5% | 2.15–11.18% | 0 pp，CI `[0, 0]` |
| Q | 5/100 = 5% | 2.15–11.18% | 0 pp，CI `[0, 0]` |

三个 arm 的逐题 correctness vector 完全相同，因此本 guardrail 没检测到相对回归。但绝对分数过低，存在明显 floor effect，不能据此声称通用能力保持不变。

## 速度与内核

### `logits_to_keep=1`

对支持该参数的模型，hidden extraction 不再物化完整 vocabulary logits；旧模型和 wrapper 通过显式 forward signature 检测保持兼容。

| Metric | Before | After | Change |
|---|---:|---:|---:|
| 64 prompts wall time | 1.3356 s | 1.2195 s | -8.7% |
| Prompt throughput | 47.92/s | 52.48/s | +9.5% |
| Peak allocated | 4.665 GiB | 4.411 GiB | -5.4% |

Tiny-Qwen hidden-state output max difference 为 0。

### Length bucketing

随机 batch 的理论 padding waste 很大，但基础 torch kernel 的实际测量并未受益：hidden extraction 慢约 10.9%，scoring 慢约 0.4%，decode 慢约 0.5%。因此该能力保留为 opt-in，默认 `False`。在 FLA hidden extraction 下，按长度排序才表现出明显收益。

### FLA kernel benchmark

`fla-core 0.5.1` 使用隔离 overlay 安装；当前环境没有 `causal-conv1d`。模型 provenance 证明 18 个 GatedDeltaNet 层实际绑定了 FLA chunk/recurrent/norm callable，而不只是“包可导入”。

| Workload | Random warm speedup | Sorted warm speedup |
|---|---:|---:|
| Hidden extraction | 1.680× | 2.152× |
| Shared-continuation scoring | 1.373× | 1.418× |
| Decode 128 tokens | 0.978× | 1.068× |

六个 workload 的 warm speedup geometric mean 为 `1.395×`，平均峰值 allocated memory 少 `216 MiB`。full-vocabulary probe 的 base→FLA KL 为 `0.0006055`，top-1 agreement 为 `100%`，但六个严格 tensor hash 均不相同。

独立的 100 harmful + 100 benign greedy generation parity 显示：

- harmful heuristic refusal：base `94/100`，FLA `94/100`，paired delta `0 pp`。
- benign explicit refusal：base `1/100`，FLA `1/100`，paired delta `0 pp`。
- response exact match：`50/200 = 25%`。
- visible token-count match：`161/200 = 80.5%`；平均绝对差 `3.2 tokens`。
- 该含首次 Triton 编译的完整 generation run：FLA throughput `0.842×` baseline，峰值 allocated memory `-5.8%`。

因此 FLA 目前只适合 opt-in 的离线 residual extraction / KL scoring。在线 decode 需要独立 warm-serving benchmark 与语义 judge 通过后再考虑。

## 已落地的工程改进

- hidden extraction 条件使用 `logits_to_keep=1`，支持 PEFT/compile wrapper signature 检测和旧模型 fallback。
- generation、hidden extraction、continuation scoring 增加稳定的 length-bucketing opt-in，并严格恢复原始顺序。
- rank-k 的第一方向固定为 canonical `mean(target) - mean(benign)`；额外方向只在其正交残差上做 SVD，并用两遍 modified Gram-Schmidt 重正交。
- winsorization 在 rank-k 路径真正生效，之后重新投影/正交；退化方向保持零向量。
- degeneracy detector 从整段 unique-char ratio 改为滑动窗口，并要求相邻低多样性窗口，减少长而正常文本被误判。
- 新增可复现 Qwen3.5 A/B、GSM8K guardrail、kernel benchmark 与 base/FLA privacy-preserving parity harness。

## 推荐的下一轮

1. 默认保留 M rank-1；Q 和 rank-3 不晋升。
2. 如果 `KL ≤0.01` 是硬 SLO，暂不发布当前 M@0.4；补测 M strength `0.25, 0.30, 0.35`，用更大 dev 或 KL 上置信界做选择。
3. 加入独立语义 judge / 人审盲测，尤其检查 harmful response 是否真正完成请求，而非只绕过关键词。
4. capability suite 至少加入 instruction following、常识/知识、代码和长上下文；GSM8K 5% 只能作为“没有相对变化”的弱 guardrail。
5. FLA 作为可选 offline extra；基础 HF decode 和 length sorting 均保持当前默认关闭状态。

## 可复核产物

- `qwen35_2b_ab_full_v2_20260710.json`：正式 A/B、所有 dev 点、locked test 与 bootstrap。
- `qwen35_2b_gsm8k_full_20260710.json`：GSM8K guardrail。
- `qwen35_kernel_base_vs_fla051_20260710.json`：FLA kernel benchmark。
- `qwen35_fla_parity_base_vs_fla051_20260710.json`：逐响应 parity comparison。
- `qwen35_fla_parity_base_20260710.json` / `qwen35_fla_parity_fla051_20260710.json`：不保存 prompt/response 明文的独立 process reports。

外部主来源：[Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)、[Transformers 5.7 Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.7.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、[FLA installation](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)、[OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k)。
