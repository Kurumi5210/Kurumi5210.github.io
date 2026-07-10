---
title: PD 分离场景下的 KV Cache 传输优化
date: 2026-07-10 17:19:56
updated: 2026-07-10 17:19:56
tags:
  - vLLM
  - Ascend
  - Mooncake
  - PD 分离
  - KV Cache
  - GQA
  - Pipeline Parallel
  - 推理优化
categories:
  - 推理优化
---

> 本文记录一次 PD 分离场景下 Mooncake Connector 的 KV Cache 传输优化，重点是 Qwen3 等 GQA/MHA/MQA 模型下的 TP 异构、PP 适配、ACL Graph 稳定性问题，以及后续融合 transpose 算子的设计。

## 背景：从 MLA 到 GQA

在 6 月 30 日，我们完成了 PD 分离场景下的 Mooncake Connector，主要用于 Prefill 节点到 Decode 节点之间的 KV Cache 传输。初版主要面向 DeepSeek 模型开发，它的 attention 使用 MLA（Multi-Head Linear Attention），KV Cache 的组织方式有两个比较友好的特点：

1. KV Cache 由 embedding 之后的 hidden states 经过下投影矩阵生成。
2. 在 TP（Tensor Parallel）域内，各个 TP rank 可以缓存全量 KV Cache。Decode 节点拉取 KV Cache 时，选择哪个 Prefill TP rank 都是可行的。

随着 Qwen3 适配推进，GQA、MHA、MQA 场景把问题复杂化了。核心变化是：TP 域内的 KV Cache 不再统一。Decode 节点不能再随便挑一个 Prefill TP rank 拉数据，而是必须选择正确的 rank，并且在 Decode 节点的 local block 内完成重排序。

这带来两个直接挑战：

- **TP 异构下的 KV Cache 不一致**：Prefill 和 Decode 的 TP 配置不同，拉到 Decode 侧的数据排布不能直接用于 forward。
- **PP 场景下的层归属不同**：拉取 KV Cache 时必须考虑 Prefill 节点的 PP stage，不同 stage 持有不同层的 KV Cache，即使 MLA 场景也不能忽略这一点。

这次优化的目标很明确：

1. 在不明显影响性能的前提下，让 PD 分离支持 GQA TP 异构的 KV Cache 传输。
2. 支持 PP（Pipeline Parallel）场景下的 KV Cache 传输。
3. 支持开启 MTP（Multi-Token Prefill）后的 PP 流水线。

## Decode 侧 KV Cache 重排序

GQA 格式下，P 节点和 D 节点 TP 配置不同，P 节点生成的 KV Cache 与 D 节点期望的存储排布不一致。以 Qwen3 235B 为例，当 Prefill 侧 TP size 为 16、`num_heads = 4` 时，单个 TP rank 只能生成一部分 head 对应的 KV Cache。Decode 侧如果直接使用这些数据，forward 的 layout 就是错的。

解决思路是把 layout 转换放到 Connector 层完成：

1. 以 block 为粒度对 KV Cache 做重排序，reshape 成符合 Decode 侧 TP size 的排布。
2. vLLM Connector 感知 TP 配置，并完成 KV Cache 的 split 和 cat。
3. 底层 Mooncake TransferEngine 只负责数据传输，不感知 TP 配置。
4. 为避免额外申请大 buffer，引入逐 block 的 zero-copy 传输，并在传输完成后逐 block transpose，达到 cat 的效果。

![GQA TP 异构下的 KV Cache 重排序](/images/KVOptimization/gqa-tp-layout-reorder.png)

重排序的执行流程如下：

![KV Cache 重排序流程](/images/KVOptimization/cache-reorder-flow.png)

初版逐 block 重排序在 4K KV Cache 上约 120ms。由于每层 Decode 间需要频繁 launch kernel，在 4K x 1.5K benchmark 中，TPOT 增加了约 28ms。

后续优化改成使用 `_npu_page_load` 读取 KV Cache，再用 `_npu_reshape_and_cache` 写回。这样每层只处理一次，只需要申请一层 KV Cache 大小的 buffer，大幅减少了 kernel launch 时间。

## 选择 Prefill Rank

Decode 节点拉取 KV Cache 时，需要选择对应的 Prefill TP rank。这个选择逻辑在 MLA 和 GQA 场景下不一样：

- MLA 场景下，同一 TP 域内的 rank 可以缓存全量 KV Cache，因此可以随机选择 Prefill TP rank，用于动态负载均衡。
- GQA 场景下，TP rank 对应的数据分片不同，Decode 侧必须选择正确的 Prefill TP rank。
- PP 场景下，需要先按 PP stage 对 Prefill 节点分组，再在每个 PP stage 内选择 TP rank。

![Prefill TP Rank 选择逻辑](/images/KVOptimization/prefill-rank-selection.png)

这部分改造的关键是让 Connector 同时感知 TP 和 PP 配置。TP 维度决定从哪个 rank 拉，PP 维度决定这个 rank 上有哪些 layer 需要拉。

## KV Cache 传输路径

真正的数据搬运仍然交给 Mooncake TransferEngine。Connector 负责根据源地址和目标地址列表构造传输任务，并把正确的地址偏移计算好。

在 GQA TP 切分下，Decode 节点每层可能需要多次拉取 KV Cache。每次拉取的数据都要写入 Decode 侧 local block 的正确位置，因此必须引入地址 offset。

PP 场景下，单个 Prefill rank 只持有一部分 layer 的 KV Cache。Connector 需要根据 PP stage 和 `get_pp_indices` 计算本次传输涉及的 layer，并将它们放到 Decode 节点对应地址。

![KV Cache 传输流程](/images/KVOptimization/kv-transfer-flow.png)

这里的职责划分很重要：TransferEngine 只接受源地址和目标地址并完成传输，复杂的 TP/PP layout 逻辑收敛在 Connector 内部。

## 手动划分 PP Layer

开启 MTP 后，KV Cache 会多出 MTP 层，原有 `get_pp_indices` 无法感知这部分 layer，容易导致部分 KV Cache 未传输，或者 PP 层切分不均。

为了解决这个问题，我们引入了手动 PP layer partition，通过 Connector 配置让 Mooncake Connector 感知层数切分。例如：

```bash
export VLLM_PP_LAYER_PARTITION=33,28
```

```json
{
  "kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": {
      "dp_size": 1,
      "tp_size": 8,
      "pp_size": 2,
      "pp_layer_partition": "33,28"
    },
    "decode": {
      "dp_size": 16,
      "tp_size": 1,
      "pp_size": 1
    }
  }
}
```

这相当于把 PP layer 的切分结果显式交给 Connector，避免 Connector 只依赖默认 `get_pp_indices` 推断。

## 开发中遇到的几个问题

### Decode 重排序精度问题

最初版本中出现过一个精度问题：前半段数据正确，后半段数据错乱。根因是重排序和结束信号的时序不清晰，部分数据还没有完成拉取或重排，就进入了后续流程。

修复方式是把流程边界收紧：所有 KV Cache 拉取完成后才开始重排序，重排序完成后再通知 Prefill 侧释放显存。

### Mooncake 对非对等层数传输的支持问题

PP KV Cache 传输初版遇到过访存越界。原因是当时 Mooncake 版本会根据首地址和本端注册的层数进行拉取，而 PP 场景下两端持有的 layer 数并不总是对等，报错信息里可以看到循环首地址。

后续 Mooncake 版本更新后解决了这个问题。

## 性能收益

优化后的收益主要体现在两部分：

1. `_reshape` 替换 `_copy` 后，TPOT 从 120ms 降至 94ms。
2. 在无异步调度时，重排序几乎没有额外 TPOT 损耗。

PP 适配完成后，DeepSeek PD 分离的性能瓶颈从 Prefill 节点转移到 Decode 节点。调整 PD 分离配比后，归一化 QPS 还能继续提升。

## ACL Graph 支持 Qwen3 235B PD 分离

在 Mooncake Connector 与 vLLM 结合的场景下，forward 流程和 Mooncake cat 方法中都使用了 `torch_npu._npu_reshape_and_cache` 算子。问题表现为：ACL Graph 场景下，调用 vLLM 的 connect 时偶发 cancel 异常。

这两个算子在功能上等价，但一个在图内调用，另一个在图外调用：

- forward 内部入图的 `_npu_reshape_and_cache` 与 Mooncake cat 中的 `_npu_reshape_and_cache` 会互相影响。
- 两处算子分别替换成等效实现都能正常跑通，精度也正常。
- 注释任意一个算子，也能正常运行。
- 同时保留两者时，可能抛出异常或出现不稳定行为。

异常形式也和调度方式有关：

- ACL Graph + `--async-scheduling` 时，vLLM connect 抛出 cancel 异常，没有更具体的错误信息。
- 关闭 `--async-scheduling` 后，异常能定位到 forward 内部的 `_npu_reshape_and_cache`。

根因在 PageAttention 算子更新过程。算子更新时每次需要额外申请 workspace，这个 workspace 来自单算子流的 pool。算子更新会把 workspace 写入图中，更新结束后 workspace 被释放。但是图重放时，单算子流正在进行 KV Cache 传输；从单算子流视角看，这段 workspace 可以复用，但从图视角看，PageAttention 计算仍然需要这段 workspace，于是发生了内存踩踏。

规避方案是在 slot mapping 生成后加入同步操作，改变流上的算子排布逻辑，以及图运行和下发的时序。

## 融合 transpose_kv_cache_by_block 算子

当前 GQA 场景下的 layout 转换依赖 `_npu_reshape_and_cache` 和 `_npu_page_load`。这个方案能跑通，但仍然存在两个问题：

1. 算子调用次数过多。目前每层需要 launch 4 个算子，一个请求会调用 376 次算子。
2. 当前依赖 Decode 间的 bubble 掩盖算子开销。未来异步调度优化 Decode bubble 后，这部分开销可能无法继续被掩盖。

本质上，这里要做的是一次 transpose，只是现有算子无法一次性完成。因此可以实现一个融合算子，一次完成所有层 KV Cache 的 layout 转换。

### 算子接口

算子的功能是：对传入的 `k_cache` 和 `v_cache` TensorList 进行 layout 转换。每隔 `block_len` 个数据视为一个 block，按照 `block_ids` 指定的 block 做转换。每个 block 内包含 `split_num` 份数据，每份表示 `num_head / split_num` 个 head 的 KV Cache。转换完成后，把这 `split_num` 份数据在 `num_head` 维度合并。

这个算子将结果写回原内存，不额外使用 workspace。

接口定义如下：

```python
transpose_kv_cache_by_block(
    k_cache,
    v_cache,
    block_ids,
    block_size,
    head_sum,
    head_dim,
    split_num,
    layer_num,
)
```

算子描述可以抽象成：

```json
[
  {
    "op": "TransposeKVCacheByBlock",
    "input_desc": [
      {
        "name": "KCache",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["fp16", "bf16"]
      },
      {
        "name": "VCache",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["fp16", "bf16"]
      },
      {
        "name": "blockIDs",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["int32", "int32"]
      }
    ],
    "output_desc": [],
    "attr": [
      {"name": "blockSize", "param_type": "required", "type": "int"},
      {"name": "headNum", "param_type": "required", "type": "int"},
      {"name": "headDim", "param_type": "required", "type": "int"},
      {"name": "splitNum", "param_type": "required", "type": "int"},
      {"name": "layerNum", "param_type": "required", "type": "int"}
    ]
  }
]
```

当 `num_head = 4`、`split_num = 4` 时，转换过程如下：

![split_num 为 4 的 KV Cache transpose](/images/KVOptimization/transpose-split4.png)

当 `num_head = 4`、`split_num = 2` 时，转换过程如下：

![split_num 为 2 的 KV Cache transpose](/images/KVOptimization/transpose-split2.png)

### 实现设计

首先根据 `blockIDs` 计算每个 layer 有多少个 block 需要做 layout 转换：

```text
block_num = blockIDs.shape[0]
total_block_num = layerNum * block_num
data_size_load_once = block_size * head_num * head_dim * sizeof(T)
```

如果 `data_size_load_once` 单个 core 就可以完成读取，则使用模板一；否则使用模板二。

#### 模板一：单 core 处理一个 block

模板一的特点是单个 core 可以完整读取一个 block 的所有数据。整体流程是把整个 block 读入 UB，在 UB 中完成重排，然后直接写回原内存位置。

负载均衡时，先计算每个 core 至少需要处理多少个 block：

```text
round = total_block_num / vector_core_num
tailCoreNum = total_block_num % vector_core_num
```

有 `tailCoreNum` 个 core 需要多处理一个 block。每个 core 负责的 block 可能分布在不同 layer 中，这样能让负载更均衡。

重排的关键是 `DataCopyParams`：

```cpp
DataCopyParams repeatParams;
repeatParams.blockCount = blockSize_;
repeatParams.blockLen =
    headNum_ / splitNum_ * headDim_ * sizeof(T) / dataBlockSize_;
repeatParams.srcStride = 0;
repeatParams.dstStride =
    (headNum_ * headDim_ - headNum_ / splitNum_ * headDim_) *
    sizeof(T) / dataBlockSize_;
```

一条 `DataCopy` 指令完成 `repeatParams.blockCount * repeatParams.blockLen` 的搬运量，也就是总数据量的 `1 / splitNum_`。因此需要执行 `splitNum_` 次 `DataCopy`，每次偏移已经完成搬运的部分：

```cpp
dstFactor_ = headNum_ / splitNum_ * headDim_;
srcFactor_ = blockSize_ * headNum_ / splitNum_ * headDim_;

for (uint32_t i = 0; i < splitNum_; ++i) {
    DataCopy(cacheLocal[i * dstFactor_],
             cacheGm[i * srcFactor_ + offsetBlock],
             repeatParams);
}
```

在 UB 内完成重排后，再连续搬出即可。

#### 模板二：多 core 协同处理一个 block

模板二解决的是单个 core 无法读取完整 block 的场景。多个 core 协同读取一个 block，这里的协同 core 数称为 `blockSizeSplitNum`。

为了充分利用算力，`blockSizeSplitNum` 取 AIV core 数的因子。以 48 个 core 为例，可选值包括 2、3、4、6、8、12、16、24、48。实现上选择能满足读取一个 block 的最小因子；如果最大因子也无法满足，就说明超出了算子能力，需要拦截并报错。

负载均衡公式变为：

```text
core_num = vector_core_num / blockSizeSplitNum
round = total_block_num / core_num
tailCoreNum = total_block_num % core_num
```

模板二和模板一的主要差异在于 block_size 维度被切分，`DataCopyParams` 需要修正：

```cpp
blockSizePerTime_ = blockSize / blockSizeSplitNum;
repeatParams.blockCount = blockSizePerTime_;
```

多个 core 计算同一个 block 时，搬入和搬出地址都需要加上对应偏移：

```cpp
uint32_t blockSizeIndex = blockIdx_ % blockSizeSplitNum_;
srcOffset = blockSizeIndex * blockSizePerTime_ * headNumSplited_ * headDim_;
dstOffset = blockSizeIndex * blockSizePerTime_ * headNum_ * headDim_;

CopyIn(kCacheGm_, offsetBlock + srcOffset, repeatParams);
CopyOut(kCacheGm_, offsetBlock + dstOffset);
```

搬出前需要做一次核间同步，保证所有 core 都已经完成读入，避免覆盖仍在读取的数据。

### 性能实测

使用融合算子后，reshape KV Cache 的时间开销从 UT 中的 7ms 降低到 0.24ms。在 PD 分离场景下，Qwen3 235B 的 TTFT 可降低约 90ms 到 110ms。

![transpose_kv_cache_by_block 性能实测](/images/KVOptimization/transpose-benchmark.jpeg)

## 小结

这次 KV Cache 传输优化的核心不是单纯把数据从 Prefill 搬到 Decode，而是把 TP、PP、模型 attention 结构和执行流同步问题都纳入 Connector 的控制面。

几个关键经验：

1. **TransferEngine 不应该感知模型并行细节**。TP/PP layout 逻辑放在 Connector 层，底层通信库只负责可靠搬运。
2. **GQA TP 异构必须显式处理 layout**。MLA 场景中随机选 rank 的假设，不能直接复用到 GQA/MHA/MQA。
3. **PP stage 是 KV Cache 路由的一部分**。只看 TP rank 不够，必须同时知道 layer 属于哪个 PP stage。
4. **性能瓶颈常常来自 kernel launch 次数，而不只是搬运带宽**。从逐 block 重排序到 `_npu_page_load` + `_npu_reshape_and_cache`，再到融合算子，本质都是在减少不必要的 launch 和中间搬运。
5. **ACL Graph 与异步流的内存生命周期需要特别谨慎**。图内 workspace、图外单算子流和 KV 传输并行时，必须保证同步边界清晰。

最终结果是，PD 分离可以覆盖 Qwen3 235B 这类 GQA 模型和 PP/MTP 场景，同时通过融合算子把 layout 转换开销压到更可控的范围内。
