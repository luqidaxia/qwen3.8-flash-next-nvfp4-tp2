# vLLM 参数详解 — Qwen3.8-Flash-Next-NVFP4 双机部署

本文逐条解释 `start-official.sh` 最终下发的 vLLM 参数，以及 `.env` 中对应变量的含义。
所有参数的**实际来源**都是 `.env`（`deploy/config.sh`），不要直接改脚本。

服务标识：模型 id = `SERVED_MODEL_NAME` = `qwen3.8-flash-next`，端口 = `PORT` = `8888`。

---

## 模型参数

### `--tensor-parallel-size 2`

- `.env`：`TENSOR_PARALLEL_SIZE=2`
- 每台 Spark 只有 1 个逻辑 GPU，因此 **TP=2 正好是 1 GPU/节点 × 2 节点**。
- 两台机器必须同时在线；单机启动请改用 `start-tp1.sh`（TP=1）。

### `--enable-expert-parallel`

- `.env`：`ENABLE_EXPERT_PARALLEL=true`
- **NVFP4 检查点的必需项**。MoE 专家按 EP 切分；关掉或改成纯 TP 会在权重加载阶段报错。
- 与 TP 组合即常见的 **TP=2 + EP** 拓扑。

### `--nnodes 2` / 分布式初始化

- head（rank 0）负责拉起 worker（rank 1），通过 `MASTER_PORT=50000` 做分布式协调，
  `HEAD_IP` / `WORKER_IP` 由 `.env` 提供（平台上由平台注入）。
- 手动场景下**不要单独启动 worker**，全部交由 `start.sh` 驱动。

### `--trust-remote-code`

- 该模型使用自定义建模代码（PLE 层、稀疏注意力等），必须开启。

### `--served-model-name qwen3.8-flash-next`

- `.env`：`SERVED_MODEL_NAME`
- `/v1/models` 返回的 id；对外请求的 `"model"` 字段用它。

---

## KV Cache 参数

### `--kv-cache-dtype fp8`

- `.env`：`KV_CACHE_DTYPE=fp8`（默认）；设 `auto` 即退回 bf16。
- **原生 QSA kernel 拒绝 fp8**，所以当值为 `fp8*` 时 `start.sh` 会自动应用
  `files/patch_qsa_fp8_kv.py`。
- 收益（262K 上下文实测）：

| dtype | KV 容量 | cache token | 相对 262K |
|---|---|---|---|
| `fp8`（默认） | 32.02 GiB | **3,652,200** | **13.93×** |
| `auto`（bf16） | 32.02 GiB | 2,131,159 | 8.13× |

即每 GiB 缓存密度提升约 **1.70×**，且实测**无解码开销、无推理/needle 回归（均 12/12）**。
但原则上仍是精度取舍：量化 key 会改变稀疏索引器选中哪些 block，不只是改变注意力输出。
请用 `bench/reasoning_check.py` 在自己的业务上复验。

### `--gpu-memory-utilization 0.835`

- `.env`：`GPU_MEMORY_UTILIZATION=0.835`
- 本套件实测的最优点，对应上表 32.02 GiB KV。
- 敏感度：**+0.01 GMU ≈ +0.83 GiB ≈ +95k fp8 cache token**；只在有余量时上调，且两台一致。

### `--max-model-len 262144`

- `.env`：`MAX_MODEL_LEN=262144`（原生上下文）。
- 要扩到 ~1M：同时设 `MAX_MODEL_LEN=1048576` 与 `YARN_ENABLE=true`（`YARN_FACTOR=4.0`，
  即 262K × 4.0），此时**必须** `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`。
- 注意：`MAX_MODEL_LEN <= 262144` 时 `start.sh` 会自动关闭 YaRN。

### YaRN 说明（重要）

- 2026-09 之前，YaRN 覆盖项被写在 `--hf-overrides` 的**顶层**，永远不会进入 `text_config`
  ——即 **静默无效**（silent no-op）。
- 现已修复，开启会**真正改变 rope 行为**（首次生效），因此长文质量必须重新评测后再上生产。
- `YARN_FACTOR`：`MAX_MODEL_LEN × FACTOR` 即目标长度。

---

## 显存 / 并发控制

### `--max-num-seqs 8`

- `.env`：`MAX_NUM_SEQS=8`
- 并发序列上限。显存紧张时优先下调它（直接影响 KV 峰值）。

### `--max-num-batched-tokens 8192`

- `.env`：`MAX_NUM_BATCHED_TOKENS=8192`
- 单个 prefill 批的 token 上限。长 prompt 场景可酌情上调。

---

## 本模型专有参数

### `--mm-encoder-tp-mode data`

- `.env`：`MM_ENCODER_TP_MODE=data`（默认）
- NVFP4 kernel 要求特征维可被 16 整除。视觉 MLP 中间维为 **4304**，TP=2 后为 **2152**，
  用 `weights` 分片会在**加载阶段崩溃**。`data` 表示每张卡各复制一份视觉编码器。

### MTP 投机解码

- `.env`：`MTP_NUM_SPECULATIVE_TOKENS=3`（设 `0` 关闭）
- MTP 草稿头自带 248,320 行 BF16 lm_head（1.18 GiB，TP=2 下 0.59 GiB/GPU），
  每个 draft step 读一次：MTP=3 的一步里 4 次 lm_head 读取中有 3 次来自它。
- 用 `MTP_DRAFT_VOCAB` 指向按频次排序的 token 子集可裁剪该头：

```bash
python3 bench/gen_draft_corpus.py --out corpus.jsonl
python3 files/build_draft_vocab.py corpus.jsonl --out draft_vocab.txt --size 32768
# 然后 .env: MTP_DRAFT_VOCAB="/path/to/draft_vocab.txt"
```

- **输出安全**：子集外的 draft 会在校验阶段被拒绝，绝不会输出；代价只是接受率下降。

### PLE n-gram embedding

- `.env`：`PLE_OFFLOAD=false`（设 `true` 把 51B PLE 表卸载到 CPU 内存）
- PLE dtype 由 `files/detect_ple_dtype.py` 探测（`nvidia/…` 只把 FP8 PLE 写在
  `quantization_config.config_groups` 内），再由 `start.sh` 通过 `--hf-overrides` 回填。
  也可用 `PLE_EMBEDDING_DTYPE` 手动覆盖。

### FP8 dense 混合检查点（实验）

- `.env`：`FP8_DENSE=false` + `FP8_DENSE_MODEL_ID=...`
- 打开后 `start.sh` 会切换 `MODEL_ID` 并把 `files/overlay` 以 bind-mount 方式注入加载器补丁
  （**无需重建镜像**）。先构建：`files/fp8dense/build.sh`。

### `QSA_PROFILE`

- `.env`：`QSA_PROFILE=stock`（镜像默认，按 GB300 / 160 SM 调优）
- `gb10` 使用 `files/qsa_gb10/qsa.py` 的 48-SM 表（未基准前不要用于生产）；
  也可传 `files/qsa_gb10/bench_qsa_kernels.py` 输出的 profile json。

### `REQUIRE_IDLE_GPU=true`

- 启动前检查两台是否有其他进程占用 GPU，有则**直接中止**，避免显存被抢导致 OOM。

---

## NCCL / 网络参数

### `--network host`

- 必需。原因见 README「NCCL 配置详解」。

### `NCCL_*` 环境变量（由脚本下发）

```
NCCL_IB_HCA=rocep1s0f0          # .env: IB_HCA="=rocep1s0f0"（精确匹配单个设备）
NCCL_IB_GID_INDEX=3             # .env: IB_GID_INDEX（RoCE 常为 3）
NCCL_SOCKET_IFNAME=enp1s0f0np0  # .env: IFACE
```

- `IB_HCA` 里的前导 `=` 表示**精确匹配**，防止误选接向其它集群的端口。
- 本平台为双路 RoCE（双卡）：`IB_GID_INDEX` 留空 + `AF_INET` 自动选 GID + `CROSS_NIC=1`，
  实际值在 `start-official.sh` 的 `NCCL_EXTRA_ENV` 中注入（head / worker 两处 heredoc）。

### `MASTER_PORT=50000`

- 分布式/NCCL 协调端口，两台需一致且空闲。

---

## 调优建议

1. **先量后调**：用 `bench/decodebench.py` 取基线，再动参数。
2. **显存换并发**：`GPU_MEMORY_UTILIZATION` ↑ → KV ↑ → `MAX_NUM_SEQS` 可 ↑，但两台必须同值。
3. **长文先验质量**：开 fp8 KV 或 YaRN 后，跑 `bench/longctx.py` + `bench/reasoning_check.py` 回归。
4. **接受率**：MTP 用 `bench/mtp_accept.py` 看接受率，再决定是否启用 `MTP_DRAFT_VOCAB`。
5. **别改脚本**：所有开关都在 `.env`；脚本内的补丁步骤是自动判定的。
