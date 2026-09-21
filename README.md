# Dual DGX Spark Deployment: Qwen3.8-Flash-Next-NVFP4

Two-node (**2 × DGX Spark / GB10**) vLLM deployment of **Qwen3.8-Flash-Next-NVFP4** with
tensor parallelism **TP=2 + expert parallelism (EP)**, **FP8 KV cache**, **MTP=3** speculative
decoding and optional **~1M context (YaRN)**.

On the **LexData AI platform** this model runs as the managed cluster instance
**`qwen3.8-flash-next-nvfp4-tp2`** — this repository is the download / start / stop / verify
runbook for it, plus the exact scripts the platform calls.

| | |
|---|---|
| Instance | `qwen3.8-flash-next-nvfp4-tp2` |
| Engine / image | `vllm/vllm-openai:qwen38-flash-next` |
| Endpoint | `http://192.168.22.155:8888` (OpenAI-compatible) |
| Model id | `qwen3.8-flash-next` |
| Health | `GET /health` |
| Weights | `/data/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` |

> 中文文档见下半部分。完整部署手册见 [`deploy/README.md`](deploy/README.md)，
> 参数逐条解释见 [`vllm-args.md`](vllm-args.md)，踩坑排查见 [`troubleshooting.md`](troubleshooting.md)，
> 并发压测结果见 [`results/`](results/)。

---

## Hardware Topology

```
        ┌───────────────────────────────┐       ┌───────────────────────────────┐
        │  HEAD  dgx-head               │       │  WORKER  dgx-work01           │
        │  DGX Spark / GB10  128GB UMA  │       │  DGX Spark / GB10  128GB UMA  │
        │  RoCE   10.12.12.11   ────────┼───────┼────────   10.12.12.21  RoCE    │
        │  Mgmt   192.168.22.155        │       │  Mgmt   192.168.22.91         │
        │  NIC    enp1s0f0np0 (CX-7)    │       │  NIC    enp1s0f0np0 (CX-7)    │
        └───────────────────────────────┘       └───────────────────────────────┘
                        └────────── ConnectX-7 RoCE ──────────┘
                            TP=2 · EP · NCCL over RoCE
```

| Item | Head (`dgx-head`) | Worker (`dgx-work01`) |
|---|---|---|
| GPU | GB10 (1 logical GPU, 128GB unified) | GB10 (1 logical GPU, 128GB unified) |
| Role | vLLM rank 0, API server, SSH driver | vLLM rank 1 |
| RoCE IP | `10.12.12.11` | `10.12.12.21` |
| Mgmt IP | `192.168.22.155` | `192.168.22.91` |
| NIC | `enp1s0f0np0` | `enp1s0f0np0` |
| SSH user | `admin` | `admin` |
| Container | `vllm-fn` | `vllm-fn` |

`--tensor-parallel-size 2` maps to **1 GPU per Spark**; NCCL moves activations over ConnectX-7 RoCE.

## Directory Structure

```
qwen3.8-flash-next-nvfp4-tp2/
├── README.md              # 本文件（中英双语）
├── deploy/
│   ├── README.md          # 详细部署手册（下载 / 启动 / 停止 / 验证 / 纳管）
│   ├── config.sh          # 配置模板 → 复制为 ../.env
│   └── prepare.sh         # 环境准备交互菜单（拉镜像 / 下权重 / 查 RoCE）
├── vllm-args.md           # vLLM 参数逐条详解
├── troubleshooting.md     # 故障排查（现象 / 根因 / 解决）
├── .env.sample            # 全部可用变量的带注释示例
├── start.sh               # ★ 平台纳管入口 → ./start-official.sh --launch
├── start-official.sh      # 双机启动主脚本（下载→同步→打补丁→启动）
├── stop.sh                # 平台纳管入口 → ./stop-official.sh
├── status.sh              # 健康 + /v1/models 检查
├── logs.sh                # docker logs -f vllm-fn
├── download.sh            # 仅下载权重到 head（--fp8 | org/repo）
├── check-weights.sh       # 校验权重完整性
├── start-fp8.sh           # 官方 FP8 检查点（NFS 共享）
├── start-tp1.sh           # 单 Spark TP=1（PLE offload）
├── stop-official.sh       # 停止双机容器
├── files/                 # 容器内即时生效的补丁与工具（无需重建镜像）
├── bench/                 # 压测 / 上下文 / MTP 接受率 / 推理抽检
├── results/               # 压测结果快照（Markdown）
├── docs/                  # 设计与移交笔记
└── tp1/                   # 单卡自包含子目录
```

## Quick Start

### Prerequisites

- 2 × DGX Spark (GB10), 128GB unified memory each, connected directly over ConnectX-7 RoCE.
- Docker + NVIDIA Container Toolkit on **both** nodes; SSH from head to worker.
- Head has the image `vllm/vllm-openai:qwen38-flash-next` and the checkpoint cached.
- An idle GPU on both nodes (`REQUIRE_IDLE_GPU=true` aborts the launch otherwise).

### 1. Environment setup (interactive menu)

```bash
cd deploy && ./prepare.sh
```

### 2. Configuration

```bash
cp .env.sample .env        # or: cp deploy/config.sh .env
vi .env
```

Minimum to edit: `HEAD_IP`, `WORKER_IP`, `WORKER_USER`, `IFACE`, `IB_HCA`, `MODEL_ID`.

### 3. Configure RoCE network

```bash
# Head
sudo ip addr add 10.12.12.11/24 dev enp1s0f0np0
# Worker
sudo ip addr add 10.12.12.21/24 dev enp1s0f0np0
# Verify bidirectional connectivity
ping -c 3 10.12.12.21      # from head
ping -c 3 10.12.12.11      # from worker
ibstat | grep -E "Port |LinkUp"
```

### 4. Launch

```bash
# The platform calls this exact path:
/opt/lexdata-ai/cluster-scripts/qwen38-flash-next/start.sh

# Manual, from the repo root:
./start.sh --launch        # weights already cached → patch + launch
./start.sh                 # full path: download → sync to worker → patch → launch
./start.sh --nfs           # distribute weights over NFS (no worker copy)
```

The head node drives everything: it patches the container, starts the worker container over
SSH, then starts its own rank 0. **Do not start the worker manually.**

### 5. Verify deployment

```bash
./status.sh
curl -s http://127.0.0.1:8888/v1/models | python3 -m json.tool
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'

python3 bench/decodebench.py --url http://127.0.0.1:8888 --model qwen3.8-flash-next
docker logs vllm-fn 2>&1 | grep -E "rank .* nranks"     # expected: rank 0/1, nranks 2
```

## Lessons Learned

- **NVFP4 requires expert parallelism.** `ENABLE_EXPERT_PARALLEL=true` is mandatory; plain TP=2 without EP will not load.
- **Vision encoder must be TP-replicated.** `MM_ENCODER_TP_MODE=data`. NVFP4 kernels need feature dims divisible by 16; the vision MLP intermediate is 4304 → 2152 after TP=2, so `weights` sharding crashes at load time.
- **Stock QSA kernels refuse FP8 KV.** `KV_CACHE_DTYPE=fp8` (default) requires `files/patch_qsa_fp8_kv.py`, applied automatically by `start.sh`. It buys **1.70×** cache density per GiB (13.93× vs 8.13× @262K) with no measured decode cost.
- **PLE dtype is declared inconsistently.** `nvidia/…` puts the FP8 PLE dtype only inside `quantization_config.config_groups`; `start.sh` recovers it with `files/detect_ple_dtype.py` and re-injects it via `--hf-overrides`.
- **YaRN was a silent no-op** before Sep 2026 — the override sat at the top level of `--hf-overrides` where it never reached `text_config`. Now fixed; enabling it genuinely changes rope, so re-validate long-context quality.
- **MTP carries its own 248,320-row lm_head** (1.18 GiB; 0.59 GiB/GPU at TP=2) and reads it once per draft step — three of the four lm_head reads in an MTP=3 step. Slicing it via `MTP_DRAFT_VOCAB` cuts that traffic; drafts outside the subset are rejected at verification (output-safe, costs acceptance).
- **All patches are bind-mounted, not baked in.** `files/` is overlaid at `docker run` time, so the image never needs a rebuild.

### Performance (measured)

| Metric | Value |
|---|---|
| `--gpu-memory-utilization` | `0.835` |
| KV cache size | **32.02 GiB** |
| KV cache tokens (fp8) | **3,652,200** → **13.93×** @262K |
| KV cache tokens (bf16, `KV_CACHE_DTYPE=auto`) | 2,131,159 → 8.13× |
| Sensitivity | +0.01 GMU ≈ +0.83 GiB ≈ +95k fp8 cache tokens |
| MTP | `MTP_NUM_SPECULATIVE_TOKENS=3` |
| Reasoning / needle regression (fp8 KV) | 12/12 both |

### Benchmark: concurrency sweep (measured)

Raw data: [`results/RESULTS-2026-09-21-concurrency.md`](results/RESULTS-2026-09-21-concurrency.md).

Measured **2026-09-21** on the live instance (fp8 KV, MTP=3, `--max-num-seqs 8`), one round per
level; each level starts `concurrency` workers, each sending one **random** question over SSE:

| Concurrency | Requests | OK / Failed | Total tokens | Session (s) | Per-stream (tok/s) | Aggregate (tok/s) |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 / 0 | 3,164 | 46 | 73.05 | **73.05** |
| 2 | 2 | 2 / 0 | 8,978 | 106 | 56.08 | **112.16** |
| 4 | 4 | 4 / 0 | 19,576 | 146 | 49.59 | **198.38** |
| 6 | 6 | 6 / 0 | 13,530 | 127 | 40.29 | **241.75** |

- **Aggregate throughput rises with concurrency**: 73.05 → 112.16 → 198.38 → 241.75 tok/s
  (**1.53× / 2.72× / 3.31×** over a single stream).
- **Per-stream speed decays**: 73.05 → 40.29 tok/s; at C=6 it is 55% of the single-stream rate
  (batch-internal competition of MTP speculative decoding plus KV bandwidth).
- **4 → 6 tapers off**: +21.9%, versus +76.8% for 2 → 4. With `--max-num-seqs 8`, 6 → 8 should
  taper further.
- **Stability**: 24/24 requests succeeded — no timeouts, no OOM.
- Caveat: questions are random, so per-level token totals are not comparable. Read the
  per-stream vs aggregate **trend**, not `total tokens ÷ session time` (prefill-polluted).

## Key vLLM Parameters

- `--tensor-parallel-size 2` — 1 GPU per Spark × 2 nodes, with `--enable-expert-parallel`.
- `--max-model-len 262144` — native context. Raise it **and** set `YARN_ENABLE=true` (`YARN_FACTOR=4.0` → ~1M); `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` is required above 262144.
- `--gpu-memory-utilization 0.835` — the measured sweet spot (32.02 GiB KV); the worker must match.
- `--max-num-seqs 8 --max-num-batched-tokens 8192`.

Full reference: [`vllm-args.md`](vllm-args.md).

## NCCL Configuration

### Why `--network host` is mandatory

NCCL must see the real ConnectX-7 devices and RoCE IPs. With bridge networking the container
gets a private IP RoCE cannot route, and NCCL hangs in `init` or silently falls back to socket.

### NCCL environment variables

```
NCCL_IB_HCA=rocep1s0f0
NCCL_IB_GID_INDEX=3
NCCL_SOCKET_IFNAME=enp1s0f0np0
```

### GID_INDEX selection logic

```bash
ibstat | grep -E "Port |Base lid"
show_gids            # GID table per device
```

On ConnectX with RoCEv2 the IPv4-mapped RoCE GID is usually index `3`. A wrong index gives
`IB device not found` or an init timeout.

### NCCL debugging

```bash
ibstat                               # RoCE devices
show_gids                            # GID table
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
```

### Common NCCL errors

| Error | Likely cause | Fix |
|---|---|---|
| `IB device not found` | wrong `NCCL_IB_HCA` / GID index | `ibstat`, set both |
| Hang in `ncclInternalInit` | RoCE IP not on NIC / no route | `ip addr`, bidirectional `ping` |
| Falls back to `socket` | bridge networking | use `--network host` |
| Timeout after ~2 min | worker started after head | let the head start the worker |

Full list: [`troubleshooting.md`](troubleshooting.md).

## Obtaining Model Weights

```bash
./download.sh                                        # MODEL_ID from .env
./download.sh Mia-AiLab/Qwen3.8-Flash-Next-NVFP4     # explicit repo
./download.sh --fp8                                  # Qwen/Qwen3.8-Flash-Next-FP8
```

Or from the platform UI: **模型管理 → 模型下载 → 来源 = ModelScope →**
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` → `/data/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`.

| Repo | Size | Make-up |
|---|---|---|
| `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` | ~99 GB | **platform default** |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | ~133 GB / 11 shards | BF16 dense + NVFP4 experts + FP8 PLE |
| `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` | ~99 GB / 37 shards | MXFP8 attention + NVFP4 PLE |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | ~126 GB / 206 shards | BF16 attention + FP8 PLE |

Default distribution is **head rsync → worker** (one-time); `--nfs` streams from the head
instead and keeps the worker copy-free.

## Platform-managed deployment

| Platform field | Value |
|---|---|
| `cluster_start_script` | `/opt/lexdata-ai/cluster-scripts/qwen38-flash-next/start.sh` |
| `cluster_stop_script` | `/opt/lexdata-ai/cluster-scripts/qwen38-flash-next/stop.sh` |
| `cluster_health_url` | `http://127.0.0.1:8888/health` |
| `cluster_access_url` | `http://192.168.22.155:8888` |
| `model_path` | `/data/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` |
| Engine / image | `vLLM-qwen38-flash-next` / `vllm/vllm-openai:qwen38-flash-next` |

The platform injects `HEAD_IP` / `WORKER_IP` / `WORKER_SSH_HOST` / `WORKER_SSH_PW` into the
script environment; `start.sh` is a thin wrapper around `./start-official.sh --launch "$@"`.
Cold-node flow: **模型下载** (ModelScope → `/data/models/…`) → **一键启动** (platform runs
`start.sh`) → **健康检查** (`/health`) → **访问** (`/v1/chat/completions`).

## Author

LexData AI Platform — 双机 DGX Spark 部署参考实现。

## License

Apache-2.0 (see [`LICENSE`](LICENSE)). Upstream vLLM is Apache-2.0; the NVFP4 checkpoint
follows the licence of its publishing repository.

---
---

# DGX Spark 双机部署 Qwen3.8-Flash-Next-NVFP4

使用 **2 台 DGX Spark（GB10）** 部署 **Qwen3.8-Flash-Next-NVFP4**：张量并行
**TP=2 + 专家并行（EP）**、**FP8 KV cache**、**MTP=3 投机解码**，可选 **~1M 上下文（YaRN）**。

在 **LexData AI 平台** 上，本模型以集群实例 **`qwen3.8-flash-next-nvfp4-tp2`** 纳管运行。
本仓库即为它的**下载 / 启动 / 停止 / 验证**完整说明，并附带平台实际调用的脚本。

| | |
|---|---|
| 实例名 | `qwen3.8-flash-next-nvfp4-tp2` |
| 引擎 / 镜像 | `vllm/vllm-openai:qwen38-flash-next` |
| 服务地址 | `http://192.168.22.155:8888`（OpenAI 兼容） |
| 模型 id | `qwen3.8-flash-next` |
| 健康检查 | `GET /health` |
| 权重目录 | `/data/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` |

## 硬件拓扑

```
        ┌───────────────────────────────┐       ┌───────────────────────────────┐
        │  HEAD  dgx-head               │       │  WORKER  dgx-work01           │
        │  DGX Spark / GB10  128GB 统一内存 │    │  DGX Spark / GB10  128GB 统一内存 │
        │  RoCE   10.12.12.11   ────────┼───────┼────────   10.12.12.21  RoCE    │
        │  管理   192.168.22.155        │       │  管理   192.168.22.91         │
        │  网卡   enp1s0f0np0 (CX-7)    │       │  网卡   enp1s0f0np0 (CX-7)    │
        └───────────────────────────────┘       └───────────────────────────────┘
                        └────────── ConnectX-7 RoCE ──────────┘
                             TP=2 · EP · NCCL 走 RoCE
```

| 项目 | Head（`dgx-head`） | Worker（`dgx-work01`） |
|---|---|---|
| GPU | GB10（1 逻辑 GPU，128GB 统一内存） | GB10（1 逻辑 GPU，128GB 统一内存） |
| 角色 | vLLM rank 0、API 服务、SSH 驱动端 | vLLM rank 1 |
| RoCE IP | `10.12.12.11` | `10.12.12.21` |
| 管理 IP | `192.168.22.155` | `192.168.22.91` |
| 网卡 | `enp1s0f0np0` | `enp1s0f0np0` |
| SSH 用户 | `admin` | `admin` |
| 容器名 | `vllm-fn` | `vllm-fn` |

`--tensor-parallel-size 2` 即**每台 Spark 1 张 GPU**；NCCL 通过 ConnectX-7 RoCE 传输激活值。

## 目录结构

```
qwen3.8-flash-next-nvfp4-tp2/
├── README.md              # 本文件（中英双语）
├── deploy/
│   ├── README.md          # 详细部署手册（下载 / 启动 / 停止 / 验证 / 纳管）
│   ├── config.sh          # 配置模板 → 复制为 ../.env
│   └── prepare.sh         # 环境准备交互菜单（拉镜像 / 下权重 / 查 RoCE）
├── vllm-args.md           # vLLM 参数逐条详解
├── troubleshooting.md     # 故障排查（现象 / 根因 / 解决）
├── .env.sample            # 全部可用变量的带注释示例
├── start.sh               # ★ 平台纳管入口 → ./start-official.sh --launch
├── start-official.sh      # 双机启动主脚本（下载→同步→打补丁→启动）
├── stop.sh                # 平台纳管入口 → ./stop-official.sh
├── status.sh              # 健康 + /v1/models 检查
├── logs.sh                # docker logs -f vllm-fn
├── download.sh            # 仅下载权重到 head（--fp8 | org/repo）
├── check-weights.sh       # 校验权重完整性
├── start-fp8.sh           # 官方 FP8 检查点（NFS 共享）
├── start-tp1.sh           # 单 Spark TP=1（PLE offload）
├── stop-official.sh       # 停止双机容器
├── files/                 # 容器内即时生效的补丁与工具（无需重建镜像）
├── bench/                 # 压测 / 上下文 / MTP 接受率 / 推理抽检
├── results/               # 压测结果快照（Markdown）
├── docs/                  # 设计与移交笔记
└── tp1/                   # 单卡自包含子目录
```

## 快速开始

### 前置条件

- 2 台 DGX Spark（GB10），各 128GB 统一内存，用 ConnectX-7 RoCE 直连。
- **两台**都装好 Docker + NVIDIA Container Toolkit；head 能 SSH 到 worker。
- head 上已具备镜像 `vllm/vllm-openai:qwen38-flash-next` 与权重缓存。
- 两台 GPU 均空闲（`REQUIRE_IDLE_GPU=true` 会因 GPU 被占用而中止启动）。

### 1. 环境准备（交互式菜单）

```bash
cd deploy && ./prepare.sh
```

菜单：拉取镜像 / 下载权重 / 检查 RoCE 与 IB / 校验权重 / 环境预检。

### 2. 配置

```bash
cp .env.sample .env        # 或 cp deploy/config.sh .env
vi .env
```

至少修改：`HEAD_IP`、`WORKER_IP`、`WORKER_USER`、`IFACE`、`IB_HCA`、`MODEL_ID`。

### 3. 配置 RoCE 网络

```bash
# Head
sudo ip addr add 10.12.12.11/24 dev enp1s0f0np0
# Worker
sudo ip addr add 10.12.12.21/24 dev enp1s0f0np0
# 验证双向互通
ping -c 3 10.12.12.21      # head 上执行
ping -c 3 10.12.12.11      # worker 上执行
ibstat | grep -E "Port |LinkUp"
```

### 4. 启动

```bash
# 平台实际调用的路径：
/opt/lexdata-ai/cluster-scripts/qwen38-flash-next/start.sh

# 手动执行（仓库根目录）：
./start.sh --launch        # 权重已缓存：仅打补丁 + 启动
./start.sh                 # 完整流程：下载 → 同步到 worker → 打补丁 → 启动
./start.sh --nfs           # 用 NFS 分发权重（worker 不落盘）
```

启动由 head 统一驱动：先给容器打补丁，再通过 SSH 拉起 worker 容器，最后启动自己的
rank 0。**不要手工启动 worker。**

### 5. 验证部署

```bash
./status.sh
curl -s http://127.0.0.1:8888/v1/models | python3 -m json.tool
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'

python3 bench/decodebench.py --url http://127.0.0.1:8888 --model qwen3.8-flash-next
docker logs vllm-fn 2>&1 | grep -E "rank .* nranks"     # 预期 rank 0/1，nranks 2
```

## 踩坑经验

- **NVFP4 必须开专家并行。** `ENABLE_EXPERT_PARALLEL=true`，否则单纯 TP=2 无法加载。
- **视觉编码器必须按 TP 复制。** `MM_ENCODER_TP_MODE=data`。NVFP4 要求特征维可被 16 整除，
  视觉 MLP 中间维 4304 在 TP=2 后为 2152，用 `weights` 分片会在加载时崩溃。
- **原生 QSA kernel 拒绝 FP8 KV。** `KV_CACHE_DTYPE=fp8`（默认）依赖
  `files/patch_qsa_fp8_kv.py`，由 `start.sh` 自动应用；每 GiB 缓存密度提升 **1.70×**
  （262K 下 13.93× vs 8.13×），且实测无解码开销。
- **PLE 数据类型声明不统一。** `nvidia/…` 只把 FP8 PLE 写在
  `quantization_config.config_groups` 里，`start.sh` 用 `files/detect_ple_dtype.py`
  探测后通过 `--hf-overrides` 回填。
- **YaRN 曾是静默无效**：覆盖项写在 `--hf-overrides` 顶层，从未进入 `text_config`。
  现已修复，开启会真正改变 rope，务必重新验证长文质量。
- **MTP 自带 248,320 行 lm_head**（1.18 GiB，TP=2 下 0.59 GiB/GPU），每个 draft step 读一次
  ——MTP=3 一步中有 4 次 lm_head 读取里的 3 次。用 `MTP_DRAFT_VOCAB` 裁剪可省这部分带宽；
  子集外的 draft 会在校验阶段被拒（**不影响输出正确性**，代价是接受率）。
- **所有补丁都是 bind-mount，不是打进镜像。** `files/` 在 `docker run` 时叠加，改行为无需重建镜像。

### 实测性能

| 指标 | 数值 |
|---|---|
| `--gpu-memory-utilization` | `0.835` |
| KV cache 容量 | **32.02 GiB** |
| KV cache token（fp8） | **3,652,200** → **13.93×** @262K |
| KV cache token（bf16，`KV_CACHE_DTYPE=auto`） | 2,131,159 → 8.13× |
| 敏感度 | +0.01 GMU ≈ +0.83 GiB ≈ +95k fp8 cache token |
| MTP | `MTP_NUM_SPECULATIVE_TOKENS=3` |
| 推理 / needle 回归（fp8 KV） | 均 12/12 |

### Benchmark：并发梯度压测（实测）

原始数据见 [`results/RESULTS-2026-09-21-concurrency.md`](results/RESULTS-2026-09-21-concurrency.md)。

**2026-09-21** 在运行中的实例上实测（fp8 KV、MTP=3、`--max-num-seqs 8`），每档 1 轮：
每档启动 `concurrency` 个 worker，各发 **1 个随机问题**（SSE 流式）：

| 并发 | 请求数 | 成功 / 失败 | 总 tokens | 会话耗时 (s) | 单流速度 (tok/s) | 合计吞吐 (tok/s) |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 / 0 | 3,164 | 46 | 73.05 | **73.05** |
| 2 | 2 | 2 / 0 | 8,978 | 106 | 56.08 | **112.16** |
| 4 | 4 | 4 / 0 | 19,576 | 146 | 49.59 | **198.38** |
| 6 | 6 | 6 / 0 | 13,530 | 127 | 40.29 | **241.75** |

- **聚合吞吐随并发单调上升**：73.05 → 112.16 → 198.38 → 241.75 tok/s，较单流提升
  **1.53× / 2.72× / 3.31×**。
- **单流速度随并发衰减**：73.05 → 40.29 tok/s，6 并发时为单流的 55%（MTP 投机解码的批内
  竞争 + KV 带宽）。
- **4 → 6 收益收窄**：+21.9%，低于 2 → 4 的 +76.8%；在 `--max-num-seqs 8` 下 6 → 8 预计继续收窄。
- **稳定性**：24 个请求全部成功，无超时、无 OOM。
- 口径说明：每档使用随机问题，各组生成长度不可比，应看「单流 vs 合计吞吐」随并发的
  **趋势**，不宜用 `总 tokens ÷ 会话耗时` 反算吞吐（含 prefill 时间，会被污染）。

## vLLM 关键参数

- `--tensor-parallel-size 2`——每台 Spark 1 张 GPU × 2 节点，配合 `--enable-expert-parallel`。
- `--max-model-len 262144`——原生上下文。要 ~1M 需同时设 `YARN_ENABLE=true`
  （`YARN_FACTOR=4.0`），且超过 262144 时必须 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`。
- `--gpu-memory-utilization 0.835`——实测最优点（32.02 GiB KV）；worker 必须一致。
- `--max-num-seqs 8 --max-num-batched-tokens 8192`。

逐条说明见 [`vllm-args.md`](vllm-args.md)。

## NCCL 配置详解

### 为什么必须用 `--network host`

NCCL 必须看到真实的 ConnectX-7 设备与 RoCE IP。桥接网络下容器拿到私有 IP，RoCE 无法路由，
NCCL 要么卡在 `init`，要么静默退化为 socket 传输。

### NCCL 环境变量

```
NCCL_IB_HCA=rocep1s0f0
NCCL_IB_GID_INDEX=3
NCCL_SOCKET_IFNAME=enp1s0f0np0
```

### GID_INDEX 选择逻辑

```bash
ibstat | grep -E "Port |Base lid"
show_gids            # 列出各设备的 GID 表
```

ConnectX 走 RoCEv2 时，IPv4 映射的 RoCE GID 通常是索引 `3`。选错会出现
`IB device not found` 或初始化超时。

### NCCL 排查

```bash
ibstat                               # 查看 RoCE 设备
show_gids                            # 查看 GID 表
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
```

### 常见 NCCL 错误

| 报错 | 可能原因 | 解决 |
|---|---|---|
| `IB device not found` | `NCCL_IB_HCA` / GID 索引不对 | `ibstat` 后正确设置两者 |
| 卡在 `ncclInternalInit` | RoCE IP 不在网卡上 / 无路由 | `ip addr` 检查，双向 `ping` |
| 退化为 `socket` | 桥接网络 | 改用 `--network host` |
| ~2 分钟后超时 | worker 在 head 之后才启动 | 交给 head 统一拉起 |

完整清单见 [`troubleshooting.md`](troubleshooting.md)。

## 模型权重获取

```bash
./download.sh                                        # 使用 .env 的 MODEL_ID
./download.sh Mia-AiLab/Qwen3.8-Flash-Next-NVFP4     # 指定仓库
./download.sh --fp8                                  # Qwen/Qwen3.8-Flash-Next-FP8
```

或走平台界面：**模型管理 → 模型下载 → 来源 = ModelScope →**
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` → 落到 `/data/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`。

| 仓库 | 体积 | 构成 |
|---|---|---|
| `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` | ~99 GB | **平台默认** |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | ~133 GB / 11 分片 | BF16 稠密 + NVFP4 专家 + FP8 PLE |
| `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` | ~99 GB / 37 分片 | MXFP8 注意力 + NVFP4 PLE |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | ~126 GB / 206 分片 | BF16 注意力 + FP8 PLE |

默认分发方式是 **head rsync → worker**（一次性）；`--nfs` 改为从 head 流式读取，worker 不落盘。

## 平台纳管部署

| 平台字段 | 取值 |
|---|---|
| `cluster_start_script` | `/opt/lexdata-ai/cluster-scripts/qwen38-flash-next/start.sh` |
| `cluster_stop_script` | `/opt/lexdata-ai/cluster-scripts/qwen38-flash-next/stop.sh` |
| `cluster_health_url` | `http://127.0.0.1:8888/health` |
| `cluster_access_url` | `http://192.168.22.155:8888` |
| `model_path` | `/data/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` |
| 引擎 / 镜像 | `vLLM-qwen38-flash-next` / `vllm/vllm-openai:qwen38-flash-next` |

平台会向脚本注入 `HEAD_IP` / `WORKER_IP` / `WORKER_SSH_HOST` / `WORKER_SSH_PW`；
`start.sh` 只是薄封装：`exec ./start-official.sh --launch "$@"`。

冷机上线流程：**模型下载**（ModelScope → `/data/models/…`）→ **一键启动**（平台调用 `start.sh`）
→ **健康检查**（`/health`）→ **对外访问**（`/v1/chat/completions`）。

## 作者

LexData AI 平台 —— 双机 DGX Spark 部署参考实现。

## 许可证

Apache-2.0，见 [`LICENSE`](LICENSE)。上游 vLLM 为 Apache-2.0；NVFP4 权重遵循其发布仓库的许可。
