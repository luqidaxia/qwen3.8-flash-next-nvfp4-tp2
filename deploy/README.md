# 部署手册 — Qwen3.8-Flash-Next-NVFP4（双机 DGX Spark）

本手册覆盖：**环境准备 → 下载权重 → 配置 → 启动 → 验证 → 停止**，以及它在
**LexData AI 平台**上作为实例 `qwen3.8-flash-next-nvfp4-tp2` 的纳管方式。

- 硬件拓扑、参数与实测数据见上级 [`../README.md`](../README.md)
- 参数逐条解释见 [`../vllm-args.md`](../vllm-args.md)
- 故障排查见 [`../troubleshooting.md`](../troubleshooting.md)

---

## 0. 目标与事实表

| 项目 | 值 |
|---|---|
| 平台实例 | `qwen3.8-flash-next-nvfp4-tp2` |
| 引擎 / 镜像 | `vLLM-qwen38-flash-next` / `vllm/vllm-openai:qwen38-flash-next` |
| 服务端口 | `8888` |
| 模型 id | `qwen3.8-flash-next` |
| 健康检查 | `http://127.0.0.1:8888/health` |
| 访问地址 | `http://192.168.22.155:8888` |
| 权重目录 | `/data/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` |
| HF 缓存 | `/home/admin/.cache/huggingface/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4` |
| 容器名 | `vllm-fn`（head / worker 同名） |
| 节点 | `dgx-head`（10.12.12.11 / 192.168.22.155）、`dgx-work01`（10.12.12.21 / 192.168.22.91） |

---

## 1. 环境准备

```bash
cd deploy
./prepare.sh          # 交互菜单
```

菜单能力：

| 选项 | 作用 |
|---|---|
| 1 | 依赖检查：docker / ssh / nvidia-smi / ibstat / 下载工具 |
| 2 | `docker pull $IMAGE` |
| 3 | 下载权重（调用 `../download.sh`，仅 head） |
| 4 | RoCE / IB 检查：网卡 IP、HCA、GID、head→worker ping |
| 5 | 校验权重（`../check-weights.sh`） |
| 6 | 生成 `.env`（来自 `deploy/config.sh`） |
| 7 | 一键预检（1 + 4 + 5） |

---

## 2. 下载权重

### 2.1 走平台界面（推荐）

**模型管理 → 模型下载**，来源选 **ModelScope**（国内更快）或 **Hugging Face**：

```
仓库:   Mia-AiLab/Qwen3.8-Flash-Next-NVFP4
目标:   /data/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4
```

平台下载完成后，会在 HF 缓存目录下建立对应布局（与 `/data/models` 共享数据块，
容器以 `-v <HF_HOME>:/root/.cache/huggingface` 挂载后即可被 vLLM 直接读取）。

### 2.2 命令行下载（仅 head）

```bash
./download.sh                                      # 用 .env 的 MODEL_ID
./download.sh Mia-AiLab/Qwen3.8-Flash-Next-NVFP4   # 指定仓库
./download.sh --fp8                                # Qwen/Qwen3.8-Flash-Next-FP8
```

`download.sh` **只把权重放到 head**，不会同步到 worker —— 分发在启动阶段由
`start.sh` 完成（rsync 或 NFS）。

### 2.3 检查点变体

| 仓库 | 体积 | 构成 |
|---|---|---|
| `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` | ~99 GB | **平台默认** |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | ~133 GB / 11 分片 | BF16 稠密 + NVFP4 专家 + FP8 PLE |
| `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` | ~99 GB / 37 分片 | MXFP8 注意力 + NVFP4 PLE |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | ~126 GB / 206 分片 | BF16 注意力 + FP8 PLE |

### 2.4 校验

```bash
./check-weights.sh
```

---

## 3. 配置

```bash
cp deploy/config.sh .env      # 或 cp ../.env.sample ../.env
vi .env
```

### 必须修改

| 变量 | 平台取值 | 说明 |
|---|---|---|
| `HEAD_IP` | `10.12.12.11` | head RoCE IP |
| `WORKER_IP` | `10.12.12.21` | worker RoCE IP |
| `WORKER_USER` | `admin` | worker 上的 SSH 用户 |
| `IFACE` | `enp1s0f0np0` | 节点间以太网卡 |
| `IB_HCA` | `=rocep1s0f0` | 精确匹配的单个 HCA（前导 `=` 表示精确） |
| `MODEL_ID` | `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` | 检查点 |

### 常用开关

| 变量 | 默认 | 作用 |
|---|---|---|
| `PORT` | `8888` | 服务端口 |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | `/v1/models` 的 id |
| `MAX_MODEL_LEN` | `262144` | 上下文；>262144 需 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |
| `YARN_ENABLE` | `false` | 扩到 ~1M（`YARN_FACTOR=4.0`） |
| `GPU_MEMORY_UTILIZATION` | `0.835` | 实测最优（32.02 GiB KV） |
| `KV_CACHE_DTYPE` | `fp8` | fp8 KV（自动打 QSA 补丁）；`auto`=bf16 |
| `TENSOR_PARALLEL_SIZE` | `2` | 1 GPU/节点 × 2 节点 |
| `ENABLE_EXPERT_PARALLEL` | `true` | NVFP4 必需 |
| `MM_ENCODER_TP_MODE` | `data` | 视觉编码器复制（不能用 `weights`） |
| `MTP_NUM_SPECULATIVE_TOKENS` | `3` | MTP 投机解码；`0` 关闭 |
| `NFS_SHARE` | `false` | `true` 改用 NFS 分发权重 |
| `PLE_OFFLOAD` | `false` | PLE 表卸载到 CPU 内存 |
| `REQUIRE_IDLE_GPU` | `true` | GPU 被占用时中止启动 |

完整清单见 `.env.sample`（带逐行注释）。

---

## 4. 启动

> 启动由 **head 统一驱动**：下载/同步 → 打补丁 → SSH 拉起 worker → 启动本机 rank 0。
> **不要手工在 worker 上启动容器。**

### 4.1 平台一键启动

平台按 `cluster_start_script` 调用：

```bash
/opt/lexdata-ai/cluster-scripts/qwen38-flash-next/start.sh
```

该脚本是薄封装：

```bash
exec ./start-official.sh --launch "$@"
```

### 4.2 手动启动

```bash
./start.sh                 # 下载 → rsync 到 worker → 打补丁 → 启动
./start.sh --no-download    # 权重已在 head：跳过下载
./start.sh --no-launch      # 只下载 + 同步，不启动
./start.sh --launch         # 跳过下载/同步：打补丁 + 启动
./start.sh --nfs            # NFS 分发权重（worker 不落盘）
./start.sh --no-nfs         # 强制 rsync（覆盖 .env 的 NFS_SHARE=true）
```

### 4.3 权重分发方式

| 方式 | 触发 | 特点 |
|---|---|---|
| rsync（默认） | `NFS_SHARE=false` | 首次同步到 worker，之后复用；worker 本地一份 |
| NFS | `NFS_SHARE=true` 或 `--nfs` | worker 只读挂载 head 缓存；head 必须常在线，冷启动每次流式读取 |

---

## 5. 验证

```bash
./status.sh

# 健康 + 模型列表
curl -s -o /dev/null -w 'health: %{http_code}\n' http://127.0.0.1:8888/health
curl -s http://127.0.0.1:8888/v1/models | python3 -m json.tool

# 对话
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"用一句话介绍你自己"}],"max_tokens":128}'

# 性能 / 长文 / MTP 接受率
python3 bench/decodebench.py --url http://127.0.0.1:8888 --model qwen3.8-flash-next
python3 bench/longctx.py     --url http://127.0.0.1:8888 --model qwen3.8-flash-next
python3 bench/mtp_accept.py  --url http://127.0.0.1:8888 --model qwen3.8-flash-next
python3 bench/reasoning_check.py --url http://127.0.0.1:8888 --model qwen3.8-flash-next

# 确认两个 rank 都加入
docker logs vllm-fn 2>&1 | grep -E "rank .* nranks"    # 预期 rank 0/1，nranks 2
```

---

## 6. 停止 / 观察

```bash
./stop.sh              # = ./stop-official.sh，停止双机容器
./stop.sh --nfs        # 停止并清理 NFS 导出
./stop.sh --all        # 全部清理
./logs.sh              # docker logs -f vllm-fn（head）
sudo docker ps | grep vllm-fn
```

平台侧对应 `cluster_stop_script = /opt/lexdata-ai/cluster-scripts/qwen38-flash-next/stop.sh`。

---

## 7. 平台纳管字段速查

| 平台字段 | 取值 |
|---|---|
| `name` | `qwen3.8-flash-next-nvfp4-tp2` |
| `engine_id` | `vLLM-qwen38-flash-next` |
| `deploy_mode` | `cluster` |
| `cluster_start_script` | `/opt/lexdata-ai/cluster-scripts/qwen38-flash-next/start.sh` |
| `cluster_stop_script` | `/opt/lexdata-ai/cluster-scripts/qwen38-flash-next/stop.sh` |
| `cluster_health_url` | `http://127.0.0.1:8888/health` |
| `cluster_access_url` | `http://192.168.22.155:8888` |
| `model_path` | `/data/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` |

平台向脚本注入：`HEAD_IP` / `WORKER_IP` / `WORKER_SSH_HOST` / `WORKER_SSH_PW`。

---

## 8. 冷机上线 Checklist

1. 两台 Spark 的 RoCE IP 已配置，`ping` 双向通。
2. head 上镜像 `vllm/vllm-openai:qwen38-flash-next` 已存在（`docker images | grep qwen38`）。
3. head 上权重已落地（`/data/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`，且 HF 缓存布局存在）。
4. `.env` 中 `HEAD_IP` / `WORKER_IP` / `WORKER_USER` / `IFACE` / `IB_HCA` 与现场一致。
5. 两台 GPU 空闲（`nvidia-smi`）。
6. `./start.sh --launch`（平台点「启动」）。
7. `./status.sh` → `/health` 200、`/v1/models` 含 `qwen3.8-flash-next`。
8. 容器日志出现 `rank 0/1 nranks 2`。
