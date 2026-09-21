# 故障排查与部署踩坑记录 — Qwen3.8-Flash-Next-NVFP4 双机部署

按「现象 → 根因 → 解决」组织。多数问题先用 `./status.sh` + `./logs.sh` 定位。

## 目录

1. [NCCL 通信故障（RoCE）](#1-nccl-通信故障roce)
2. [容器看不到 GPU](#2-容器看不到-gpu)
3. [worker 容器启动后自动退出](#3-worker-容器启动后自动退出)
4. [加载超慢 / 每次重启都重新同步](#4-加载超慢--每次重启都重新同步)
5. [fp8 KV 报 kernel 不支持](#5-fp8-kv-报-kernel-不支持)
6. [视觉输入崩溃（MM_ENCODER_TP_MODE）](#6-视觉输入崩溃mm_encoder_tp_mode)
7. [NVFP4 权重加载失败](#7-nvfp4-权重加载失败)
8. [PLE dtype 报错 / PLE 层加载失败](#8-ple-dtype-报错--ple-层加载失败)
9. [启动顺序 / 卡在 NCCL init](#9-启动顺序--卡在-nccl-init)
10. [端口不通 / 服务访问不到](#10-端口不通--服务访问不到)
11. [显存不足（OOM）](#11-显存不足oom)
12. [GPU 被占用导致启动中止](#12-gpu-被占用导致启动中止)

---

## 1. NCCL 通信故障（RoCE）

### 现象

启动卡住或报 `IB device not found` / `Failed to initialize NCCL` / 退化为 `socket`。

### 根因

`NCCL_IB_HCA` 或 `NCCL_IB_GID_INDEX` 不对；RoCE IP 没加到 ConnectX 网卡上；或用了桥接网络。

### 解决

```bash
# 两节点都执行
ibstat | grep -E "Port |LinkUp"     # 确认 HCA 名与 LinkUp
ip -o -4 addr show enp1s0f0np0      # 确认 RoCE IP 在网卡上
show_gids                            # 确认 GID 索引（RoCEv2 常为 3）

# 验证双向互通
ping -c 3 10.12.12.21   # head → worker
ping -c 3 10.12.12.11   # worker → head
```

在 `.env` 中修正 `IB_HCA` / `IB_GID_INDEX` / `IFACE`，并确保使用 `--network host`。

---

## 2. 容器看不到 GPU

### 现象

`docker exec vllm-fn nvidia-smi` 报错，或 `No CUDA GPUs are available`。

### 根因

未安装 NVIDIA Container Toolkit，或 `docker run` 缺少 `--gpus all` / `--runtime nvidia`。

### 解决

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

本套件由 `start-official.sh` 统一拼装 `docker run` 参数，通常无需手工处理。

---

## 3. worker 容器启动后自动退出

### 现象

worker 端 `docker ps` 看不到 `vllm-fn`，或刚起就退出。

### 根因

worker 缺权重（默认 rsync 分发未完成）、端口/`MASTER_PORT` 与 head 冲突、
或 worker 上已有同名容器残留。

### 解决

```bash
# 在 worker 上检查权重是否存在
ls /home/admin/.cache/huggingface/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4
# 清理残留容器（两节点）
sudo docker rm -f vllm-fn
# 重新同步（不带 --launch）
./start.sh --no-launch
```

---

## 4. 加载超慢 / 每次重启都重新同步

### 现象

每次启动都要等很久；`du -sh` 观察 worker 在重复拷贝。

### 根因

checkpoint 未在本地缓存（`--launch` 之前没有先同步），或选错了权重分发方式。

### 解决

- 默认（rsync）分发**只会同步一次**；之后 `./start.sh --launch` 会直接跳过同步。
- 若每次都要等，用 NFS 让 worker 直接从 head 读：

```bash
./start.sh --nfs        # head 通过 ConnectX 导出 HF 缓存，worker 只读挂载
```

NFS 的取舍：worker 不落盘、无需 rsync 等待，但 **head 必须常在线**，且每次冷启动
worker 都要经 ConnectX 流式读取全部分片。

---

## 5. fp8 KV 报 kernel 不支持

### 现象

设了 `KV_CACHE_DTYPE=fp8` 后启动失败，日志提示 QSA kernel 不支持 fp8。

### 根因

原生 QSA kernel 不接受 fp8 KV，需要打补丁。

### 解决

`start.sh` 在 `KV_CACHE_DTYPE=fp8*` 时会**自动**应用 `files/patch_qsa_fp8_kv.py`。
若仍失败，确认 `files/` 目录完整、且以 `./start.sh`（而非手工 `docker run`）启动。
临时退回 bf16：`.env` 设 `KV_CACHE_DTYPE=auto`。

---

## 6. 视觉输入崩溃（MM_ENCODER_TP_MODE）

### 现象

加载或首次多模态请求时崩溃，涉及 MLP 维度。

### 根因

NVFP4 kernel 要求特征维可被 16 整除。视觉 MLP 中间维 4304，TP=2 后为 2152，不能被 16 整除。

### 解决

`.env` 保持 `MM_ENCODER_TP_MODE=data`（默认，每卡复制视觉编码器）。
**不要**设为 `weights`。

---

## 7. NVFP4 权重加载失败

### 现象

加载阶段报专家/量化相关错误，或提示不支持的 dtype。

### 根因

未开启专家并行（EP），或用了不支持 NVFP4 的镜像。

### 解决

- `.env`：`ENABLE_EXPERT_PARALLEL=true`。
- 确认镜像为 `IMAGE="vllm/vllm-openai:qwen38-flash-next"`。
- 需要 modelopt 相关补丁时，`files/` 会在运行时注入
  （`patch_modelopt_mxfp8.py` / `patch_modelopt_fp8_block_moe.py`）。

---

## 8. PLE dtype 报错 / PLE 层加载失败

### 现象

PLE 层加载报 dtype 不匹配。

### 根因

`nvidia/…` 只在 `quantization_config.config_groups` 内声明 FP8 PLE，常规读取取不到。

### 解决

`start.sh` 会用 `files/detect_ple_dtype.py` 探测并回填 `--hf-overrides`。
如仍异常，在 `.env` 显式指定 `PLE_EMBEDDING_DTYPE`。
显存紧张时可用 `PLE_OFFLOAD=true` 把 51B PLE 表卸载到 CPU 内存。

---

## 9. 启动顺序 / 卡在 NCCL init

### 现象

NCCL 初始化长时间无进展，约 2 分钟后超时。

### 根因

worker 未就绪，或手工先启动了 head。

### 解决

**只启动 head**：`./start.sh --launch`，由 head 负责拉起 worker。
排查时：

```bash
ping -c 3 10.12.12.21
ibstat
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET ./start.sh --launch
```

---

## 10. 端口不通 / 服务访问不到

### 现象

`curl http://192.168.22.155:8888/health` 不通。

### 根因

服务未起、`PORT` 不一致，或未用 host 网络。

### 解决

```bash
./status.sh
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8888/health
sudo docker ps | grep vllm-fn
```

注意 `status.sh` 会先 source `.env` 取 `PORT`；若 `.env` 缺失则回退 `4109`，容易误判。

---

## 11. 显存不足（OOM）

### 现象

启动或压测时 OOM。

### 根因

`GPU_MEMORY_UTILIZATION` 过高、`MAX_NUM_SEQS` 过大，或权重分发方式导致冗余占用。

### 解决

- 下调 `GPU_MEMORY_UTILIZATION`（**两台同值**），或下调 `MAX_NUM_SEQS`。
- 参考 README 的实测：0.835 → 32.02 GiB KV。
- `PLE_OFFLOAD=true` 可把 PLE 表挪到 CPU 内存。

---

## 12. GPU 被占用导致启动中止

### 现象

启动直接中止，提示 GPU 非空闲。

### 根因

`.env` 中 `REQUIRE_IDLE_GPU=true`（默认）会在检测到有进程占用 GPU 时中止，防止显存被抢。

### 解决

```bash
nvidia-smi          # 找到占用进程
# 清理后重试；确认无冲突时也可临时设 REQUIRE_IDLE_GPU=false
```
