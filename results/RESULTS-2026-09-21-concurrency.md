# Qwen3.8-Flash-Next（NVFP4，TP=2）并发梯度压测 — 1 / 2 / 4 / 6

> 2026-09-21 12:17–12:25，对 2 节点 RoCE 直连、TP=2 的 `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`
> 实例做 1 / 2 / 4 / 6 四档并发梯度压测，经平台并发测试入口发起，**24 个请求全部成功，0 失败**。

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| 模型 | `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`（served name `qwen3.8-flash-next`） |
| 平台实例 | id=1 `qwen3.8-flash-next-nvfp4-tp2`，`deploy_mode=cluster` |
| 拓扑 | HEAD `10.12.12.11` + WORKER `10.12.12.21`，`--nnodes 2` / `--tensor-parallel-size 2` |
| 推理服务 | `http://127.0.0.1:8888`，`/health` 200，容器 `vllm-fn`（`vllm/vllm-openai:qwen38-flash-next`） |
| 测试入口 | 平台并发测试 `http://127.0.0.1:8889/api/concurrent-test`（平台后端转发至 8888） |
| 采样方式 | 每档启动 `concurrency` 个 worker，各发 **1 个随机问题**（SSE 流式），等全部返回后统计 |
| 采样轮次 | 每档 1 轮 |

### 引擎启动参数（运行中容器的实际 `Cmd`）

```
--tensor-parallel-size 2          --max-model-len 262144
--gpu-memory-utilization 0.835    --max-num-seqs 8
--max-num-batched-tokens 8192     --kv-cache-dtype fp8
--enable-chunked-prefill          --load-format safetensors --safetensors-load-strategy lazy
--speculative-config {"method":"mtp","num_speculative_tokens":3}
--enable-expert-parallel --all2all-backend allgather_reducescatter
--compilation-config {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}
--nnodes 2 --master-addr 10.12.12.11 --master-port 50000
```

`--max-num-seqs 8` ≥ 本次最高并发 6，因此各档均未触及调度上限，梯度反映的是模型自身的
decode 吞吐特性，而非排队截断。

## 2. 测试数据

| 并发 | 请求数 | 成功 / 失败 | 总 tokens | 会话耗时 (s) | 单流速度 (tok/s) | 合计吞吐 (tok/s) |
|------|--------|-------------|-----------|--------------|------------------|------------------|
| 1 | 1 | 1 / 0 | 3164 | 46 | 73.05 | **73.05** |
| 2 | 2 | 2 / 0 | 8978 | 106 | 56.08 | **112.16** |
| 4 | 4 | 4 / 0 | 19576 | 146 | 49.59 | **198.38** |
| 6 | 6 | 6 / 0 | 13530 | 127 | 40.29 | **241.75** |

指标口径：

- **单流速度** = 各 worker 本次请求的生成速度（completion tokens / 解码耗时）的算术平均；
- **合计吞吐** = 各 worker 生成速度之和，即该并发下的聚合 decode 吞吐；
- **会话耗时** = 会话启动到全部 worker 请求返回的墙钟时间，含 prefill 与排队。

## 3. 结论

1. **聚合吞吐随并发单调上升**：73.05 → 112.16 → 198.38 → 241.75 tok/s，
   较单流提升 1.53× / 2.72× / 3.31×。
2. **单流速度随并发衰减**：73.05 → 56.08 → 49.59 → 40.29 tok/s，6 并发时为单流的 55%。
   衰减主要来自 MTP 投机解码的 batch 内批处理与 KV 带宽竞争。
3. **4 → 6 档收益开始收窄**：吞吐 +21.9%（198.38 → 241.75），低于 2 → 4 档的 +76.8%。
   结合 `--max-num-seqs 8`，预计 6 → 8 进一步收窄。
4. **稳定性**：四档共 24 请求全部成功，无超时、无 OOM。

## 4. 口径说明与局限

- 每档使用**随机问题**，各组生成长度不可比（如 C=1 为 3164 tokens / C=4 为 19576 tokens），
  因此**不宜用 `总 tokens ÷ 会话耗时` 反算吞吐**（该值会被 prompt 长度与 prefill 时间污染）；
  横向对比应以「单流 vs 合计吞吐」随并发的相对趋势为准。
- 每档仅 1 轮采样，属**快速梯度**而非多 cell 矩阵压测，未测 TTFT 分位与固定输入输出长度矩阵。
- 如需与 TP=4 / 4 节点拓扑横向对比，建议改用固定 prompt 长度 × 固定 `max_tokens` 的矩阵脚本。

## 5. 复现步骤

1. 平台登录（admin）后启动实例 1，等待 `/health` 返回 200 且权重加载完成；
2. 对每个并发档位调用一次：

```bash
# 启动一档并发压测
curl -s -b cookie.txt -X POST http://127.0.0.1:8889/api/concurrent-test/start \
     -H 'Content-Type: application/json' \
     -d '{"model_id":1,"concurrency":6,"question_mode":"random"}'
# 轮询该 session_id 的 status，直到 running=false，再取 total_requests / success_count / workers[].last_speed
curl -s -b cookie.txt "http://127.0.0.1:8889/api/concurrent-test/status?session_id=<SID>"
```

3. 档位之间确认引擎空闲再开始下一档：

```bash
curl -s http://127.0.0.1:8888/metrics | grep -E '^vllm:num_requests_(running|waiting)\{'
```

---
*测试时间：2026-09-21 12:17:50 / 12:18:41 / 12:20:34 / 12:23:06（对应 C=1/2/4/6）*
