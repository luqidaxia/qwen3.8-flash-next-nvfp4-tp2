#!/usr/bin/env bash
# Platform cluster wrapper: Qwen3.8-Flash-Next-NVFP4 (官方 MiaAI dual recipe c2325b2 + 本地 NCCL 适配)
# Called by Lexdata-AI-Platform with HEAD_IP/WORKER_IP/WORKER_SSH_HOST/WORKER_SSH_PW env.
# 本地适配摘要:
#   - .env: 双路RoCE (IB_HCA 双卡 + IB_GID_INDEX 空 + AF_INET 自动GID + CROSS_NIC=1)
#   - start-official.sh: 补丁①必需变量去掉 IB_GID_INDEX(空值会卡校验)
#                        补丁②worker/head 两处 heredoc 注入 NCCL_EXTRA_ENV
#   - 权重: /data/models/nvidia/Qwen3.8-Flash-Next-NVFP4 (flat) → /data/hf/hub 符号链接布局
#   - 平台 wrapper: start.sh/stop.sh/status.sh/logs.sh
set -uo pipefail
cd "$(dirname "$0")"
exec ./start-official.sh --launch "$@"
