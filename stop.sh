#!/usr/bin/env bash
# Platform cluster wrapper: stop Qwen3.8-Flash-Next-NVFP4 (vllm-fn 双机)
set -uo pipefail
cd "$(dirname "$0")"
exec ./stop-official.sh "$@"
