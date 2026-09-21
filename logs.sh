#!/usr/bin/env bash
# Platform cluster wrapper: logs Qwen3.8-Flash-Next-NVFP4 (head vllm-fn)
set -uo pipefail
cd "$(dirname "$0")"
exec docker logs -f vllm-fn "$@"
