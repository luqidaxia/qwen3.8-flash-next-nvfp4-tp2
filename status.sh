#!/usr/bin/env bash
# Platform cluster wrapper: status Qwen3.8-Flash-Next-NVFP4 @4109
set -uo pipefail
cd "$(dirname "$0")"
source .env 2>/dev/null
PORT="${PORT:-4109}"
echo "== health =="
curl -s -o /dev/null -w "http %{http_code}\n" --max-time 5 "http://127.0.0.1:${PORT}/health" 2>&1 || echo "health 不可达"
echo "== /v1/models =="
curl -s --max-time 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print([m['id'] for m in d.get('data',[])])" 2>/dev/null || echo "(模型列表不可达)"
echo "(docker 详情需 root: sudo docker ps | grep vllm-fn)"
