#!/usr/bin/env bash
# ============================================================================
# prepare.sh — Qwen3.8-Flash-Next-NVFP4 双机部署环境准备（交互式菜单）
#
#   1) 依赖检查    docker / nvidia-smi / ssh / ibstat 是否就绪
#   2) 拉取镜像    IMAGE from .env
#   3) 下载权重    调用 ../download.sh（仅 head）
#   4) RoCE/IB 检查 网卡 IP、HCA、GID、双向 ping
#   5) 校验权重    ../check-weights.sh
#   6) 生成 .env   从 config.sh 复制（若尚不存在）
#
# 用法: cd deploy && ./prepare.sh
# ============================================================================
set -uo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$DEPLOY_DIR/.." && pwd)"
cd "$ROOT_DIR"

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; }

load_env() {
    if [[ -f .env ]]; then
        # shellcheck disable=SC1091
        source .env
    else
        warn ".env 不存在，先执行菜单项 6 生成。"
    fi
}

pause() { read -r -p "按回车返回菜单..." _ || true; }

check_deps() {
    echo "== 依赖检查 =="
    for c in docker ssh; do
        if command -v "$c" >/dev/null 2>&1; then ok "$c: $(command -v "$c")"; else err "$c 未安装"; fi
    done
    if command -v nvidia-smi >/dev/null 2>&1; then
        ok "nvidia-smi: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    else
        warn "nvidia-smi 不可用（当前用户或环境未暴露 GPU）"
    fi
    if command -v ibstat >/dev/null 2>&1; then
        ok "ibstat 可用"
    else
        warn "ibstat 不可用（缺 rdma-core/infiniband-diags）"
    fi
    for t in uvx hf huggingface-cli; do
        if command -v "$t" >/dev/null 2>&1; then ok "下载工具: $t"; fi
    done
}

pull_image() {
    load_env
    [[ -z "${IMAGE:-}" ]] && { err "IMAGE 未设置（检查 .env）"; return; }
    info "docker pull $IMAGE"
    docker pull "$IMAGE" && ok "镜像就绪: $IMAGE"
}

download_weights() {
    load_env
    if [[ -n "${MODEL_ID:-}" ]]; then info "MODEL_ID=$MODEL_ID"; fi
    info "调用 ../download.sh（仅下载到 head）"
    ./download.sh "$@"
}

check_roce() {
    load_env
    echo "== RoCE / IB 检查 =="
    local iface="${IFACE:-}"
    local hca="${IB_HCA:-}"
    echo "IFACE=${iface:-未设置}   IB_HCA=${hca:-未设置}   IB_GID_INDEX=${IB_GID_INDEX:-未设置}"
    if [[ -n "$iface" ]] && command -v ip >/dev/null 2>&1; then
        ip -o -4 addr show "$iface" 2>/dev/null || warn "网卡 $iface 上没有 IPv4 地址"
    fi
    if command -v ibstat >/dev/null 2>&1; then
        ibstat 2>/dev/null | grep -E "CA '|Port |State:|LinkUp" | head -20
    fi
    if command -v show_gids >/dev/null 2>&1; then
        show_gids 2>/dev/null | head -20
    else
        warn "show_gids 不可用（可选）"
    fi
    if [[ -n "${WORKER_IP:-}" ]]; then
        if ping -c 3 -W 2 "$WORKER_IP" >/dev/null 2>&1; then
            ok "head → worker ($WORKER_IP) 可达"
        else
            err "head → worker ($WORKER_IP) 不可达（检查 RoCE IP / 线缆）"
        fi
    fi
}

verify_weights() {
    if [[ -x ./check-weights.sh ]]; then
        ./check-weights.sh
    else
        load_env
        local hub="${HF_HOME:-$HOME/.cache/huggingface}/hub"
        local org="${MODEL_ID%%/*}" name="${MODEL_ID##*/}"
        local p="$hub/models--${org}--${name}"
        echo "== 权重检查 =="
        if [[ -d "$p" ]]; then
            ok "$p ($(du -sh "$p" 2>/dev/null | cut -f1))"
        else
            warn "未找到 $p"
        fi
    fi
}

gen_env() {
    if [[ -f .env ]]; then
        warn ".env 已存在，未覆盖。如需重置: rm .env && cp deploy/config.sh .env"
        return
    fi
    cp "$DEPLOY_DIR/config.sh" .env
    ok "已生成 .env（来自 deploy/config.sh）—— 请按实际环境修改 HEAD_IP/WORKER_IP/IFACE/IB_HCA"
}

menu() {
    cat <<'EOF'

============================================================
 Qwen3.8-Flash-Next-NVFP4 · 双机部署环境准备
 实例: qwen3.8-flash-next-nvfp4-tp2   端口: 8888
============================================================
  1) 依赖检查 (docker/ssh/nvidia-smi/ibstat)
  2) 拉取镜像 (docker pull IMAGE)
  3) 下载权重 (仅 head)
  4) RoCE / IB 检查
  5) 校验权重
  6) 生成 .env (来自 deploy/config.sh)
  7) 一键预检 (1 + 4 + 5)
  0) 退出
EOF
}

main() {
    while true; do
        menu
        read -r -p "请选择 [0-7]: " choice || exit 0
        case "$choice" in
            1) check_deps; pause ;;
            2) pull_image; pause ;;
            3) download_weights; pause ;;
            4) check_roce; pause ;;
            5) verify_weights; pause ;;
            6) gen_env; pause ;;
            7) check_deps; echo; check_roce; echo; verify_weights; pause ;;
            0) exit 0 ;;
            *) warn "无效选项" ;;
        esac
    done
}

main "$@"
