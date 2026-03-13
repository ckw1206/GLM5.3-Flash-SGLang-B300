#!/usr/bin/env bash
set -euo pipefail

# GLM-5.3-Flash on SGLang (8x B300 288GB)
#
# Recipe from the SGLang cookbook (Blackwell column):
#   https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash
#
# Notes:
#   - MoE model, 320B total / 18B active, FP8 weights (~330GB). Fits on
#     4x B300; 8 GPUs give more KV cache. Max context 1,048,576 tokens.
#   - Needs SGLang >= v0.5.20. B300 (sm_103) wants the CUDA 13 image.
#   - Blackwell path: TRT-LLM DSA backends for prefill/decode with FP8 KV
#     cache; MoE runner is flashinfer_trtllm (SGLang default on Blackwell).
#   - Parsers: --reasoning-parser glm45, --tool-call-parser glm47 (what
#     "auto" resolves to). SGLang needs no --enable-auto-tool-choice, just
#     send `tools` in the request.
#   - MTP=1 (default) turns on the built-in MTP head via EAGLE, cookbook's
#     low-latency settings (5 steps, topk 1, 6 draft tokens).

# Load optional .env overrides. Shell env wins; .env fills gaps; defaults last.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${SCRIPT_DIR}/.env"
fi

MODEL_ID="${MODEL_ID:-zai-org/GLM-5.3-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.3-Flash}"
IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.22-cu130}"
GPU_ID="${GPU_ID:-0,1,2,3,4,5,6,7}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
# Free-form extra SGLang flags, appended LAST (argparse last-wins):
#   EXTRA_ARGS="--context-length 262144" ./start.sh
read -ra EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

TP_SIZE="$(tr ',' '\n' <<< "${GPU_ID}" | grep -c '[0-9]' || true)"
case "${TP_SIZE}" in
  4|8) PARALLEL_ARGS=(--tp-size "${TP_SIZE}") ;;
  *) echo "GPU_ID='${GPU_ID}' gives ${TP_SIZE} GPUs; GLM-5.3-Flash recipe supports 4 or 8"; exit 1 ;;
esac

MTP="${MTP:-1}"
case "${MTP}" in
  0) SPEC_ARGS=() ;;
  1) SPEC_ARGS=(--speculative-algorithm EAGLE
       --speculative-num-steps 5
       --speculative-eagle-topk 1
       --speculative-num-draft-tokens 6) ;;
  *) echo "MTP must be 0 or 1, got '${MTP}'"; exit 1 ;;
esac

CONTAINER_NAME="glm-5.3-flash-sglang"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }

mkdir -p "${HF_CACHE}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting SGLang container for ${MODEL_ID}"
echo "GPUs: ${GPU_ID} (${PARALLEL_ARGS[*]})"
echo "Spec decode: $( (( MTP )) && echo "MTP (EAGLE)" || echo off)"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

echo "[$(date -Is)] launching SGLang container" > "${LOG_FILE}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --cap-add SYS_NICE \
  --gpus "\"device=${GPU_ID}\"" \
  --shm-size 32g \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  "${PARALLEL_ARGS[@]}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --dsa-prefill-backend trtllm \
  --dsa-decode-backend trtllm \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend flashinfer_trtllm \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  "${SPEC_ARGS[@]}" \
  --enable-metrics \
  --host "${HOST}" \
  --port "${PORT}" \
  "${EXTRA_ARGS_ARR[@]}" \
  >/dev/null

docker inspect -f '{{.Id}}' "${CONTAINER_NAME}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} ($(cat "${PID_FILE}"))"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

docker logs -f "${CONTAINER_NAME}" 2>&1 | tee -a "${LOG_FILE}" &
log_follow_pid=$!

# First boot downloads ~330GB of weights; that alone can take a while.
echo "Waiting for HTTP readiness at ${READY_URL}"
heartbeat=0
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "SGLang container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  if (( heartbeat % 6 == 0 )); then
    echo "  still starting..."
  fi
  heartbeat=$((heartbeat + 1))
  sleep 5
done

echo "SGLang is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Stop with ./stop.sh; logs: docker logs -f ${CONTAINER_NAME}"
