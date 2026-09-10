#!/bin/bash
# Stage 1: text -> clean underwater scene latent (+ optional Stage 2 clean pass and depth)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/stage1}"
PROMPT="${PROMPT:-an underwater scene of a shark swimming in the ocean near a rock}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT_PREFIX="${PROMPT_PREFIX:-crystal clear underwater, }"
LORA_SCALE="${LORA_SCALE:-0.9}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
RESOLUTION="${RESOLUTION:-512}"
SEED="${SEED:-42}"

STAGE1_LORA="${CKPT_DIR}/stage1"
STAGE2_CKPT="${CKPT_DIR}/stage2/model.pth"

if [ ! -f "${STAGE1_LORA}/pytorch_lora_weights.safetensors" ]; then
    echo "Stage 1 weights not found. Run: python scripts/download_weights.py"
    exit 1
fi
if [ ! -f "${STAGE2_CKPT}" ]; then
    echo "Stage 2 weights not found. Run: python scripts/download_weights.py"
    exit 1
fi

PROMPT_ARGS=()
if [ -n "${PROMPT_FILE}" ]; then
    PROMPT_ARGS+=(--prompt_file="${PROMPT_FILE}")
else
    PROMPT_ARGS+=(--prompt="${PROMPT}")
fi

python inference_stage1_clean_only_sdxl.py \
    --stage1_model="stabilityai/stable-diffusion-xl-base-1.0" \
    --stage1_vae="madebyollin/sdxl-vae-fp16-fix" \
    --stage1_lora="${STAGE1_LORA}" \
    --lora_scale="${LORA_SCALE}" \
    --stage2_checkpoint="${STAGE2_CKPT}" \
    --prompt_prefix="${PROMPT_PREFIX}" \
    --guidance_scale="${GUIDANCE_SCALE}" \
    --num_inference_steps="${NUM_INFERENCE_STEPS}" \
    --resolution="${RESOLUTION}" \
    --output_dir="${OUTPUT_DIR}" \
    --seed="${SEED}" \
    --skip_existing \
    "${PROMPT_ARGS[@]}"
