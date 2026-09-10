#!/bin/bash
# Stage 2: decode a saved clean latent under UF7D-style water types
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints}"
INPUT_DIR="${INPUT_DIR:-${ROOT}/outputs/stage1}"
STAGE2_CKPT="${CKPT_DIR}/stage2/model.pth"
RESOLUTION="${RESOLUTION:-512}"
SEED="${SEED:-42}"

if [ ! -f "${STAGE2_CKPT}" ]; then
    echo "Stage 2 weights not found. Run: python scripts/download_weights.py"
    exit 1
fi
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Input directory not found: ${INPUT_DIR}"
    echo "Run Stage 1 first: bash scripts/run_stage1.sh"
    exit 1
fi

python inference_stage2_underwater_only_sdxl_fixed_waters_BL.py \
    --input_dir="${INPUT_DIR}" \
    --stage2_checkpoint="${STAGE2_CKPT}" \
    --resolution="${RESOLUTION}" \
    --seed="${SEED}"
