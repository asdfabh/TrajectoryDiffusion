#!/usr/bin/env bash
set -euo pipefail

# Test-only ablation runner (no training):
# 1) evaluate_fut for A-only and A+B checkpoints
# 2) diffusion diagnostics for A-only and A+B checkpoints

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/ye/anaconda3/envs/tame/bin/python}"

DATA_ROOT="${DATA_ROOT:-/mnt/datasets/ngsimdata}"
NUM_WORKERS="${NUM_WORKERS:-8}"

EVAL_TEST_RATIO="${EVAL_TEST_RATIO:-0.1}"
DIAG_MAX_BATCHES="${DIAG_MAX_BATCHES:-100}"
DIAG_STEPS="${DIAG_STEPS:-100}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_DIAG="${RUN_DIAG:-1}"

EXP_TAG="${EXP_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/method_diffusion/eval_runs/${EXP_TAG}}"
LOG_DIR="${EXP_ROOT}/logs"

# Preferred input: direct checkpoint files.
# Backward-compatible input: checkpoint directories.
#
# Example (direct):
# A_CKPT=/path/to/checkpoint_epoch_10_AO.pth
# B_CKPT=/path/to/checkpoint_epoch_10_AB.pth
#
# Example (directory):
# A_CKPT_DIR=/path/to/a_only
# B_CKPT_DIR=/path/to/a_plus_b
A_CKPT="${A_CKPT:-}"
B_CKPT="${B_CKPT:-}"
A_CKPT_DIR="${A_CKPT_DIR:-}"
B_CKPT_DIR="${B_CKPT_DIR:-}"

resolve_ckpt_path() {
  local file_path="$1"
  local dir_path="$2"
  local tag="$3"

  if [[ -n "${file_path}" ]]; then
    echo "${file_path}"
    return 0
  fi

  if [[ -n "${dir_path}" ]]; then
    if [[ -f "${dir_path}/fut/checkpoint_best.pth" ]]; then
      echo "${dir_path}/fut/checkpoint_best.pth"
      return 0
    fi
    if [[ -f "${dir_path}/checkpoint_best.pth" ]]; then
      echo "${dir_path}/checkpoint_best.pth"
      return 0
    fi
    echo "[Error] ${tag}: cannot find checkpoint_best.pth under '${dir_path}' or '${dir_path}/fut'." >&2
    return 1
  fi

  echo "[Error] ${tag}: please provide ${tag}_CKPT (file) or ${tag}_CKPT_DIR (directory)." >&2
  return 1
}

A_RESUME_PATH="$(resolve_ckpt_path "${A_CKPT}" "${A_CKPT_DIR}" "A")"
B_RESUME_PATH="$(resolve_ckpt_path "${B_CKPT}" "${B_CKPT_DIR}" "B")"

if [[ ! -f "${A_RESUME_PATH}" ]]; then
  echo "[Error] Missing A checkpoint file: ${A_RESUME_PATH}"
  exit 1
fi
if [[ ! -f "${B_RESUME_PATH}" ]]; then
  echo "[Error] Missing B checkpoint file: ${B_RESUME_PATH}"
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "[Run] Root: ${ROOT_DIR}"
echo "[Run] Python: ${PYTHON_BIN}"
echo "[Run] Data root: ${DATA_ROOT}"
echo "[Run] A checkpoint: ${A_RESUME_PATH}"
echo "[Run] B checkpoint: ${B_RESUME_PATH}"
echo "[Run] Experiment root: ${EXP_ROOT}"
echo "[Run] Stages: eval=${RUN_EVAL}, diag=${RUN_DIAG}"

run_eval() {
  local tag="$1"
  local resume_path="$2"
  local log_file="${LOG_DIR}/eval_${tag}.log"

  echo "[Run][${tag}] evaluate_fut (test_ratio=${EVAL_TEST_RATIO})..."
  "${PYTHON_BIN}" method_diffusion/evaluate_fut.py \
    --resume_fut "${resume_path}" \
    --checkpoint_dir "${ROOT_DIR}/method_diffusion/checkpoints" \
    --data_root "${DATA_ROOT}" \
    --eval_mode fut_only \
    --test_ratio "${EVAL_TEST_RATIO}" \
    --visualize_samples 0 \
    2>&1 | tee "${log_file}"
}

run_diag() {
  local tag="$1"
  local resume_path="$2"
  local log_file="${LOG_DIR}/diag_${tag}.log"

  echo "[Run][${tag}] diffusion diagnostics..."
  "${PYTHON_BIN}" scripts/test_fut_diffusion_diagnostics.py \
    --resume_fut "${resume_path}" \
    --checkpoint_dir "${ROOT_DIR}/method_diffusion/checkpoints" \
    --data_root "${DATA_ROOT}" \
    --batch_size 256 \
    --num_workers "${NUM_WORKERS}" \
    --diag_steps "${DIAG_STEPS}" \
    --t_profile_probe_count 9 \
    --sweep_steps 20,30,50,100 \
    --sampler_spacings leading,trailing \
    --sampler_etas 0.0,0.1,0.2 \
    --max_batches "${DIAG_MAX_BATCHES}" \
    2>&1 | tee "${log_file}"
}

# A-only
if [[ "${RUN_EVAL}" == "1" ]]; then
  run_eval "a_only" "${A_RESUME_PATH}"
fi
if [[ "${RUN_DIAG}" == "1" ]]; then
  run_diag "a_only" "${A_RESUME_PATH}"
fi

# A+B
if [[ "${RUN_EVAL}" == "1" ]]; then
  run_eval "a_plus_b" "${B_RESUME_PATH}"
fi
if [[ "${RUN_DIAG}" == "1" ]]; then
  run_diag "a_plus_b" "${B_RESUME_PATH}"
fi

echo "[Done] All experiments finished."
echo "[Done] Logs: ${LOG_DIR}"


#  A_CKPT=/home/ye/Project/WorkSpace/TrajectoryDiffusion/method_diffusion/checkpoints/fut/checkpoint_epoch_10_AO.pth \
#  B_CKPT=/home/ye/Project/WorkSpace/TrajectoryDiffusion/method_diffusion/checkpoints/fut/checkpoint_epoch_10_AB.pth \
#  bash scripts/run_ablation_ddp4.sh
