#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    # bf16 weights pass through CPU RAM on load (~16 GB for Qwen3-8B)
#SBATCH --time=12:00:00
#SBATCH --requeue                    # preemption on mit_preemptable requeues the job
#SBATCH --signal=B:USR1@600          # 10 min before the time limit, signal this script (see trap below)
#SBATCH --open-mode=append           # one log file across all attempts (%j is stable across requeues)
#SBATCH --output=cluster/logs/%x_%j.out
#
# Requeue-safe headless run of kv_cache_consolidated.ipynb. Submit from the repo root:
#   sbatch -J kvc-<lane> cluster/run.sbatch <lane> [preset] [model] [extra papermill args...]
#   lanes: all | smoke | final | exp-a | exp-b | bench | optional   (see the notebook's §0)
# Every arm is saved as it finishes, so a requeued attempt re-runs the notebook from the top and loads
# finished arms instead of running them. A cell error exits non-zero: the job FAILS instead of looping.

set -uo pipefail
LANE=${1:?usage: sbatch cluster/run.sbatch <lane> [preset] [model] [extra papermill args]}
PRESET=${2:-full}
MODEL_ID=${3:-Qwen/Qwen3-8B}
shift $(( $# < 3 ? $# : 3 ))
ATTEMPT=${SLURM_RESTART_COUNT:-0}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source cluster/env.sh
mkdir -p runs cluster/logs
# mit_preemptable nodes have internet: top up the model cache first (only HEAD requests when it is complete).
# Plain HTTP, because the xet transfer stalled on this cluster. Then run offline from the cache.
HF_HUB_DISABLE_XET=1 timeout 3600 hf download "$MODEL_ID" > "cluster/logs/fetch_${SLURM_JOB_ID:-local}.log" 2>&1 \
    || echo "WARNING: could not refresh $MODEL_ID in the cache (see cluster/logs/fetch_${SLURM_JOB_ID:-local}.log)"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== $(date -Is) job=${SLURM_JOB_ID:-none} attempt=$ATTEMPT host=$(hostname) lane=$LANE preset=$PRESET model=$MODEL_ID"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Slurm requeues preempted jobs itself; a job that reaches its time limit is not requeued, so do it here.
trap 'echo "=== $(date -Is) near the time limit: requeueing job $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"' USR1

OUT_NB="runs/${LANE}_${PRESET}_${MODEL_ID//\//__}_${SLURM_JOB_ID:-local}_a${ATTEMPT}.ipynb"
papermill kv_cache_consolidated.ipynb "$OUT_NB" -k kvcache \
    -p PRESET "$PRESET" -p MODEL_ID "$MODEL_ID" -p LANE "$LANE" "$@" \
    --log-output --no-progress-bar &
pid=$!
wait "$pid"; rc=$?
while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done   # a trapped signal interrupts `wait`
echo "=== $(date -Is) papermill exited with $rc (executed notebook: $OUT_NB)"
exit "$rc"
