# Sourced by cluster/setup.sh and cluster/run.sbatch. Paths can be overridden from the environment.
export KV_VENV="${KV_VENV:-$HOME/orcd/pool/venvs/kvcache}"
# ~/.cache/huggingface is a dangling symlink on this account, so always set HF_HOME explicitly.
export HF_HOME="${KV_HF_HOME:-$HOME/orcd/pool/hf}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
unset TRANSFORMERS_CACHE  # deprecated alias; HF_HOME covers it
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-$USER"
if [ -f "$KV_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$KV_VENV/bin/activate"
fi
