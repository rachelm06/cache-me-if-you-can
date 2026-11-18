#!/bin/bash
# One-time setup, on the login node (it needs internet; jobs run offline):
#   bash cluster/setup.sh            venv + packages + model/dataset prefetch
#   bash cluster/setup.sh --no-fetch venv + packages only
# Re-running is safe: existing packages and cached files are kept.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"  # cu128 needs driver >= 570
# uv 0.11.16 segfaults on this login node (venv and pip alike), so use the stdlib venv + pip.
PYTHON="${KV_PYTHON:-$HOME/.local/bin/python3.12}"

if [ ! -x "$KV_VENV/bin/python" ]; then
    mkdir -p "$(dirname "$KV_VENV")"
    "$PYTHON" -m venv "$KV_VENV"
fi
source "$KV_VENV/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install "torch==2.11.0" --index-url "$TORCH_INDEX"
# transformers 4.57.1 needs huggingface_hub < 1.0; datasets < 4 still loads script-based datasets (LongBench);
# pandas < 3 matches Colab (the notebook relies on pandas 2 string and copy semantics).
python -m pip install "transformers==4.57.1" "accelerate>=0.26.0" "lm_eval[hf,longbench]==0.4.11" \
    "huggingface_hub[hf_xet]>=0.34,<1.0" "datasets>=3.0,<4.0" sentencepiece "pandas>=2.2,<3" matplotlib papermill ipykernel
python -m ipykernel install --sys-prefix --name kvcache --display-name "Python (kvcache)"
python -c "import torch, transformers, lm_eval, papermill; print('torch', torch.__version__, '| transformers', transformers.__version__, '| lm_eval', lm_eval.__version__, '| papermill', papermill.__version__)"

if [ "${1:-}" != "--no-fetch" ]; then
    for m in Qwen/Qwen2.5-0.5B Qwen/Qwen3-8B; do
        hf download "$m"
    done
    python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
print("wikitext-2 test rows:", len(ds))
EOF
fi
echo "setup done: venv $KV_VENV, HF_HOME $HF_HOME"
