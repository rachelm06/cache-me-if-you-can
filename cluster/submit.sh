#!/bin/bash
# Full run: two experiment lanes and the memory benchmark on three GPUs, then a final pass (everything loaded
# from disk) that writes the complete executed notebook, summary.csv, the figure index and the zip.
#   bash cluster/submit.sh [model]
# If a lane fails, the final job waits forever (DependencyNeverSatisfied): cancel it, fix the problem,
# resubmit the lane (finished arms are kept), then resubmit the final job with the new job ids.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_ID=${1:-Qwen/Qwen3-8B}
mkdir -p cluster/logs runs
a=$(sbatch --parsable -J kvc-exp-a cluster/run.sbatch exp-a full "$MODEL_ID")
b=$(sbatch --parsable -J kvc-exp-b cluster/run.sbatch exp-b full "$MODEL_ID")
c=$(sbatch --parsable -J kvc-bench --time=03:00:00 cluster/run.sbatch bench full "$MODEL_ID")
f=$(sbatch --parsable -J kvc-final --time=03:00:00 --dependency=afterok:$a:$b:$c cluster/run.sbatch final full "$MODEL_ID")
echo "exp-a=$a exp-b=$b bench=$c final=$f (after the other three succeed)"
echo "watch: squeue -u $USER; tail -f cluster/logs/kvc-exp-a_$a.out"
