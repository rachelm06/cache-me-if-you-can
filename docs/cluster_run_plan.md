# Running the consolidated notebook on ORCD: sbatch, `mit_preemptable`, L40S

> **Status (2026-11-18):** carried out. The Qwen3-8B run is in `kv_cache_results/Qwen__Qwen3-8B/` (see the README).
> The repo was reorganized afterwards: the first run's files, the root `e4_*`/`e5_*` CSVs, `kv_cache_starter.ipynb`
> and `writeup/diagnostics.py` are gone from this branch (the first three are on `origin/Ali`). Paths below that mention
> them describe the repo as it was when this plan was written.

This plan goes with `consolidated_notebook_plan.md`. That plan covers what the notebook computes. This one covers how to build the notebook and run it headless under Slurm so that a preemption costs at most one arm.

## Short answer: it can stay a notebook

Resuming doesn't need the kernel to survive. All progress is saved to disk as one file per finished arm, so to resume you just run the notebook again from the top. Setup takes a few minutes, and every arm that already has a file is loaded instead of run.

Papermill runs the notebook headless. It takes parameters (`PRESET`, `MODEL_ID`, `LANE`), streams cell output to the Slurm log, and exits non-zero when a cell raises, so a failed gate stops the job instead of requeueing it forever. The same `.ipynb` still runs top to bottom on Colab.

A plain script would only be worth it for a graceful SIGTERM handler, which this partition barely allows (next section), or for checkpoints smaller than one arm.

## What the cluster and repo look like (checked 2026-11-18)

| Finding | Consequence |
| --- | --- |
| `mit_preemptable` has no plain L40. Its 48 GB cards are L40S (`gpu:l40s`, ~350 GPUs) | Request `--gres=gpu:l40s:1`: same 48 GB as an L40, more than the ~21–23 GB peak |
| `PreemptMode=REQUEUE`, `GraceTime=0`, `KillWait=30s` | Slurm requeues a preempted job automatically, but allows at most ~30 s between SIGTERM and SIGKILL. Assume a hard kill: atomic writes, no shutdown hook |
| `MaxTime=2-00:00:00`; QOS cap of 4 GPUs per user | A job that hits its time limit is *not* requeued, so the job requeues itself via `--signal`. Up to 4 jobs at once |
| Your past batch jobs on this partition show `Restarts` 1–3, then `COMPLETED` | Automatic requeue works for your account. Interactive sessions end `PREEMPTED`, the same failure as a Colab disconnect |
| The E0–E6 code exists only on `origin/Ali` (commit `ccc0817`, 54-cell `kv_cache_starter.ipynb`). Local `main` has the older 38-cell notebook | Build the consolidated notebook from `origin/Ali` |
| `writeup/diagnostics.py` defines the fact-eviction ratio as (1 − `needle_kept_frac`) / (1 − `tokens_retained_frac`) and reads `e4_haystack_raw.csv` / `e5_tier_split.csv` from the repo root | E4 and E6 report that ratio; the root CSVs stay where they are |
| Existing conda envs don't match the pins (`Super_Experts`: torch 2.5.1, transformers 4.51.3, no lm_eval). `uv` 0.11.16 segfaults on the login node (`uv venv` and `uv pip` alike) | New stdlib venv + pip on pool, with the notebook's pins |
| `HF_HOME=~/orcd/pool/hf` (740 GB free on pool). Neither Qwen3-8B nor Qwen2.5-0.5B is cached. `~/.cache/huggingface` is a dangling symlink | Download ahead of time on the login node, and export `HF_HOME` explicitly in the job |
| The login node reaches huggingface.co. Compute-node internet is untested | Jobs run with `HF_HUB_OFFLINE=1` from the prefetched cache |
| Random selectors seed their own `torch.Generator().manual_seed(cfg.seed)`. Documents and knock-out plans use seeded `random.Random` | An arm's result doesn't depend on which arms ran before it in the process, so skipping cached arms is safe |
| `torch==2.11.0` (the Colab run's version) has cu126, cu128 and cu130 wheels | Install the cu128 build (driver ≥ 570). Switch to cu130 if the first job shows a driver ≥ 580 |

## How resume works

1. **The arm is the unit of checkpointing.** After each arm, `run_arms` writes `<exp>/arms/<arm_key>.csv`, plus `<arm_key>.retention.npy` when E0 asks for retention maps. It writes to a temp file in the same folder and then calls `os.replace`, so a kill mid-write never leaves a partial file that looks finished. The CSV is written last and marks the arm as done.
2. **`arm_key` is a readable name plus a hash of everything that changes the result:** `asdict(cfg)`, the scoring signal, the tags, the haystack, `MODEL_ID`, `EXP_DOCS`, `RETRIEVAL_FACTS`, `EXP_TOKENS` and the decode flag. The plan callables in E1 can't be hashed, but their tags and config names already identify them. Changing any knob gives a new key, so a stale arm is never mixed into new results.
3. **On start**, `run_arms` loads every arm that has a file, prints it with a `[cached]` mark and its headline numbers, and runs only the rest. Summaries, paired tables and plots are rebuilt from all arm files every time.
4. **Each arm row also records** host, GPU, job id, restart count and seconds per arm. Seconds per arm is the number the consolidated plan wants in place of its runtime estimates.
5. **E0 writes `e0/probe_budgets.json`.** E2, E3 and E6 read it, so they can run in a different job from E0. If the file is missing they fall back to 0.6 and 0.8, print a warning, and record the fallback in the manifest.
6. **§17 checkpoints per repeat** (`bench/repeat_<r>.csv`). Every process start reruns the untimed warm-up pass before its next repeat, so a resumed benchmark still follows the timing protocol. Each repeat records its node.
7. **Worst case per preemption:** one arm plus a model load of about 2–5 min. The longest arms are E3's question-scored arms, at several times an ordinary arm.

## Files to add

| File | Purpose |
| --- | --- |
| `kv_cache_consolidated.ipynb` | The notebook: `origin/Ali` + the consolidated plan's 14 changes + the cluster changes below |
| `cluster/setup.sh` | Run once on the login node: venv, pinned installs, register the kernel, prefetch models and WikiText |
| `cluster/env.sh` | Sourced by every job: activate the venv, set `HF_HOME`, offline flags, `KV_RESULTS_DIR`, `PYTHONUNBUFFERED=1` |
| `cluster/run.sbatch` | The requeue-safe job: `sbatch -J kvc-<lane> cluster/run.sbatch <lane> [preset] [model]` |
| `cluster/submit.sh` | Submits the three lanes, then the final pass with an `afterok` dependency |
| `cluster/logs/`, `runs/` | Slurm logs (appended across restarts) and one executed notebook per attempt. Both gitignored |

## Notebook changes for the cluster

These come on top of the consolidated plan's 14 changes.

1. **Parameters cell** tagged `parameters`: `PRESET`, `MODEL_ID`, `LANE`, `RESULTS_ROOT`. The defaults (`LANE = "all"`) reproduce the Colab behaviour.
2. **Environment detection.** If `IN_COLAB`, run `%pip install` and mount Drive. Otherwise do neither, and take `RESULTS_ROOT` from `$KV_RESULTS_DIR`. The `lm_eval` import moves behind `RUN_MMLU or RUN_LONGBENCH`.
3. **`LANE` sets the run flags** (Lanes table below).
4. **Checkpointing** as in "How resume works". `run_arms` gains `exp=` and `haystack=` arguments, and retention maps are saved with their arm.
5. **Smoke runs write to their own root** (`kv_cache_results_smoke/`). The arm key already keeps smoke and full results apart; this also keeps the folders clean.
6. **One manifest per lane** (`manifest/<lane>.json`), because lanes run at the same time. Each attempt appends an entry: job id, restart count, node, GPU, driver, package versions, and start and end times per experiment. §19 merges them into `run_manifest.json`.
7. **The headroom gate is E0's `full` arm**, so after the first run it costs nothing. In the `full` preset it raises if full-cache tf-exact on natural prose is below 0.7. In `smoke` it only warns, so the pipeline check can finish.

## Lanes

The experiments are independent, except that E2, E3 and E6 need E0's probe budgets. They split across 3 GPUs (the QOS allows 4), and each lane keeps the consolidated plan's priority order.

| Lane | Runs | Arms | Rough L40S time |
| --- | --- | --- | --- |
| `exp-a` | E0 → E2 → E3 → E6 | 19 + 33 + 31 + 44 | 4–9 h |
| `exp-b` | E5 → E1 → E4 | 64 + 81 + 60 | 4–9 h |
| `bench` | §17 only, in a fresh process | 22 methods × (warm-up + 3 repeats) | < 1 h |
| `final` | everything, all from cache, plus §19 summary and zip | 0 new | ~15 min |
| `smoke` | everything, `PRESET=smoke` | tiny | minutes |

The times come from the consolidated plan's 40–75 s per arm on an A100, multiplied by 1.5–2 for the L40S. E0's arms also decode, and E3's question-scored arms compress each document several times, so both lanes come to roughly 200 ordinary-arm equivalents. Replace these numbers with the seconds per arm the 8B smoke run prints.

Each lane requests `--time=12:00:00`. Shorter requests start sooner through backfill, and a lane that runs long requeues itself and carries on. The `final` lane produces one complete executed notebook with every figure, which is the copy to share with the team.

## Job scripts

These are built and live in `cluster/`; read them there rather than here.

- `cluster/run.sbatch <lane> [preset] [model] [extra papermill args]`: `mit_preemptable`, `gpu:l40s:1`, 8 CPUs, 64 GB, 12 h, `--requeue`, `--open-mode=append`, and `--signal=B:USR1@600` with a trap that runs `scontrol requeue` on itself. It runs papermill in the background and waits, so the trap fires promptly. It exits with papermill's code, so a failed cell marks the job FAILED instead of requeueing it. Jobs run with `HF_HUB_OFFLINE=1`.
- `cluster/submit.sh [model]`: `exp-a`, `exp-b` and `bench` at once, then `final` with `--dependency=afterok` on all three.
- `cluster/setup.sh`: the venv and the prefetch, as below.
- `cluster/env.sh`: venv, `HF_HOME`, `PYTHONUNBUFFERED`, `MPLCONFIGDIR`. Sourced by the other three.

`cluster/setup.sh` (login node, run once):

```bash
VENV=~/orcd/pool/venvs/kvcache
~/.local/bin/python3.12 -m venv "$VENV" && source "$VENV/bin/activate"     # uv segfaults on this node
pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.57.1" "accelerate>=0.26.0" "lm_eval[hf,longbench]==0.4.11" \
    "huggingface_hub[hf_xet]>=0.34,<1.0" "datasets>=3.0,<4.0" sentencepiece "pandas>=2.2,<3" matplotlib papermill ipykernel
export HF_HOME=~/orcd/pool/hf
hf download Qwen/Qwen2.5-0.5B && hf download Qwen/Qwen3-8B          # ~17 GB
python -c "from datasets import load_dataset as l; l('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')"
```

## Run sequence

1. **Branch.** `git switch -c consolidated-qwen3 origin/Ali`. It is `main` plus one commit. Leave the `e4_*` and `e5_*` CSVs in the repo root: `writeup/diagnostics.py` reads them there.
2. **Build the notebook and cluster files** (Files to add, Notebook changes).
3. **Setup.** `bash cluster/setup.sh` on the login node.
4. **Smoke on 0.5B.** `sbatch -J kvc-smoke05 --time=01:00:00 cluster/run.sbatch smoke smoke Qwen/Qwen2.5-0.5B`. This checks the environment, the offline cache, the driver line in the log, and every cell end to end.
5. **Resume tests.** Neither has been tried on this cluster yet.
   - *Preemption.* Rerun the smoke job into a fresh results root. Once the log shows E0 finished, run `scontrol requeue <jobid>`, the same path a preemption takes. Pass if `sacct -j <id> --format=State,Restarts` shows `Restarts=1`, the log has an `attempt=1` header and `[cached]` lines for E0, and the final CSVs match step 4's, exactly or within bf16 noise, ignoring timing columns.
   - *Time limit.* `sbatch --time=00:10:00 --signal=B:USR1@180 ...` must log "near time limit: requeueing" and come back as attempt 1.
6. **Smoke on Qwen3-8B.** `sbatch -J kvc-smoke8b --time=02:00:00 cluster/run.sbatch smoke smoke Qwen/Qwen3-8B`. This runs the five load-time asserts on the real model, the SDPA prefill-peak check, and prints the seconds per arm to update the Lanes table.
7. **Full run.** `bash cluster/submit.sh`. The headroom gate runs at the start of each experiment lane.
8. **Final pass.** It starts automatically once all three lanes succeed, and leaves the complete notebook in `runs/final_<id>_a*.ipynb`, `summary.csv`, `figures_index.md` and the zip.

## Monitoring and recovery

- **Status.** `squeue -u $USER` and `sacct -j <id> -X --format=JobID,JobName%14,State,Restarts,Elapsed,NodeList`.
- **Log.** `tail -f cluster/logs/kvc-exp-a_<id>.out`. Each attempt starts with a `=== ... attempt=N` line.
- **Progress.** Count arm files against the expected totals (E0 19, E1 81, E2 33, E3 31, E4 60, E5 64, E6 44):
  `for e in e0 e1 e2 e3 e4 e5 e6; do echo "$e $(ls kv_cache_results/Qwen__Qwen3-8B/$e/arms/*.csv 2>/dev/null | wc -l)"; done`
- **A lane failed** (a cell raised). Read the end of its log and that attempt's `runs/*.ipynb`, fix the problem, and resubmit the same lane. Finished arms are kept. The `final` job then sits in `DependencyNeverSatisfied`: cancel it and resubmit it with the new job id.
- **Rerun one arm:** delete its file and resubmit its lane. **Rerun one experiment:** delete its folder.
- **A job shows held** after many requeues (`JobHeldAdmin`): `scontrol release <id>`.
- **Don't run two jobs of the same lane at once.** Nothing gets corrupted, since writes are atomic and the content is identical, but the work is duplicated.

## Risks and unknowns

- **L40S driver version.** Unknown until the first job prints it. The cu128 build needs ≥ 570; if smoke fails at CUDA init, that's the first thing to check.
- **Compute-node internet.** Untested. Running offline from the prefetched cache avoids depending on it. A cache miss fails fast with a clear error.
- **`MaxBatchRequeue=5`.** I believe it counts batch-launch failures, not preemptions. Past jobs here show up to 3 restarts, but I haven't confirmed this. The held-job recovery above covers it.
- **Latency noise.** `bench` gets its own GPU and cgroup cores, but other jobs share the node. The median of 3 shuffled repeats absorbs most of it. For cleaner numbers, delete `bench/` and rerun the lane alone later.
- **Shared reference arm.** Both experiment lanes may compute the gate's `full` arm at the same moment. That's harmless: one atomic write replaces an identical one.

## Open items

1. **Where results are shared.** Drive was the team's shared folder. On the cluster, results land in `kv_cache_results/Qwen__Qwen3-8B/` in this repo. Commit the summaries and figures, and keep the `arms/` folders gitignored unless the team wants them.
2. **Consolidated plan decisions** 1 (Base or post-trained model), 3, 4 and 5 are still the team's to make. `MODEL_ID` is a job argument, so decision 1 doesn't change anything here. Decision 2 is settled: L40S.
