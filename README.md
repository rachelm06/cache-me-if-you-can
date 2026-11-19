# KV-cache compression diagnostics: ILF vs attention selection (6.4610 project)

Does inverse-layer-frequency (ILF) token selection keep a compressed KV cache more useful than plain attention
selection? This branch holds the two main runs of the diagnostic notebook (experiments E0–E6), both from
2026-11-18. The project proposal is `docs/6_4610_project_proposal.pdf`.

| | Engaging run | Colab run |
|---|---|---|
| Model | `Qwen/Qwen3-8B` (bf16) | `Qwen/Qwen3-4B` (bf16) |
| Hardware | MIT ORCD Engaging, `mit_preemptable`, 1× NVIDIA L40S (48 GB) | Google Colab, 1× NVIDIA L4 (22 GiB) |
| Notebook | `kv_cache_consolidated.ipynb` | `kv_cache_qwen3_4b_t4_l4.ipynb` |
| Results | `kv_cache_results/Qwen__Qwen3-8B/` | `kv_cache_results/Qwen__Qwen3-4B/` |
| E0–E6 | complete (332 arms) | complete (332 arms) |
| Memory and latency benchmark | complete (22 methods, batch 4 × 4096 tokens, 3 repeats) | started, no results saved |
| LongBench | 5 tasks, ≤ 200 questions each, 10 methods, inputs up to 32768 tokens | not run |
| MMLU | not run | not run |
| Observation-window screen | complete (137 arms: scoring windows of 8 tokens to the whole prefix, attention and ILF, 4 budgets) | not run |
| Probe budgets (from E0) | 0.7, 0.9 | 0.7, 0.9 |

Both runs use the `full` preset: 30 documents × 20 facts × 3000 tokens of natural prose (WikiText-2). The cache
is compressed before any question is seen; E3 also tests scoring with the question.

## Layout

```
README.md
kv_cache_consolidated.ipynb       notebook of the Engaging run (runs headless under Slurm, or on Colab)
kv_cache_qwen3_4b_t4_l4.ipynb     notebook of the Colab run, as it ran
kv_cache_results/
  Qwen__Qwen3-8B/                 Engaging run
    e0_* … e6_*                   per-experiment tables (*_raw.csv: one row per question) and figures
    memory_latency*.csv           memory and decode-latency benchmark
    longbench.csv, longbench/     LongBench scores (one file per method in longbench/)
    window_screen*                observation-window screen (notebook §16b, lane `window`): tables, paired table, figures
    summary.csv, figures_index.md every experiment's summary (window screen included) in one table, with
                                  benchmark and LongBench columns by method
    run_manifest.json             versions, GPU, job attempts, probe budgets, every knob
    notebooks/                    executed notebooks: final_* has every table and figure;
                                  exp-a / exp-b / bench are the lanes as they ran; optional_* is LongBench;
                                  window_* is the window screen (run after final_*, so final_* does not show it)
    logs/                         Slurm logs of those jobs (the LongBench job was preempted twice)
  Qwen__Qwen3-4B/                 Colab run (copied from Drive, MyDrive/kv_cache_results2_l4/)
    e0_* … e6_*, run_manifest.json
    summary_all.csv, figures_index.md   written by cluster/summarize.py (the run stopped before its own section 19)
cluster/                          Slurm scripts for the Engaging run (setup, job script, submit), plus
                                  summarize.py (section 19 without a model or GPU) and compare_runs.py
docs/
  consolidated_notebook_plan.md   what the consolidated notebook computes, and why
  cluster_run_plan.md             how it runs on Engaging (lanes, requeue, resume, monitoring)
  6_4610_project_proposal.pdf
```

Not in git: the `arms/` folder of each experiment (one checkpoint per arm, which the notebooks reload instead of
recomputing) and the Engaging run's `manifest/` folder. They exist only in the working copy on Engaging.

## Reproducing

**Engaging.** Once, on a node with internet: `bash cluster/setup.sh` (venv on pool, model and dataset download).
Then from the repo root:

```bash
bash cluster/submit.sh Qwen/Qwen3-8B        # E0–E6 in two lanes, the benchmark, then a final pass
sbatch -J kvc-longbench --time=24:00:00 cluster/run.sbatch optional full Qwen/Qwen3-8B -p RUN_LONGBENCH True
sbatch -J kvc-window --time=04:00:00 cluster/run.sbatch window full Qwen/Qwen3-8B   # scoring-window screen
source cluster/env.sh && python cluster/summarize.py kv_cache_results/Qwen__Qwen3-8B   # rebuild summaries, no GPU
```

Jobs requeue after preemption and resume from the saved arms. Details are in `docs/cluster_run_plan.md`.

**Colab.** Open `kv_cache_qwen3_4b_t4_l4.ipynb` with an L4 (or T4) runtime and run all cells. Results go to
`MyDrive/kv_cache_results2_<profile>/`. A rerun reloads finished arms from there.

## Older material

The first run (Qwen2.5-0.5B, `kv_cache_results/Qwen__Qwen2.5-0.5B/`, `kv_cache_colab_filled.ipynb`), the
`kv_cache_starter.ipynb` that the consolidated notebook was built from, and the partial E4/E5 CSVs that were in
the repo root are on `origin/Ali` (commit `ccc0817`); `main` has the first run.
