# Consolidated notebook plan: E0–E6 + §13 on Qwen3-8B

Nov 18, 2026 · @Ali Alsaleh

## Goals and scope

One notebook, one model per run: it runs diagnostics E0–E6 and the §13 memory and latency benchmark on Qwen3-8B, with MMLU and LongBench behind flags that default to off. Natural prose is the default haystack. Repeated filler runs only where an experiment compares the two (E4, E6).

| Part | Question it answers | Write-up claim it backs | Default |
| --- | --- | --- | --- |
| E0 budget sweep | Where do methods differ, and does ILF beat attention there? | P2 (the ILF test); sets the probe budgets | on |
| E1 layer coverage | Which layers must still hold a fact for retrieval to work? | D2 ("right layers") | on |
| E2 consistency sweep | Does retrieval rise with cross-layer agreement? Do `ilf_gated` or `lf` help? | D2, D4, P4 | on |
| E3 scoring signal | What must the scorer see? Does a cache scored for one question serve another? | D3; ILF's fairest test | on |
| E4 haystack | Does natural prose change the picture? | D1 | on (both haystacks) |
| E5 tier split | Fewer tokens at high precision, or more at low precision? | D5; the next-step method | on |
| E6 position and content diversity | Do `segment` or `idf` selection help on prose? | D4 discussion | on, lowest priority |
| §13 memory and latency | Bytes kept, measured MiB, decode ms/token at batch 4 × 4096 | R5 | on |
| MMLU | Does compression hurt short-context reasoning? | R4 | off |
| LongBench | Real long-context QA | R4 | off |

Dropped from the starter notebook:

- §10–§12 as standalone sections; E0 covers the §12 retrieval test.
- The §14 λ sweep at 10%, where every method sat at chance; E2 replaces it.
- The §16 multi-model loop, which skipped E0–E6. Change `MODEL_ID` instead.

Qwen2.5-0.5B stays as the smoke-test model, so the whole pipeline can be checked in minutes before the 8B run.

## Hardware, memory and runtime budget

Qwen3-8B fits a 40–48 GB GPU with about half the memory to spare, so time is the limit, not memory. The heaviest step, the memory benchmark's batch 4 × 4,096 prefill, needs roughly 21–23 GB.

KV cache per token, from the architecture in the next section:

```latex
2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot 2\ \text{bytes} = 2 \cdot 36 \cdot 8 \cdot 128 \cdot 2 = 147{,}456\ \text{bytes} = 144\ \text{KiB}
```

| Item | Size | Basis |
| --- | --- | --- |
| Weights, bf16 | ≈ 16.4 GB | ≈ 8.2B parameters × 2 bytes |
| KV, one \~3k-token retrieval document | ≈ 0.42 GiB | 3,000 × 144 KiB |
| KV, E3 re-reading signal (\~6.1k tokens) | ≈ 0.84 GiB | the document read twice |
| KV, memory benchmark (4 × 4,096 tokens) | 2.25 GiB | 16,384 × 144 KiB |
| Prefill activations at 4 × 4,096 | ≈ 1.5–2.5 GB (estimate) | 0.5B run measured ≈ 0.57 GB, scaled by model width |
| Peak, memory benchmark | ≈ 21–23 GB (estimate) | the rows above plus ≈ 1 GB CUDA context |

Two things keep it that way:

- Prefill keeps `logits_to_keep=1`, as it does now. Full logits at 4 × 4,096 would add ≈ 5 GB (16,384 positions × a 152k vocabulary × 2 bytes).
- The custom SDPA backend passes an explicit mask. If PyTorch falls back to its math kernel, each layer builds a 4 × 32 × 4,096 × 4,096 score tensor (≈ 4.3 GB). The smoke test checks the prefill peak to catch this.

A full pass is 332 arms (counts under Experiment specs), each over 30 documents with 20 questions:

```latex
\text{time} \approx \sum_{\text{arms}} \text{docs} \times \left(\text{prefills per doc} \times t_{\text{prefill}} + 20 \times t_{\text{question}}\right)
```

On one A100, estimate t\_prefill ≈ 0.3–0.6 s for 3k tokens and t\_question ≈ 30–60 ms. That is ≈ 40–75 s per arm and roughly 5–10 hours for E0–E6; the top-priority trio (E0, E5, E2: 116 arms) takes about a third of that. A 48 GB L40S or A6000 has less memory bandwidth, so allow up to 1.5–2× longer. Replace these estimates with the seconds per arm the first E0 arms print.

## Qwen3-8B specifics and compatibility checks

Qwen3-8B uses the same attention pattern as Qwen2.5 (full causal attention with grouped KV heads), so the compression code carries over unchanged. Five properties are checked at load time instead of assumed.

| Property | Qwen2.5-0.5B (first run) | Qwen3-8B | What changes |
| --- | --- | --- | --- |
| Layers L | 24 | 36 | ILF's top boost rises from log 25 ≈ 3.2 to log 37 ≈ 3.6 (4.2× to 4.6× at λ = 1). E1's layer grid becomes 0, 3, 6, 12, 18, 24, 30, 36 |
| Query / KV heads | 14 / 2 | 32 / 8 | a\_{i,l} averages 32 query heads; 4 query heads share each KV head |
| Head dim | 64 | 128 | Byte accounting unchanged (quantization groups of 32) |
| Hidden / MLP width | 896 / 4,864 | 4,096 / 12,288 | Larger activations |
| Max positions | 32,768 | 40,960 | E3's re-reading signal (2 × 3k tokens) fits, so it runs |
| KV per token | 12 KiB | 144 KiB | 12× more cache per token |

Qwen3 also normalizes queries and keys per head (QK-norm) before RoPE. That happens before the attention interface is called, so the recorded scores and cached keys already include it.

Load-time checks, each an assert in the sanity-check cell:

1. Print `model.config` and assert (L, H\_kv, d\_head) = (36, 8, 128).
2. Assert every digit of a sample of 5-digit codes is one token. Otherwise chance NLL is no longer ln 10 and the teacher-forced metrics change meaning.
3. Compare uncompressed prefill + extend with a plain forward pass by argmax agreement and mean |Δlogit|. bf16 noise exceeds the 0.074 max seen in fp16 on 0.5B, so a fixed max-difference threshold would misfire.
4. Assert measured KV memory matches the byte accounting within 5%.
5. Headroom gate: full-cache teacher-forced exact match on natural prose must be at least 0.7.

`Qwen/Qwen3-8B` is the post-trained model with a switchable thinking mode; `Qwen/Qwen3-8B-Base` is the pretrained checkpoint. E0–E6 use raw-text prompts and teacher-forced scoring, which work on either and never trigger thinking. Thinking only matters for the chat-templated benchmarks (Optional benchmarks).

The pinned `transformers==4.57.1` should already support Qwen3 (support arrived in 4.51, from memory); loading the model is the test. The Hugging Face config and model card could not be opened from this session. The Qwen3-8B numbers above come from search results quoting the [config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json), so check 1 is what confirms them.

## Notebook layout

Twenty sections in run order: setup once (§0–§9), then one section per experiment in priority order, so a run that dies late has already saved the results that matter most. Every experiment section is guarded by its own flag.

1. **§0 Read me.** How to run, presets, expected runtime, where results land.
2. **§1 Install and imports.** Pinned `transformers==4.57.1`, `torch>=2.6`, `accelerate`. `lm_eval` installs only when MMLU or LongBench is on. Prints versions, GPU name and memory.
3. **§2 Configuration.** One cell holding every knob (see Configuration reference): `MODEL_ID`, `PRESET`, run flags, budgets, documents, haystack.
4. **§3 Results I/O and resume.** Results folder on Google Drive, `run_manifest.json`, per-arm checkpoint helpers.
5. **§4 Attention backend.** The `kv_compress` SDPA backend that records a\_{i,l}, unchanged.
6. **§5 Scoring and bit allocation.** `selection_scores` (ilf, ilf\_gated, lf, mix, segment, idf, random), `allocate_bits`, byte accounting including 2-bit.
7. **§6 Compressed cache.** Packed 16/8/4/2-bit storage and `CompressedPrefix.make_cache()` for reuse across questions.
8. **§7 Prefill, compress, extend.** `prefill_and_compress`, `extend`, `greedy_generate`.
9. **§8 Load model and sanity checks.** bf16 load, the five load-time checks, and the headroom gate.
10. **§9 Experiment harness.** Natural and repeated-filler documents, `Arm`, `run_arms` with checkpointing, teacher-forced scoring, `summarize_arms`, `paired_compare`, plotting helpers.
11. **§10 E0 budget sweep.** Ends by setting `PROBE_BUDGETS`.
12. **§11 E5 tier split.**
13. **§12 E2 consistency sweep.**
14. **§13 E1 layer coverage.**
15. **§14 E3 scoring signal.**
16. **§15 E4 haystack.**
17. **§16 E6 position and content diversity.**
18. **§17 Memory and latency** (the starter notebook's §13).
19. **§18 Optional benchmarks:** MMLU, LongBench.
20. **§19 Summary and export.** One table joining every experiment, a figure index, and a zip of the results folder.

## Configuration reference

One `PRESET` switch replaces `SMOKE_TEST`: `full` runs the real experiments on Qwen3-8B, and `smoke` runs everything on tiny settings to check the pipeline. Values marked new do not exist in the starter notebook.

| Knob | full (Qwen3-8B) | smoke | Note |
| --- | --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen3-8B` | `Qwen/Qwen2.5-0.5B`, then `Qwen/Qwen3-8B` | Smoke on both before the full run |
| `PRESET` (new) | `full` | `smoke` | Replaces `SMOKE_TEST` |
| `RUN_E0` … `RUN_E6`, `RUN_BENCH` | on | on | Listed in run order |
| `RUN_MMLU`, `RUN_LONGBENCH` | off | off | Optional benchmarks |
| `HAYSTACK` (new) | `natural` | `natural` | Primary haystack for E0–E3 and E5; E4 and E6 run both |
| `EXP_DOCS` | 30 | 1 | Documents per arm |
| `RETRIEVAL_FACTS` | 20 | 6 | Facts (questions) per document |
| `RETRIEVAL_TOKENS` | 3,000 | 800 | Document length; kept for comparability with the 0.5B run |
| `KEEP_GRID` | 0.3, 0.4, …, 0.9 | 0.5 | E0 and E4 budgets (E4 adds 0.1 and 0.2) |
| `E5_BUDGETS` (new name) | 0.1, 0.2, `int4` | 0.2 | Byte budgets for the tier split |
| `PROBE_BUDGETS` | set by E0 (fallback 0.6, 0.8) | 0.5 | Used by E2, E3, E6 |
| `AUTO_PROBE_BUDGETS` | on | on | E0 picks budgets where attention scores 20–80% of the full cache |
| `LAMBDA` | 1.0 | 1.0 | Main ILF weight |
| `SINK_TOKENS`, `OBS_WINDOW`, `POOL_KERNEL` | 4, 32, 7 | 4, 32, 7 | Unchanged |
| `EXP_GENERATE` | off | off | E0 always decodes; the others are teacher-forced only |
| `BENCH_BATCH`, `BENCH_SEQ_LEN`, `BENCH_DECODE_TOKENS` | 4, 4,096, 64 | 1, 512, 8 | Memory and latency workload |
| `BENCH_REPEATS` (new) | 3 | 1 | Shuffled repeats for latency |
| `MAX_INPUT_TOKENS` | 4,096 | 2,048 | Raise to 8,192 only for LongBench, if wanted |
| `MMLU_LIMIT` | 10 per subject (570 questions) | 4 | No-harm check |
| `LONGBENCH_LIMIT` | 200 per task | 2 | Up from 50 |
| `QUERY_AWARE` | off | off | Prefix compressed before the question |
| `SEED` | 42 | 42 |  |
| `RESULTS_DIR` | Drive: `kv_cache_results/` | local `results/` | Per-model subfolder |

## Experiment specs (E0–E6)

The seven experiments total 332 arms on Qwen3-8B. Every arm compresses each document once and answers all 20 questions on that cache. All arms report teacher-forced metrics: gold-code NLL, per-digit accuracy and exact match.

| Exp | Arms | Haystack | Budgets | Headline metric | Outputs | Feeds |
| --- | --- | --- | --- | --- | --- | --- |
| E0 budget sweep | 19: full, int8, int4, plus attention and ILF at 7 grid budgets and `int8` | natural | `KEEP_GRID` + `int8` | free-running exact match, gold NLL; paired ILF − attention | `e0_*` | `PROBE_BUDGETS` for E2, E3, E6 |
| E5 tier split | 64: uniform 8/4/2-bit + 3 budgets × 5 splits × 4 rankings | natural | 10%, 20%, `int4` | per-digit accuracy vs share of tokens kept | `e5_*` | the next-step method |
| E2 consistency sweep | 33: full + 16 selectors per probe budget | natural | 2 probe budgets | metric vs mean layer Jaccard | `e2_*` | "consistency, not diversity" |
| E1 layer coverage | 81: full + 40 knock-out + 36 single-layer + 4 oracle | natural | full cache; 10%, 20% for oracle | exact match vs layers kept; NLL per evicted layer | `e1_*` | "right layers"; layer-aware budgets |
| E3 scoring signal | 31: full + 2 budgets × 5 signals × 3 selectors | natural | 2 probe budgets | exact match by signal | `e3_*` | which score the next method uses |
| E4 haystack | 60: 2 haystacks × (3 references + 9 budgets × 3 selectors) | both | 10–90% | exact match, gold NLL, fact-eviction ratio | `e4_*` | whether the filler result holds at 8B |
| E6 position and content diversity | 44: 2 haystacks × (full + 3 budgets × 7 selectors) | both | 20% + 2 probe budgets | exact match | `e6_*` | discussion |

Notes per experiment:

- **E0** decodes as well as scoring teacher-forced, so its arms cost more. It picks the probe budgets where attention selection scores 20–80% of the full cache; record them in the manifest. If none qualify, it takes the budget closest to 50%. If E0 is skipped, E2, E3 and E6 fall back to 0.6 and 0.8.
- **E5** must plot per-digit accuracy and gold NLL, not exact match, which sits at 0–4% for every evicting arm.
- **E2** selectors per budget: ILF at λ 4, 1, 0.25; `ilf_gated` at λ 4, 1; ILF with a tighter τ; attention; `mix` at α 0.25, 0.5, 0.75, 1; `lf` at λ 0.25, 1, 4; random shared and random per layer.
- **E1** single-layer scan is 36 arms, the largest block. If time is short, scan every second layer first.
- **E3** question-scored arms compress each document 10 times (one cache per scoring question), so each costs about 10 ordinary arms. The re-reading signal needs 2 × document + 32 tokens, well under 40,960, so it runs.
- **E4 and E6** add the fact-eviction ratio from `writeup/diagnostics.py` to their summaries.
- **E4, E5 and E6** gain a paired table against attention with document-clustered bootstrap intervals, which E0 and E2 already have.

## Memory and latency protocol (§13)

Memory needs no change, but latency needs a new protocol. In the first run, the first seven methods measured 46–97 ms/token and later selection methods 27–33 ms/token, so the order of the runs set the numbers.

Workload: batch 4 × 4,096 tokens, 64 timed decode steps. Methods: the starter notebook's 19 plus uniform 2-bit and consistent selection at 10% and 20%, 22 in all.

| Column | What it measures |
| --- | --- |
| `kv_bytes_pct`, `kv_MiB_logical` | Stored KV bytes from the byte accounting, metadata included |
| `kv_MiB_measured` | GPU memory still allocated after prefill and compression |
| `prefill_compress_ms` | Prefill plus scoring plus compression |
| `prefill_peak_MiB` | Peak extra memory during prefill; the same for every method, because compression runs after prefill |
| `decode_ms_per_token` | Mean wall time per decode step for the whole batch |
| `decode_peak_MiB` | Peak during decoding, including the dequantized copy of one layer |

Timing protocol:

1. Run the memory benchmark in a fresh runtime with nothing else on the GPU.
2. Do one untimed pass over every method to warm kernels and the allocator.
3. Run 3 repeats. Within each, shuffle the method order with a fixed seed.
4. Per method: 3 warm-up decode steps, then 64 timed steps, with `torch.cuda.synchronize()` around each.
5. Report the median and interquartile range across repeats, plus GPU name and driver in the manifest.

Expected on Qwen3-8B: about 2.25 GiB of KV for the full cache and about 230 MiB at 10%. Decoding with quantized tokens dequantizes in plain PyTorch at every step, so state that the latency cost of quantization is our unfused implementation's, not the method's. Flag any method whose measured KV memory is more than 5% off the logical count; select@int4 was 12% over in the first run.

## Optional benchmarks: MMLU and LongBench

Both stay off by default, and each runs only after the full cache passes a gate on the new model. The first run showed why: on 0.5B, LongBench averaged 0.127 and passage retrieval scored 0.043 against a chance level of 1/30.

|  | MMLU | LongBench |
| --- | --- | --- |
| Role | No-harm check on short contexts | Real long-context QA |
| Gate | none; always informative as a check | Drop any task where the full cache is within noise of chance |
| Size | 10 per subject, 570 questions (SE ≈ 0.02) | 200 per task, 5 tasks |
| Methods | full, int8, int4, int2, plus attention, ILF and consistent selection at the 2 probe budgets | same 10 |
| Prompting | 5-shot, log-likelihood scoring | chat template, thinking off, short answers |
| Cost | small | the largest optional block: 10 methods × 1,000 long prompts with generation |

LongBench details:

- Qwen3-8B can emit a reasoning block before answering. For lm-eval, turn thinking off in the chat template (`enable_thinking=False`, from memory; confirm on the model card). Otherwise the reasoning eats the generation budget and breaks F1.
- Check whether our `CompressedKVLM` adapter applies the chat template at all before trusting any score.
- With `QUERY_AWARE` off, the prompt is split where the question starts. Under a chat template, the split must fall inside the user turn: the cached prefix ends before the question, and the question plus the assistant header run on the compressed cache.
- Keep `MAX_INPUT_TOKENS` at 4,096 for comparability, or raise it to 8,192 if the team wants LongBench closer to its native lengths.

## Run order, checkpointing and outputs

Run in priority order, checkpoint after every arm, and stop at a gate rather than spend hours on a broken setup. A Colab disconnect should cost at most one arm.

&#91;embedded content: run order · 2 smoke tests, 1 gate, 7 experiments, 3 closing steps\]

The gate is the five load-time checks plus full-cache exact match of at least 0.7 on natural prose. E2, E3 and E6 wait for E0 only because they use its probe budgets; the rest follow priority.

Checkpointing (new; the starter notebook has none):

- After each arm, write its rows to `<experiment>/arms/<arm_key>.csv` on Drive. `arm_key` hashes the config, scoring signal, plan, haystack, document count and decode flag; the result CSVs already carry this string.
- On a rerun, an arm whose file exists is loaded, not run. Each experiment's `_raw.csv` is rebuilt from its arm files.
- To rerun one arm, delete its file. To rerun an experiment, delete its folder.
- Experiments are independent apart from the probe budgets, so two runtimes can run different experiments into the same Drive folder.

Output layout:

```
kv_cache_results/Qwen__Qwen3-8B/
  run_manifest.json
  e0/arms/*.csv
  e0_budget_sweep_raw.csv, e0_budget_sweep.csv, e0_paired_ilf_vs_attention.csv, e0_*.png
  e1/ ... e6/              same pattern
  memory_latency.csv
  mmlu.csv, longbench.csv  only if run
  summary.csv, figures_index.md
```

`run_manifest.json` records the model ID and revision hash; the `torch`, `transformers` and `lm_eval` versions; GPU name and memory; the preset and every knob; the probe budgets actually used; the notebook's git commit; and start and end times per experiment.

## Changes from the current starter notebook

Fourteen changes turn the starter notebook into the consolidated one; tick them off as they land.

- [ ] Merge into one notebook: setup, E0–E6, memory and latency, optional MMLU and LongBench. Remove §10–§12, §14 and §16.
- [ ] Set `MODEL_ID = "Qwen/Qwen3-8B"`, and replace `SMOKE_TEST` with a `PRESET` holding smoke and full values.
- [ ] Build `exp_docs` from natural prose for E0–E3 and E5.
- [ ] Add per-arm checkpointing and resume on Drive, plus `run_manifest.json`.
- [ ] Order run flags by priority, with E0 first so the probe budgets exist before E2, E3 and E6.
- [ ] Add the load-time asserts: config dimensions, one token per digit, efficient SDPA kernel, bf16 logit match by argmax agreement.
- [ ] Add the headroom gate before E0.
- [ ] Plot E5 with per-digit accuracy and gold NLL instead of exact match.
- [ ] Add paired tables against attention, with bootstrap intervals, to E4, E5 and E6.
- [ ] Add the fact-eviction ratio to the E4 and E6 summaries.
- [ ] Memory and latency: add uniform 2-bit and consistent selection; shuffle order, add a warm-up pass and 3 repeats; report median and interquartile range.
- [ ] Put MMLU and LongBench behind flags. LongBench gets the chat template with thinking off, 200 examples per task, and the full-cache gate.
- [ ] Guard E0's first-run regression check so it only runs for Qwen2.5-0.5B on repeated filler.
- [ ] Make the summary cell join every experiment that ran.

## Open decisions

Six choices are the team's to make before the full run.

1. **Qwen3-8B or Qwen3-8B-Base for E0–E6?** Base avoids post-training effects on retrieval; the post-trained model is needed for LongBench. Using one model for both keeps the paper simpler.
2. **Which GPU?** A 40 GB A100 and a 48 GB card both fit; the choice changes runtime, not feasibility.
3. **Rerun the 0.5B diagnostics on natural prose?** It is cheap and gives a like-for-like 0.5B vs 8B comparison, since the 0.5B E0–E3 and E5 used repeated filler.
4. **Document length: keep \~3k tokens or raise to 8k?** 3k matches the first run; 8k gives eviction more room to matter on a model that retrieves better.
5. **Add the 2-bit-floor method (E7) to this notebook now, or as a follow-up?** The code change is small, and it would land the next-step method in the same run.
6. **Who runs what?** Arms are independent, so experiments can be split across team members' Colab runtimes writing to one Drive folder.
