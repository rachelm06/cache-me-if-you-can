"""Rebuild a results folder's summary files from the CSVs on disk: no model, no GPU, little memory.

    python cluster/summarize.py kv_cache_results/Qwen__Qwen3-8B   # consolidated notebook: summary.csv
    python cluster/summarize.py kv_cache_results/Qwen__Qwen3-4B   # Colab notebook: summary_all.csv

It runs the summary step (section 19) of the notebook that produced the folder, recognized from
run_manifest.json: the consolidated notebook writes a "lanes" key, the Colab notebook a "last_session_started"
key. Two differences from the notebooks: no zip is written, and the Colab run_manifest.json is left as it is
(the notebook would stamp "last_session_finished", but that run stopped during the memory benchmark).
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write(path, write):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    write(tmp)
    os.replace(tmp, path)


def read_table(out, name):
    path = out / f"{name}.csv"
    return pd.read_csv(path, dtype={"budget": str}) if path.exists() else None


def save_table(out, df, name):
    atomic_write(out / f"{name}.csv", lambda p: df.to_csv(p, index=False))
    print(f"Saved {out / f'{name}.csv'} ({len(df)} rows)")


# ---- consolidated notebook (kv_cache_consolidated.ipynb, section 19) ----
SUMMARY_SPECS = [("E0 budget sweep", "e0_budget_sweep"), ("E5 tier split", "e5_tier_split"),
                 ("E2 consistency", "e2_consistency"), ("E1 knock-out", "e1_knockout"),
                 ("E1 single layer", "e1_single_layer"), ("E1 oracle-keep", "e1_oracle_keep"),
                 ("E3 scoring signal", "e3_scoring_signal"), ("E4 haystack", "e4_haystack"),
                 ("E6 position/content diversity", "e6_content_diversity")]
SUMMARY_LEAD = ["experiment", "method", "arm", "selector", "signal", "scheme", "layers_kept", "layer", "ranking", "split",
                "haystack", "budget", "kind", "kv_bytes_pct", "tf_exact", "tf_digit_acc", "gold_nll", "correct",
                "digit_acc", "needle_kept_frac", "evict_ratio", "code_layers_full", "mean_layer_jaccard",
                "unique_positions_frac", "tokens_retained_frac"]
FIGURE_SECTIONS = [("e0", "E0 budget sweep"), ("e5", "E5 tier split"), ("e2", "E2 consistency"),
                   ("e1", "E1 layer coverage"), ("e3", "E3 scoring signal"), ("e4", "E4 haystack"),
                   ("e6", "E6 position and content diversity")]


def consolidated(out, manifest):
    model, preset = manifest["model"], manifest["preset"]
    parts = [df.assign(experiment=label) for label, name in SUMMARY_SPECS if (df := read_table(out, name)) is not None]
    if parts:
        summary = pd.concat(parts, ignore_index=True)
        summary = summary[[c for c in SUMMARY_LEAD if c in summary] + [c for c in summary if c not in SUMMARY_LEAD]]
        bench = read_table(out, "memory_latency")
        if bench is not None:
            cols = ["method"] + [c for c in ("kv_MiB_measured", "decode_ms_per_token") if c in bench]
            summary = summary.merge(bench[cols].rename(columns={c: f"bench_{c}" for c in cols[1:]}), on="method", how="left")
        for name, col in (("mmlu", "mmlu_acc"), ("longbench", "longbench_score")):
            df = read_table(out, name)
            if df is not None:
                summary = summary.merge(df.groupby("method", sort=False)["value"].mean().rename(col).reset_index(),
                                        on="method", how="left")
        save_table(out, summary, "summary")
        print(summary.groupby("experiment", sort=False).size().rename("rows").to_string())
    else:
        print("no experiment summaries on disk")
    figs = sorted(p.name for p in out.glob("*.png"))
    lines = [f"# Figures for {model} ({preset} preset)", ""]
    for prefix, title in FIGURE_SECTIONS:
        mine = [f for f in figs if f.startswith(prefix + "_")]
        if mine:
            lines += [f"## {title}", ""] + [f"- [{f}]({f})" for f in mine] + [""]
    atomic_write(out / "figures_index.md", lambda p: Path(p).write_text("\n".join(lines)))
    lanes = {p.stem: json.loads(p.read_text()) for p in sorted((out / "manifest").glob("*.json"))}
    probe = out / "e0" / "probe_budgets.json"
    merged = {"model": model, "preset": preset, "written": now(),
              "probe_budgets": json.loads(probe.read_text()) if probe.exists() else None,
              "config": manifest["config"],
              "config_note": "config of the last notebook lane that ran section 19; every lane attempt under "
                             "'lanes' records its own config",
              "lanes": lanes}
    atomic_write(out / "run_manifest.json", lambda p: Path(p).write_text(json.dumps(merged, indent=2, default=str)))
    print(f"figures: {len(figs)} | lanes merged into run_manifest.json: {', '.join(lanes)}")


# ---- Colab notebook (kv_cache_qwen3_4b_t4_l4.ipynb, section 19) ----
COLAB_FILES = {"E0": "e0_budget_sweep", "E1 knock-out": "e1_knockout", "E1 single layer": "e1_single_layer",
               "E1 oracle": "e1_oracle_keep", "E2": "e2_consistency", "E3": "e3_scoring_signal",
               "E4": "e4_haystack", "E5": "e5_tier_split", "E6": "e6_content_diversity",
               "memory and latency": "memory_latency", "MMLU": "mmlu", "LongBench": "longbench"}
COLAB_FIRST = ["experiment", "method", "haystack", "budget", "tf_exact", "tf_digit_acc", "gold_nll", "kv_bytes_pct"]


def colab(out, manifest):
    model, preset = manifest["model_id"], manifest["preset"]
    parts = [df.assign(experiment=label) for label, name in COLAB_FILES.items() if (df := read_table(out, name)) is not None]
    if parts:
        summary = pd.concat(parts, ignore_index=True)
        summary = summary[[c for c in COLAB_FIRST if c in summary] + [c for c in summary if c not in COLAB_FIRST]]
        save_table(out, summary, "summary_all")
        print(summary.groupby("experiment", sort=False).size().rename("rows").to_string())
    else:
        print("no experiment results found in", out)
    figs = sorted(p.name for p in out.glob("*.png"))
    text = f"# Figures for {model} ({preset})\n\n" + "\n".join(f"- [{n}]({n})" for n in figs) + "\n"
    atomic_write(out / "figures_index.md", lambda p: Path(p).write_text(text))
    print(f"{len(figs)} figures listed in figures_index.md")


if __name__ == "__main__":
    for arg in sys.argv[1:] or sys.exit(__doc__):
        out = Path(arg)
        manifest = json.loads((out / "run_manifest.json").read_text())
        style = "consolidated" if "lanes" in manifest else "colab" if "last_session_started" in manifest else None
        if style is None:
            sys.exit(f"{out}: run_manifest.json is from neither notebook")
        print(f"== {out} ({style} notebook)")
        (consolidated if style == "consolidated" else colab)(out, manifest)
