"""compare_runs.py <results_dir_A> <results_dir_B>: do two runs of the notebook produce the same tables?
Compares every experiment CSV present in both (timing, host and job columns ignored; floats to 1e-6)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", "All-NaN slice")

IGNORE = {"arm_seconds", "host", "gpu", "job_id", "attempt", "prefill_compress_s", "prefill_compress_ms",
          "decode_ms_per_token", "decode_ms_per_token_iqr", "prefill_compress_ms_iqr", "position", "kv_MiB_measured",
          "prefill_peak_MiB", "decode_peak_MiB", "measured_vs_logical_pct", "memory_flag",
          "bench_decode_ms_per_token", "bench_kv_MiB_measured"}
a, b = Path(sys.argv[1]), Path(sys.argv[2])
names = sorted({p.name for p in a.glob("*.csv")} & {p.name for p in b.glob("*.csv")})
only = sorted({p.name for p in a.glob("*.csv")} ^ {p.name for p in b.glob("*.csv")})
bad = 0
for n in names:
    x, y = pd.read_csv(a / n, dtype={"budget": str}), pd.read_csv(b / n, dtype={"budget": str})
    cols = [c for c in x.columns if c not in IGNORE]
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        print(f"DIFF {n}: shape {x.shape} vs {y.shape}")
        bad += 1
        continue
    worst, where = 0.0, None
    for c in cols:
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            d = np.nanmax(np.abs(x[c].to_numpy(float) - y[c].to_numpy(float))) if len(x) else 0.0
            same_nan = (x[c].isna() == y[c].isna()).all()
            if not same_nan or d > 1e-6:
                worst, where = max(worst, d if same_nan else np.inf), c
        elif not (x[c].astype(str) == y[c].astype(str)).all():
            worst, where = np.inf, c
    status = "same" if where is None else f"DIFF (max {worst:.3g} in {where})"
    bad += where is not None
    print(f"{status:>32}  {n}  ({len(x)} rows)")
print(f"{len(names)} tables compared, {bad} differ; only in one run: {only or 'none'}")
