#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

df = pd.read_csv("bench_stage_summary.tsv", sep="\t")
idx = ["tag", "snps", "n", "pve", "structure", "method", "cores"]

wide = df.pivot_table(
    index=idx,
    columns="stage",
    values=["elapsed_sec", "max_rss_gb", "cpu_percent"],
    aggfunc="first",
).reset_index()

wide.columns = [
    "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
    if isinstance(col, tuple) else col
    for col in wide.columns
]

wide["total_elapsed_sec"] = wide.get("elapsed_sec_grm", 0).fillna(0) + wide.get("elapsed_sec_gwas", 0).fillna(0)
wide["peak_rss_gb"] = wide[["max_rss_gb_grm", "max_rss_gb_gwas"]].max(axis=1)

wide = wide.sort_values(["snps", "n", "method", "cores"])
wide.to_csv("bench_two_stage_summary.tsv", sep="\t", index=False)

print(wide.to_csv(sep="\t", index=False))
