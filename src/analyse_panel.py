"""Across-dataset test on the 33-dataset panel, top-k rules only.

Set size is held fixed by the top-k rule, so ambiguity is comparable across
training sizes. Draws are averaged within each (dataset, size, rule) cell
before anything is tested: draws inside a cell share a test set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# auto_price and autoPrice are the same OpenML data under two names; the
# signature guard missed it because one-hot encoding gave different widths.
# parkinsons and sonar already appear in the seven-dataset study.
DROP = {"autoPrice", "parkinsons", "sonar"}

raw = pd.read_csv(RES / "multiplicity_panel.csv")
raw = raw[~raw.dataset.isin(DROP)]
raw = raw[raw.rule.str.startswith("top_")]

cells = (raw.groupby(["dataset", "rule", "n_train"], as_index=False)
         [["ambiguity", "discrepancy", "score_spread", "rashomon_size"]].mean())

for rule in sorted(cells.rule.unique()):
    c = cells[cells.rule == rule]
    rows = []
    for ds, g in c.groupby("dataset"):
        g = g.sort_values("n_train")
        if len(g) < 4:
            continue
        rho, _ = spearmanr(g.n_train, g.ambiguity)
        lo, hi = g.iloc[0], g.iloc[-1]
        rows.append({"dataset": ds, "n_small": int(lo.n_train),
                     "n_large": int(hi.n_train),
                     "amb_small": round(float(lo.ambiguity), 3),
                     "amb_large": round(float(hi.ambiguity), 3),
                     "rho": round(float(rho), 3)})
    t = pd.DataFrame(rows)
    t.to_csv(RES / f"panel_{rule}.csv", index=False)
    d = t.amb_small - t.amb_large
    stat, p = wilcoxon(t.amb_small, t.amb_large)
    print("=" * 70)
    print(f"{rule}  ({len(t)} datasets, set size held fixed)")
    print("=" * 70)
    print(t.sort_values("rho").to_string(index=False))
    print(f"\n  higher at small n in {(d > 0).sum()}/{len(d)} datasets")
    print(f"  mean difference {d.mean():+.4f}")
    print(f"  Wilcoxon signed-rank p = {p:.5f}")
    print(f"  median rho vs n = {t.rho.median():+.3f}, "
          f"negative in {(t.rho < 0).sum()}/{len(t)}\n")
