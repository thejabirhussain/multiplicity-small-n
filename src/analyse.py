"""Read the experiment output and test the decoupling hypothesis.

The claim under test: as n falls, models inside the Rashomon set look more
alike in aggregate and less alike on individual cases. Aggregate agreement is
read from the spread of test scores inside the set; individual agreement from
ambiguity. If the two move in opposite directions as n falls, they decouple.

Nothing here fits a model of dataset properties. Seven datasets give seven
observations, so the per-dataset trends are reported separately and the
pooled test treats the dataset as the unit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "multiplicity.csv")


def per_cell(frame: pd.DataFrame) -> pd.DataFrame:
    """Average the draws within each (dataset, size, epsilon) cell.

    Draws inside a cell share a test set, so they are repeated measures and
    averaging first keeps the later tests from counting them as independent.
    """
    keys = ["dataset", "clinical", "epsilon", "n_train"]
    return (frame.groupby(keys, as_index=False)
            [["ambiguity", "discrepancy", "score_spread",
              "rashomon_size", "selection_regret", "best_test"]]
            .mean())


def trends(cells: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each quantity against n, per dataset."""
    out = []
    for (ds, eps), g in cells.groupby(["dataset", "epsilon"]):
        if len(g) < 4:
            continue
        row = {"dataset": ds, "epsilon": eps, "n_sizes": len(g)}
        for col in ["ambiguity", "discrepancy", "score_spread", "rashomon_size"]:
            rho, p = spearmanr(g.n_train, g[col])
            row[f"rho_{col}"] = round(float(rho), 3)
            row[f"p_{col}"] = float(p)
        out.append(row)
    return pd.DataFrame(out)


def extremes(cells: pd.DataFrame) -> pd.DataFrame:
    """Smallest versus largest size, per dataset and epsilon."""
    out = []
    for (ds, eps), g in cells.groupby(["dataset", "epsilon"]):
        g = g.sort_values("n_train")
        lo, hi = g.iloc[0], g.iloc[-1]
        out.append({
            "dataset": ds, "epsilon": eps,
            "n_small": int(lo.n_train), "n_large": int(hi.n_train),
            "amb_small": round(float(lo.ambiguity), 3),
            "amb_large": round(float(hi.ambiguity), 3),
            "disc_small": round(float(lo.discrepancy), 3),
            "disc_large": round(float(hi.discrepancy), 3),
            "spread_small": round(float(lo.score_spread), 3),
            "spread_large": round(float(hi.score_spread), 3),
            "set_small": round(float(lo.rashomon_size), 2),
            "set_large": round(float(hi.rashomon_size), 2),
        })
    return pd.DataFrame(out)


def main() -> None:
    raw = load()
    cells = per_cell(raw)
    cells.to_csv(RESULTS / "cells.csv", index=False)

    tr = trends(cells)
    tr.to_csv(RESULTS / "trends.csv", index=False)
    ex = extremes(cells)
    ex.to_csv(RESULTS / "extremes.csv", index=False)

    print("=" * 74)
    print("TREND AGAINST n  (Spearman rho; negative = falls as n grows)")
    print("=" * 74)
    for eps in sorted(cells.epsilon.unique()):
        t = tr[tr.epsilon == eps]
        print(f"\nepsilon = {eps}")
        print(t[["dataset", "n_sizes", "rho_ambiguity", "rho_discrepancy",
                 "rho_score_spread", "rho_rashomon_size"]].to_string(index=False))

    print("\n" + "=" * 74)
    print("SMALLEST vs LARGEST TRAINING SIZE")
    print("=" * 74)
    for eps in sorted(cells.epsilon.unique()):
        print(f"\nepsilon = {eps}")
        print(ex[ex.epsilon == eps].drop(columns=["epsilon"]).to_string(index=False))

    print("\n" + "=" * 74)
    print("PAIRED TESTS ACROSS DATASETS (dataset is the unit, n = 7)")
    print("=" * 74)
    for eps in sorted(cells.epsilon.unique()):
        e = ex[ex.epsilon == eps]
        print(f"\nepsilon = {eps}")
        for label, a, b in [
            ("ambiguity", "amb_small", "amb_large"),
            ("discrepancy", "disc_small", "disc_large"),
            ("score spread", "spread_small", "spread_large"),
        ]:
            d = e[a].to_numpy() - e[b].to_numpy()
            if np.allclose(d, 0):
                print(f"  {label:14s} all differences zero")
                continue
            stat, p = wilcoxon(e[a], e[b])
            direction = "higher" if d.mean() > 0 else "lower"
            print(f"  {label:14s} small-n {direction} by "
                  f"{abs(d.mean()):.3f} on average, Wilcoxon p = {p:.4f} "
                  f"({int((d > 0).sum())}/{len(d)} datasets higher at small n)")

    # The decoupling question, stated as a ratio: how much individual-level
    # disagreement accompanies one point of aggregate score spread?
    print("\n" + "=" * 74)
    print("AMBIGUITY PER UNIT OF SCORE SPREAD")
    print("=" * 74)
    safe = cells[cells.score_spread > 1e-9].copy()
    safe["ratio"] = safe.ambiguity / safe.score_spread
    for eps in sorted(safe.epsilon.unique()):
        s = safe[safe.epsilon == eps]
        rho, p = spearmanr(s.n_train, s.ratio)
        print(f"  epsilon={eps}: median ratio {s.ratio.median():.2f}, "
              f"Spearman(ratio, n) rho={rho:.3f}, p={p:.2e}")


if __name__ == "__main__":
    main()
