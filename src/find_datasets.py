"""Pick a larger panel of datasets so the across-dataset test has power.

The unit of analysis is the dataset, so seven datasets means n = 7 and a sign
test cannot clear 0.125 however clean the effect looks. This queries OpenML for
small binary tabular problems and writes the shortlist to a file, which the
experiment then reads. Selection is by fixed numeric criteria applied before
anything is fitted, so it cannot be tuned to the result.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data"

# Pre-specified selection rules.
MIN_ROWS, MAX_ROWS = 150, 1200
MAX_FEATURES = 80
MIN_MINORITY = 0.15          # avoid severe imbalance; stratified draws need both classes
TARGET_COUNT = 32

# Datasets used in the ECS8051 coursework. Excluded by name so the two
# studies stay separate.
COURSEWORK = {"iris", "wine", "breast-w", "wdbc", "optdigits", "glass",
              "kr-vs-kp", "zoo", "breast-cancer-wisconsin"}


if __name__ == "__main__":
    # The OpenML REST listing is simpler and avoids a heavy dependency.
    import urllib.request

    url = ("https://www.openml.org/api/v1/json/data/list/"
           f"number_classes/2/number_instances/{MIN_ROWS}..{MAX_ROWS}/"
           f"number_features/4..{MAX_FEATURES}/status/active/limit/400")
    with urllib.request.urlopen(url, timeout=60) as fh:
        payload = json.load(fh)

    rows = []
    for rec in payload["data"]["dataset"]:
        qualities = {q["name"]: float(q["value"]) for q in rec.get("quality", [])}
        n = qualities.get("NumberOfInstances")
        p = qualities.get("NumberOfFeatures")
        miss = qualities.get("NumberOfInstancesWithMissingValues", 0.0)
        minor_size = qualities.get("MinorityClassSize")
        if None in (n, p) or minor_size is None or not n:
            continue
        minority = minor_size / n
        if minority < MIN_MINORITY:
            continue
        if miss and n and miss / n > 0.5:
            continue
        name = rec["name"].lower()
        if any(c in name for c in COURSEWORK):
            continue
        rows.append({"data_id": int(rec["did"]), "name": rec["name"],
                     "n": int(n), "p": int(p),
                     "minority_pct": round(minority * 100, 1)})

    table = pd.DataFrame(rows).drop_duplicates(subset="name")
    # deterministic order, then take the head: no cherry-picking
    table = table.sort_values(["n", "data_id"]).reset_index(drop=True)
    out = ROOT / "results" / "candidate_datasets.csv"
    table.to_csv(out, index=False)
    print(f"{len(table)} candidates matched the rules; wrote {out}")
    print(table.head(40).to_string(index=False))
