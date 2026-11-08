"""Load the larger dataset panel, skipping any that fail to parse.

Selection happened in find_datasets.py against fixed numeric rules. This only
loads, cleans and records what actually arrived, so the paper can state how
many of the shortlisted datasets survived and why the others did not.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data import Dataset, _clean, CACHE

ROOT = Path(__file__).resolve().parent.parent
TARGET = 32
MIN_ROWS = 150


def load_panel(limit: int = TARGET) -> tuple[list[Dataset], pd.DataFrame]:
    from sklearn.datasets import fetch_openml
    import numpy as np

    cand = pd.read_csv(ROOT / "results" / "candidate_datasets.csv")
    loaded: list[Dataset] = []
    seen_signatures: set[tuple[int, int, int]] = set()
    log: list[dict] = []

    for rec in cand.itertuples():
        if len(loaded) >= limit:
            break
        bunch = None
        for kwargs in (dict(as_frame=True, parser="pandas"),
                       dict(as_frame=False, parser="liac-arff")):
            try:
                bunch = fetch_openml(data_id=int(rec.data_id),
                                     data_home=str(CACHE), **kwargs)
                break
            except Exception:
                continue
        if bunch is None:
            log.append({"name": rec.name, "status": "fetch failed"})
            continue
        try:
            if isinstance(bunch.data, pd.DataFrame):
                frame, target = bunch.data, bunch.target
            else:
                frame = pd.DataFrame(bunch.data, columns=list(bunch.feature_names))
                target = pd.Series(np.asarray(bunch.target).ravel())
            X, y = _clean(frame, target)
        except Exception as exc:
            log.append({"name": rec.name, "status": f"clean failed: {type(exc).__name__}"})
            continue

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2 or counts.min() < 20 or X.shape[0] < MIN_ROWS \
                or X.shape[1] < 2:
            log.append({"name": rec.name, "status": "failed post-clean checks"})
            continue

        # Guard against the same data appearing under two OpenML names
        # (auto_price / autoPrice are one dataset). Counting it twice would
        # inflate the across-dataset test this panel exists to power.
        sig = (X.shape[0], X.shape[1], int(counts.min()))
        if sig in seen_signatures:
            log.append({"name": rec.name, "status": "duplicate signature"})
            continue
        seen_signatures.add(sig)

        loaded.append(Dataset(name=str(rec.name), clinical=False, X=X, y=y))
        log.append({"name": rec.name, "status": "ok", "n": X.shape[0],
                    "p": X.shape[1], "minority": round(float(counts.min() / counts.sum()), 3)})

    return loaded, pd.DataFrame(log)


if __name__ == "__main__":
    ds, log = load_panel()
    out = ROOT / "results"
    log.to_csv(out / "panel_log.csv", index=False)
    print(f"loaded {len(ds)} datasets")
    print(log[log.status == "ok"].to_string(index=False))
    bad = log[log.status != "ok"]
    if len(bad):
        print(f"\nskipped {len(bad)}:")
        print(bad.to_string(index=False))
