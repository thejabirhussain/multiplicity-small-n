"""Dataset loading for the multiplicity-versus-sample-size study.

Seven public tabular binary-classification datasets, fetched from OpenML by
numeric data_id so that a rerun gets the same bytes. Five are clinical; two
non-clinical sets are kept as controls so that anything we see is not an
artefact of medical data in particular.

None of these overlap with the ECS8051 coursework datasets (Iris, Wine, Breast
Cancer Wisconsin, Optical Digits, Glass2, Chess, Zoo). That separation is
deliberate and should stay.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

CACHE = Path(__file__).resolve().parent.parent / "data"

# data_id -> (short name, clinical?)
SPEC = {
    37:   ("pima_diabetes", True),
    53:   ("heart_statlog", True),
    43:   ("haberman_survival", True),
    1480: ("indian_liver", True),
    1488: ("parkinsons", True),
    59:   ("ionosphere", False),
    40:   ("sonar", False),
}


@dataclass
class Dataset:
    name: str
    clinical: bool
    X: np.ndarray
    y: np.ndarray

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]

    @property
    def minority_frac(self) -> float:
        _, counts = np.unique(self.y, return_counts=True)
        return float(counts.min() / counts.sum())


def _clean(frame: pd.DataFrame, target: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Numeric-only design matrix, median-imputed, with a 0/1 target.

    Categorical predictors are one-hot encoded. Rows are never dropped: these
    datasets are small and discarding rows would change the sample sizes the
    study is about.
    """
    frame = frame.copy()
    cat = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if cat:
        frame = pd.get_dummies(frame, columns=cat, drop_first=True)
    frame = frame.astype(float)
    frame = frame.fillna(frame.median(numeric_only=True))
    # a column that is constant carries no information and breaks scaling
    frame = frame.loc[:, frame.std(axis=0) > 0]

    codes, _ = pd.factorize(target)
    y = np.asarray(codes, dtype=int)
    # factorize orders by appearance; fix 1 as the minority class so that
    # "positive" means the same thing everywhere
    if (y == 1).sum() > (y == 0).sum():
        y = 1 - y
    return frame.to_numpy(dtype=float), y


def load(data_id: int) -> Dataset:
    name, clinical = SPEC[data_id]
    CACHE.mkdir(parents=True, exist_ok=True)
    # sklearn 1.9 / pandas 3 disagree in the ARFF parser on some categorical
    # targets, and not the same way for every file, so try the options in turn.
    attempts = [
        dict(as_frame=True, parser="pandas"),
        dict(as_frame=False, parser="liac-arff"),
    ]
    bunch = None
    for kwargs in attempts:
        try:
            bunch = fetch_openml(data_id=data_id, data_home=str(CACHE), **kwargs)
            break
        except Exception:
            continue
    if bunch is None:
        raise RuntimeError(f"{name}: no OpenML parser option succeeded")

    if isinstance(bunch.data, pd.DataFrame):
        frame, target = bunch.data, bunch.target
    else:
        frame = pd.DataFrame(bunch.data, columns=list(bunch.feature_names))
        target = pd.Series(np.asarray(bunch.target).ravel())
    X, y = _clean(frame, target)
    if len(np.unique(y)) != 2:
        raise ValueError(f"{name}: expected 2 classes, got {len(np.unique(y))}")
    return Dataset(name=name, clinical=clinical, X=X, y=y)


def load_all() -> list[Dataset]:
    return [load(i) for i in SPEC]


if __name__ == "__main__":
    rows = []
    for ds in load_all():
        rows.append({
            "dataset": ds.name,
            "clinical": ds.clinical,
            "n": ds.n,
            "predictors": ds.p,
            "minority_frac": round(ds.minority_frac, 3),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "dataset_summary.csv", index=False)
