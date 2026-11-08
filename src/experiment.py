"""Does predictive multiplicity grow as the training set shrinks?

The question is about what happens to individual predictions in the regime
where aggregate scores stop separating models. At each training size we

  1. fit a pool of candidate classifiers on the same subsample,
  2. keep those within epsilon of the best validation score (the empirical
     Rashomon set, following Marx, Calmon and Ustun, ICML 2020),
  3. measure, on a held-out test set that never changes size, how far the
     surviving models disagree about individual cases.

Ambiguity and discrepancy are theirs; what is new here is reading them as a
function of n. Two quantities are recorded side by side at every size: the
spread of test scores inside the Rashomon set (the aggregate view, which is
what a practitioner sees) and the ambiguity (the individual view, which they
do not). The hypothesis is that these come apart as n falls.

Every model is selected on validation data and measured on test data, so the
selection never sees the numbers it is judged by.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data import load_all

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Pre-specified before any result was looked at.
EPSILONS = (0.01, 0.05)       # Rashomon tolerance, in balanced accuracy
TOP_K = (3, 5)                # fixed-size near-optimal sets, to control set size
TEST_FRACTION = 0.30
N_TEST_SPLITS = 5             # independent test sets
N_TRAIN_DRAWS = 20            # training subsamples per (test split, size)
SEED = 20261103


def candidate_pool(seed: int) -> dict[str, Pipeline]:
    """A deliberately heterogeneous pool.

    Multiplicity that only appears between random seeds of one model family is
    a weaker claim than multiplicity across families, so the pool spans linear,
    kernel, instance-based, tree and probabilistic learners, with a few
    hyperparameter settings each.
    """
    spec: dict[str, object] = {
        "logreg_C0.1": LogisticRegression(C=0.1, max_iter=5000, random_state=seed),
        "logreg_C1": LogisticRegression(C=1.0, max_iter=5000, random_state=seed),
        "logreg_C10": LogisticRegression(C=10.0, max_iter=5000, random_state=seed),
        "lda": LinearDiscriminantAnalysis(),
        "gnb": GaussianNB(),
        "knn3": KNeighborsClassifier(n_neighbors=3),
        "knn5": KNeighborsClassifier(n_neighbors=5),
        "svc_rbf_C1": SVC(C=1.0, kernel="rbf", random_state=seed),
        "svc_rbf_C10": SVC(C=10.0, kernel="rbf", random_state=seed),
        "tree_d3": DecisionTreeClassifier(max_depth=3, random_state=seed),
        "tree_d5": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "rf_100": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
    }
    return {k: Pipeline([("scale", StandardScaler()), ("clf", v)]) for k, v in spec.items()}


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    out = []
    for c in (0, 1):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        out.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(out)) if out else float("nan")


def size_grid(pool_n: int, minority_frac: float) -> list[int]:
    """Geometric grid from the smallest viable size up to the pool.

    The floor is four per class: below that a stratified split cannot also
    hold out a validation set with both classes present.
    """
    floor = int(np.ceil(8 / max(minority_frac, 1e-9)))
    floor = max(floor, 20)
    if pool_n <= floor:
        return [pool_n]
    grid = np.geomspace(floor, pool_n, num=7)
    sizes = sorted({int(round(v)) for v in grid})
    return [s for s in sizes if s <= pool_n]


@dataclass
class Row:
    dataset: str
    clinical: bool
    n_train: int
    test_split: int
    train_draw: int
    rule: str
    param: float
    rashomon_size: int
    ambiguity: float
    discrepancy: float
    score_spread: float
    best_test: float
    worst_in_set_test: float
    selection_regret: float


def one_cell(X, y, train_idx, val_idx, test_idx, seed) -> list[dict]:
    """Fit the pool once, then read off multiplicity at each epsilon."""
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    Xte, yte = X[test_idx], y[test_idx]

    names, val_scores, test_scores, preds = [], [], [], []
    for name, model in candidate_pool(seed).items():
        try:
            model.fit(Xtr, ytr)
            pv = model.predict(Xva)
            pt = model.predict(Xte)
        except Exception:
            continue
        names.append(name)
        val_scores.append(balanced_accuracy(yva, pv))
        test_scores.append(balanced_accuracy(yte, pt))
        preds.append(pt)

    if len(names) < 2:
        return []

    val_scores = np.asarray(val_scores)
    test_scores = np.asarray(test_scores)
    preds = np.vstack(preds)                       # (models, test points)

    # the model a practitioner would deploy: best on validation
    baseline = int(np.argmax(val_scores))
    out = []

    # Two ways of choosing the set of near-optimal models.
    #
    #  - epsilon-ball: the definition in the literature. Its membership count
    #    moves with n, which makes ambiguity incomparable across sizes, so it
    #    cannot answer the sample-size question on its own.
    #  - top-k: a fixed number of models, which holds set size constant and
    #    isolates the effect of n.
    selections: list[tuple[str, float, np.ndarray]] = []
    for eps in EPSILONS:
        selections.append(
            (f"eps_{eps}", eps,
             np.flatnonzero(val_scores >= val_scores[baseline] - eps)))
    order = np.argsort(-val_scores, kind="stable")
    for k in TOP_K:
        if k <= len(val_scores):
            selections.append((f"top_{k}", float(k), order[:k]))

    for rule, param, keep in selections:
        if keep.size < 2:
            # a set of one has no multiplicity by definition; record it as zero
            amb = disc = 0.0
            spread = 0.0
            worst = float(test_scores[baseline])
        else:
            sub = preds[keep]
            # ambiguity: share of test points on which the set disagrees at all
            amb = float(np.mean(sub.min(axis=0) != sub.max(axis=0)))
            # discrepancy: largest share of flips against the deployed model
            flips = (sub != preds[baseline]).mean(axis=1)
            disc = float(flips.max())
            spread = float(test_scores[keep].max() - test_scores[keep].min())
            worst = float(test_scores[keep].min())
        out.append(dict(
            rule=rule,
            param=param,
            rashomon_size=int(max(keep.size, 1)),
            ambiguity=amb,
            discrepancy=disc,
            score_spread=spread,
            best_test=float(test_scores.max()),
            worst_in_set_test=worst,
            selection_regret=float(test_scores.max() - test_scores[baseline]),
        ))
    return out


def run(smoke: bool = False, panel: bool = False) -> pd.DataFrame:
    rng_master = np.random.default_rng(SEED)
    if panel:
        from panel import load_panel
        datasets, _log = load_panel()
    else:
        datasets = load_all()
    n_splits = 2 if smoke else N_TEST_SPLITS
    n_draws = 3 if smoke else N_TRAIN_DRAWS

    rows: list[Row] = []
    for ds in datasets:
        outer = StratifiedShuffleSplit(n_splits=n_splits, test_size=TEST_FRACTION,
                                       random_state=SEED)
        for t, (pool_idx, test_idx) in enumerate(outer.split(ds.X, ds.y)):
            y_pool = ds.y[pool_idx]
            sizes = size_grid(len(pool_idx), ds.minority_frac)
            if smoke:
                sizes = sizes[:3]
            for n_train in sizes:
                for r in range(n_draws):
                    seed = int(rng_master.integers(0, 2**31 - 1))
                    # carve train+validation out of the pool, stratified
                    inner = StratifiedShuffleSplit(
                        n_splits=1, train_size=n_train, random_state=seed)
                    try:
                        sub_tr, sub_rest = next(inner.split(ds.X[pool_idx], y_pool))
                    except ValueError:
                        continue
                    # validation: a fixed slice of what is left, capped so it
                    # does not grow without bound on the larger pools
                    n_val = int(min(max(20, 0.5 * n_train), len(sub_rest)))
                    val_pick = np.random.default_rng(seed).choice(
                        sub_rest, size=n_val, replace=False)
                    if len(np.unique(y_pool[val_pick])) < 2:
                        continue
                    cells = one_cell(ds.X, ds.y, pool_idx[sub_tr], pool_idx[val_pick],
                                     test_idx, seed)
                    for c in cells:
                        rows.append(Row(dataset=ds.name, clinical=ds.clinical,
                                        n_train=n_train, test_split=t, train_draw=r,
                                        **c))
        print(f"  {ds.name}: done", flush=True)

    frame = pd.DataFrame([asdict(r) for r in rows])
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = ("multiplicity_smoke.csv" if smoke else
            "multiplicity_panel.csv" if panel else "multiplicity_v2.csv")
    frame.to_csv(RESULTS / name, index=False)
    meta = dict(seed=SEED, epsilons=list(EPSILONS), top_k=list(TOP_K), test_fraction=TEST_FRACTION,
                n_test_splits=n_splits, n_train_draws=n_draws,
                n_rows=len(frame), smoke=smoke)
    (RESULTS / ("meta_smoke.json" if smoke else "meta_panel.json" if panel else "meta.json")).write_text(
        json.dumps(meta, indent=2))
    print(f"wrote {RESULTS / name}  ({len(frame)} rows)")
    return frame


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--panel", action="store_true")
    args = ap.parse_args()
    run(smoke=args.smoke, panel=args.panel)
