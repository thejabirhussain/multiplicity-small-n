# Does predictive multiplicity grow as the training set shrinks?

Two models can score the same and still disagree about which patients they
flag. That disagreement is **predictive multiplicity**, defined by Marx, Calmon
and Ustun (ICML 2020). It is measured at a fixed sample size in every study I
could find.

This repository asks what happens to it **as a function of n**, which matters
because clinical prediction models are routinely built on a few hundred
patients, and because a 2022 systematic review of machine-learning prognostic
models in oncology found 81% at high risk of bias with insufficient sample size
among the main drivers.

## Result

Across **30 public tabular datasets**, with the size of the near-optimal model
set held fixed, ambiguity is higher at the smallest training size than at the
largest in **28 of 30 datasets**.

| Selection rule | Direction | Mean difference | Wilcoxon signed-rank | Median rho vs n |
|---|---|---|---|---|
| top-3 | 28/30 | +0.037 | **p = 4.6e-06** | -0.935 |
| top-5 | 28/30 | +0.060 | **p = 1.2e-05** | -0.943 |

Eleven datasets under top-5 have rho = -1.000: ambiguity falls at every step of
the size grid without a single reversal.

Two datasets go the other way, and both are reported rather than dropped.
`divorce_prediction` has ambiguity between 0.003 and 0.030, so the classes are
nearly separable and there is no disagreement to measure at any size.
`analcatdata_gsssexsurvey` is a genuine counter-example.

## The methodological finding, which came first

The initial design used the epsilon-ball from the literature: keep every model
within epsilon of the best validation score. That produced a null result.

The reason is a confound. The number of models admitted by an epsilon-ball
**changes with n** (on Pima at epsilon=0.05, from 3.06 models at n=23 to 8.02 at
n=318), and ambiguity correlates with set size at rho = +0.538, p = 2.4e-04. A
larger set disagrees more simply because it has more members, so ambiguity
measured this way is not comparable across sample sizes.

Holding the set size fixed with a top-k rule removes the confound, and the
effect appears.

**This has a consequence beyond this study: published ambiguity figures are not
comparable across papers unless the size of the Rashomon set is reported, and
it usually is not.**

## Method

- 12 classifiers spanning linear, kernel, instance-based, tree and
  probabilistic families, each in a pipeline with standardisation
- At each training size, fit the pool, select the near-optimal set on a
  validation split, measure disagreement on a held-out test set whose size
  never changes
- **Ambiguity**: share of test points on which the selected set disagrees
- **Discrepancy**: largest share of flips against the deployed model
- 5 independent test splits, 20 training subsamples per cell, geometric size grid
- Selection always on validation data, measurement always on test data
- Draws within a cell share a test set, so they are averaged before any test is
  applied. The unit of analysis is the dataset.
- Every constant was fixed before results were inspected (`SEED = 20261103`)

## Datasets

Selected from OpenML by numeric rules applied before anything was fitted: binary
target, 150 to 1200 rows, 4 to 80 features, minority class at least 15%, and no
overlap with the datasets used in my taught coursework. 227 candidates matched;
the panel takes the first 32 in a deterministic order.

Of those, `auto_price` and `autoPrice` are the same OpenML data under two names
and `parkinsons` and `sonar` appear in the earlier seven-dataset study, so all
four are excluded from the across-dataset test, leaving 30.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/find_datasets.py     # writes the candidate shortlist
.venv/bin/python src/experiment.py --panel  # the 32-dataset run
.venv/bin/python src/analyse_panel.py     # the across-dataset tests
```

`--smoke` runs a fast reduced version. Data is fetched from OpenML by numeric
`data_id` and cached under `data/`, which is not tracked.

## Layout

| Path | Contents |
|---|---|
| `src/data.py` | The original seven datasets, loading and cleaning |
| `src/find_datasets.py` | OpenML query and the pre-specified selection rules |
| `src/panel.py` | Loads the larger panel, logs what failed and why |
| `src/experiment.py` | The experiment |
| `src/analyse.py` | Analysis of the seven-dataset run |
| `src/analyse_panel.py` | Across-dataset tests on the panel |
| `results/` | Raw per-cell output and the derived tables |

## Status

Work in progress. The analysis is complete and the manuscript is being written.
Findings here are reported as they stand, including the null result that came
first and the two datasets that go against the effect.

## Reference

Marx, C., Calmon, F. and Ustun, B. (2020). Predictive Multiplicity in
Classification. *ICML*. [arXiv:1909.06677](https://arxiv.org/abs/1909.06677)

## Author

Mohammed Jabir Hussain Shaik, MSc Artificial Intelligence, Queen's University
Belfast.
