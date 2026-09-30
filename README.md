# ExAt-MLP: attention-enhanced MLP for DGA detection — artifact

Code for (i) feature extraction, (ii) training and evaluation of ExAt-MLP, the
MLP, XGBoost and autoencoder baselines, (iii) the SHAP analysis, and (iv) the
inference benchmark. DGArchive domains are not redistributed; see *Data*.

## Requirements

Python 3.11 on Linux (the inference benchmark uses `sched_setaffinity`; WSL2
works). `pip install -r requirements.txt`. A GPU is optional
(`tensorflow[and-cuda]==2.15.1`). All commands run from the repository root.

## Data

**DGArchive (not included).** Access to [DGArchive](https://dgarchive.caad.fkie.fraunhofer.de/)
is granted on request by its maintainers (Fraunhofer FKIE). We used the full
snapshot `2020-06-19-dgarchive_full.tgz`. With credentials:

```bash
export DGARCHIVE_USER=... DGARCHIVE_PASSWORD=...
```

`process_dgarchive.py` downloads the snapshot into `Data/Raw/dgarchive/` (or
uses it if already there; set `DGARCHIVE_FILE` for another snapshot name),
keeps families with at least 30,000 names (the last 30,000 per family) and
writes `<family>_dga-top.csv` and `dga_families.txt`. **Expected format** if
you supply the files yourself: one CSV per family, `Data/Raw/<family>_dga-top.csv`,
with the domain name in the first column (the DGArchive CSV layout), and
`Data/Raw/dga_families.txt` with one family name per line; then run the
pipeline from `reduce_and_label.py` on. `reduce_and_label.py` keeps 15,000
unique names per family (families with fewer are dropped).

**Shipped in `Data/Raw`:**

| File | Use |
|---|---|
| `tranco_top100k.txt` | Top-100k Tranco names: n-gram whitelist for the `Reputation` feature |
| `tranco_remaining.txt` | Tranco names after the top 100k (first 900,693 lines); `reduce_and_label.py` takes the first 900,000 unique suffix-stripped names as the benign class |
| `public_suffixes_list_v2.csv` | Mozilla Public Suffix List (MPL 2.0), comments removed |

Both Tranco files come from Tranco list `7XX6X` (https://tranco-list.eu), with
every name that appears in DGArchive removed (`filter_tranco.py`). The
`Words_Freq`/`Words_Mean` features use the English word-frequency model bundled
with `wordninja` 2.0.0 (MIT license); no separate word list is needed.

## (i) Feature extraction — `Scripts/Preprocessing/`

```bash
sh Scripts/Preprocessing/pipeline.sh
```

| Script | Output |
|---|---|
| `process_dgarchive.py` | per-family DGA files, `dga_families.txt` |
| `filter_tranco.py`, `suffix_list.py` | the shipped Tranco / PSL files (optional rebuild) |
| `reduce_and_label.py` | `labeled_dataset.csv`: `prefix,name,label,family` (public suffix stripped; label 1 = DGA) |
| `feature_extractor.py` | `labeled_dataset_features.csv`: 50 lexical features + `Name,Label,Family` |
| `build_dataset.py` | drops `Ratio_DeciDig` (removed by the correlation screen at 0.9; the screen is re-run and reported), 80/20 split, Min-Max fitted on the training split only → `Data/Processed/{train,test}_data.csv` |

## (ii) Training and evaluation — `Scripts/`, `Models/`

```bash
# hyperparameter search on validation ROC-AUC (optional)
python -m Scripts.tune --model exat_mlp
python -m Scripts.tune --model plain_mlp

# ExAt-MLP, its ablations, MLP and XGBoost
python -m Scripts.run_experiments --output-dir Results/full_run \
    --models exat_mlp no_attention single_head plain_mlp xgboost

python -m Scripts.run_autoencoder --output-dir Results/full_run
python -m Scripts.make_results_table Results/full_run
```

`tune.py` writes `best_hyperparameters.json`, which `run_experiments.py` can use
via `--hyperparameters` (ExAt-MLP and ablations) and `--baseline-hyperparameters`
(MLP); without them the defaults in `Models/exat_mlp.py` are used. Run
`python -m Scripts.<name> --help` for all options.

`Models/exat_mlp.py` defines ExAt-MLP and its variants; `Models/AutoEncoder.py`
the autoencoder. Validation is held out before SMOTE, and SMOTE is applied to
the training subset only. `Scripts/metrics.py` reports DGA-class precision,
recall and F1, and ROC-AUC from scores. Each run writes `metrics.json`,
per-family metrics, test predictions and the trained model.

## (iii) SHAP analysis — `Scripts/explain.py`

```bash
python -m Scripts.explain --results-dir Results/full_run   # MLP and ExAt-MLP
```

KernelSHAP on the predicted DGA probability, with 50 K-means centroids of the
training set as background instances and 250 test instances explained. SHAP
values of correlated features (pairwise Pearson, `--corr-threshold`) are also
consolidated. Writes the summary plot of the top-20 features (Fig. 2), the
dependence plots for Entropy and other key features (Fig. 3), and importance
tables to `Results/shap/`. Add `--models xgboost` for the other baselines.

## (iv) Inference benchmark — `Scripts/realtime/`

```bash
python -m Scripts.realtime.build_preprocessor      # recover the scaler, verify the online featurizer
python -m Scripts.realtime.run_benchmark           # add --quick for a ~10 min check
python -m Scripts.realtime.report Results/realtime/<run>
```

Times classification from the raw domain name (suffix stripping, the 50
features, scaling, model) using the trained models in `Results/full_run`:
model-only latency and throughput per backend (Keras, `tf.function`, XLA,
ONNX Runtime, XGBoost), device, batch size and CPU budget; feature-extraction
cost; and an open-loop Poisson streaming test with micro-batching and 1–10
single-core shards. `report.py` writes the figures and tables.

## Smoke tests

```bash
python -m Scripts.run_experiments --smoke-test --models exat_mlp plain_mlp xgboost
python -m Scripts.run_autoencoder --smoke-test
python -m Scripts.explain --smoke-test --results-dir <run dir>
```
