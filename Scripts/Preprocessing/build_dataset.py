"""Turn the 50 extracted features into the model-ready train/test files.

    python -m Scripts.Preprocessing.build_dataset

Input:  Data/Raw/labeled_dataset_features.csv  (feature_extractor.py output:
        50 features, then Name, Label, Family)
Output: Data/Processed/train_data.csv  49 features + Label   (original rows only)
        Data/Processed/test_data.csv   49 features + Name, Label, Family

Steps, using the functions in Preprocessing.py:

  1. drop ``Ratio_DeciDig`` (50 -> 49 features), the feature removed by the
     Pearson-correlation screen at |r| >= 0.9 on the paper's dataset.  The
     screen is re-run and reported; a warning is printed if on your data it
     would remove a different set;
  2. 80/20 split, ``train_test_split(test_size=0.2, random_state=2345)``;
  3. Min-Max scaling fitted on the TRAINING split only and applied to both.

SMOTE is not applied here: run_experiments.py holds out validation first and
oversamples only the remaining training subset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.Preprocessing.Preprocessing import (  # noqa: E402
    drop_features_by_correlation, load_dataset, scale_dataset, split_dataset,
)

DROPPED_FEATURES = ("Ratio_DeciDig",)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("Data/Raw/labeled_dataset_features.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("Data/Processed"))
    args = ap.parse_args()

    df, features = load_dataset(args.input)
    print(f"{len(df):,} rows, {len(features)} features")
    screened, _, _ = drop_features_by_correlation(df)
    if list(screened) != list(DROPPED_FEATURES):
        print(f"WARNING: on this data the |r| >= 0.9 screen would drop {screened}; "
              f"dropping {DROPPED_FEATURES} as in the paper.")
    df = df.drop(columns=list(DROPPED_FEATURES))
    print(f"dropped {DROPPED_FEATURES}; {df.shape[1] - 3} features remain")

    X_train, y_train, X_test, y_test = split_dataset(df)
    X_train, X_test = scale_dataset(X_train, X_test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    X_train.join(y_train).to_csv(args.output_dir / "train_data.csv", index=False)
    X_test.join(y_test).to_csv(args.output_dir / "test_data.csv", index=False)
    print(f"train {len(X_train):,} rows {y_train['Label'].value_counts().to_dict()}; "
          f"test {len(X_test):,} rows {y_test['Label'].value_counts().to_dict()}")
    print(f"written to {args.output_dir}")


if __name__ == "__main__":
    main()
