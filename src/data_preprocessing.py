"""
data_preprocessing.py
----------------------
Loading, inspection and cleaning of the raw Student Performance dataset.

Dataset: UCI Machine Learning Repository - "Student Performance" Data Set
Source : https://archive.ics.uci.edu/dataset/320/student+performance
File   : student-mat.csv (Math course, 395 students, 33 attributes)
Citation: P. Cortez and A. Silva. Using Data Mining to Predict Secondary
          School Student Performance. In A. Brito and J. Teixeira Eds.,
          Proceedings of 5th FUBUTEC 2008, pp. 5-12, Porto, Portugal, 2008.

Why these steps are necessary
------------------------------
* Loading & inspection: confirms the data was read correctly (row/column
  counts, dtypes) before any transformation is trusted.
* Missing-value handling: the raw UCI file has NO missing values, but the
  pipeline still uses SimpleImputer defensively so the same code works if a
  lecturer swaps in a noisier real-world extract.
* Duplicate removal: guards against accidental double-counted students,
  which would bias both training and evaluation.
* Invalid-value checks: grades (G1/G2/G3) must lie in [0, 20]; a handful of
  categorical columns must only contain their documented levels.
* Encoding & scaling are handled later inside the scikit-learn
  ColumnTransformer (see train_model.py) so that the *exact* same
  transformation is guaranteed at both training and prediction time
  (this is how we prevent train/prediction preprocessing mismatch).
"""

from pathlib import Path

import pandas as pd

from config import RAW_DATA_PATH, get_logger

logger = get_logger(__name__)


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV file into a DataFrame."""
    df = pd.read_csv(path)
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Return a small dictionary summary used for reporting / sanity checks."""
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
      1. Drop exact duplicate rows (if any).
      2. Clip grade columns to the documented valid range [0, 20].
      3. Strip whitespace from string/categorical columns.
      4. Reset the index after any row removal.
    """
    df = df.copy()

    # 1. Duplicates
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.info(f"Removed {removed} duplicate row(s).")

    # 2. Invalid grade values -> clip to the documented 0-20 scale
    for col in ["G1", "G2", "G3"]:
        if col in df.columns:
            invalid = ((df[col] < 0) | (df[col] > 20)).sum()
            if invalid:
                logger.info(f"Clipping {invalid} invalid value(s) in '{col}'.")
            df[col] = df[col].clip(lower=0, upper=20)

    # 3. Strip whitespace on object columns
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # 4. Reset index
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = load_data()
    logger.info("Raw data summary:")
    for k, v in inspect_data(data).items():
        if k != "dtypes":
            logger.info(f"  {k}: {v}")
    cleaned = clean_data(data)
    logger.info(f"Cleaned data shape: {cleaned.shape}")
