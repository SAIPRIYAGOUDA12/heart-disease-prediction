"""Data loading and preprocessing for the heart disease datasets.

Mirrors the methodology of the original notebook so results are reproducible:
  * '?' placeholders -> NaN, remaining NaNs filled with the column mean
  * UCI: target ``num`` is binarised (0 = no disease, 1-4 = disease)
  * UCI: categorical columns are dropped (as in the original notebook)
  * 80/20 train/test split with ``random_state=42`` and standard scaling
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"

UCI_CATEGORICAL = ["sex", "cp", "restecg", "slope", "thal"]
UCI_DROP = ["id", "dataset"]


def _fill_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")
    return df.fillna(df.mean())


def load_uci(path: Path = DATA_DIR / "UCIdataset.csv"):
    """Return (X, y) for the UCI Heart Disease dataset (binary target)."""
    df = pd.read_csv(path)
    df = df.drop(columns=UCI_DROP + UCI_CATEGORICAL)
    df = _fill_with_mean(df)
    y = (df["num"] > 0).astype(int)
    X = df.drop(columns="num")
    return X, y


def load_framingham(path: Path = DATA_DIR / "FraminghamDataset.csv"):
    """Return (X, y) for the Framingham dataset (target: TenYearCHD)."""
    df = _fill_with_mean(pd.read_csv(path))
    y = df["TenYearCHD"].astype(int)
    X = df.drop(columns="TenYearCHD")
    return X, y


LOADERS = {"uci": load_uci, "framingham": load_framingham}


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """80/20 split plus a StandardScaler fitted on the training set only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_s, X_test_s
