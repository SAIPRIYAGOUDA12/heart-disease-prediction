"""Train and compare heart disease classifiers.

Usage:
    python src/train.py                      # UCI dataset (default)
    python src/train.py --dataset framingham
    python src/train.py --skip-dnn           # skip TensorFlow model

Outputs (per dataset) are written to ``results/``:
    metrics_<dataset>.csv, feature_importance_<dataset>.csv, figures/*.png
"""
import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier

from preprocess import LOADERS, ROOT, split_and_scale

SEED = 42
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def evaluate(name, y_test, y_pred, y_prob, tag, store):
    """Print metrics, save confusion-matrix + ROC figures, record a row."""
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    store.append(row)
    print(f"\n===== {name.upper()} =====")
    print({k: round(v, 4) for k, v in row.items() if k != "Model"})
    print(classification_report(y_test, y_pred))

    slug = f"{tag}_{name.lower().replace(' ', '_')}"
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(FIGURES / f"{slug}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    RocCurveDisplay.from_predictions(y_test, y_prob, name=name)
    plt.title(f"{name} - ROC Curve")
    plt.savefig(FIGURES / f"{slug}_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=LOADERS, default="uci")
    ap.add_argument("--skip-dnn", action="store_true", help="skip the TensorFlow model")
    ap.add_argument("--skip-xgboost", action="store_true", help="skip XGBoost")
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    tag = args.dataset

    X, y = LOADERS[tag]()
    print(f"Dataset: {tag}  |  samples: {len(X)}  |  features: {list(X.columns)}")
    print(f"Positive class rate: {y.mean():.3f}")
    Xtr, Xte, ytr, yte, Xtr_s, Xte_s = split_and_scale(X, y, random_state=SEED)

    rows = []

    # Logistic Regression (scaled)
    lr = LogisticRegression(max_iter=1000).fit(Xtr_s, ytr)
    evaluate("Logistic Regression", yte, lr.predict(Xte_s), lr.predict_proba(Xte_s)[:, 1], tag, rows)

    # Random Forest (unscaled) + feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(Xtr, ytr)
    evaluate("Random Forest", yte, rf.predict(Xte), rf.predict_proba(Xte)[:, 1], tag, rows)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.to_csv(RESULTS / f"feature_importance_{tag}.csv", header=["importance"], index_label="feature")
    imp.sort_values().plot(kind="barh", title="Random Forest Feature Importance")
    plt.savefig(FIGURES / f"{tag}_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # KNN (scaled)
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr_s, ytr)
    evaluate("KNN", yte, knn.predict(Xte_s), knn.predict_proba(Xte_s)[:, 1], tag, rows)

    # XGBoost (optional dependency)
    if not args.skip_xgboost:
        try:
            from xgboost import XGBClassifier

            xgb = XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=4,
                random_state=SEED, eval_metric="logloss",
            ).fit(Xtr, ytr)
            evaluate("XGBoost", yte, xgb.predict(Xte), xgb.predict_proba(Xte)[:, 1], tag, rows)
        except ImportError:
            warnings.warn("xgboost not installed - skipping XGBoost")

    # Deep neural network (optional dependency)
    if not args.skip_dnn:
        try:
            import tensorflow as tf
            from tensorflow.keras.layers import Dense, Input
            from tensorflow.keras.models import Sequential

            tf.keras.utils.set_random_seed(SEED)
            nn = Sequential([
                Input(shape=(Xtr_s.shape[1],)),
                Dense(32, activation="relu"),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid"),
            ])
            nn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            hist = nn.fit(Xtr_s, ytr, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
            prob = nn.predict(Xte_s, verbose=0).ravel()
            evaluate("Deep Neural Network", yte, (prob > 0.5).astype(int), prob, tag, rows)

            plt.figure(figsize=(8, 5))
            plt.plot(hist.history["accuracy"], label="Training")
            plt.plot(hist.history["val_accuracy"], label="Validation")
            plt.title("Neural Network: Training vs Validation Accuracy")
            plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
            plt.savefig(FIGURES / f"{tag}_nn_training_curve.png", dpi=150, bbox_inches="tight")
            plt.close()
        except ImportError:
            warnings.warn("tensorflow not installed - skipping the neural network")

    # Comparison table + bar chart
    results = pd.DataFrame(rows).set_index("Model")
    results.round(4).to_csv(RESULTS / f"metrics_{tag}.csv")
    print("\n===== MODEL COMPARISON =====")
    print(results.round(4))
    results[["Accuracy", "ROC-AUC"]].plot(kind="bar", figsize=(10, 5), ylim=(0, 1),
                                          title=f"Model comparison ({tag})")
    plt.xticks(rotation=20, ha="right")
    plt.savefig(FIGURES / f"{tag}_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
