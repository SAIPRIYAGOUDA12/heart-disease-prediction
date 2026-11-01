# Heart Disease Prediction

Machine-learning pipeline that predicts the presence of heart disease from clinical measurements. Five classifiers (Logistic Regression, Random Forest, KNN, XGBoost and a small Deep Neural Network) are trained and compared on the **UCI Heart Disease** dataset. The **Framingham** dataset is included as a second experiment.

> **Disclaimer:** educational project only. It is not a medical device and must not be used for diagnosis.

## Results (UCI, 80/20 split, `random_state=42`, 184 test patients)

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.884 | 0.771 | 0.858 |
| Random Forest | 0.810 | 0.878 | 0.789 | **0.879** |
| KNN (k=5) | 0.793 | 0.845 | 0.798 | 0.849 |
| **XGBoost** | **0.832** | 0.882 | **0.826** | 0.870 |
| Deep Neural Network | 0.804 | 0.876 | 0.780 | 0.866 |

XGBoost has the best accuracy and recall; Random Forest has the best ROC-AUC. With only 184 test patients, differences of one or two points are within noise. The neural network's numbers vary a little from run to run.

Top Random Forest features: `chol`, `thalch` (max heart rate), `age`, `oldpeak`, `exang`.

![Model comparison](results/figures/uci_model_comparison.png)

More plots (confusion matrices, ROC curves, feature importance, NN training curve) are in [`results/figures/`](results/figures).

### Framingham (extra experiment)

Logistic Regression, Random Forest and KNN reach 83–86% accuracy but **recall of only about 5–7%** (ROC-AUC 0.59–0.71), because just 15% of patients are positive and no class balancing is applied. Accuracy is misleading here; handling the imbalance (class weights, resampling, threshold tuning) is the main next step. See `results/metrics_framingham.csv`. XGBoost and the neural network are supported by the script for this dataset but were not run for the reported numbers.

## Project structure

```
heart-disease-prediction/
├── data/
│   ├── raw/                  # UCIdataset.csv, FraminghamDataset.csv
│   └── README.md             # column descriptions
├── notebooks/
│   └── HeartDiseasePrediction.ipynb   # step-by-step UCI walkthrough
├── src/
│   ├── preprocess.py         # loading, cleaning, split, scaling
│   └── train.py              # trains/evaluates all models, saves results
├── results/
│   ├── metrics_uci.csv
│   ├── metrics_framingham.csv
│   ├── feature_importance_*.csv
│   └── figures/
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting started

```bash
git clone https://github.com/SAIPRIYAGOUDA12/heart-disease-prediction.git
cd heart-disease-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Notebook:** `jupyter notebook notebooks/HeartDiseasePrediction.ipynb`

**Script:**

```bash
python src/train.py                        # UCI dataset
python src/train.py --dataset framingham   # Framingham dataset
python src/train.py --skip-dnn             # skip TensorFlow
python src/train.py --skip-xgboost         # skip XGBoost
```

If `xgboost` or `tensorflow` is not installed, `train.py` skips those models with a warning. Outputs go to `results/`.

## Methodology

1. Load the dataset; replace `'?'` with `NaN` and fill missing values with the column mean.
2. UCI: drop `id`, `dataset` and the categorical columns (`sex`, `cp`, `restecg`, `slope`, `thal`); binarise the target (`num > 0`). The eight remaining numeric features are `age`, `trestbps`, `chol`, `fbs`, `thalch`, `exang`, `oldpeak`, `ca`.
3. 80/20 train/test split; standardise (fit on train only) for Logistic Regression, KNN and the neural network. Tree models use raw features.
4. Evaluate with accuracy, precision, recall, ROC-AUC, confusion matrices and ROC curves.

Models: Logistic Regression (`max_iter=1000`), Random Forest (100 trees), KNN (k=5), XGBoost (100 trees, depth 4, lr 0.1), and a neural network (32 → 16 → 1, Adam, 50 epochs, batch 16, 20% validation split).

## Limitations and next steps

- Categorical UCI features are dropped; one-hot encoding them would likely help.
- `chol = 0` in 172 UCI rows is really "missing" but is treated as a value.
- Mean imputation is computed on the full dataset before the split (slight leakage), and results come from a single split. Cross-validation and a `Pipeline` would give more reliable estimates.
- Address class imbalance on Framingham and report recall / PR-AUC there.
- Add hyperparameter tuning and model persistence.

## Data sources

- UCI Machine Learning Repository: Heart Disease dataset (Cleveland, Hungary, Switzerland, VA Long Beach).
- Framingham Heart Study cardiovascular risk dataset.

## License

Code is released under the [MIT License](LICENSE). Datasets remain subject to their original terms.
