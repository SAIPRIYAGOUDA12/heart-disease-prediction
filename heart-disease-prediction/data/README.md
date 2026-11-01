# Data

Raw datasets live in `data/raw/`. The data files keep the terms of their original sources; the repo's MIT license covers the code only.

## `UCIdataset.csv` — UCI Heart Disease (920 rows × 16 columns)

Patients from four sources: Cleveland (304), Hungary (293), VA Long Beach (200) and Switzerland (123).

| Column | Description |
|---|---|
| `id` | Patient identifier (dropped) |
| `age` | Age in years |
| `sex` | Male / Female |
| `dataset` | Source of the record (dropped) |
| `cp` | Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG result |
| `thalch` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak-exercise ST segment |
| `ca` | Number of major vessels (0–3) coloured by fluoroscopy |
| `thal` | Thalassemia (normal, fixed defect, reversible defect) |
| `num` | **Target**: 0 = no disease, 1–4 = disease severity (binarised to 0/1 in this project) |

Known quirks: `ca`, `thal` and `slope` are mostly missing; `chol` is recorded as `0` in 172 rows and `oldpeak` as `0` in 370 rows (many `chol = 0` values are almost certainly "not measured").

## `FraminghamDataset.csv` — Framingham Heart Study (4,240 rows × 16 columns)

Demographic, behavioural and clinical risk factors (`male`, `age`, `education`, `currentSmoker`, `cigsPerDay`, `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes`, `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose`).

**Target:** `TenYearCHD` — 10-year risk of coronary heart disease (1 = yes). Only about 15% of patients are positive, so the classes are strongly imbalanced.
