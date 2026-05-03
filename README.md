# Bank Customer Churn — ML Classification

Binary classification project predicting whether a bank customer **churns** (`Exited = 1`) or **stays** (`Exited = 0`). The pipeline covers preprocessing, then two supervised models evaluated with standard classification metrics.

## Dataset

| File | Description |
|------|-------------|
| `data/bank_churn.csv` | Raw tabular data (~10k rows): demographics, account behavior, geography, and target `Exited`. |
| `data/x_train.csv`, `data/x_test.csv` | Preprocessed **features** (scaled numeric columns + one-hot geography). Train: 8,000 rows; test: 2,000 rows; **12** features. |
| `data/y_train.csv`, `data/y_test.csv` | Target column **`Exited`** aligned with the corresponding `x_*` splits. |

**Feature columns** in `x_train.csv` / `x_test.csv`: `CreditScore`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, and one-hot `Geography_France`, `Geography_Germany`, `Geography_Spain`.

## Project layout

```
bank-churn-ml-classification/
├── data/                      # Raw CSV and train/test splits
├── notebooks/
│   ├── 00_preprocessing.ipynb # Load raw data, split, scale → x_*/y_* CSVs
│   ├── 01_logistic_regression.ipynb
│   ├── 02_random_forest.ipynb
│   ├── 03_knn.ipynb
│   └── 04_decision_tree.ipynb
├── README.md
├── members.txt                # Team IDs / emails (per assignment)
└── submission.txt             # Dataset + GitHub + YouTube links (per assignment)
```

## Requirements

- Python **3.9+** (3.10+ recommended)
- [Jupyter](https://jupyter.org/) or [VS Code](https://code.visualstudio.com/) with Jupyter support

Install dependencies (use the same Python as your Jupyter kernel):

```bash
python -m pip install -r requirements.txt
```

Or without the file: `pip install pandas numpy matplotlib seaborn scikit-learn jupyter`

## Running the notebooks

1. **Preprocessing** — Open `notebooks/00_preprocessing.ipynb` and run all cells. It expects the raw CSV (e.g. `bank_churn.csv`) and writes or refreshes the files under `data/`.

2. **Models** — Run `01_logistic_regression.ipynb`, `02_random_forest.ipynb`, and `03_knn.ipynb` after splits exist.

**Paths:** Some cells use **Google Colab** + Google Drive paths (e.g. `/content/drive/...`). For local runs, change `read_csv` paths to point at this repo, for example:

```python
from pathlib import Path

ROOT = Path("..")  # or Path("." ) if the notebook cwd is the repo root
X_train = pd.read_csv(ROOT / "data" / "x_train.csv")
```

## Models

| Notebook | Algorithm | Typical use |
|----------|-----------|-------------|
| `01_logistic_regression.ipynb` | Logistic regression | Linear decision boundary, calibrated probabilities |
| `02_random_forest.ipynb` | Random forest | Nonlinear interactions, feature importance |
| `03_knn.ipynb` | K-nearest neighbors | Distance-based voting; uses the same scaled features as other models |
| `04_decision_tree.ipynb` | Decision tree | Interpretable splits, handles nonlinear patterns, no scaling required |

These notebooks report metrics such as accuracy, precision, recall, F1, ROC-AUC, and confusion matrices where implemented.
