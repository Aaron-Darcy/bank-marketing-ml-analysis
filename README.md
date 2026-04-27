# Bank Marketing ML Analysis

MSc in Data Science — Machine Learning Project, ATU Donegal, Semester 2 2025-2026.

Predicting whether a client will subscribe to a bank term deposit using the UCI Bank Marketing dataset. The project covers end-to-end machine learning: preprocessing, exploratory data analysis, dimensionality reduction, model training with hyperparameter tuning, and evaluation.

**Team:** Aaron Darcy, Abey Mathew

---

## Dataset

UCI Bank Marketing dataset — `bank-full.csv` (45,211 rows, 16 features).

Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)

The target variable is `y` — whether the client subscribed to a term deposit (yes/no). The dataset has a significant class imbalance: approximately 88% no, 12% yes.

---

## Project Structure

```
bank-marketing-ml-analysis/
├── data/
│   └── bank-full.csv          # Raw dataset (included so notebooks run out of the box)
├── notebooks/
│   ├── 01_preprocessing_eda.ipynb   # Preprocessing and exploratory data analysis
│   ├── 02_modelling.ipynb           # Dimensionality reduction and model training
│   └── 03_evaluation.ipynb          # Model evaluation and comparison
├── figures/                   # Generated plots (created when notebooks are run)
├── requirements.txt
└── README.md
```

---

## Notebooks

### 01 — Preprocessing and EDA

- Removes the `duration` feature (data leakage — only known after the call ends)
- Handles 'unknown' values: mode imputation for `job` and `education` only; `contact` and `poutcome` unknowns are kept as valid categories
- Handles the `pdays` sentinel value (-1 = never contacted before) by adding a binary `contacted_before` flag
- One-hot encodes all categorical features (drop_first=True)
- Stratified 80/20 train/test split (36,168 train / 9,043 test, 40 features)
- EDA: class distribution, numeric feature histograms, correlation heatmap, subscription rates by category, box plots

### 02 — Dimensionality Reduction and Modelling

**Dimensionality reduction:**
- PCA: full component analysis, scree plot, cumulative variance, 2D scatter. 31 components retain 95% of variance.
- t-SNE: PCA pre-reduction to 40 components, 5,000-row sample, 2D embedding coloured by class

**Modelling:**
- Three classifiers trained using scikit-learn Pipelines (StandardScaler + classifier)
- Cross-validation: StratifiedKFold(5), scoring on ROC-AUC
- Random Forest — RandomizedSearchCV, 30 iterations
- Gradient Boosted Trees — RandomizedSearchCV, 30 iterations
- Support Vector Machine (RBF kernel) — GridSearchCV, 12 parameter combinations
- Models saved to `data/*.pkl`

### 03 — Evaluation

- Confusion matrices, classification reports
- ROC curves and Precision-Recall curves for all three models
- Feature importance plots (RF and GBT)
- Model comparison table and bar chart across key metrics
- Conclusions and limitations

---

## Results Summary

| Model | ROC-AUC | Recall (Yes) | F1 (Yes) | Avg Precision |
|---|---|---|---|---|
| Random Forest | **0.8054** | 0.5388 | **0.4734** | 0.4532 |
| Gradient Boosted Trees | 0.8036 | 0.2656 | 0.3752 | **0.4661** |
| SVM | 0.7881 | **0.5907** | 0.4575 | 0.3935 |

Random Forest is the recommended model — it achieves the highest ROC-AUC and best F1 for the positive class, making it the strongest choice for identifying potential subscribers in a campaign with significant class imbalance.

---

## How to Run

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the notebooks in order:

```
notebooks/01_preprocessing_eda.ipynb
notebooks/02_modelling.ipynb
notebooks/03_evaluation.ipynb
```

Notebook 01 saves preprocessed splits to `data/`. Notebook 02 trains models and saves them to `data/*.pkl`. Notebook 03 loads those files for evaluation — so they must be run in sequence.

> Note: The SVM training cell in notebook 02 can take 30-60 minutes depending on hardware.

---

## Requirements

- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, scipy, jupyter

Full list in `requirements.txt`.
