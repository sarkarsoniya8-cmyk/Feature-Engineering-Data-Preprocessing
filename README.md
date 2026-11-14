# Feature Engineering & Data Preprocessing

Internship submission demonstrating missing value imputation, domain feature engineering, categorical encoding (`OrdinalEncoder`, `OneHotEncoder`, `LabelEncoder`), feature scaling (`StandardScaler`, `MinMaxScaler`, `RobustScaler`), and feature selection (`SelectKBest` ANOVA F-test & Mutual Information) on the Breast Cancer Wisconsin Diagnostic Dataset.

## Repository Structure
```text
├── data/
│   ├── breast_cancer_raw_with_categories.csv       # Raw dataset prior to preprocessing
│   └── breast_cancer_preprocessed_engineered.csv   # Model-ready preprocessed & engineered dataset
├── notebooks/
│   └── feature_engineering_preprocessing.ipynb     # Executable Jupyter / Google Colab notebook
├── reports/
│   ├── preprocessing_and_feature_engineering_summary.md  # Explanatory summary document
│   ├── feature_selection_scores.csv                # ANOVA F-scores & Mutual Information rankings
│   ├── top15_feature_selection.png                 # Top 15 selected features chart
│   └── scaling_methods_comparison.png              # Boxplot comparison across scaling methods
├── .gitignore
├── README.md
└── requirements.txt
```

## Quick Start
```bash
pip install -r requirements.txt
jupyter notebook notebooks/feature_engineering_preprocessing.ipynb
```
