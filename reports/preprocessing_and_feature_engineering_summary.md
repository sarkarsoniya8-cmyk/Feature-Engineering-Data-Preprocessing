# Feature Engineering & Data Preprocessing Documentation

## 1. Why Data Preprocessing Was Required
Raw diagnostic measurements cannot be fed directly into machine learning algorithms without degrading convergence speed, numerical stability, and predictive accuracy:

1. **Extreme Scale Disparities Across Features:** Raw numerical attributes vary by more than four orders of magnitude—for example, `mean area` ranges from `143.5` to `2501.0`, whereas `mean smoothness` ranges from `0.05` to `0.16`. Without standardization (`StandardScaler`), gradient-based optimizers (Logistic Regression, Neural Networks) and distance-based algorithms (SVM, KNN) allow high-magnitude features to dominate distance calculations and gradient updates.
2. **Non-Numeric Categorical Attributes:** Machine learning estimators require numeric tensors. Raw string categories (`tumor_size_grade`, `texture_pattern`, `diagnosis_label`) had to be encoded using mathematically appropriate schemes—preserving order for ordinal grades (`Small < Medium < Large`) while avoiding false rank assumptions on nominal texture categories (`Smooth`, `Irregular`, `Rough`).
3. **Missing Values:** Real-world clinical exports often contain incomplete measurements (`mean smoothness` and `mean compactness` had missing records). Median imputation prevented row deletion while remaining robust against right-skewed outliers.
4. **Redundancy & Dimensionality:** Across 39 total engineered and encoded columns, univariate statistical filtering (`SelectKBest` with ANOVA F-test and Mutual Information) identified the highest-signal predictors and highlighted newly created features that outperform several raw baseline columns.

---

## 2. Summary of Newly Engineered Features

| Engineered Feature | Formula / Construction | Domain Rationale & Statistical Impact |
| :--- | :--- | :--- |
| **`area_perimeter_ratio`** | `mean area / mean perimeter` | Captures tumor compactness and mass density relative to boundary length. Ranked **#9 overall** (`ANOVA F = 596.37`, $p = 1.5 \times 10^{-90}$), outperforming raw `mean area` (`F = 573.06`). |
| **`nuclear_malformation_score`** | `worst concavity + worst concave points + worst compactness` | Combines three correlated worst-case morphological severity indicators into a single composite risk index (`ANOVA F = 497.43`). |
| **`concavity_smoothness_index`** | `mean concavity * mean smoothness` | Non-linear interaction term capturing tumors that simultaneously exhibit deep boundary indentations and rapid surface variation (`ANOVA F = 469.72`). |
| **`circularity_deviation`** | `abs((mean perimeter / (2 * pi * mean radius)) - 1)` | Measures structural departure from an ideal circular geometry ($P / 2\pi r = 1$); malignant tumors exhibit irregular, lobulated contours (`ANOVA F = 427.64`). |
| **`radius_worst_to_mean_ratio`** | `worst radius / mean radius` | Quantifies localized asymmetric tumor protrusion relative to its average cross-sectional radius. |

---

## 3. Encoding & Scaling Techniques Applied

- **Ordinal Encoding (`tumor_size_grade_encoded`):** Mapped `Small -> 0`, `Medium -> 1`, `Large -> 2`, achieving an ANOVA F-Score of **599.33** (#8 overall).
- **One-Hot Encoding (`texture_pattern_*`):** Expanded nominal `texture_pattern` into binary indicator columns (`Smooth`, `Irregular`, `Rough`) via `OneHotEncoder`.
- **Scaling Comparison:** Evaluated `StandardScaler` (zero mean, unit variance), `MinMaxScaler` (`[0, 1]` bounded range), and `RobustScaler` (median/IQR centering). `StandardScaler` was selected for the final exported dataset (`data/breast_cancer_preprocessed_engineered.csv`), boosting 5-Fold Cross-Validation accuracy from **95.43%** (raw unscaled data) to **97.37%**.
