# House Price Prediction – Ames Housing Dataset

A beginner-friendly, end-to-end machine learning project that predicts house sale prices using the Ames Housing dataset. It follows the workflow from Chapter 2 of *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (Aurélien Géron), from exploring the data through to evaluating the final model.

📓 **Notebook:** [`Ames_Housing_2(ML)_py.ipynb`](Ames_Housing_2(ML)_py.ipynb)

## Dataset

- **Ames Housing** (Dean De Cock): 2,930 houses sold in Ames, Iowa, described by 82 columns.
- **Target:** `SalePrice`
- The CSV is **not included** in this repo. Download `AmesHousing.csv` (it is available on Kaggle and in other public mirrors) and place it next to the notebook.

## Workflow

1. **Data exploration:** shape, data types, summary statistics, missing values, category counts.
2. **EDA:** histograms, correlation with `SalePrice`, living area vs. price scatter plot.
3. **Train/test split:** 80/20, `random_state=42`.
4. **Preprocessing:**
   - Numeric features: median imputation, then `StandardScaler`.
   - Categorical features: most-frequent imputation, then `OneHotEncoder`.
5. **Models:**
   - Linear Regression
   - Decision Tree
   - Random Forest (10-fold cross-validation)
   - Random Forest tuned with `GridSearchCV`
6. **Evaluation:** RMSE and R², feature importances, predicted vs. actual plots, residual analysis and the largest errors.

## Results (test set)

| Model | RMSE ($) | R² |
| --- | --- | --- |
| Linear Regression | 29,635 | 0.890 |
| Decision Tree | 35,482 | 0.843 |
| Random Forest (default) | 26,424 | – |
| Random Forest (grid-searched: `n_estimators=30`, `max_features=8`) | 31,423 | 0.877 |

Cross-validation RMSE on the training set was about **38,321** for the Decision Tree and about **25,965** for the default Random Forest.

The most important features were `Garage Cars`, `Overall Qual`, `Gr Liv Area`, `Garage Yr Blt` and `Total Bsmt SF`.

> **Note:** the grid search only tried `max_features` values up to 8, far fewer than the ~300 one-hot-encoded features. That is why the "tuned" forest scores worse than the default one. A wider search, such as `max_features` values of `'sqrt'`, `0.3` and `0.5` with more trees, would be a natural next step.

## Getting started

```bash
git clone https://github.com/buvann-11/Ames-Housing.git
cd Ames-Housing
pip install -r requirements.txt
# put AmesHousing.csv in this folder, then:
jupyter notebook "Ames_Housing_2(ML)_py.ipynb"
```

## Tech stack

Python · NumPy · pandas · Matplotlib · scikit-learn · Jupyter
