<<<<<<< HEAD
# Task 5 - Feature Engineering & Tuned Ensembles (Ames House Prices)

Engineer new features, then compare a tuned Random Forest and a tuned XGBoost
against a Task 3 style baseline, using 5-fold cross-validation.

## Folder

| File | What it is |
|---|---|
| `House_Prices_Task5.ipynb` | Step-by-step notebook with explanations (submit this) |
| `run_experiments.py` | Same experiment from the command line, writes `results/` |
| `task5_lib.py` | All the code (features, pipelines, CV, tuning, plots) |
| `data/train.csv` | Ames House Prices training data (1460 rows, 81 columns) |
| `results/` | Comparison table, report, plots, best hyperparameters |
| `requirements.txt` | Python packages |

## How to run

```bash
pip install -r requirements.txt

# Option A: notebook (recommended for submission)
jupyter notebook House_Prices_Task5.ipynb      # then Kernel > Restart & Run All

# Option B: command line
python run_experiments.py            # full run, about 5-10 minutes
python run_experiments.py --quick    # smoke test, about 1-2 minutes
```

Run both from inside this folder. Needs Python 3.9 or newer.

If `xgboost` is missing the code does not crash: it falls back to scikit-learn's
`GradientBoostingRegressor` and labels that row "XGBoost NOT installed - stand-in".
For the real task, install xgboost (it is in `requirements.txt`) and run again.

## What the code does

1. One 80/20 hold-out split (seed 42). The hold-out set is used once, at the end.
2. 7 engineered features, each justified in writing: `TotalSF`, `HouseAge`,
   `TotalBath`, `HasRemodeled`, `RemodAge`, `TotalPorchSF`, `HasPool_or_Fireplace`.
3. Target is `log1p(SalePrice)`, so every RMSE is on the log scale
   (0.13 is roughly a 13% typical error).
4. Feature engineering, imputation and encoding are inside a scikit-learn `Pipeline`,
   so they are re-fitted on the training part of every fold (no leakage).
5. 5-fold CV (same folds for every model), reported as mean +- std of RMSE.
6. Random Forest (5 hyperparameters) and XGBoost (7 hyperparameters), each tuned
   with `RandomizedSearchCV` on the same 5 folds.
7. One comparison table (model, mean CV RMSE, mean CV R2, training time, note),
   a "is it more than noise?" check, and a verdict.

## Before you submit

- The baseline is rebuilt here as a default Random Forest on the original
  features, scored on the same 5 folds. If your real Task 3 baseline was a
  different model, quote its number next to this one.
- In the notebook, write your own 2-4 sentence verdict in the last cell.
  `results/report.md` already contains a verdict written from the numbers.
=======
# AMES-HOUSE-PRICE-PREDICTION
>>>>>>> a45ed1a9d4f144003c124cbf8a314794dcd78f9c
