# Task 5 - Feature Engineering & Tuned Ensembles (Ames House Prices)

> **Note:** the `xgboost` package was not available when this report was produced, so sklearn's `GradientBoostingRegressor` was used as a stand-in for the boosting model. Install xgboost (`pip install xgboost`) and re-run `python run_experiments.py` to get the real XGBoost row.

## 1. Setup

- Data: Ames House Prices `train.csv`, 1460 rows, 81 columns.
- Target: `log1p(SalePrice)`. Sale prices are right-skewed; the log makes errors relative.
  **All RMSE values below are on the log scale** (an RMSE of 0.13 means a typical error of roughly 13% of the price).
- Split: 1168 train / 292 hold-out (80/20, seed 42). The hold-out set is evaluated **once**, at the end.
- Validation: the same 5-fold shuffled `KFold` (seed 42) for every model, run on the train split only.
- Leakage control: feature engineering, imputation and one-hot encoding are steps inside a scikit-learn
  `Pipeline`, so they are re-fitted on the training part of every fold. No feature uses `SalePrice`.

## 2. Engineered features (each justified)

- **TotalSF**: TotalBsmtSF + 1stFlrSF + 2ndFlrSF. Buyers think in total usable area, not in separate floors. One strong, additive size signal is easier for a tree to split on than three partial ones.
- **HouseAge**: YrSold - YearBuilt (clipped at 0). The raw year does not say how old the house was when it sold; age at sale does, and older houses usually sell for less, all else equal.
- **TotalBath**: FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath. Bathrooms are split over four columns; a half bath is worth roughly half a full one. The total is what a buyer counts.
- **HasRemodeled**: 1 if YearRemodAdd != YearBuilt else 0. A remodel flag captures 'updated house' separately from raw age.
- **RemodAge**: YrSold - YearRemodAdd (clipped at 0). Years since the last update; a recent remodel makes an old house feel newer.
- **TotalPorchSF**: OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF. Five sparse outdoor-area columns summed into one usable 'outdoor living space' number.
- **HasPool_or_Fireplace**: 1 if the house has a pool or at least one fireplace. Both are rare luxury amenities, so a single binary flag is less noisy than the raw counts.

Check (training split only, shown for reporting and not used to pick features):

| feature | corr_with_log_price | min | max |
|---|---|---|---|
| TotalSF | 0.762 | 334.0 | 11752.0 |
| HouseAge | -0.577 | 0.0 | 136.0 |
| TotalBath | 0.659 | 1.0 | 6.0 |
| HasRemodeled | -0.074 | 0.0 | 1.0 |
| RemodAge | -0.564 | 0.0 | 60.0 |
| TotalPorchSF | 0.396 | 0.0 | 1027.0 |
| HasPool_or_Fireplace | 0.507 | 0.0 | 1.0 |

All features are computed from columns known before the sale (areas, years, counts). None is built from the target or from price bins.

## 3. Hyperparameter tuning

`RandomizedSearchCV`, n_iter=30, same 5 folds, scoring = RMSE, best model refit on the full train split.

- **Random Forest (tuned) + features**: `{'max_depth': 15, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300}`
- **GradientBoosting (tuned)**: `{'learning_rate': 0.10196967939171485, 'max_depth': 3, 'min_samples_leaf': 2, 'n_estimators': 645, 'subsample': 0.9706635463175177}`

## 4. Comparison table (5-fold CV on the train split)

| Model | Mean CV RMSE (log) | Mean CV R² | Training time (s) | Final fit time (s) | Note |
|---|---|---|---|---|---|
| Baseline RF (default, original features) | 0.1475 ± 0.0177 | 0.8568 ± 0.0225 | 17.2 | 5.32 | Task 3 style baseline: no new features, no tuning. |
| RF (default) + engineered features | 0.1433 ± 0.0166 | 0.8649 ± 0.0185 | 17.4 | 5.73 | Same algorithm and settings; only the features changed. |
| Random Forest (tuned) + features | 0.1381 ± 0.0183 | 0.8746 ± 0.0215 | 210.9 | 3.52 | Bagging: many deep trees averaged, robust, low variance. |
| GradientBoosting (tuned) [XGBoost NOT installed - stand-in] | 0.1220 ± 0.0155 | 0.9020 ± 0.0168 | 215.3 | 5.98 | Boosting: shallow trees added one by one to fix earlier errors; more tuning-sensitive. |

"Training time" is the wall-clock time of the whole 5-fold CV (untuned rows) or the whole random search (tuned rows).
"Final fit time" is the time to fit one model once on the full train split.

Per-fold RMSE (identical folds for every model):

|  | Baseline RF (default, original features) | RF (default) + engineered features | Random Forest (tuned) + features | GradientBoosting (tuned) |
|---|---|---|---|---|
| fold 1 | 0.1482 | 0.1480 | 0.1434 | 0.1265 |
| fold 2 | 0.1519 | 0.1517 | 0.1473 | 0.1201 |
| fold 3 | 0.1741 | 0.1638 | 0.1610 | 0.1462 |
| fold 4 | 0.1353 | 0.1305 | 0.1203 | 0.1088 |
| fold 5 | 0.1280 | 0.1227 | 0.1185 | 0.1084 |

## 5. Do the improvements beat noise?

Adding the engineered features alone (same default Random Forest) reduced CV RMSE by 0.0042 (2.8%), improving 5/5 folds. The best final model reduced it by 0.0255 (17.3%) versus the baseline, so the features account for roughly 16% of the total gain and tuning/algorithm choice for the rest.

- Random Forest (tuned) vs baseline: RMSE drop +0.0094 (+6.4%), 5/5 folds improved, paired t p≈0.011.
- GradientBoosting (tuned) vs baseline: RMSE drop +0.0255 (+17.3%), 5/5 folds improved, paired t p≈0.000.

Rule used for "meaningful": all folds improve, mean drop > 2 standard errors of the paired fold differences, and relative gain ≥ 3%.
The p-values are only a guide: folds share training data, so they are not independent.
The tuned models' CV scores are slightly optimistic, because the search picked the winner using those same folds. The hold-out result below is the honest number.

## 6. Final hold-out result (evaluated once)

| Model | Hold-out RMSE (log) | Hold-out R² | Hold-out RMSE ($) |
|---|---|---|---|
| Shipped: GradientBoosting (tuned) | 0.1358 | 0.9011 | 27,624 |
| Task 3 style baseline (reference only) | 0.1461 | 0.8855 | 29,564 |

The baseline was scored on the hold-out only for reference. It played no part in choosing the model.
On the hold-out the shipped model has 7.1% lower log-RMSE than the baseline (typical error about 14.6% vs about 15.7%).

## 7. Verdict

**Model I would ship: GradientBoosting (tuned).** GradientBoosting (tuned) [XGBoost NOT installed - stand-in] has the lowest mean CV RMSE (0.1220) and no faster model is within one standard error of it.

Compared with the baseline it lowers CV RMSE by 17.3% (hold-out: 7.1% lower) while the full tuning run took about 12.5x the baseline's CV time (one-off cost; the final model fits in 6.0s). The improvement over the baseline is **both statistically and practically meaningful** by the rule used here: it improves all 5 folds, the mean drop (0.0255) is more than 2 standard errors of the fold differences, and the relative gain (17.3%) is at least 3%. Verdict: the extra complexity and training time are justified.

Honest caveats:
- With ~1,170 training rows and one 80/20 split, differences of a few thousandths of RMSE can be noise; repeat with other seeds before trusting small gaps.
- The hold-out score is one draw from one split; it gives a believable ballpark, not an exact figure.
- The tuned models' CV scores are slightly optimistic (the search chose the winner on those folds); trust the hold-out number more, and treat both as estimates.

Figures: `cv_rmse_comparison.png`, `feature_importance.png`, `holdout_predictions.png`.
