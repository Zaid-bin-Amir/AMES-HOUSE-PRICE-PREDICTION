| Model | Mean CV RMSE (log) | Mean CV R² | Training time (s) | Final fit time (s) | Note |
|---|---|---|---|---|---|
| Baseline RF (default, original features) | 0.1475 ± 0.0177 | 0.8568 ± 0.0225 | 17.2 | 5.32 | Task 3 style baseline: no new features, no tuning. |
| RF (default) + engineered features | 0.1433 ± 0.0166 | 0.8649 ± 0.0185 | 17.4 | 5.73 | Same algorithm and settings; only the features changed. |
| Random Forest (tuned) + features | 0.1381 ± 0.0183 | 0.8746 ± 0.0215 | 210.9 | 3.52 | Bagging: many deep trees averaged, robust, low variance. |
| GradientBoosting (tuned) [XGBoost NOT installed - stand-in] | 0.1220 ± 0.0155 | 0.9020 ± 0.0168 | 215.3 | 5.98 | Boosting: shallow trees added one by one to fix earlier errors; more tuning-sensitive. |