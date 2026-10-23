"""
Run the complete Task 5 experiment from the command line.

    python run_experiments.py              # full run (about 5-15 min)
    python run_experiments.py --quick      # fast smoke test (about 1-2 min)
    python run_experiments.py --n-iter 40  # bigger random search

Everything is written to the results/ folder:
    comparison_table.csv / .md   the single comparison table
    report.md                    features, method, table, verdict
    best_params.json             tuned hyperparameters
    cv_rmse_comparison.png, feature_importance.png, holdout_predictions.png
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window needed, just save the images

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import task5_lib as lib  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main(n_iter: int, quick: bool) -> int:
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    RESULTS_DIR.mkdir(exist_ok=True)
    t_start = time.perf_counter()

    # ---- data -----------------------------------------------------------
    df = lib.load_data()
    X_train, X_test, y_train, y_test = lib.split_data(df)
    print(f"Data: {df.shape[0]} rows, {df.shape[1]} cols | "
          f"train={len(X_train)}  hold-out={len(X_test)} (hold-out untouched until the end)")

    feat_table = lib.feature_check_table(X_train, y_train)
    print("\nEngineered features (correlation with log price, TRAIN only):")
    print(feat_table.to_string(index=False))

    # ---- experiments ----------------------------------------------------
    results = []

    print("\n[1/4] Baseline: default Random Forest, original features only ...")
    base = lib.cv_untuned(
        "Baseline RF (default, original features)",
        lib.make_pipeline(lib.rf_estimator().set_params(n_estimators=300), add_features=False),
        X_train, y_train,
        "Task 3 style baseline: no new features, no tuning.",
    )
    results.append(base)

    print("[2/4] Same default RF + engineered features (isolates the effect of features) ...")
    rf_fe = lib.cv_untuned(
        "RF (default) + engineered features",
        lib.make_pipeline(lib.rf_estimator().set_params(n_estimators=300), add_features=True),
        X_train, y_train,
        "Same algorithm and settings; only the features changed.",
    )
    results.append(rf_fe)

    print(f"[3/4] Random Forest + features, RandomizedSearchCV (n_iter={n_iter}, 5-fold) ...")
    rf_tuned = lib.cv_tuned(
        "Random Forest (tuned) + features",
        lib.make_pipeline(lib.rf_estimator(), add_features=True),
        lib.rf_param_distributions(), X_train, y_train, n_iter,
        "Bagging: many deep trees averaged, robust, low variance.",
    )
    results.append(rf_tuned)

    boost_name, boost_est, boost_dist, is_xgb = lib.boosting_model()
    print(f"[4/4] {boost_name} + features, RandomizedSearchCV (n_iter={n_iter}, 5-fold) ...")
    xgb_tuned = lib.cv_tuned(
        boost_name,
        lib.make_pipeline(boost_est, add_features=True),
        boost_dist, X_train, y_train, n_iter,
        "Boosting: shallow trees added one by one to fix earlier errors; "
        "more tuning-sensitive.",
    )
    results.append(xgb_tuned)

    # ---- comparison table ----------------------------------------------
    table = lib.comparison_table(results)
    table.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    (RESULTS_DIR / "comparison_table.md").write_text(md_table(table), encoding="utf-8")
    print("\n" + table.drop(columns=["Note"]).to_string(index=False))

    # ---- significance + choice -----------------------------------------
    imp_fe_only = lib.paired_improvement(base, rf_fe)
    imp_rf = lib.paired_improvement(base, rf_tuned)
    imp_xgb = lib.paired_improvement(base, xgb_tuned)
    chosen, why = lib.choose_model_to_ship([rf_tuned, xgb_tuned])
    imp_chosen = lib.paired_improvement(base, chosen)

    # ---- final hold-out: evaluated ONCE --------------------------------
    print("\nFinal hold-out evaluation (once) ...")
    chosen_fit = lib.fit_on_train(chosen["fitted"], X_train, y_train)
    base_fit = lib.fit_on_train(base["pipeline"], X_train, y_train)
    hold_chosen = lib.evaluate_holdout(chosen_fit, X_test, y_test)
    hold_base = lib.evaluate_holdout(base_fit, X_test, y_test)  # reference only, not used to choose
    print(f"  shipped model  : RMSE(log)={hold_chosen['rmse_log']:.4f}  R2={hold_chosen['r2']:.4f}  "
          f"RMSE=${hold_chosen['rmse_usd']:,.0f}")
    print(f"  Task 3 baseline: RMSE(log)={hold_base['rmse_log']:.4f}  R2={hold_base['r2']:.4f}  "
          f"RMSE=${hold_base['rmse_usd']:,.0f}")

    # ---- plots ----------------------------------------------------------
    lib.plot_cv_comparison(results, RESULTS_DIR / "cv_rmse_comparison.png")
    lib.plot_feature_importance(chosen_fit, RESULTS_DIR / "feature_importance.png")
    lib.plot_holdout(y_test, hold_chosen["pred_log"],
                     f"Hold-out: {chosen['model'].split(' [')[0]}",
                     RESULTS_DIR / "holdout_predictions.png")

    # ---- json -----------------------------------------------------------
    (RESULTS_DIR / "best_params.json").write_text(
        json.dumps({r["model"]: r["best_params"] for r in results if r["best_params"]},
                   indent=2, default=str),
        encoding="utf-8",
    )

    # ---- report ---------------------------------------------------------
    report = build_report(
        df, X_train, X_test, feat_table, table, results, base, rf_fe, rf_tuned, xgb_tuned,
        imp_fe_only, imp_rf, imp_xgb, chosen, why, imp_chosen, hold_chosen, hold_base,
        is_xgb, n_iter, quick,
    )
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")

    print(f"\nDone in {(time.perf_counter() - t_start) / 60:.1f} min. "
          f"Open results/report.md for the full write-up.")
    if not is_xgb:
        print("\nNOTE: xgboost was not available, a GradientBoosting stand-in was used. "
              "Run `pip install xgboost` and re-run for the real XGBoost numbers.")
    return 0


def md_table(df: pd.DataFrame) -> str:
    """Tiny markdown table writer (no extra dependency needed)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_report(df, X_train, X_test, feat_table, table, results, base, rf_fe, rf_tuned, xgb_tuned,
                 imp_fe_only, imp_rf, imp_xgb, chosen, why, imp_chosen, hold_chosen, hold_base,
                 is_xgb, n_iter, quick) -> str:
    def pct_err(rmse_log):  # RMSE in log space ~ typical relative price error
        return f"about {np.expm1(rmse_log) * 100:.1f}%"

    gain_vs_base = (base["cv_rmse_mean"] - chosen["cv_rmse_mean"]) / base["cv_rmse_mean"] * 100
    hold_gain = (hold_base["rmse_log"] - hold_chosen["rmse_log"]) / hold_base["rmse_log"] * 100
    time_ratio = chosen["total_time_s"] / max(base["total_time_s"], 1e-9)

    # --- verdict wording is derived from the numbers, not hard-coded ---
    if imp_chosen["meaningful"]:
        meaning = (
            "The improvement over the baseline is **both statistically and practically meaningful** "
            f"by the rule used here: it improves all {imp_chosen['n_folds']} folds, the mean drop "
            f"({imp_chosen['mean_rmse_drop']:.4f}) is more than 2 standard errors of the fold differences, "
            f"and the relative gain ({imp_chosen['relative_improvement_pct']:.1f}%) is at least 3%."
        )
        justify = "the extra complexity and training time are justified."
    elif imp_chosen["mean_rmse_drop"] > 0:
        meaning = (
            "The improvement over the baseline is **real in direction but modest by the rule used here**: "
            f"it improved {imp_chosen['folds_improved']} of {imp_chosen['n_folds']} folds, mean drop "
            f"{imp_chosen['mean_rmse_drop']:.4f} ({imp_chosen['relative_improvement_pct']:.1f}%), "
            "which does not clear all three conditions (all folds improved, mean drop > 2 SE, relative gain ≥ 3%)."
        )
        justify = "the extra complexity is only partly justified; weigh it against the extra training time."
    else:
        meaning = (
            "The tuned model did **not** beat the baseline on cross-validation: "
            f"mean change {imp_chosen['mean_rmse_drop']:.4f} RMSE."
        )
        justify = "the added complexity is not justified."

    share_fe = (imp_fe_only["mean_rmse_drop"] / imp_chosen["mean_rmse_drop"] * 100
                if imp_chosen["mean_rmse_drop"] > 0 else float("nan"))
    fe_part = (
        f"Adding the engineered features alone (same default Random Forest) reduced CV RMSE by "
        f"{imp_fe_only['mean_rmse_drop']:.4f} ({imp_fe_only['relative_improvement_pct']:.1f}%), "
        f"improving {imp_fe_only['folds_improved']}/{imp_fe_only['n_folds']} folds. "
        f"The best final model reduced it by {imp_chosen['mean_rmse_drop']:.4f} "
        f"({imp_chosen['relative_improvement_pct']:.1f}%) versus the baseline"
        + (f", so the features account for roughly {share_fe:.0f}% of the total gain and "
           f"tuning/algorithm choice for the rest." if share_fe == share_fe else ".")
    )

    xgb_warn = ""
    if not is_xgb:
        xgb_warn = (
            "\n> **Note:** the `xgboost` package was not available when this report was produced, "
            "so sklearn's `GradientBoostingRegressor` was used as a stand-in for the boosting model. "
            "Install xgboost (`pip install xgboost`) and re-run `python run_experiments.py` "
            "to get the real XGBoost row.\n"
        )
    quick_warn = ""
    if quick:
        quick_warn = (
            f"\n> **Note:** this report was produced with `--quick` (n_iter={n_iter}); "
            "run without `--quick` for the full search.\n"
        )

    fold_rows = pd.DataFrame(
        {
            r["model"].split(" [")[0]: [f"{v:.4f}" for v in r["rmse_folds"]]
            for r in results
        },
        index=[f"fold {i + 1}" for i in range(lib.N_FOLDS)],
    ).reset_index().rename(columns={"index": ""})

    feat_lines = "\n".join(
        f"- **{name}**: {text}" for name, text in lib.FEATURE_JUSTIFICATIONS.items()
    )

    params_lines = "\n".join(
        f"- **{r['model'].split(' [')[0]}**: `{r['best_params']}`" for r in results if r["best_params"]
    )

    return f"""# Task 5 - Feature Engineering & Tuned Ensembles (Ames House Prices)
{xgb_warn}{quick_warn}
## 1. Setup

- Data: Ames House Prices `train.csv`, {df.shape[0]} rows, {df.shape[1]} columns.
- Target: `log1p(SalePrice)`. Sale prices are right-skewed; the log makes errors relative.
  **All RMSE values below are on the log scale** (an RMSE of 0.13 means a typical error of roughly 13% of the price).
- Split: {len(X_train)} train / {len(X_test)} hold-out (80/20, seed 42). The hold-out set is evaluated **once**, at the end.
- Validation: the same 5-fold shuffled `KFold` (seed 42) for every model, run on the train split only.
- Leakage control: feature engineering, imputation and one-hot encoding are steps inside a scikit-learn
  `Pipeline`, so they are re-fitted on the training part of every fold. No feature uses `SalePrice`.

## 2. Engineered features (each justified)

{feat_lines}

Check (training split only, shown for reporting and not used to pick features):

{md_table(feat_table)}

All features are computed from columns known before the sale (areas, years, counts). None is built from the target or from price bins.

## 3. Hyperparameter tuning

`RandomizedSearchCV`, n_iter={n_iter}, same 5 folds, scoring = RMSE, best model refit on the full train split.

{params_lines}

## 4. Comparison table (5-fold CV on the train split)

{md_table(table)}

"Training time" is the wall-clock time of the whole 5-fold CV (untuned rows) or the whole random search (tuned rows).
"Final fit time" is the time to fit one model once on the full train split.

Per-fold RMSE (identical folds for every model):

{md_table(fold_rows)}

## 5. Do the improvements beat noise?

{fe_part}

- Random Forest (tuned) vs baseline: RMSE drop {imp_rf['mean_rmse_drop']:+.4f} ({imp_rf['relative_improvement_pct']:+.1f}%), {imp_rf['folds_improved']}/{imp_rf['n_folds']} folds improved, paired t p≈{imp_rf['p_value']:.3f}.
- {xgb_tuned['model'].split(' [')[0]} vs baseline: RMSE drop {imp_xgb['mean_rmse_drop']:+.4f} ({imp_xgb['relative_improvement_pct']:+.1f}%), {imp_xgb['folds_improved']}/{imp_xgb['n_folds']} folds improved, paired t p≈{imp_xgb['p_value']:.3f}.

Rule used for "meaningful": all folds improve, mean drop > 2 standard errors of the paired fold differences, and relative gain ≥ 3%.
The p-values are only a guide: folds share training data, so they are not independent.
The tuned models' CV scores are slightly optimistic, because the search picked the winner using those same folds. The hold-out result below is the honest number.

## 6. Final hold-out result (evaluated once)

| Model | Hold-out RMSE (log) | Hold-out R² | Hold-out RMSE ($) |
|---|---|---|---|
| Shipped: {chosen['model'].split(' [')[0]} | {hold_chosen['rmse_log']:.4f} | {hold_chosen['r2']:.4f} | {hold_chosen['rmse_usd']:,.0f} |
| Task 3 style baseline (reference only) | {hold_base['rmse_log']:.4f} | {hold_base['r2']:.4f} | {hold_base['rmse_usd']:,.0f} |

The baseline was scored on the hold-out only for reference. It played no part in choosing the model.
On the hold-out the shipped model has {abs(hold_gain):.1f}% {'lower' if hold_gain >= 0 else 'higher'} log-RMSE than the baseline (typical error {pct_err(hold_chosen['rmse_log'])} vs {pct_err(hold_base['rmse_log'])}).

## 7. Verdict

**Model I would ship: {chosen['model'].split(' [')[0]}.** {why}

Compared with the baseline it lowers CV RMSE by {gain_vs_base:.1f}% (hold-out: {hold_gain:.1f}% lower) while the full tuning run took about {time_ratio:.1f}x the baseline's CV time (one-off cost; the final model fits in {chosen['final_fit_time_s']:.1f}s). {meaning} Verdict: {justify}

Honest caveats:
- With ~1,170 training rows and one 80/20 split, differences of a few thousandths of RMSE can be noise; repeat with other seeds before trusting small gaps.
- The hold-out score is one draw from one split; it gives a believable ballpark, not an exact figure.
- The tuned models' CV scores are slightly optimistic (the search chose the winner on those folds); trust the hold-out number more, and treat both as estimates.

Figures: `cv_rmse_comparison.png`, `feature_importance.png`, `holdout_predictions.png`.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Task 5: features + tuned ensembles on Ames")
    ap.add_argument("--quick", action="store_true", help="fast smoke test (n_iter=4)")
    ap.add_argument("--n-iter", type=int, default=30, help="random search iterations per model")
    args = ap.parse_args()
    n_iter = 4 if args.quick else args.n_iter
    sys.exit(main(n_iter, args.quick))
