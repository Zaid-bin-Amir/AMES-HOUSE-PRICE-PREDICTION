"""
Task 5 - Feature Engineering & Tuned Ensembles for Regression (Ames House Prices)

All reusable logic lives here. It is used by both:
  - run_experiments.py            (command line, writes results/ folder)
  - House_Prices_Task5.ipynb      (step by step, with explanations)

Leakage rules followed everywhere in this project
-------------------------------------------------
1. The target (SalePrice) is separated from X before anything else happens.
2. Feature engineering, imputation and encoding are steps INSIDE a sklearn
   Pipeline, so they are re-fitted on the training portion of every CV fold.
3. Hyperparameters are tuned only with CV on the training split.
4. The hold-out test split is touched exactly once, at the very end.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
N_FOLDS = 5
TARGET = "SalePrice"
DATA_PATH = Path(__file__).resolve().parent / "data" / "train.csv"

# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the Ames training csv."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"'{TARGET}' column not found in {path}")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    Hold-out split (same idea as Task 3: 80 / 20, fixed seed).

    The target is log-transformed: log1p(SalePrice). Prices are right-skewed,
    so modelling the log makes errors relative (a 10k error on a cheap house
    matters more than on an expensive one) and is what Kaggle scores on.
    Returns X_train, X_test, y_train, y_test (y values are on the log scale).
    """
    y = np.log1p(df[TARGET])
    X = df.drop(columns=[TARGET])
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


# ----------------------------------------------------------------------------
# 2. Feature engineering (stateless -> cannot leak, but still lives in Pipeline)
# ----------------------------------------------------------------------------

# Written justification for every engineered feature (used in notebook + report)
FEATURE_JUSTIFICATIONS = {
    "TotalSF": (
        "TotalBsmtSF + 1stFlrSF + 2ndFlrSF. Buyers think in total usable area, "
        "not in separate floors. One strong, additive size signal is easier for "
        "a tree to split on than three partial ones."
    ),
    "HouseAge": (
        "YrSold - YearBuilt (clipped at 0). The raw year does not say how old "
        "the house was when it sold; age at sale does, and older houses "
        "usually sell for less, all else equal."
    ),
    "TotalBath": (
        "FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath. Bathrooms "
        "are split over four columns; a half bath is worth roughly half a full "
        "one. The total is what a buyer counts."
    ),
    "HasRemodeled": (
        "1 if YearRemodAdd != YearBuilt else 0. A remodel flag captures "
        "'updated house' separately from raw age."
    ),
    "RemodAge": (
        "YrSold - YearRemodAdd (clipped at 0). Years since the last update; a "
        "recent remodel makes an old house feel newer."
    ),
    "TotalPorchSF": (
        "OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF. "
        "Five sparse outdoor-area columns summed into one usable "
        "'outdoor living space' number."
    ),
    "HasPool_or_Fireplace": (
        "1 if the house has a pool or at least one fireplace. Both are rare "
        "luxury amenities, so a single binary flag is less noisy than the raw "
        "counts."
    ),
}

ENGINEERED_FEATURES = list(FEATURE_JUSTIFICATIONS.keys())


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Cleans the raw frame and (optionally) adds domain features.

    - Drops the Id column (a row number, not information).
    - Treats MSSubClass as a category (it is a code, not a quantity).
    - Fills missing categorical values with the string 'None'. In Ames most
      "missing" categoricals mean "this house does not have the thing"
      (no alley, no pool, no garage ...). A constant fill uses no statistics
      from the data, so it cannot leak.
    - Adds engineered features only from other input columns, never from the
      target, so every feature is computable before the sale price is known.

    Numeric NaNs are left alone: the imputer inside the Pipeline fills them
    with the median of the TRAINING fold only.
    """

    def __init__(self, add_features: bool = True):
        self.add_features = add_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=["Id"], errors="ignore")
        X["MSSubClass"] = X["MSSubClass"].astype(str)

        # make every non-numeric column a plain object column without NaN
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype(object).where(X[col].notna(), "None")

        if self.add_features:
            num = lambda c: pd.to_numeric(X[c], errors="coerce").fillna(0)  # noqa: E731

            X["TotalSF"] = num("TotalBsmtSF") + num("1stFlrSF") + num("2ndFlrSF")
            X["HouseAge"] = (num("YrSold") - num("YearBuilt")).clip(lower=0)
            X["RemodAge"] = (num("YrSold") - num("YearRemodAdd")).clip(lower=0)
            X["TotalBath"] = (
                num("FullBath")
                + 0.5 * num("HalfBath")
                + num("BsmtFullBath")
                + 0.5 * num("BsmtHalfBath")
            )
            X["HasRemodeled"] = (
                num("YearRemodAdd") != num("YearBuilt")
            ).astype(int)
            X["TotalPorchSF"] = (
                num("OpenPorchSF")
                + num("EnclosedPorch")
                + num("3SsnPorch")
                + num("ScreenPorch")
                + num("WoodDeckSF")
            )
            X["HasPool_or_Fireplace"] = (
                (num("PoolArea") > 0) | (num("Fireplaces") > 0)
            ).astype(int)
        return X


def feature_check_table(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Sanity table for the written justification: shows each engineered feature
    next to its correlation with the (log) price on the TRAINING split only.
    This is for reporting, NOT used to select features.
    """
    fe = FeatureEngineer(add_features=True).transform(X_train)
    rows = []
    for name in ENGINEERED_FEATURES:
        rows.append(
            {
                "feature": name,
                "corr_with_log_price": round(float(np.corrcoef(fe[name], y_train)[0, 1]), 3),
                "min": float(fe[name].min()),
                "max": float(fe[name].max()),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. Pipelines
# ----------------------------------------------------------------------------


def make_preprocessor() -> ColumnTransformer:
    """Median-impute numbers, one-hot encode categories (both fitted per fold)."""
    return ColumnTransformer(
        [
            (
                "num",
                SimpleImputer(strategy="median"),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        verbose_feature_names_out=True,
    )


def make_pipeline(model, add_features: bool = True) -> Pipeline:
    """feature engineering -> preprocessing -> model, all refitted per fold."""
    return Pipeline(
        [
            ("fe", FeatureEngineer(add_features=add_features)),
            ("prep", make_preprocessor()),
            ("model", model),
        ]
    )


def rf_estimator() -> RandomForestRegressor:
    return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)


def rf_param_distributions() -> dict:
    """Random Forest search space: 5 hyperparameters."""
    return {
        "model__n_estimators": randint(200, 601),
        "model__max_depth": [None, 10, 15, 20, 30],
        "model__min_samples_split": randint(2, 13),
        "model__min_samples_leaf": randint(1, 6),
        "model__max_features": ["sqrt", 0.3, 0.5, 1.0],
    }


def _xgboost_works() -> bool:
    """True only if xgboost imports AND can fit a tiny model in this environment."""
    try:
        from xgboost import XGBRegressor

        rng = np.random.RandomState(0)
        Xs, ys = rng.rand(30, 3), rng.rand(30)
        XGBRegressor(n_estimators=3, n_jobs=1, verbosity=0).fit(Xs, ys)
        # clone() must also work, RandomizedSearchCV needs it
        from sklearn.base import clone

        clone(XGBRegressor())
        return True
    except Exception:  # ImportError or any sklearn/xgboost version mismatch
        return False


def boosting_model():
    """
    Returns (display_name, estimator, param_distributions, is_real_xgboost).

    Uses XGBRegressor when available. If xgboost is not installed (or is
    broken in this environment) it falls back to sklearn's
    GradientBoostingRegressor, so the script never crashes. The fallback is
    clearly labelled in every output, because Task 5 asks for XGBoost.
    """
    if _xgboost_works():
        from xgboost import XGBRegressor

        est = XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            objective="reg:squarederror",
            verbosity=0,
        )
        dist = {  # 7 hyperparameters
            "model__n_estimators": randint(300, 1201),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": randint(2, 7),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__min_child_weight": randint(1, 8),
            "model__reg_lambda": loguniform(0.5, 10),
        }
        return "XGBoost (tuned)", est, dist, True

    warnings.warn(
        "xgboost is not installed / not working -> using sklearn "
        "GradientBoostingRegressor as a stand-in. Run `pip install xgboost` "
        "and re-run to get the real XGBoost result.",
        stacklevel=2,
    )
    est = GradientBoostingRegressor(random_state=RANDOM_STATE)
    dist = {
        "model__n_estimators": randint(200, 701),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(2, 6),
        "model__subsample": uniform(0.6, 0.4),
        "model__min_samples_leaf": randint(1, 8),
    }
    return "GradientBoosting (tuned) [XGBoost NOT installed - stand-in]", est, dist, False


# ----------------------------------------------------------------------------
# 4. Cross-validation helpers
# ----------------------------------------------------------------------------

SCORING = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}


def make_cv() -> KFold:
    """The SAME 5 folds are used for every model (shuffle + fixed seed)."""
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def cv_untuned(name: str, pipe: Pipeline, X, y, note: str) -> dict:
    """5-fold CV of an untuned pipeline. Returns a result dict."""
    t0 = time.perf_counter()
    res = cross_validate(
        pipe, X, y, cv=make_cv(), scoring=SCORING, n_jobs=-1, error_score="raise"
    )
    wall = time.perf_counter() - t0
    rmse = -res["test_rmse"]
    r2 = res["test_r2"]
    return {
        "model": name,
        "rmse_folds": rmse,
        "r2_folds": r2,
        "cv_rmse_mean": rmse.mean(),
        "cv_rmse_std": rmse.std(ddof=1),
        "cv_r2_mean": r2.mean(),
        "cv_r2_std": r2.std(ddof=1),
        "total_time_s": wall,
        "final_fit_time_s": float(res["fit_time"].mean()),
        "best_params": {},
        "note": note,
        "fitted": None,  # untuned models are refit only if needed
        "pipeline": pipe,
    }


def cv_tuned(name: str, pipe: Pipeline, param_dist: dict, X, y, n_iter: int, note: str) -> dict:
    """
    RandomizedSearchCV with the same 5 folds. The reported CV numbers are the
    per-fold scores of the best parameter set (so mean +- std is real).
    NOTE: because the search picked the winner on these same folds, the CV
    score is slightly optimistic; the untouched hold-out set gives the honest
    final number.
    """
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=SCORING,
        refit="rmse",
        cv=make_cv(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
        return_train_score=False,
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    wall = time.perf_counter() - t0

    i = search.best_index_
    cvr = search.cv_results_
    rmse = np.array([-cvr[f"split{k}_test_rmse"][i] for k in range(N_FOLDS)])
    r2 = np.array([cvr[f"split{k}_test_r2"][i] for k in range(N_FOLDS)])

    clean_params = {}
    for k, v in search.best_params_.items():
        v = v.item() if hasattr(v, "item") else v
        clean_params[k.replace("model__", "")] = v

    return {
        "model": name,
        "rmse_folds": rmse,
        "r2_folds": r2,
        "cv_rmse_mean": rmse.mean(),
        "cv_rmse_std": rmse.std(ddof=1),
        "cv_r2_mean": r2.mean(),
        "cv_r2_std": r2.std(ddof=1),
        "total_time_s": wall,
        "final_fit_time_s": float(search.refit_time_),
        "best_params": clean_params,
        "note": note,
        "fitted": search.best_estimator_,
        "pipeline": pipe,
        "search": search,
    }


# ----------------------------------------------------------------------------
# 5. Comparison, significance, verdict
# ----------------------------------------------------------------------------


def comparison_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Model": r["model"],
                "Mean CV RMSE (log)": f"{r['cv_rmse_mean']:.4f} ± {r['cv_rmse_std']:.4f}",
                "Mean CV R²": f"{r['cv_r2_mean']:.4f} ± {r['cv_r2_std']:.4f}",
                "Training time (s)": round(r["total_time_s"], 1),
                "Final fit time (s)": round(r["final_fit_time_s"], 2),
                "Note": r["note"],
            }
        )
    return pd.DataFrame(rows)


def paired_improvement(baseline: dict, other: dict) -> dict:
    """
    Fold-by-fold comparison on identical folds. Positive diff = `other` better.
    A rough paired t-test is reported, but folds share training data, so the
    p-value is only a guide, not a strict proof.
    """
    diff = baseline["rmse_folds"] - other["rmse_folds"]
    mean_diff = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(len(diff)) if sd > 0 else 0.0
    rel = mean_diff / baseline["cv_rmse_mean"] * 100
    t_stat, p_val = stats.ttest_rel(baseline["rmse_folds"], other["rmse_folds"])
    return {
        "mean_rmse_drop": mean_diff,
        "relative_improvement_pct": float(rel),
        "folds_improved": int((diff > 0).sum()),
        "n_folds": len(diff),
        "std_of_diffs": sd,
        "se_of_diffs": float(se),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        # transparent rule used in the report (stated openly there):
        "meaningful": bool(
            rel >= 3.0 and mean_diff > 2 * se and (diff > 0).sum() == len(diff)
        ),
    }


def choose_model_to_ship(tuned_results: list[dict]) -> tuple[dict, str]:
    """
    One-standard-error rule: take the lowest CV RMSE model, but if another
    model is within one standard error of it AND is faster to fit, ship the
    faster one (simpler, cheaper, same accuracy within noise).
    """
    ordered = sorted(tuned_results, key=lambda r: r["cv_rmse_mean"])
    best = ordered[0]
    se_best = best["cv_rmse_std"] / np.sqrt(N_FOLDS)
    for other in ordered[1:]:
        within = other["cv_rmse_mean"] - best["cv_rmse_mean"] <= se_best
        faster = other["final_fit_time_s"] < best["final_fit_time_s"]
        if within and faster:
            return other, (
                f"{other['model']} is within one standard error of "
                f"{best['model']} ({other['cv_rmse_mean']:.4f} vs "
                f"{best['cv_rmse_mean']:.4f}, SE ≈ {se_best:.4f}) and fits faster "
                f"({other['final_fit_time_s']:.1f}s vs {best['final_fit_time_s']:.1f}s), "
                "so it is the better engineering choice."
            )
    return best, (
        f"{best['model']} has the lowest mean CV RMSE "
        f"({best['cv_rmse_mean']:.4f}) and no faster model is within one "
        "standard error of it."
    )


# ----------------------------------------------------------------------------
# 6. Final hold-out evaluation (call ONCE)
# ----------------------------------------------------------------------------


def evaluate_holdout(fitted_pipe: Pipeline, X_test, y_test) -> dict:
    """Score on the untouched hold-out. RMSE on log scale and in dollars."""
    pred_log = fitted_pipe.predict(X_test)
    rmse_log = float(np.sqrt(mean_squared_error(y_test, pred_log)))
    r2 = float(r2_score(y_test, pred_log))
    rmse_usd = float(np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred_log))))
    return {
        "rmse_log": rmse_log,
        "r2": r2,
        "rmse_usd": rmse_usd,
        "pred_log": pred_log,
    }


def fit_on_train(pipe: Pipeline, X_train, y_train) -> Pipeline:
    """Fit a (already tuned or default) pipeline on the full training split."""
    from sklearn.base import clone

    p = clone(pipe)
    p.fit(X_train, y_train)
    return p


# ----------------------------------------------------------------------------
# 7. Plots (matplotlib imported lazily so notebook/CLI can pick the backend)
# ----------------------------------------------------------------------------


def plot_cv_comparison(results: list[dict], path: Path | None = None):
    import matplotlib.pyplot as plt

    names = [r["model"].split(" [")[0] for r in results]
    means = [r["cv_rmse_mean"] for r in results]
    stds = [r["cv_rmse_std"] for r in results]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=6, color="#4C78A8")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax.set_ylabel("5-fold CV RMSE (log price, lower is better)")
    ax.set_title("Model comparison: mean ± std of RMSE across 5 folds")
    for b, m, s in zip(bars, means, stds):
        # put the value above the error bar so the whisker does not cross the text
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.002, f"{m:.4f}", ha="center", va="bottom", fontsize=8)
    lo = max(0, min(means) - 2.5 * max(stds) - 0.01)
    ax.set_ylim(lo, max(means) + 3 * max(stds) + 0.01)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_feature_importance(fitted_pipe: Pipeline, path: Path | None = None, top: int = 15):
    import matplotlib.pyplot as plt

    model = fitted_pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    names = fitted_pipe.named_steps["prep"].get_feature_names_out()
    names = [n.replace("num__", "").replace("cat__", "") for n in names]
    imp = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#E45756" if n in ENGINEERED_FEATURES else "#4C78A8" for n in imp.index]
    ax.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1])
    ax.set_title(f"Top {top} features (red = engineered by us)")
    ax.set_xlabel("importance")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_holdout(y_true_log, y_pred_log, title: str, path: Path | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    actual, pred = np.expm1(y_true_log), np.expm1(y_pred_log)
    ax.scatter(actual, pred, s=14, alpha=0.6, color="#4C78A8")
    lim = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Actual SalePrice ($)")
    ax.set_ylabel("Predicted SalePrice ($)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
