# feature_importance_utils.py

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Tuple
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

# --- constants (match your project) ---
TREAT_COL   = "treatment_group"
TARGET_COL  = "total_30d_post_campaign_spend"
CAMPAIGN_COL= "campaign_id"
COST_PER_POSTCARD = 0.70

# # -------------------------
# # iROAS summarizer from tau
# # -------------------------
# def _iroas_from_tau(df: pd.DataFrame, tau_hat: np.ndarray) -> pd.DataFrame:
#     """Return per-campaign predicted incremental spend & iROAS from row-level tau."""
#     tmp = df.copy()
#     tmp["tau_hat"] = np.asarray(tau_hat)
#     g = tmp.groupby(CAMPAIGN_COL, observed=True, dropna=False)
#     n_treat = g[TREAT_COL].sum()
#     ad_cost = n_treat * COST_PER_POSTCARD

#     incr_pred = (
#         tmp.loc[tmp[TREAT_COL] == 1]
#            .groupby(CAMPAIGN_COL, observed=True, dropna=False)["tau_hat"]
#            .sum()
#     ).reindex(n_treat.index, fill_value=0)

#     iroas_pred = incr_pred / ad_cost.replace(0, np.nan)
#     return pd.DataFrame({
#         CAMPAIGN_COL: n_treat.index.astype(str),
#         "incremental_pred_spend": incr_pred.values,
#         "iROAS_pred": iroas_pred.values,
#     })

# ----------------------------------------------------
# Generic permutation importance on tau or on iROAS
# ----------------------------------------------------
def permute_importance_tau(
    df: pd.DataFrame,
    predict_tau_df: Callable[[pd.DataFrame], np.ndarray],
    features: List[str],
    on: str = "iROAS",              # "iROAS" or "tau"
    metric: str = "MAE",            # "MAE" or "RMSE"
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance for a tau model.
    - on="iROAS": compute iROAS per campaign from tau and measure MAE/RMSE vs baseline iROAS.
    - on="tau":   measure MAE/RMSE vs baseline tau per row.
    Importance = average degradation (>= 0) in the chosen metric.
    """
    rng = np.random.default_rng(random_state)
    mode = (on or "").strip().lower()
    metric = metric.upper()

    def to_iroas(tau_vec: np.ndarray) -> np.ndarray:
        tmp = df[[CAMPAIGN_COL, TREAT_COL]].copy()
        tmp["tau"] = tau_vec
        g = tmp.groupby(CAMPAIGN_COL, observed=True, dropna=False)
        n_treat = g[TREAT_COL].sum()
        ad_cost = (n_treat * COST_PER_POSTCARD).replace(0, np.nan)
        incr = (
            tmp.loc[tmp[TREAT_COL] == 1]
               .groupby(CAMPAIGN_COL, observed=True, dropna=False)["tau"]
               .sum()
        ).reindex(n_treat.index, fill_value=0)
        return (incr / ad_cost).values  # one iROAS per campaign, aligned to n_treat.index

    def agg_mae(arr: np.ndarray, ref: np.ndarray) -> float:
        return float(np.nanmean(np.abs(arr - ref)))

    def agg_rmse(arr: np.ndarray, ref: np.ndarray) -> float:
        return float(np.sqrt(np.nanmean((arr - ref) ** 2)))

    scorer = agg_mae if metric == "MAE" else agg_rmse

    # --- Baseline predictions and reference ---
    tau_base = predict_tau_df(df)
    if mode == "tau":
        ref = tau_base
    elif mode == "iroas":
        ref = to_iroas(tau_base)
    else:
        raise ValueError("on must be 'iROAS' or 'tau'")

    rows = []
    for f in features:
        deltas = []
        for _ in range(n_repeats):
            df_p = df.copy()
            # Permute safely for any dtype, including categoricals
            idx = rng.permutation(len(df_p))
            df_p.loc[:, f] = df_p[f].iloc[idx].to_numpy()

            tau_p = predict_tau_df(df_p)
            if mode == "tau":
                s = scorer(tau_p, ref)
            else:
                s = scorer(to_iroas(tau_p), ref)
            deltas.append(s)  # baseline is 0 by construction

        rows.append({
            "feature": f,
            "importance": float(np.mean(deltas)),  # >= 0, in chosen metric units
            "metric": metric,
            "on": "iROAS" if mode == "iroas" else "tau",
            "n_repeats": n_repeats,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

def plot_perm_importance(imp_df: pd.DataFrame, top: int | None = None, title: str = None):
    dfp = imp_df.sort_values("importance", ascending=True)
    if top is not None:
        dfp = dfp.tail(top)
    plt.figure(figsize=(8, max(3, 0.4*len(dfp))))
    plt.barh(dfp["feature"], dfp["importance"])
    plt.xlabel(f"Permutation importance ({dfp['metric'].iloc[0]} on {dfp['on'].iloc[0]})")
    plt.title(title or "Feature importance (permutation)")
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------
# CatBoost built-ins for individual models
# ----------------------------------------------------
def _cb_importance(model: CatBoostRegressor, X: pd.DataFrame, feature_names: List[str]) -> pd.Series:
    """
    CatBoost feature importance (PredictionValuesChange), aligned to feature_names.
    """
    pool = Pool(X[feature_names], cat_features=[i for i, c in enumerate(feature_names) if str(X[c].dtype).startswith("category")])
    imp = model.get_feature_importance(type="PredictionValuesChange", data=pool)
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)

def catboost_importance_table_DR_X(
    model_dr: Dict,
    df: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    """
    For DR_X dict model: report importances for tau1, tau0, and a blended view
    using mean g(x) as weight if available; otherwise by treated share.
    """
    feats = [f for f in features if f in df.columns]
    tau1_imp = _cb_importance(model_dr["tau1"], df, feats)
    tau0_imp = _cb_importance(model_dr["tau0"], df, feats)

    if model_dr.get("g", None) is not None:
        from catboost import Pool
        g_hat = model_dr["g"].predict_proba(Pool(df[feats], cat_features=[i for i,c in enumerate(feats) if str(df[c].dtype).startswith("category")]))[:,1]
        w = float(np.clip(np.mean(g_hat), 1e-3, 1-1e-3))
    else:
        w = float(df[TREAT_COL].mean())

    blended = (1.0 - w) * tau1_imp + w * tau0_imp
    out = pd.DataFrame({
        "tau1_importance": tau1_imp,
        "tau0_importance": tau0_imp,
        "blended_tau_importance": blended,
    }).sort_values("blended_tau_importance", ascending=False)
    return out

def catboost_importance_table_X(
    model_x: Dict,
    df: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    """Same idea for the plain X-learner (tau1/tau0 then blend by mean g)."""
    feats = [f for f in features if f in df.columns]
    tau1_imp = _cb_importance(model_x["tau1"], df, feats)
    tau0_imp = _cb_importance(model_x["tau0"], df, feats)

    g_hat = model_x["g"].predict_proba(df[feats])[:,1]
    w = float(np.clip(np.mean(g_hat), 1e-3, 1-1e-3))
    blended = (1.0 - w) * tau1_imp + w * tau0_imp

    return pd.DataFrame({
        "tau1_importance": tau1_imp,
        "tau0_importance": tau0_imp,
        "blended_tau_importance": blended,
    }).sort_values("blended_tau_importance", ascending=False)

def catboost_importance_table_PlainReg(
    model_plain: Dict,
    df: pd.DataFrame,
) -> pd.Series:
    """
    For the plain CatBoost regressor model (log-target baseline).
    This is *regressor* importance, not uplift; useful as a sanity check.
    """
    feats = [f for f in model_plain["features"] if f in df.columns]
    return _cb_importance(model_plain["reg"], df, feats).rename("reg_feature_importance")
