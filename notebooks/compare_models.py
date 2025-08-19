"""
Model comparison utilities (log-target by default + Duan smearing).

Includes:
  1) DR CatBoost X-learner (cap → log1p target for m1/m0; tau in dollars; optional propensity)
  2) Plain CatBoost X-learner (log1p target for m1/m0; tau in dollars)
  3) Plain CatBoost baseline (predict spend directly; log1p target; includes treatment by default)

Common API:
  - fit_dr_xlearner(df, feats, cats, ...)
  - fit_plain_xlearner(df, feats, cats, ...)
  - fit_plain_catboost(df, feats, cats, include_treatment=True, ...)
  - summarize_campaigns(df, model)  # campaign-level iROAS
  - evaluate_all_models(df, {"DR_X": m1, "X": m2, "Plain": m3}, feats, cats, cap_actual_at=...)
  - plotting + winsorized error helpers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split

# ---- Columns / constants ----
TREAT_COL = "treatment_group"
TARGET_COL = "total_30d_post_campaign_spend"
CAMPAIGN_COL = "campaign_id"
DATE_COL = "campaign_send_date"
COST_PER_POSTCARD = 0.70

pd.set_option("display.float_format", "{:,.2f}".format)

# ==============================
# Feature engineering
# ==============================
# --- constants assumed available ---
# TREAT_COL = "treatment_group"
# TARGET_COL = "total_30d_post_campaign_spend"
# CAMPAIGN_COL = "campaign_id"
# DATE_COL = "campaign_send_date"

import numpy as np
import pandas as pd
from typing import List, Tuple

def build_features(
    df: pd.DataFrame,
    *,
    include_campaign_id: bool = False
) -> pd.DataFrame:
    """
    One-step feature builder:
      - cleans treatment
      - adds seasonality (day_of_week, month, quarter)
      - base transforms (log_hist_spend, inv_days_since)
      - engineered continuous & interactions
      - categorical recency_bin

    NOTE: keeps campaign_id column in the frame for grouping, but by default
    it is NOT intended to be used as a model feature (see choose_features()).
    """
    out = df.copy()

    # --- treatment -> 0/1 (robust mapping)
    mapping = {"control": 0, "treatment": 1, "0": 0, "1": 1, 0: 0, 1: 1, True: 1, False: 0}
    out[TREAT_COL] = out[TREAT_COL].apply(lambda x: mapping.get(x, mapping.get(str(x).lower(), np.nan)))
    out = out.dropna(subset=[TREAT_COL]).copy()
    out[TREAT_COL] = out[TREAT_COL].astype(int)

    # --- seasonality from date
    dt = pd.to_datetime(out[DATE_COL], errors="coerce")
    out["day_of_week"] = dt.dt.dayofweek.astype("Int64").astype("category")
    out["month"] = dt.dt.month.astype("Int64").astype("category")
    out["quarter"] = dt.dt.quarter.astype("Int64").astype("category")

    # --- base transforms
    hist = pd.to_numeric(out.get("historical_lifetime_spend"), errors="coerce").fillna(0)
    out["log_hist_spend"] = np.log1p(hist.clip(lower=0))

    dslp = pd.to_numeric(out.get("days_since_last_purchase"), errors="coerce").fillna(0)
    out["inv_days_since"] = 1.0 / (dslp.clip(lower=0) + 1.0)

    # --- engineered features
    web   = pd.to_numeric(out.get("web_activity_score"), errors="coerce")
    openr = pd.to_numeric(out.get("email_open_rate"), errors="coerce")
    aff   = pd.to_numeric(out.get("brand_affinity_score"), errors="coerce")
    price = pd.to_numeric(out.get("price_sensitivity"), errors="coerce")
    tenure= pd.to_numeric(out.get("customer_tenure_days"), errors="coerce")

    if web.notna().any() and openr.notna().any():
        out["engagement_score"] = 0.5 * web.fillna(0) + 0.5 * openr.fillna(0)
        out["engagement_mult"]  = web.fillna(0) * openr.fillna(0)

    if aff.notna().any():
        out["affinity_x_recency"] = aff.fillna(0) * out["inv_days_since"].astype(float)

    if tenure.notna().any():
        out["log_tenure"] = np.log1p(tenure.fillna(0).clip(lower=0))

    if price.notna().any():
        p = price.fillna(0)
        out["price_sensitivity_sq"] = p ** 2
        out["affinity_over_price"]  = aff.fillna(0) / (p + 1e-3)

    # --- categorical recency bin
    edges  = [-np.inf, 0, 7, 14, 30, 60, 90, 180, 365, 730, np.inf]
    labels = ["=0","1-7","8-14","15-30","31-60","61-90","91-180","181-365","366-730",">730"]
    out["recency_bin"] = pd.cut(dslp, bins=edges, labels=labels, include_lowest=True).astype("category")

    # Keep campaign_id as string for grouping/reporting; not used as feature unless requested
    out[CAMPAIGN_COL] = out[CAMPAIGN_COL].astype(str)

    # Optional: if you *do* want to let models see the campaign_id, convert to category
    if include_campaign_id:
        out[CAMPAIGN_COL] = out[CAMPAIGN_COL].astype("category")

    return out


def choose_features(
    df: pd.DataFrame,
    *,
    include_campaign_id: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Returns (features, categorical_features) for modeling.
    By default, EXCLUDES `campaign_id` from the feature set to avoid ID memorization.
    """
    numeric = [
        "brand_affinity_score",
        "web_activity_score",
        "price_sensitivity",
        "email_open_rate",
        "customer_tenure_days",
        "historical_lifetime_spend",
        "days_since_last_purchase",
        "log_hist_spend",
        "inv_days_since",
        # engineered
        "engagement_score",
        "engagement_mult",
        "affinity_x_recency",
        "log_tenure",
        "affinity_over_price",
        "price_sensitivity_sq",
    ]

    cats = ["day_of_week", "month", "quarter", "recency_bin"]
    if include_campaign_id:
        cats = [CAMPAIGN_COL] + cats

    # keep only those present
    feats = [c for c in numeric if c in df.columns] + [c for c in cats if c in df.columns]
    cat_features = [c for c in cats if c in feats]
    return feats, cat_features


# ==============================
# Shared helpers
# ==============================
_REG = dict(
    depth=6,
    learning_rate=0.04,
    l2_leaf_reg=6.0,
    random_strength=1.5,
    subsample=0.8,
    random_seed=42,
    verbose=False,
)
_IT_OUTCOME, _IT_TAU, _ESR = 4000, 3000, 200

def _split_arm(df_arm: pd.DataFrame, features: List[str], label_col: str):
    X_tr, X_va, y_tr, y_va = train_test_split(
        df_arm[features], df_arm[label_col].values, test_size=0.15, random_state=42
    )
    return (X_tr, y_tr), (X_va, y_va)

def _cap_target(df: pd.DataFrame, cap_abs: Optional[float] = None, cap_q: float = 0.995) -> Tuple[pd.DataFrame, float]:
    out = df.copy()
    if cap_abs is None:
        cap_val = float(np.quantile(out[TARGET_COL].astype(float).values, cap_q))
    else:
        cap_val = float(cap_abs)
    out[TARGET_COL] = np.clip(out[TARGET_COL].astype(float).values, 0.0, cap_val)
    return out, cap_val

# ==============================
# Actual & predicted aggregation
# ==============================
def _campaign_actual(df: pd.DataFrame) -> pd.DataFrame:
    g_all = df.groupby(CAMPAIGN_COL, observed=True, dropna=False)
    n_treat = g_all[TREAT_COL].sum().rename("n_treat")
    ad_cost = (n_treat * COST_PER_POSTCARD).rename("ad_cost")

    treat = (
        df.loc[df[TREAT_COL] == 1]
          .groupby(CAMPAIGN_COL, observed=True, dropna=False)[TARGET_COL]
          .sum().rename("treat_spend")
    ).reindex(n_treat.index, fill_value=0)

    ctrl = (
        df.loc[df[TREAT_COL] == 0]
          .groupby(CAMPAIGN_COL, observed=True, dropna=False)[TARGET_COL]
          .sum().rename("ctrl_spend")
    ).reindex(n_treat.index, fill_value=0)

    incr = (treat - ctrl).rename("incremental_spend")
    iroas = (incr / ad_cost.replace(0, np.nan)).rename("iROAS_actual")
    return pd.concat([n_treat, ad_cost, treat, ctrl, incr, iroas], axis=1).reset_index()

def summarize_campaigns_from_tau(df: pd.DataFrame, tau_hat: np.ndarray) -> pd.DataFrame:
    tmp = df.copy()
    tmp["tau_hat"] = np.asarray(tau_hat)
    g_all = tmp.groupby(CAMPAIGN_COL, observed=True, dropna=False)
    n_treat = g_all[TREAT_COL].sum()
    ad_cost = n_treat * COST_PER_POSTCARD

    incr_pred = (
        tmp.loc[tmp[TREAT_COL] == 1]
           .groupby(CAMPAIGN_COL, observed=True, dropna=False)["tau_hat"]
           .sum()
    ).reindex(n_treat.index, fill_value=0)

    iroas_pred = incr_pred / ad_cost.replace(0, np.nan)

    return (
        pd.DataFrame({
            CAMPAIGN_COL: n_treat.index.astype(str),
            "n_treat": n_treat.values,
            "ad_cost": ad_cost.values,
            "incremental_pred_spend": incr_pred.values,
            "iROAS_pred": iroas_pred.values,
        })
        .sort_values(CAMPAIGN_COL)
        .reset_index(drop=True)
    )

# ==============================
# DR CatBoost X-learner (log-target for m1/m0 + smearing)
# ==============================
def fit_dr_xlearner(
    df_f: pd.DataFrame,
    features: List[str],
    cat_features: List[str],
    cap_abs: Optional[float] = None,
    cap_q: float = 0.995,
    use_propensity: bool = True,
    seed: int = 42,
) -> Dict:
    """
    DR-flavored X-learner with LOG-TARGET outcome models.

    1) Cap y in dollars (optional), then y_log = log1p(y_cap)
    2) Fit m1/m0 on y_log (RMSE); compute Duan smearing s1/s0 from val residuals
    3) Back-transform outcome preds to dollars: mu = s * exp(yhat_log) - 1
       Build D1/D0 on dollar scale; fit tau models (RMSE)
    4) Optional propensity g(x) to blend tau1/tau0
    """
    df_cap, cap_val = _cap_target(df_f, cap_abs=cap_abs, cap_q=cap_q)
    df_t1 = df_cap[df_cap[TREAT_COL] == 1].copy()
    df_t0 = df_cap[df_cap[TREAT_COL] == 0].copy()

    g_model = None
    if use_propensity:
        g_model = CatBoostClassifier(
            iterations=5000, loss_function="Logloss", auto_class_weights="Balanced",
            **{k: v for k, v in _REG.items() if k != "loss_function"}
        )
        Xtr, Xva, ytr, yva = train_test_split(
            df_cap[features], df_cap[TREAT_COL].values,
            test_size=0.15, random_state=seed, stratify=df_cap[TREAT_COL].values
        )
        g_model.fit(Xtr, ytr, cat_features=cat_features, eval_set=(Xva, yva), early_stopping_rounds=_ESR)

    # treated arm
    (X1tr, y1tr_raw), (X1va, y1va_raw) = _split_arm(df_t1, features, TARGET_COL)
    y1tr = np.log1p(y1tr_raw.astype(float))
    y1va = np.log1p(y1va_raw.astype(float))
    m1 = CatBoostRegressor(iterations=_IT_OUTCOME, loss_function="RMSE", **_REG)
    m1.fit(X1tr, y1tr, cat_features=cat_features, eval_set=(X1va, y1va), early_stopping_rounds=_ESR)
    y1va_hat = m1.predict(X1va)
    smear1 = float(np.mean(np.exp(y1va - y1va_hat)))

    # control arm
    (X0tr, y0tr_raw), (X0va, y0va_raw) = _split_arm(df_t0, features, TARGET_COL)
    y0tr = np.log1p(y0tr_raw.astype(float))
    y0va = np.log1p(y0va_raw.astype(float))
    m0 = CatBoostRegressor(iterations=_IT_OUTCOME, loss_function="RMSE", **_REG)
    m0.fit(X0tr, y0tr, cat_features=cat_features, eval_set=(X0va, y0va), early_stopping_rounds=_ESR)
    y0va_hat = m0.predict(X0va)
    smear0 = float(np.mean(np.exp(y0va - y0va_hat)))

    # back-transform for all rows
    mu1_all = smear1 * np.exp(m1.predict(df_cap[features])) - 1.0
    mu0_all = smear0 * np.exp(m0.predict(df_cap[features])) - 1.0
    y_cap = df_cap[TARGET_COL].astype(float).values

    # uplifts (dollar scale)
    mask_t1 = df_cap[TREAT_COL].values == 1
    D1 = y_cap[mask_t1] - mu0_all[mask_t1]  # treated
    D0 = mu1_all[~mask_t1] - y_cap[~mask_t1]  # control

    dfD1 = df_t1.copy(); dfD1["D"] = D1
    dfD0 = df_t0.copy(); dfD0["D"] = D0

    (Xt1tr, Dt1tr), (Xt1va, Dt1va) = _split_arm(dfD1, features, "D")
    tau1 = CatBoostRegressor(iterations=_IT_TAU, loss_function="RMSE", **_REG)
    tau1.fit(Xt1tr, Dt1tr, cat_features=cat_features, eval_set=(Xt1va, Dt1va), early_stopping_rounds=_ESR)

    (Xt0tr, Dt0tr), (Xt0va, Dt0va) = _split_arm(dfD0, features, "D")
    tau0 = CatBoostRegressor(iterations=_IT_TAU, loss_function="RMSE", **_REG)
    tau0.fit(Xt0tr, Dt0tr, cat_features=cat_features, eval_set=(Xt0va, Dt0va), early_stopping_rounds=_ESR)

    return {
        "type": "DR_X",
        "m1": m1, "m0": m0,
        "tau1": tau1, "tau0": tau0,
        "g": g_model,
        "features": features, "cat_features": cat_features,
        "cap_value": cap_val,
        "use_propensity": use_propensity,
        "log_target": True,
        "smear1": smear1, "smear0": smear0,
        "seed": seed,
    }

def predict_tau_dr_x(df_f: pd.DataFrame, model: Dict) -> np.ndarray:
    feats = model["features"]
    tau1_hat = model["tau1"].predict(df_f[feats])
    tau0_hat = model["tau0"].predict(df_f[feats])
    if model.get("use_propensity", False) and model.get("g") is not None:
        g = model["g"].predict_proba(df_f[feats])[:, 1]
        g = np.clip(g, 1e-3, 1 - 1e-3)
        return g * tau0_hat + (1 - g) * tau1_hat
    else:
        w = float(df_f[TREAT_COL].mean())
        return w * tau0_hat + (1 - w) * tau1_hat

# ==============================
# Plain X-learner (log-target for m1/m0 + smearing)
# ==============================
def fit_plain_xlearner(
    df_f: pd.DataFrame,
    features: List[str],
    cat_features: List[str],
    seed: int = 42,
) -> Dict:
    df_t1 = df_f[df_f[TREAT_COL] == 1].copy()
    df_t0 = df_f[df_f[TREAT_COL] == 0].copy()

    # treated
    X1tr, X1va, y1tr_raw, y1va_raw = train_test_split(
        df_t1[features], df_t1[TARGET_COL].astype(float).values,
        test_size=0.15, random_state=seed
    )
    y1tr = np.log1p(y1tr_raw); y1va = np.log1p(y1va_raw)
    m1 = CatBoostRegressor(depth=6, iterations=2000, learning_rate=0.04, loss_function="RMSE",
                           random_seed=seed, verbose=False)
    m1.fit(X1tr, y1tr, cat_features=cat_features, eval_set=(X1va, y1va), early_stopping_rounds=200)
    smear1 = float(np.mean(np.exp(y1va - m1.predict(X1va))))

    # control
    X0tr, X0va, y0tr_raw, y0va_raw = train_test_split(
        df_t0[features], df_t0[TARGET_COL].astype(float).values,
        test_size=0.15, random_state=seed
    )
    y0tr = np.log1p(y0tr_raw); y0va = np.log1p(y0va_raw)
    m0 = CatBoostRegressor(depth=6, iterations=2000, learning_rate=0.04, loss_function="RMSE",
                           random_seed=seed, verbose=False)
    m0.fit(X0tr, y0tr, cat_features=cat_features, eval_set=(X0va, y0va), early_stopping_rounds=200)
    smear0 = float(np.mean(np.exp(y0va - m0.predict(X0va))))

    # dollar-scale mu
    mu1_all = smear1 * np.exp(m1.predict(df_f[features])) - 1.0
    mu0_all = smear0 * np.exp(m0.predict(df_f[features])) - 1.0
    y = df_f[TARGET_COL].astype(float).values
    mask_t1 = df_f[TREAT_COL].values == 1

    D1 = y[mask_t1] - mu0_all[mask_t1]
    D0 = mu1_all[~mask_t1] - y[~mask_t1]

    tau1 = CatBoostRegressor(depth=6, iterations=1200, learning_rate=0.05, loss_function="RMSE",
                             random_seed=seed, verbose=False)
    tau0 = CatBoostRegressor(depth=6, iterations=1200, learning_rate=0.05, loss_function="RMSE",
                             random_seed=seed, verbose=False)
    tau1.fit(df_f.loc[mask_t1, features], D1, cat_features=cat_features)
    tau0.fit(df_f.loc[~mask_t1, features], D0, cat_features=cat_features)

    # simple propensity
    g_model = CatBoostClassifier(depth=6, iterations=1000, learning_rate=0.04, loss_function="Logloss",
                                 random_seed=seed, verbose=False)
    g_model.fit(df_f[features], df_f[TREAT_COL].astype(int), cat_features=cat_features)

    return {
        "type": "X",
        "m1": m1, "m0": m0, "tau1": tau1, "tau0": tau0, "g": g_model,
        "features": features, "cat_features": cat_features,
        "log_target": True, "smear1": smear1, "smear0": smear0,
    }

def predict_tau_plain_x(df_f: pd.DataFrame, model: Dict) -> np.ndarray:
    feats = model["features"]
    g = model["g"].predict_proba(df_f[feats])[:, 1]
    tau1_hat = model["tau1"].predict(df_f[feats])
    tau0_hat = model["tau0"].predict(df_f[feats])
    return g * tau0_hat + (1 - g) * tau1_hat

# ==============================
# Plain baseline (predict spend directly; log-target + smearing)
# ==============================
def fit_plain_catboost(
    df_f: pd.DataFrame,
    features: List[str],
    cat_features: List[str],
    include_treatment: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Plain CatBoost baseline on log1p(y). Treatment can be included (default)
    as a feature. During summary we back-transform with Duan smearing.
    """
    feats = list(dict.fromkeys([*features, TREAT_COL] if include_treatment else [*features]))
    feats = [f for f in feats if f in df_f.columns]
    cats  = [c for c in cat_features if c in feats]

    y_log = np.log1p(df_f[TARGET_COL].astype(float).values)
    X_tr, X_va, y_tr, y_va = train_test_split(df_f[feats], y_log, test_size=0.15, random_state=seed)

    # reg = CatBoostRegressor(
    #     loss_function="RMSE",
    #     depth=6,
    #     iterations=4000,
    #     learning_rate=0.03,
    #     l2_leaf_reg=10.0,
    #     subsample=0.8,
    #     rsm=0.7,
    #     random_seed=seed,
    #     verbose=False,
    # )
    reg = CatBoostRegressor(
        loss_function="RMSE",
        depth=4,
        iterations=4000,
        learning_rate=0.02,
        l2_leaf_reg=24.0,
        subsample=0.7,
        rsm=0.6,
        min_data_in_leaf=80,
        random_seed=seed,
        verbose=False,
    )

    reg.fit(X_tr, y_tr, cat_features=cats, eval_set=(X_va, y_va), early_stopping_rounds=300)

    # Duan smearing
    smear = float(np.mean(np.exp(y_va - reg.predict(X_va))))

    return {"type": "Plain", "reg": reg, "features": feats, "cat_features": cats,
            "log_target": True, "smear": smear}

def _summarize_campaigns_plain(df: pd.DataFrame, model: Dict) -> pd.DataFrame:
    # back-transform if log-target
    if model.get("log_target", False):
        yhat_log = model["reg"].predict(df[model["features"]])
        yhat = np.expm1(yhat_log) * float(model.get("smear", 1.0))
    else:
        yhat = model["reg"].predict(df[model["features"]])

    tmp = df.copy(); tmp["yhat"] = yhat
    g_all = tmp.groupby(CAMPAIGN_COL, observed=True, dropna=False)
    n_treat = g_all[TREAT_COL].sum()
    ad_cost = n_treat * COST_PER_POSTCARD

    yhat_treat = (
        tmp.loc[tmp[TREAT_COL] == 1].groupby(CAMPAIGN_COL, observed=True, dropna=False)["yhat"].sum()
    ).reindex(n_treat.index, fill_value=0)
    yhat_ctrl = (
        tmp.loc[tmp[TREAT_COL] == 0].groupby(CAMPAIGN_COL, observed=True, dropna=False)["yhat"].sum()
    ).reindex(n_treat.index, fill_value=0)

    incr_pred = yhat_treat - yhat_ctrl
    iroas_pred = incr_pred / ad_cost.replace(0, np.nan)

    return (
        pd.DataFrame({
            CAMPAIGN_COL: n_treat.index.astype(str),
            "n_treat": n_treat.values,
            "ad_cost": ad_cost.values,
            "incremental_pred_spend": incr_pred.values,
            "iROAS_pred": iroas_pred.values,
        })
        .sort_values(CAMPAIGN_COL)
        .reset_index(drop=True)
    )

# ==============================
# Router & evaluation
# ==============================
def summarize_campaigns(df: pd.DataFrame, model: Dict) -> pd.DataFrame:
    mtype = model.get("type")
    if mtype == "Plain":
        return _summarize_campaigns_plain(df, model)
    elif mtype == "DR_X":
        return summarize_campaigns_from_tau(df, predict_tau_dr_x(df, model))
    elif mtype == "X":
        return summarize_campaigns_from_tau(df, predict_tau_plain_x(df, model))
    else:
        raise ValueError(f"Unknown model type: {mtype}")

def evaluate_all_models(
    df_f: pd.DataFrame,
    models: Dict[str, Dict],
    features: List[str],
    cat_features: List[str],
    cap_actual_at: Optional[float] = None,
) -> pd.DataFrame:
    """
    Per-campaign comparison with actual vs each model's predicted iROAS.

    If cap_actual_at is provided, actual spend is capped (dollars) before iROAS
    to align with DR training caps.
    """
    df_actual = df_f.copy()
    if cap_actual_at is not None:
        df_actual[TARGET_COL] = np.clip(df_actual[TARGET_COL].astype(float).values, 0.0, float(cap_actual_at))
    act = _campaign_actual(df_actual).rename(columns={
        "treat_spend": "treat_spend_actual",
        "ctrl_spend": "ctrl_spend_actual",
        "incremental_spend": "incremental_spend_actual",
        "iROAS_actual": "iROAS_actual",
    })

    outs = [act[[CAMPAIGN_COL, "n_treat", "ad_cost", "treat_spend_actual", "ctrl_spend_actual", "incremental_spend_actual", "iROAS_actual"]]]
    for name, mdl in models.items():
        summ = summarize_campaigns(df_f, mdl).rename(columns={
            "incremental_pred_spend": f"incremental_pred_spend_{name}",
            "iROAS_pred": f"iROAS_pred_{name}",
        })
        outs.append(summ[[CAMPAIGN_COL, f"incremental_pred_spend_{name}", f"iROAS_pred_{name}"]])

    out = outs[0]
    for tbl in outs[1:]:
        out = out.merge(tbl, on=CAMPAIGN_COL, how="left")

    for name in models.keys():
        out[f"abs_err_iROAS_{name}"] = (out[f"iROAS_pred_{name}"] - out["iROAS_actual"]).abs()

    return out.sort_values(CAMPAIGN_COL).reset_index(drop=True)

# ==============================
# Plots & robust metrics
# ==============================
def scatter_pred_vs_actual_iROAS(comparison_df: pd.DataFrame, model_name: str):
    import matplotlib.pyplot as plt
    x = comparison_df["iROAS_actual"].values
    y = comparison_df[f"iROAS_pred_{model_name}"].values
    labels = comparison_df[CAMPAIGN_COL].astype(str).values
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
    ax.plot(lims, lims)
    ax.set_xlabel("iROAS_actual")
    ax.set_ylabel(f"iROAS_pred_{model_name}")
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    return fig

def _winsorize(arr: np.ndarray, lo_q: float = 0.0, hi_q: float = 0.995) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    lo = np.nanquantile(arr, lo_q)
    hi = np.nanquantile(arr, hi_q)
    return np.clip(arr, lo, hi)

def winsorized_error_table(
    comparison_df: pd.DataFrame,
    model_names: List[str] = ("DR_X","X","Plain"),
    target_col: str = "iROAS_actual",
    hi_q: float = 0.995,
) -> pd.DataFrame:
    out = []
    y = comparison_df[target_col].values
    y_w = _winsorize(y, 0.0, hi_q)
    for name in model_names:
        yhat = comparison_df[f"iROAS_pred_{name}"].values
        yhat_w = _winsorize(yhat, 0.0, hi_q)
        mae = float(np.nanmean(np.abs(yhat_w - y_w)))
        rmse = float(np.sqrt(np.nanmean((yhat_w - y_w) ** 2)))
        out.append({"model": name, "MAE_winsor": mae, "RMSE_winsor": rmse, "hi_q": hi_q})
    return pd.DataFrame(out).sort_values("MAE_winsor").reset_index(drop=True)

def campaigns_with_whales(df_rows: pd.DataFrame, spend_threshold: float = 50000.0) -> List[str]:
    mask = pd.to_numeric(df_rows.get(TARGET_COL, pd.Series(dtype=float)), errors="coerce") > float(spend_threshold)
    return df_rows.loc[mask, CAMPAIGN_COL].astype(str).unique().tolist()

def scatter_pred_vs_actual_iROAS_whales(
    comparison_df: pd.DataFrame,
    model_name: str,
    df_rows: Optional[pd.DataFrame] = None,
    whale_threshold: float = 50000.0,
    winsor_hi_q: Optional[float] = None,
):
    import matplotlib.pyplot as plt
    x = comparison_df["iROAS_actual"].values
    y = comparison_df[f"iROAS_pred_{model_name}"].values
    labels = comparison_df[CAMPAIGN_COL].astype(str).values

    if winsor_hi_q is not None:
        x = _winsorize(x, 0.0, winsor_hi_q)
        y = _winsorize(y, 0.0, winsor_hi_q)

    whale_ids = set(campaigns_with_whales(df_rows, whale_threshold)) if df_rows is not None else set()

    fig, ax = plt.subplots()
    idx_reg = [i for i, cid in enumerate(labels) if cid not in whale_ids]
    ax.scatter(x[idx_reg], y[idx_reg])

    if whale_ids:
        idx_wh = [i for i, cid in enumerate(labels) if cid in whale_ids]
        if idx_wh:
            ax.scatter(x[idx_wh], y[idx_wh])  # distinct color by default

    lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
    ax.plot(lims, lims)
    ax.set_xlabel("iROAS_actual")
    ax.set_ylabel(f"iROAS_pred_{model_name}")

    resid = np.abs(y - x)
    top_k = np.argsort(resid)[-min(8, len(resid)):]
    for i in top_k:
        ax.annotate(labels[i], (x[i], y[i]), xytext=(3, 3), textcoords="offset points", fontsize=8)

    fig.tight_layout()
    return fig

# ==============================
# Ensembles on campaign incremental spend
# ==============================
def _get_Xy_from_comparison(comp_df: pd.DataFrame, model_keys: List[str]):
    """Build X (pred incremental spend) and y (actual incremental spend) from comparison table."""
    y = comp_df["incremental_spend_actual"].values.astype(float)
    X_cols = [f"incremental_pred_spend_{k}" for k in model_keys]
    X = comp_df[X_cols].values.astype(float)
    return X, y

def fit_convex_ensemble_simplex(
    comp_train: pd.DataFrame,
    model_keys: List[str] = ("DR_X","X","Plain"),
    step: float = 0.05,
    weight_col: str = "n_treat",
) -> Dict:
    """
    Find non-negative weights that sum to 1 by grid search on the simplex.
    Minimizes weighted squared error on incremental spend (campaign level).
    """
    X, y = _get_Xy_from_comparison(comp_train, list(model_keys))
    wts = comp_train[weight_col].values.astype(float) if weight_col in comp_train.columns else None

    best = {"loss": np.inf, "w": None}
    # 3-model simplex
    for w1 in np.arange(0.0, 1.0 + 1e-9, step):
        for w2 in np.arange(0.0, 1.0 - w1 + 1e-9, step):
            w3 = 1.0 - w1 - w2
            w = np.array([w1, w2, w3], dtype=float)
            yhat = X @ w
            if wts is None:
                loss = float(np.mean((y - yhat) ** 2))
            else:
                loss = float(np.sum(wts * (y - yhat) ** 2) / np.sum(wts))
            if loss < best["loss"]:
                best = {"loss": loss, "w": w}
    return {"type": "ensemble_convex", "weights": best["w"], "intercept": 0.0, "model_keys": list(model_keys)}

def fit_ridge_ensemble(
    comp_train: pd.DataFrame,
    model_keys: List[str] = ("DR_X","X","Plain"),
    ridge: float = 1e-6,
    weight_col: str = "n_treat",
) -> Dict:
    """
    Weighted ridge regression on incremental spend: y ≈ X @ w + b.
    Returns weights (can be negative) and an intercept.
    """
    X, y = _get_Xy_from_comparison(comp_train, list(model_keys))
    n, p = X.shape
    # add intercept column
    X1 = np.c_[X, np.ones(n)]
    # weights
    if weight_col in comp_train.columns:
        wts = comp_train[weight_col].values.astype(float)
        W = np.diag(wts)
    else:
        W = np.eye(n)
    # ridge (no reg on intercept)
    I = np.eye(p + 1)
    I[-1, -1] = 0.0
    XtWX = X1.T @ W @ X1 + ridge * I
    XtWy = X1.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)
    return {"type": "ensemble_ridge", "weights": beta[:p], "intercept": float(beta[-1]), "model_keys": list(model_keys)}

def apply_ensemble_to_comparison(
    comp_df: pd.DataFrame,
    ens: Dict,
    out_name: str = "Ensemble",
) -> pd.DataFrame:
    """
    Adds incremental_pred_spend_{out_name} and iROAS_pred_{out_name} columns to comp_df.
    """
    keys = ens["model_keys"]
    X, _ = _get_Xy_from_comparison(comp_df, keys)
    w = ens["weights"]
    b = ens.get("intercept", 0.0)

    incr_pred = X @ w + b
    out = comp_df.copy()
    out[f"incremental_pred_spend_{out_name}"] = incr_pred
    out[f"iROAS_pred_{out_name}"] = incr_pred / out["ad_cost"].values
    # add absolute error if actual present
    if "iROAS_actual" in out.columns:
        out[f"abs_err_iROAS_{out_name}"] = (out[f"iROAS_pred_{out_name}"] - out["iROAS_actual"]).abs()
    return out

def print_ensemble(ens: Dict):
    keys = ens["model_keys"]
    w = ens["weights"]
    b = ens.get("intercept", 0.0)
    pieces = [f"{k}:{w[i]:.3f}" for i,k in enumerate(keys)]
    print(f"Ensemble -> {' + '.join(pieces)} + intercept:{b:.3f}")
