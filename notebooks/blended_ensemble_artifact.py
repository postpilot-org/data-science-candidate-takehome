# ---- blended_ensemble_artifact.py (paste into a notebook cell) ----
import os, joblib, numpy as np, pandas as pd
from typing import Dict, Optional, List, Tuple
from catboost import Pool

# Must match your code
TREAT_COL = "treatment_group"
TARGET_COL = "total_30d_post_campaign_spend"
CAMPAIGN_COL = "campaign_id"
DATE_COL = "campaign_send_date"


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature builder consistent with your training (no campaign_id as feature).
    Keeps campaign_id as string for grouping; not used as a feature here.
    """
    out = df.copy()

    # treatment -> 0/1
    mapping = {
        "control": 0,
        "treatment": 1,
        "0": 0,
        "1": 1,
        0: 0,
        1: 1,
        True: 1,
        False: 0,
    }
    out[TREAT_COL] = out[TREAT_COL].apply(
        lambda x: mapping.get(x, mapping.get(str(x).lower(), np.nan))
    )
    out = out.dropna(subset=[TREAT_COL]).copy()
    out[TREAT_COL] = out[TREAT_COL].astype(int)

    # seasonality
    dt = pd.to_datetime(out[DATE_COL], errors="coerce")
    out["day_of_week"] = dt.dt.dayofweek.astype("Int64").astype("category")
    out["month"] = dt.dt.month.astype("Int64").astype("category")
    out["quarter"] = dt.dt.quarter.astype("Int64").astype("category")

    # base transforms
    hist = pd.to_numeric(out.get("historical_lifetime_spend"), errors="coerce").fillna(
        0
    )
    out["log_hist_spend"] = np.log1p(hist.clip(lower=0))
    dslp = pd.to_numeric(out.get("days_since_last_purchase"), errors="coerce").fillna(0)
    out["inv_days_since"] = 1.0 / (dslp.clip(lower=0) + 1.0)

    # engineered
    web = pd.to_numeric(out.get("web_activity_score"), errors="coerce").fillna(0)
    openr = pd.to_numeric(out.get("email_open_rate"), errors="coerce").fillna(0)
    aff = pd.to_numeric(out.get("brand_affinity_score"), errors="coerce").fillna(0)
    price = pd.to_numeric(out.get("price_sensitivity"), errors="coerce").fillna(0)
    tenure = pd.to_numeric(out.get("customer_tenure_days"), errors="coerce").fillna(0)

    out["engagement_score"] = 0.5 * web + 0.5 * openr
    out["engagement_mult"] = web * openr
    out["affinity_x_recency"] = aff * out["inv_days_since"].astype(float)
    out["log_tenure"] = np.log1p(tenure.clip(lower=0))
    out["price_sensitivity_sq"] = price**2
    out["affinity_over_price"] = aff / (price + 1e-3)

    # recency bin
    edges = [-np.inf, 0, 7, 14, 30, 60, 90, 180, 365, 730, np.inf]
    labels = [
        "=0",
        "1-7",
        "8-14",
        "15-30",
        "31-60",
        "61-90",
        "91-180",
        "181-365",
        "366-730",
        ">730",
    ]
    out["recency_bin"] = pd.cut(
        dslp, bins=edges, labels=labels, include_lowest=True
    ).astype("category")

    # keep for grouping only
    out[CAMPAIGN_COL] = out[CAMPAIGN_COL].astype(str)
    return out


def _extract_weights(
    ens, order=("DR_X", "X", "Plain")
) -> Tuple[Dict[str, float], float]:
    """
    Accepts:
      - dict like {"weights": {"DR_X":..., "X":..., "Plain":...}, "intercept": ...}
      - plain dict {"DR_X":..., "X":..., "Plain":..., "intercept": ...}
      - list/tuple/np.ndarray of weights in the given `order`
      - object with attributes .weights or named attributes
    Returns (weights_map, intercept_float)
    """
    # dict variants
    if isinstance(ens, dict):
        if "weights" in ens:
            w = ens["weights"]
            b = float(ens.get("intercept", 0.0))
            if isinstance(w, (list, tuple, np.ndarray)):
                w = np.asarray(w, dtype=float)
                if w.shape[0] != len(order):
                    raise ValueError(
                        "Weights array length does not match expected order."
                    )
                wmap = {k: float(w[i]) for i, k in enumerate(order)}
            else:
                wmap = {k: float(w.get(k, 0.0)) for k in order}
            return wmap, b
        else:
            b = float(ens.get("intercept", 0.0))
            wmap = {k: float(ens.get(k, 0.0)) for k in order}
            return wmap, b

    # list/tuple/array
    if isinstance(ens, (list, tuple, np.ndarray)):
        arr = np.asarray(ens, dtype=float)
        if arr.shape[0] != len(order):
            raise ValueError("Weights array length does not match expected order.")
        wmap = {k: float(arr[i]) for i, k in enumerate(order)}
        return wmap, 0.0

    # object with .weights
    if hasattr(ens, "weights"):
        return _extract_weights(getattr(ens, "weights"), order)

    # object with named attributes
    try:
        wmap = {k: float(getattr(ens, k)) for k in order}
        b = float(getattr(ens, "intercept", 0.0))
        return wmap, b
    except Exception as e:
        raise TypeError(f"Unsupported ens_cvx format: {type(ens)}") from e


class BlendedTauDF:
    """
    Deployable blended learner:
      - holds your three base models (m_dr, m_x, m_plain)
      - builds features internally
      - outputs per-row uplift: predict_tau_df(df) -> np.ndarray
    """

    def __init__(
        self,
        m_dr: Optional[Dict],
        m_x: Optional[Dict],
        m_plain: Optional[Dict],
        weights,
        intercept: float = 0.0,
        order: Tuple[str, ...] = ("DR_X", "X", "Plain"),
    ):
        self.m_dr = m_dr
        self.m_x = m_x
        self.m_plain = m_plain
        self.order = list(order)

        # Robust weight parsing
        if isinstance(weights, (list, tuple, np.ndarray)):
            wvec = np.asarray(weights, dtype=float)
            if wvec.shape[0] != len(self.order):
                raise ValueError("Weights array length mismatch.")
            self.w = wvec
        elif isinstance(weights, dict):
            self.w = np.array(
                [float(weights.get(k, 0.0)) for k in self.order], dtype=float
            )
        else:
            # treat as an object; reuse extractor
            wmap, _ = _extract_weights(weights, order=self.order)
            self.w = np.array(
                [float(wmap.get(k, 0.0)) for k in self.order], dtype=float
            )

        self.b = float(intercept)

    # ---- base components ----
    def _tau_dr(self, df_f: pd.DataFrame) -> np.ndarray:
        if self.m_dr is None:
            return np.zeros(len(df_f), float)
        feats = self.m_dr["features"]
        cats = self.m_dr["cat_features"]
        tau1_hat = self.m_dr["tau1"].predict(Pool(df_f[feats], cat_features=cats))
        tau0_hat = self.m_dr["tau0"].predict(Pool(df_f[feats], cat_features=cats))
        if self.m_dr.get("use_propensity", False) and self.m_dr.get("g") is not None:
            g = self.m_dr["g"].predict_proba(Pool(df_f[feats], cat_features=cats))[:, 1]
            g = np.clip(g, 1e-3, 1 - 1e-3)
            return g * tau0_hat + (1 - g) * tau1_hat
        else:
            w = float(df_f[TREAT_COL].mean())
            return w * tau0_hat + (1 - w) * tau1_hat

    def _tau_x(self, df_f: pd.DataFrame) -> np.ndarray:
        if self.m_x is None:
            return np.zeros(len(df_f), float)
        feats = self.m_x["features"]
        cats = self.m_x["cat_features"]
        tau1_hat = self.m_x["tau1"].predict(Pool(df_f[feats], cat_features=cats))
        tau0_hat = self.m_x["tau0"].predict(Pool(df_f[feats], cat_features=cats))
        g = self.m_x["g"].predict_proba(Pool(df_f[feats], cat_features=cats))[:, 1]
        return g * tau0_hat + (1 - g) * tau1_hat

    def _tau_plain(self, df_f: pd.DataFrame) -> np.ndarray:
        if self.m_plain is None:
            return np.zeros(len(df_f), float)
        feats = list(self.m_plain["features"])
        if TREAT_COL not in feats:
            raise AssertionError(
                "Plain model must include treatment_group to produce uplift by toggle."
            )
        X1 = df_f[feats].copy()
        X1[TREAT_COL] = 1
        X0 = df_f[feats].copy()
        X0[TREAT_COL] = 0
        y1 = self.m_plain["reg"].predict(X1)
        y0 = self.m_plain["reg"].predict(X0)
        if self.m_plain.get("log_target", False):
            smear = float(self.m_plain.get("smear", 1.0))
            y1 = np.expm1(y1) * smear
            y0 = np.expm1(y0) * smear
        return y1 - y0

    # ---- public API ----
    def predict_tau_df(self, df: pd.DataFrame) -> np.ndarray:
        df_f = _build_features(df)  # ensure features exist
        parts = []
        if self.w[0] != 0.0:
            parts.append(self.w[0] * self._tau_dr(df_f))
        if self.w[1] != 0.0:
            parts.append(self.w[1] * self._tau_x(df_f))
        if self.w[2] != 0.0:
            parts.append(self.w[2] * self._tau_plain(df_f))
        if not parts:
            return self.b + np.zeros(len(df_f), float)
        return np.sum(parts, axis=0) + self.b