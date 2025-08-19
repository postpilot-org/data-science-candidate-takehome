# app.py
from __future__ import annotations

import io
import os
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

# ---------- Constants (must match your training) ----------
TREAT_COL = "treatment_group"
TARGET_COL = "total_30d_post_campaign_spend"
CAMPAIGN_COL = "campaign_id"
DATE_COL = "campaign_send_date"
COST_PER_POSTCARD = 0.70

# Import the runtime class so joblib can resolve it on load
from blended_ensemble_artifact import BlendedTauDF  # noqa: F401

# ---------- Load artifacts ----------
ART_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")
LEARNER_PATH = os.path.join(ART_DIR, "learner.joblib")
INFO_PATH = os.path.join(ART_DIR, "info.joblib")

if not os.path.exists(LEARNER_PATH):
    raise RuntimeError(f"Missing model artifact: {LEARNER_PATH}")
if not os.path.exists(INFO_PATH):
    raise RuntimeError(f"Missing info artifact: {INFO_PATH}")

learner = joblib.load(LEARNER_PATH)  # BlendedTauDF object
info: Dict = joblib.load(INFO_PATH)  # e.g. {"cap_val":..., "weights":..., ...}

MODEL_VERSION = os.environ.get("MODEL_VERSION", str(info.get("model_version", "1.0.0")))
API_TOKEN = os.environ.get("API_TOKEN")  # optional bearer token

# ---------- Auth ----------
def require_auth(authorization: Optional[str] = Header(None)):
    if API_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        if authorization.split(" ", 1)[1] != API_TOKEN:
            raise HTTPException(status_code=403, detail="Forbidden")

# ---------- Core helpers ----------
def predict_tau_internal(df: pd.DataFrame) -> np.ndarray:
    """
    The blended learner builds features internally from raw columns.
    It does NOT use campaign_id as a feature (only for summarization).
    """
    if hasattr(learner, "predict_tau_df"):
        return learner.predict_tau_df(df)
    raise HTTPException(status_code=500, detail="Loaded learner does not support predict_tau_df().")

def summarize_campaigns_internal(
    df: pd.DataFrame,
    tau_hat: np.ndarray,
    cap_val: Optional[float],
    winsor_q: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """
    Aggregate per-row uplifts to campaign-level iROAS.
    - Keeps ad_cost = 0.70 * (# treated)
    - Optionally caps observed spend for reporting parity
    - Winsorizes tau within campaign before summing, to reduce whale domination
    """
    out = df.copy()
    y_obs = out[TARGET_COL].astype(float).values if TARGET_COL in out.columns else np.zeros(len(out))
    if cap_val is not None:
        y_obs = np.clip(y_obs, 0.0, float(cap_val))
    out["spend_cap"] = y_obs
    out["tau_hat"] = tau_hat

    ql, qh = winsor_q
    if not (0.0 <= ql < qh <= 1.0):
        raise HTTPException(status_code=400, detail="winsor_low/winsor_high must satisfy 0.0 <= low < high <= 1.0")

    def _sum_tau_wins(g: pd.DataFrame) -> float:
        vals = g["tau_hat"].values
        lo, hi = np.quantile(vals, [ql, qh]) if len(vals) > 1 else (vals[0], vals[0])
        vals = np.clip(vals, lo, hi)
        return vals[g[TREAT_COL].values == 1].sum()

    g = out.groupby(CAMPAIGN_COL, dropna=False)
    n_treat = g[TREAT_COL].sum()
    ad_cost = n_treat * COST_PER_POSTCARD

    # Observed (for reference)
    treat_spend = g.apply(lambda x: x.loc[x[TREAT_COL] == 1, "spend_cap"].sum())
    ctrl_spend = g.apply(lambda x: x.loc[x[TREAT_COL] == 0, "spend_cap"].sum())
    incr_actual = treat_spend - ctrl_spend
    iroas_actual = incr_actual / ad_cost.replace(0, np.nan)

    # Predicted
    incr_pred = g.apply(_sum_tau_wins)
    iroas_pred = incr_pred / ad_cost.replace(0, np.nan)

    return (
        pd.DataFrame(
            {
                CAMPAIGN_COL: n_treat.index.astype(str),
                "n_treat": n_treat.values,
                "ad_cost": ad_cost.values.round(2),
                "treat_spend_cap": treat_spend.values,
                "ctrl_spend_cap": ctrl_spend.values,
                "incremental_spend_cap": incr_actual.values,
                "iROAS_actual": iroas_actual.values,
                "incremental_pred_spend": incr_pred.values,
                "iROAS_pred": iroas_pred.values,
                "cap_value": float(cap_val) if cap_val is not None else np.nan,
            }
        )
        .sort_values(CAMPAIGN_COL)
        .reset_index(drop=True)
    )

# ---------- FastAPI app ----------
class RecordsPayload(BaseModel):
    records: List[Dict[str, Union[int, float, str]]]
    winsor_low: float = Field(0.01, ge=0.0, le=0.5)
    winsor_high: float = Field(0.99, ge=0.5, le=1.0)
    return_tau_only: bool = False

app = FastAPI(title="iROAS DR-Blend API", version=MODEL_VERSION)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.get("/metadata")
def metadata():
    return {
        "learner_kind": info.get("learner_kind", "df_blend"),
        "weights": info.get("weights"),
        "intercept": info.get("intercept"),
        "cap_val": info.get("cap_val"),
        "uses_campaign_id_feature": info.get("uses_campaign_id_feature", False),
        "model_version": MODEL_VERSION,
    }

# --- JSON records (small/medium batches) ---
@app.post("/predict_tau", dependencies=[Depends(require_auth)])
def predict_tau_json(payload: RecordsPayload):
    df = pd.DataFrame(payload.records)
    tau = predict_tau_internal(df)

    if payload.return_tau_only:
        return {"tau_hat": tau.tolist(), "model_version": MODEL_VERSION}

    # Summaries require campaign_id for grouping
    if CAMPAIGN_COL not in df.columns:
        raise HTTPException(status_code=400, detail=f"Missing '{CAMPAIGN_COL}' for summarization")

    summary = summarize_campaigns_internal(
        df,
        tau,
        cap_val=info.get("cap_val"),
        winsor_q=(payload.winsor_low, payload.winsor_high),
    )
    return {"tau_hat": tau.tolist(), "summary": summary.to_dict(orient="records"), "model_version": MODEL_VERSION}

# --- CSV upload (large batches) ---
@app.post("/summarize_campaigns", dependencies=[Depends(require_auth)])
async def summarize_campaigns_csv(
    file: UploadFile = File(...),
    winsor_low: float = 0.01,
    winsor_high: float = 0.99,
):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    if CAMPAIGN_COL not in df.columns:
        raise HTTPException(status_code=400, detail=f"CSV must include '{CAMPAIGN_COL}'")

    tau = predict_tau_internal(df)
    summary = summarize_campaigns_internal(df, tau, cap_val=info.get("cap_val"), winsor_q=(winsor_low, winsor_high))
    return {"rows": len(df), "summary": summary.to_dict(orient="records"), "model_version": MODEL_VERSION}
