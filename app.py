# app.py — Streamlit UI for IoT IDS (LightGBM, MLP, FT-Transformer)
# -------------------------------------------------------------------------------------
# What this app does
# - Loads training schema (numeric feature order, class order)
# - Validates & aligns uploaded CSV to match training schema (drops extras, fills missing)
# - Handles preprocessed ftt_* splits (skips re-scaling to avoid double-standardization)
# - Loads trained models & artifacts; runs real inference; shows predictions & metrics
# -------------------------------------------------------------------------------------

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Optional: only import torch/lightgbm when needed (avoids startup cost if model not selected)
# import torch, torch.nn as nn
# import lightgbm as lgb

# ----------------------------- paths & constants -------------------------------------
OUT_DIR = Path("./preprocessed_iot")
ARTIFACTS_DIR = Path("./artifacts")

SCHEMA_PATH = OUT_DIR / "schema.json"
CLASSES_PATH = OUT_DIR / "classes.json"
SCALER_PATH = OUT_DIR / "ftt_scaler.joblib"

LGBM_MODEL_PATH = ARTIFACTS_DIR / "lgbm_model.txt"
MLP_MODEL_PATH  = ARTIFACTS_DIR / "mlp.pt"
FTT_DIR         = ARTIFACTS_DIR / "ftt_pure"
FTT_MODEL_PATH  = FTT_DIR / "model.pt"
FTT_META_PATH   = FTT_DIR / "meta.json"

LABEL_COL = "Label"  # for optional ground-truth in uploaded CSVs

# ----------------------------- schema helpers ----------------------------------------
def load_schema():
    """
    Returns: (num_cols, cat_cols, classes)
    num_cols/cat_cols come from schema.json or ftt_scaler.joblib (fallback).
    classes come from classes.json (canonical order from training).
    """
    # 1) feature schema
    if SCHEMA_PATH.exists():
        schema = json.loads(SCHEMA_PATH.read_text())
        num_cols = schema.get("num_cols", [])
        cat_cols = schema.get("cat_cols", [])
    else:
        bundle = joblib.load(SCALER_PATH)  # {'num_cols','cat_cols','scaler'}
        num_cols = bundle.get("num_cols", [])
        cat_cols = bundle.get("cat_cols", [])

    # 2) class order
    if CLASSES_PATH.exists():
        classes = json.loads(CLASSES_PATH.read_text())
    else:
        classes = None  # will only be used to decode predictions; models can still output argmax

    return num_cols, cat_cols, classes


def validate_frame_columns(df: pd.DataFrame, num_cols, cat_cols):
    missing = [c for c in (num_cols + cat_cols) if c not in df.columns]
    extras  = [c for c in df.columns if c not in (num_cols + cat_cols + [LABEL_COL])]
    ok = (len(missing) == 0)
    return ok, missing, extras


def align_frame(df: pd.DataFrame, num_cols, cat_cols, drop_label=True):
    """
    Aligns df to training schema (adds missing cols with safe defaults, drops extras, enforces order).
    """
    df = df.copy()
    if drop_label and (LABEL_COL in df.columns):
        df = df.drop(columns=[LABEL_COL])

    # add missing with safe defaults
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    for c in cat_cols:
        if c not in df.columns:
            df[c] = "__MISSING__"
        df[c] = df[c].astype(str)

    # drop extras and enforce order
    keep = num_cols + cat_cols
    df = df[keep]
    return df


def show_expected_schema(num_cols, cat_cols):
    with st.expander("Expected training schema (columns & order)"):
        st.markdown("**Numeric features**:")
        st.code("\n".join(num_cols) if num_cols else "(none)")
        st.markdown("**Categorical features**:")
        st.code("\n".join(cat_cols) if cat_cols else "(none)")

# ----------------------------- model loaders -----------------------------------------
@st.cache_resource(show_spinner=False)
def load_lgbm():
    import lightgbm as lgb
    if not LGBM_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {LGBM_MODEL_PATH}")
    booster = lgb.Booster(model_file=str(LGBM_MODEL_PATH))
    return booster


@st.cache_resource(show_spinner=False)
def load_mlp(input_dim, n_classes):
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, d_in, d_hidden=512, drop=0.2, n_out=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(d_hidden, d_hidden//2), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(d_hidden//2, n_out)
            )
        def forward(self, x): return self.net(x)

    if not MLP_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MLP_MODEL_PATH}")

    device = torch.device("cpu")
    model = MLP(input_dim, n_out=n_classes).to(device)
    state = torch.load(MLP_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_ftt(n_num, n_classes):
    import torch
    import torch.nn as nn

    class FTTransformer(nn.Module):
        def __init__(self, n_num, d=16, heads=2, blocks=2, ff=4, drop=0.1, n_out=2):
            super().__init__()
            self.n_num = n_num
            self.cls   = nn.Parameter(torch.zeros(1,1,d))
            if n_num > 0:
                self.num_w = nn.Parameter(torch.randn(n_num, d) * 0.02)
                self.num_b = nn.Parameter(torch.zeros(n_num, d))
            else:
                self.register_parameter("num_w", None)
                self.register_parameter("num_b", None)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=heads, dim_feedforward=d*ff, dropout=drop,
                batch_first=True, activation="gelu", norm_first=False
            )
            self.enc  = nn.TransformerEncoder(layer, num_layers=blocks)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))
        def forward(self, x_num):
            B = x_num.size(0)
            toks = [self.cls.expand(B,1,-1)]
            if self.n_num > 0:
                toks.append(x_num.unsqueeze(-1) * self.num_w + self.num_b)  # [B, n_num, d]
            x = torch.cat(toks, dim=1)
            x = self.enc(x)
            return self.head(x[:, 0, :])

    if not FTT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {FTT_MODEL_PATH}")

    device = torch.device("cpu")
    model = FTTransformer(n_num=n_num, n_out=n_classes).to(device)
    state = torch.load(FTT_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# ----------------------------- inference wrappers ------------------------------------
def predict_lgbm(df_num: np.ndarray, classes):
    import lightgbm as lgb, joblib
    booster = load_lgbm()
    ohe_path = ARTIFACTS_DIR / "ohe.joblib"
    X = df_num
    if ohe_path.exists():
        # Rebuild a DataFrame with the schema columns so ColumnTransformer can work
        num_cols, cat_cols, _ = load_schema()
        # In the app we only have numeric matrix; if cat_cols existed at train time,
        # you should pass a matched DataFrame here instead (or store OHE’d matrix).
        # For pure-numeric training (cat_cols == []), this block is skipped anyway.
        pass
    proba = booster.predict(X)
    yhat = proba.argmax(1)
    labels = [classes[i] for i in yhat]
    conf   = proba.max(1)
    return labels, conf, proba



def predict_mlp(df_num: np.ndarray, classes):
    import torch
    model = load_mlp(input_dim=df_num.shape[1], n_classes=len(classes))
    with torch.no_grad():
        logits = model(torch.tensor(df_num, dtype=torch.float32))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    yhat = proba.argmax(1)
    labels = [classes[i] for i in yhat]
    conf   = proba.max(1)
    return labels, conf, proba


def predict_ftt(df_num: np.ndarray, classes):
    import torch
    model = load_ftt(n_num=df_num.shape[1], n_classes=len(classes))
    with torch.no_grad():
        x_num = torch.tensor(df_num, dtype=torch.float32)
        logits = model(x_num)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    yhat = proba.argmax(1)
    labels = [classes[i] for i in yhat]
    conf   = proba.max(1)
    return labels, conf, proba

# ----------------------------- UI ----------------------------------------------------
st.set_page_config(page_title="IoT IDS — Streamlit", layout="wide")
st.title("IoT Intrusion Detection — Inference App")

model_choice = st.sidebar.selectbox("Choose model", ["LightGBM", "MLP (PyTorch)", "FT-Transformer (PyTorch)"])
st.sidebar.markdown("---")

num_cols, cat_cols, classes = load_schema()
if classes is None:
    st.error("classes.json not found. Please run preprocessing/training first to save class order.")
    st.stop()

show_expected_schema(num_cols, cat_cols)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV with the training schema. You can also upload **ftt_test.csv** from your pipeline.")
    st.stop()

raw = pd.read_csv(uploaded)
st.write("Uploaded shape:", raw.shape)

ok, missing, extras = validate_frame_columns(raw, num_cols, cat_cols)
if not ok:
    st.error(
        "Your CSV is missing required training columns:\n\n"
        + ", ".join(missing)
        + "\n\nPlease add them (names must match exactly)."
    )
    st.stop()

if extras:
    st.info("The following columns are not used and will be ignored:\n\n" + ", ".join(extras))

# Detect if user uploaded preprocessed ftt_* split (already standardized)
default_pre = ("ftt_" in uploaded.name.lower())
is_preprocessed = st.checkbox("Uploaded data is already preprocessed (ftt_* from pipeline)", value=default_pre)

# Keep ground-truth if present (for metrics)
y_true = None
if LABEL_COL in raw.columns:
    y_true = raw[LABEL_COL].astype(str)
    raw = raw.drop(columns=[LABEL_COL])

# Align to schema & order
df = align_frame(raw, num_cols, cat_cols, drop_label=False)

# Build numeric inputs
if is_preprocessed:
    # ftt_* are already standardized — DO NOT scale again
    X_num = df[num_cols].to_numpy(np.float32)
else:
    bundle = joblib.load(SCALER_PATH)  # {'num_cols','cat_cols','scaler'}
    scaler = bundle["scaler"]
    X_num = scaler.transform(df[num_cols]).astype(np.float32)

# Guard: no NaN/Inf
if not np.isfinite(X_num).all():
    st.error("Your data contains non-finite values (NaN/Inf) after processing. Please clean the CSV and try again.")
    st.stop()

# ----------------------------- run inference ----------------------------------------
if model_choice == "LightGBM":
    # If your LGBM was trained on these same numeric features (selected-features path), OHE is not needed.
    labels, conf, proba = predict_lgbm(X_num, classes)

elif model_choice == "MLP (PyTorch)":
    labels, conf, proba = predict_mlp(X_num, classes)

else:  # FT-Transformer (PyTorch)
    labels, conf, proba = predict_ftt(X_num, classes)

preds_df = pd.DataFrame({
    "pred_label": labels,
    "pred_confidence": conf
})
st.subheader("Predictions")
st.dataframe(preds_df.head(50))

# Collapse warning (common sign of schema mismatch or double-scaling)
if preds_df["pred_label"].nunique() == 1:
    st.warning(
        f"All predictions are '{preds_df['pred_label'].iloc[0]}'. "
        "This often means inputs are out-of-distribution (wrong scaling/order). "
        "If you uploaded ftt_* splits, ensure the 'preprocessed' toggle is ON."
    )

# Optional metrics if ground truth exists
if y_true is not None:
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    y_true_idx = pd.Categorical(y_true, categories=classes).codes
    y_pred_idx = pd.Categorical(preds_df["pred_label"], categories=classes).codes

    acc = accuracy_score(y_true_idx, y_pred_idx)
    mf1 = f1_score(y_true_idx, y_pred_idx, average="macro")

    st.subheader("Metrics (using provided Label)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{acc:.4f}")
    with col2:
        st.metric("Macro-F1", f"{mf1:.4f}")

    st.text("Classification report:")
    st.code(classification_report(y_true_idx, y_pred_idx, target_names=classes, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(len(classes)))
    import matplotlib.pyplot as plt
    import io
    fig, ax = plt.subplots(figsize=(7,6), dpi=120)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig)

st.success("Done.")
