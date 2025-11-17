import os, json, io, gc, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ---------- Paths ----------
ART = Path("artifacts")                # root for saved models
FTT_DIR = ART / "ftt_pure"             # FT-Transformer subdir
CLASSES = json.loads((ART/"classes.json").read_text())
CLS2ID = {c:i for i,c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

st.set_page_config(page_title="IoT IDS — LGBM / MLP / FT-Transformer", layout="wide")

st.title("🔐 IoT Intrusion Detection — LGBM / MLP / FT-Transformer")
st.caption("Batch inference + quick evaluation — uses your trained artifacts and mirrors training-time preprocessing.")

# ---------- Utilities ----------
@st.cache_resource
def load_common():
    # Shared scaler & column lists (from training)
    ftt_scaler_bundle = joblib.load(FTT_DIR/"ftt_scaler.joblib")
    num_cols = ftt_scaler_bundle.get("num_cols", []) or []
    cat_cols = ftt_scaler_bundle.get("cat_cols", []) or []

    # For LGBM & MLP OHE flow
    ohe = joblib.load(ART/"ohe.joblib")

    return ftt_scaler_bundle, num_cols, cat_cols, ohe

@st.cache_resource
def load_lgbm():
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(ART/"lgbm_model.txt"))
    return booster

@st.cache_resource
def load_mlp(input_dim=None):
    import torch, torch.nn as nn
    meta = json.loads((ART/"mlp_meta.json").read_text())
    if input_dim is None:
        input_dim = meta["input_dim"]
    class MLP(nn.Module):
        def __init__(self, d_in, d_hidden=512, drop=0.2, n_out=N_CLASSES):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(d_hidden, d_hidden//2), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(d_hidden//2, n_out)
            )
        def forward(self, x): return self.net(x)
    device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    model = MLP(input_dim, n_out=N_CLASSES).to(device)
    model.load_state_dict(torch.load(ART/"mlp.pt", map_location=device))
    model.eval()
    return model, device

@st.cache_resource
def load_ftt():
    import torch, torch.nn as nn
    meta = json.loads((FTT_DIR/"meta.json").read_text())
    CAT = meta["cat_cols"]; NUM = meta["num_cols"]
    CARD = meta["cat_cardinalities"]
    hp = meta["model_hparams"]

    class FTT(nn.Module):
        def __init__(self,n_num,card,d=hp["d_token"],heads=hp["n_heads"],blocks=hp["n_blocks"],
                     ff=hp["ff_mult"],drop=hp["dropout"], n_out=N_CLASSES):
            super().__init__()
            self.n_num=n_num; self.cls=nn.Parameter(torch.zeros(1,1,d))
            if n_num>0:
                self.num_w=nn.Parameter(torch.randn(n_num,d)*0.02); self.num_b=nn.Parameter(torch.zeros(n_num,d))
            else:
                self.register_parameter("num_w",None); self.register_parameter("num_b",None)
            self.emb=nn.ModuleList([nn.Embedding(c,d) for c in card])
            layer=nn.TransformerEncoderLayer(d_model=d,nhead=heads,dim_feedforward=d*ff,
                                             dropout=drop,batch_first=True,activation="gelu",norm_first=True)
            self.enc=nn.TransformerEncoder(layer,num_layers=blocks)
            self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,n_out))
        def forward(self,xn,xc):
            B=xn.size(0); toks=[self.cls.expand(B,1,-1)]
            if self.n_num>0: toks.append(xn.unsqueeze(-1)*self.num_w + self.num_b)
            for i,e in enumerate(self.emb): toks.append(e(xc[:,i]).unsqueeze(1))
            x=torch.cat(toks,dim=1); x=self.enc(x); return self.head(x[:,0,:])

    device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    model = FTT(len(NUM), CARD).to(device)
    model.load_state_dict(torch.load(FTT_DIR/"model.pt", map_location=device))
    model.eval()
    return model, device, CAT, NUM, hp

def clean_like_training(df, label_col="Label"):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if label_col in df.columns:
        # inference: we ignore provided Label unless user wants eval
        pass
    df = df.replace([np.inf, -np.inf], np.nan)
    # Fill NA similarly
    for c in df.columns:
        if c == label_col: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].astype(str).fillna("Unknown")
    return df

def apply_saved_scaler(df, ftt_scaler_bundle, num_cols):
    # Your training scaled numerics; reuse saved scaler/mean,std
    if ftt_scaler_bundle.get("scaler") is not None:
        scaler = ftt_scaler_bundle["scaler"]
        if all(c in df.columns for c in num_cols) and len(num_cols):
            df[num_cols] = scaler.transform(df[num_cols]).astype("float32")
    else:
        # torch-computed mean/std saved as arrays
        mean_ = ftt_scaler_bundle.get("mean_")
        scale_= ftt_scaler_bundle.get("scale_")
        if mean_ is not None and scale_ is not None and len(num_cols):
            X = df[num_cols].to_numpy(dtype=np.float32)
            X = (X - np.asarray(mean_, dtype=np.float32)) / np.clip(np.asarray(scale_, dtype=np.float32), 1e-8, None)
            df[num_cols] = X.astype("float32")
    return df

def to_ftt_tensors(df, cat_cols, num_cols, cat_maps):
    # Build per-col integer indices for categories as in training (unknown -> 0)
    mats=[]
    for c, m in zip(cat_cols, cat_maps):
        m_int = {k:int(v) for k,v in m.items()}
        idx = df[c].astype(str).map(m_int).fillna(0).astype(np.int64).to_numpy().reshape(-1,1)
        mats.append(idx)
    Xc = np.concatenate(mats, axis=1) if mats else np.zeros((len(df),0), dtype=np.int64)
    Xn = df[num_cols].to_numpy(dtype=np.float32) if num_cols else np.zeros((len(df),0), dtype=np.float32)
    return Xn, Xc

def ohe_features(df, ohe, cat_cols, num_cols):
    # Keep only seen columns in the expected order
    want = list(ohe.transformers_[0][2]) + list(ohe.transformers_[1][2])
    # Fill missing expected columns if any (with 0 or "Unknown")
    for c in cat_cols:
        if c not in df.columns: df[c] = "Unknown"
        df[c] = df[c].astype(str)
    for c in num_cols:
        if c not in df.columns: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    X = ohe.transform(df[want])
    return X

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Inference Settings")
model_choice = st.sidebar.selectbox("Choose model", ["LightGBM", "MLP", "FT-Transformer"])
show_examples = st.sidebar.checkbox("Show training-time summary plots", value=True)
threshold_note = st.sidebar.caption("Multiclass outputs use argmax; confidence shown from predicted probability.")

# ---------- Load artifacts ----------
ftt_scaler_bundle, NUM_COLS, CAT_COLS, OHE = load_common()

# Model-specific lazy loads
if model_choice == "LightGBM":
    booster = load_lgbm()
elif model_choice == "MLP":
    MLP_MODEL, MLP_DEVICE = load_mlp()
else:
    FTT_MODEL, FTT_DEVICE, FTT_CAT, FTT_NUM, _hp = load_ftt()
    # cat maps live in meta.json
    META = json.loads((FTT_DIR/"meta.json").read_text())
    CAT_MAPS = META["cat_maps"]

# ---------- Inputs ----------
st.subheader("📥 Input data")
up = st.file_uploader("Upload CSV with the same schema used for training (no Label needed for prediction).", type=["csv"])

if up is not None:
    df = pd.read_csv(up)
else:
    st.info("No CSV uploaded yet. You can still see model summaries below.")
    df = None

# ---------- Inference ----------
def run_inference(df_in: pd.DataFrame):
    df_clean = clean_like_training(df_in, label_col="Label")
    # Apply saved scaling to numerics (same as training)
    df_scaled = apply_saved_scaler(df_clean, ftt_scaler_bundle, NUM_COLS)

    if model_choice == "LightGBM":
        # OHE + passthrough numerics (already scaled)
        X = ohe_features(df_scaled, OHE, CAT_COLS, NUM_COLS)
        proba = booster.predict(X)
        yhat = proba.argmax(1)
        conf = proba.max(1)

    elif model_choice == "MLP":
        import torch
        X = ohe_features(df_scaled, OHE, CAT_COLS, NUM_COLS)
        with torch.no_grad():
            logits = MLP_MODEL(torch.tensor(X, dtype=torch.float32, device=MLP_DEVICE))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            yhat = proba.argmax(1)
            conf = proba.max(1)

    else:  # FT-Transformer
        import torch
        # Ensure expected cols exist
        for c in FTT_CAT:
            if c not in df_scaled.columns: df_scaled[c] = "Unknown"
            df_scaled[c] = df_scaled[c].astype(str)
        for c in FTT_NUM:
            if c not in df_scaled.columns: df_scaled[c] = 0.0
            df_scaled[c] = pd.to_numeric(df_scaled[c], errors="coerce").fillna(0.0)

        Xn, Xc = to_ftt_tensors(df_scaled, FTT_CAT, FTT_NUM, CAT_MAPS)
        with torch.no_grad():
            xn = torch.tensor(Xn, dtype=torch.float32, device=FTT_DEVICE)
            xc = torch.tensor(Xc, dtype=torch.long, device=FTT_DEVICE)
            logits = FTT_MODEL(xn, xc)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            yhat = proba.argmax(1)
            conf = proba.max(1)

    pred_labels = [CLASSES[i] for i in yhat]
    out = df_in.copy()
    out["pred_label"] = pred_labels
    out["pred_confidence"] = conf
    return out, proba

# ---------- Run & Display ----------
if df is not None and len(df):
    with st.spinner("Running inference…"):
        preds, proba = run_inference(df)
    st.success(f"Done. Predicted {len(preds)} rows.")

    st.subheader("🔎 Predictions")
    st.dataframe(preds.head(50))
    # Download CSV
    buf = io.BytesIO()
    preds.to_csv(buf, index=False)
    st.download_button("⬇️ Download predictions CSV", data=buf.getvalue(),
                       file_name=f"predictions_{model_choice.lower().replace(' ','_')}.csv",
                       mime="text/csv")

# ---------- Metrics & Plots (from training) ----------
st.subheader("📊 Training-time Metrics & Plots")
col1, col2, col3 = st.columns(3)
def show_img(p: Path, label: str):
    if p.exists():
        st.image(str(p), caption=label, use_container_width=True)
    else:
        st.caption(f"({label} not found)")

with col1:
    show_img(ART/"perf_accuracy_by_model.png", "Accuracy by Model")
with col2:
    show_img(ART/"perf_macrof1_by_model.png", "Macro-F1 by Model")
with col3:
    show_img(ART/"perf_perclass_f1_grouped.png", "Per-class F1")

c1, c2, c3 = st.columns(3)
with c1:
    show_img(ART/"lgbm_confusion_matrix.png", "LightGBM — Confusion Matrix")
with c2:
    show_img(FTT_DIR/"ftt_confusion_matrix.png", "FT-Transformer — Confusion Matrix")
with c3:
    show_img(ART/"mlp_confusion_matrix.png", "MLP — Confusion Matrix")

st.divider()
st.caption("Tip: the app mirrors the training pipeline: clean → scale numerics (saved scaler) → model-specific encoding (OHE or categorical indices).")
