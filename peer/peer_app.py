#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
peer/peer_app.py — Peer GUI (A or B)

- Generates local NHS-like data (non-IID)
- Optional fictional LLM-style clinical notes (Scheme A)
- Trains local model for 1 epoch (LogReg or MLP, chosen by coordinator)
- Applies SMPC-style mask (+ for A, - for B), encrypts masked delta (Paillier)
- Submits encrypted update to coordinator
- Sends optional pitch metadata:
    * 2 note samples (fictional, truncated)
    * site_stats: aggregate means/rates (non-sensitive)

Environment:
  PEER_ID=A  (or B)
"""
import os, time, hashlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import httpx

from common.data_gen import generate_peer_data, attach_clinical_notes, standardize, FEATURES
from common.fl_model import predict_proba_model, local_train_one_epoch_model
from common.crypto import mask_id as make_mask_id, apply_mask, encrypt_vector

st.set_page_config(page_title="Peer — Client", layout="wide")

def safe_pyplot(fig):
    st.pyplot(fig)
    try:
        plt.close(fig)
    except Exception:
        pass

def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def df_show(df: pd.DataFrame, height: int = 260):
    try:
        st.dataframe(df, width="stretch", height=height)
    except TypeError:
        st.dataframe(df, use_container_width=True, height=height)

PEER_ID = os.environ.get("PEER_ID", "A").strip().upper()
if PEER_ID not in ("A","B"):
    PEER_ID = "A"
SIGN = +1 if PEER_ID == "A" else -1

st.title(f"🏥 Peer {PEER_ID} — Local Data + Encrypted FL Client")
st.caption("Raw data stays local. Only encrypted, masked model updates are sent.")

with st.sidebar:
    base = st.text_input("Coordinator URL", value="http://127.0.0.1:8000")

    st.markdown("---")
    st.subheader("Local data")
    n = st.number_input("Patients (N)", value=1500, step=100)
    seed = st.number_input("Seed", value=42, step=1)

    st.markdown("---")
    st.subheader("LLM-style notes (text only)")
    use_notes = st.checkbox("Generate clinical notes", value=True)
    use_ollama = st.checkbox("Use local Ollama (Llama) for a few notes", value=False)
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    ollama_model = st.text_input("Ollama model", value="llama3.1")
    note_k = st.slider("LLM note count (speed tradeoff)", 0, 20, 8, step=1)

    st.markdown("---")
    st.subheader("Local training")
    lr = st.number_input("LR", value=0.05, step=0.01, format="%.3f")
    batch = st.number_input("Batch size", value=256, step=64)
    l2 = st.number_input("L2", value=0.001, step=0.001, format="%.3f")
    grad_clip = st.number_input("Grad clip", value=5.0, step=1.0, format="%.1f")

    st.markdown("---")
    st.subheader("Security")
    secret = st.text_input("Shared secret (A/B)", value="shared-demo-secret")
    show_cipher_sample = st.checkbox("Show ciphertext sample (demo)", value=False)
    attack_wrong_secret = st.checkbox("⚠️ Attack demo: wrong secret (next submit)", value=False)
    attack_wrong_sign = st.checkbox("⚠️ Attack demo: wrong sign (next submit)", value=False)

    st.markdown("---")
    st.subheader("Actions")
    btn_gen = st.button("Generate / Reset local data", type="primary")
    btn_submit = st.button("Train + Encrypt + Submit (1 round)")
    auto_submit = st.checkbox("Auto submit as rounds advance", value=False)
    auto_sleep = st.slider("Auto loop seconds", 1, 10, 2)

for k, v in [("ready", False), ("last_round", -1), ("last_message", None), ("last_response", None), ("status_msg", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

def init_data():
    df = generate_peer_data(PEER_ID, n=int(n), seed=int(seed))
    if use_notes:
        df = attach_clinical_notes(
            df, site=PEER_ID,
            use_ollama=bool(use_ollama),
            ollama_url=str(ollama_url),
            ollama_model=str(ollama_model),
            llm_note_count=int(note_k),
            seed=int(seed)
        )
    X, y, _, _ = standardize(df)
    st.session_state.df = df
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.ready = True
    st.session_state.last_round = -1
    st.session_state.last_message = None
    st.session_state.last_response = None
    if use_notes and df.attrs.get("ollama_replaced", 0) > 0:
        st.session_state.status_msg = f"Data+notes ready. Ollama replaced {df.attrs.get('ollama_replaced')} notes."
    elif use_notes:
        st.session_state.status_msg = "Data+notes ready (template notes used)."
    else:
        st.session_state.status_msg = "Data ready (no notes)."

if btn_gen or (not st.session_state.ready):
    init_data()

df: pd.DataFrame = st.session_state.df
X: np.ndarray = st.session_state.X
y: np.ndarray = st.session_state.y

try:
    means = df[FEATURES].mean().to_numpy()
    data_sig = hashlib.sha256(str(list(means.round(3))).encode("utf-8")).hexdigest()[:12]
except Exception:
    data_sig = "na"

coord = {}
try:
    with httpx.Client(timeout=5.0) as http:
        coord = http.get(f"{base}/status").json()
except Exception:
    coord = {}
model_type = str(coord.get("model_type","logreg")).lower()
hidden_dim = int(coord.get("hidden_dim", 0) or 0)

top = st.columns(7)
top[0].metric("Coordinator round", str(coord.get("current_round","?")))
top[1].metric("Received", f"{coord.get('received','?')}/{coord.get('required',2)}")
top[2].metric("HE bits", str(coord.get("he_bits","?")))
top[3].metric("Model", model_type.upper())
top[4].metric("Hidden", str(hidden_dim) if model_type=="mlp" else "—")
top[5].metric("SMPC sign", str(SIGN))
top[6].metric("data_sig", data_sig)

if st.session_state.status_msg:
    st.info(st.session_state.status_msg)

c1, c2 = st.columns([1,1])
with c1:
    st.subheader("Local dataset snapshot")
    df_show(df.head(8), height=240)
with c2:
    st.subheader("Local feature distribution")
    feat = st.selectbox("Feature", FEATURES, index=0)
    fig, ax = plt.subplots()
    ax.hist(df[feat], bins=30, alpha=0.85)
    ax.set_title(f"Peer {PEER_ID}: {feat}")
    safe_pyplot(fig)

if use_notes and "clinical_note" in df.columns:
    st.markdown("---")
    st.subheader("📝 Example fictional NHS-style notes (pitch-friendly)")
    for i, t in enumerate(df["clinical_note"].head(5).tolist(), start=1):
        st.markdown(f"**Note {i}:** {t}")

st.markdown("---")
st.subheader("Local eval of GLOBAL model on local data")
try:
    with httpx.Client(timeout=10.0) as http:
        g = http.get(f"{base}/global").json()
    params = np.array(g["params"], dtype=np.float32)
    prob = predict_proba_model(params, X, model_type, hidden=hidden_dim if hidden_dim>0 else 16)
    pred = (prob >= 0.5).astype(int)
    acc_local = float((pred == y).mean())
    st.write({"local_eval_acc": round(acc_local, 4), "model_type": model_type, "hidden_dim": hidden_dim})
except Exception as e:
    st.warning(f"Global model not available yet: {e}")

st.markdown("---")
st.subheader("📤 Train + Encrypt + Submit")

def do_submit_once():
    with httpx.Client(timeout=120.0) as http:
        s = http.get(f"{base}/status").json()
        if not s.get("initialised", False):
            st.session_state.status_msg = "Coordinator not initialised. Open Dashboard and click Init."
            return

        g = http.get(f"{base}/global").json()
        pk = http.get(f"{base}/pubkey").json()
        round_idx = int(g["current_round"])
        mt = str(g.get("model_type","logreg")).lower()
        hd = int(g.get("hidden_dim", 0) or 0)

        if st.session_state.last_round == round_idx:
            st.session_state.status_msg = f"[{PEER_ID}] Already submitted round {round_idx}. Waiting…"
            return

        params0 = np.array(g["params"], dtype=np.float32)

        t0 = time.perf_counter()
        delta = local_train_one_epoch_model(
            params0, X, y,
            model_type=mt,
            hidden=hd if hd>0 else 16,
            lr=float(lr),
            batch_size=int(batch),
            l2=float(l2),
            grad_clip=float(grad_clip),
            seed=int(seed) + round_idx
        )
        t_train = float(time.perf_counter() - t0)

        use_secret2 = secret + "-WRONG" if attack_wrong_secret else secret
        use_sign = (-SIGN) if attack_wrong_sign else SIGN
        mid = make_mask_id(use_secret2, round_idx)
        masked = apply_mask(delta, use_secret2, round_idx, use_sign)

        # encrypt masked delta
        from phe import paillier
        pub = paillier.PaillierPublicKey(int(pk["n"]))
        t1 = time.perf_counter()
        payload = encrypt_vector(pub, masked)
        t_enc = float(time.perf_counter() - t1)

        bytes_payload = len(str(payload).encode("utf-8"))
        cipher_sample = payload[:3] if show_cipher_sample else "(hidden)"

        # Pitch metadata (non-sensitive)
        note_samples = [str(x)[:240] for x in df["clinical_note"].head(2).tolist()] if "clinical_note" in df.columns else []
        site_stats = {
            "age_mean": float(df["age"].mean()),
            "sys_bp_mean": float(df["sys_bp"].mean()),
            "hba1c_mean": float(df["hba1c"].mean()),
            "smoker_rate": float(df["smoker"].mean()),
        }

        preview = {
            "peer": PEER_ID,
            "round": round_idx,
            "mask_id": mid,
            "sign": use_sign,
            "model_type": mt,
            "hidden_dim": hd,
            "delta_norm": float(np.linalg.norm(delta)),
            "masked_norm": float(np.linalg.norm(masked)),
            "t_train": t_train,
            "sample_count": int(len(df)),
            "data_sig": data_sig,
            "ciphertext_sample": cipher_sample,
            "note_samples": note_samples,
            "site_stats": site_stats,
        }

        msg = {
            "peer_id": PEER_ID,
            "round": round_idx,
            "mask_id": mid,
            "sign": use_sign,
            "bytes_payload": bytes_payload,
            "t_train": t_train,
            "t_encrypt": t_enc,
            "payload_len": len(payload),
            "preview": preview,
        }

        r = http.post(f"{base}/submit", json={
            "peer_id": PEER_ID,
            "round": round_idx,
            "enc_masked_delta": payload,
            "bytes_payload": bytes_payload,
            "t_encrypt": t_enc,
            "mask_id": mid,
            "sign": use_sign,
            "preview": preview,
        })
        try:
            resp = r.json()
        except Exception:
            resp = {"ok": False, "error": r.text}

        st.session_state.last_message = msg
        st.session_state.last_response = resp

        if resp.get("ok"):
            st.session_state.last_round = round_idx
            st.session_state.status_msg = f"[{PEER_ID}] ✅ Submitted round {round_idx} ({bytes_payload/1024:.1f} KB)"
        else:
            st.session_state.status_msg = f"[{PEER_ID}] ❌ Submit failed: {resp.get('error','unknown')}"

if btn_submit:
    do_submit_once()
    rerun()

if auto_submit:
    do_submit_once()
    time.sleep(auto_sleep)
    rerun()


# st.markdown("---")
# st.subheader("Last packet / response (this session)")
# if st.session_state.get("last_message"):
#     st.json(st.session_state["last_message"])
#     st.json(st.session_state.get("last_response"))
# else:
#     st.info("No packet yet. Click submit or enable Auto submit.")
