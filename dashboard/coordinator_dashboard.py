#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dashboard/coordinator_dashboard.py — Pitch dashboard (Coordinator view)

Shows:
- Model chooser (LogReg / MLP) on init/reset
- Public Mode (non-technical wording)
- Learning card: update dimension, per-peer delta norms, combined avg_delta_norm
- A/B note compare: fictional note samples + site snapshot (aggregates)
- Training progress chart (acc/auc)
- Encrypted packets + protocol timeline
- Animated swimlane (message-flow) to show FL + HE + SMPC story

Tip:
- Turn off auto refresh while playing swimlane animation.
"""
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import httpx
import os
from common.data_gen import FEATURES, generate_peer_data, make_peer_dataset, standardize
from common.fl_model import init_params, local_train_one_epoch_model, predict_proba_model

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

def df_show(df: pd.DataFrame, height: int = 320):
    try:
        st.dataframe(df, width="stretch", height=height)
    except TypeError:
        st.dataframe(df, use_container_width=True, height=height)

LANES = ["Peer A", "Peer B", "Coordinator"]
Y = {"Peer A": 2, "Peer B": 1, "Coordinator": 0}
C_BG = "#9aa0a6"; C_OK = "#34a853"; C_BAD = "#ea4335"; C_INFO = "#4285f4"; C_WARN = "#fbbc05"

PUBLIC_LABELS = {
    "recv_ciphertext": "Secure message received (encrypted update)",
    "mask_check": "Safety check (mask pairing)",
    "he_add": "Secure combine (encrypted)",
    "decrypt_sum": "Unlock combined update",
    "fedavg_update_applied": "Global model updated (Federated Learning)",
    "model_update": "Model quality updated",
    "round_complete": "Round complete",
    "round_blocked": "Blocked (mismatch detected)",
    "init": "Demo initialised",
    "round_start": "Round started",
    "demo_complete": "Demo finished"
}

def lane_of_event(row: dict) -> str:
    ev = row.get("event","")
    if ev == "recv_ciphertext":
        pid = row.get("peer_id","")
        return "Peer A" if pid == "A" else ("Peer B" if pid == "B" else "Coordinator")
    return "Coordinator"

def event_color(ev_name: str, row: dict) -> str:
    if ev_name == "round_blocked":
        return C_BAD
    if ev_name == "mask_check":
        ok = row.get("ok", None)
        if ok is True: return C_OK
        if ok is False: return C_BAD
        return C_WARN
    if ev_name in ("recv_ciphertext","he_add","decrypt_sum","fedavg_update_applied","model_update","round_complete","demo_complete","round_start","init"):
        return C_INFO
    return C_BG

def pretty_event_name(ev: str, public_mode: bool) -> str:
    if public_mode and ev in PUBLIC_LABELS:
        return PUBLIC_LABELS[ev]
    return ev

def draw_swimlane(ev_all: pd.DataFrame, upto: int, round_filter: int | None, highlight_idx: int | None, blink_on: bool, public_mode: bool):
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.set_title("Animated Swimlane — 'Data stays local, only encrypted learning updates move'" if public_mode
                 else "Animated Swimlane — HE transport + SMPC cancel + FedAvg")

    if round_filter is not None and "round" in ev_all.columns:
        ev = ev_all[ev_all["round"] == round_filter].copy()
    else:
        ev = ev_all.copy()
    ev = ev.reset_index(drop=True)
    upto = min(upto, len(ev))

    ax.hlines([Y[l] for l in LANES], xmin=0, xmax=max(1, len(ev)), linewidth=2, color="#d0d0d0")
    for l in LANES:
        ax.text(-0.3, Y[l], l, va="center", ha="right", fontsize=11)

    for i in range(upto):
        row = ev.iloc[i].to_dict()
        e = row.get("event","")
        lane = lane_of_event(row)
        x = i + 1
        col = event_color(e, row)
        alpha = 0.35
        lw = 1.5

        if highlight_idx is not None and i == highlight_idx:
            alpha = 1.0 if blink_on else 0.25
            lw = 3.0

        if e == "recv_ciphertext":
            src = lane; dst = "Coordinator"
            ax.annotate("", xy=(x, Y[dst]), xytext=(x, Y[src]),
                        arrowprops=dict(arrowstyle="->", lw=lw, color=col, alpha=alpha))
            label = "encrypted update" if public_mode else f"ciphertext\nmask={row.get('mask_id','')}\nsign={row.get('sign','')}"
            ax.text(x, (Y[src]+Y[dst])/2 + 0.08, label, ha="center", va="bottom", fontsize=8, alpha=alpha)
        else:
            y = Y[lane]
            if e in ("round_complete","demo_complete"):
                ax.plot([x],[y], marker="*", markersize=16, color=col, alpha=alpha)
            elif e == "round_blocked":
                ax.plot([x],[y], marker="X", markersize=12, color=col, alpha=alpha)
            else:
                ax.plot([x],[y], marker="o", markersize=9, color=col, alpha=alpha)

            label = pretty_event_name(e, public_mode)
            if not public_mode and e == "fedavg_update_applied":
                label = f"FedAvg applied\n||Δ||={float(row.get('avg_delta_norm',0.0)):.3f}\n||w||={float(row.get('w_norm',0.0)):.3f}"
            elif public_mode and e == "fedavg_update_applied":
                label = "Global model updated\n(Federated Learning)"
            ax.text(x, y + 0.20, label, ha="center", va="bottom", fontsize=8, alpha=alpha)

    ax.set_xlim(-0.5, max(1.5, len(ev)+0.5))
    ax.set_ylim(-0.7, 2.7)
    ax.set_yticks([])
    ax.set_xticks(range(1, max(2, len(ev)+1)))
    ax.set_xlabel("Event order")
    ax.grid(True, axis="x", alpha=0.12)
    return fig

# st.set_page_config(page_title="MediVault Dashboard", layout="wide")
# st.title("MediVault Dashboard")

# --- page config ---
BASE_DIR = os.path.dirname(__file__)
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

st.set_page_config(page_title="MediVault Dashboard", layout="wide")

# --- header row (logo + title) ---
c1, c2 = st.columns([1, 8], vertical_alignment="center")
with c1:
    st.image(LOGO_PATH, width=200)   
with c2:
    st.title("Dashboard")

with st.sidebar:
    base = st.text_input("Coordinator URL", value="http://127.0.0.1:8000")
    refresh = st.slider("Refresh seconds", 1, 15, 6)
    auto = st.checkbox("Auto refresh", value=False)
    st.markdown("---")
    public_mode = st.toggle("Public Mode", value=True)

    st.markdown("---")
    st.subheader("Init / Reset")
    rounds_total = st.number_input("rounds_total", value=5, step=1)
    he_bits = st.selectbox("Paillier bits", [1024, 2048, 3072, 4096], index=1)
    strict = st.checkbox("Strict mask check", value=True)

    model_ui = st.selectbox("Model", ["LogReg (fast)", "MLP (1 hidden layer)"], index=0)
    model_type = "logreg" if model_ui.startswith("LogReg") else "mlp"
    hidden_dim = 16
    if model_type == "mlp":
        hidden_dim = st.slider("MLP hidden dim", 8, 64, 16, step=8)

    if st.button("Init Coordinator", type="primary"):
        with httpx.Client(timeout=30.0) as http:
            r = http.post(f"{base}/reset", json={
                "rounds_total": int(rounds_total),
                "he_bits": int(he_bits),
                "strict_mask_check": bool(strict),
                "model_type": model_type,
                "hidden_dim": int(hidden_dim),
            })
            st.success("Initialised" if r.status_code == 200 else f"Init failed: {r.text}")

# Fetch
try:
    with httpx.Client(timeout=10.0) as http:
        status = http.get(f"{base}/status").json()
        events = http.get(f"{base}/events", params={"limit": 2400}).json()
        metrics = http.get(f"{base}/metrics").json()
        try:
            glob = http.get(f"{base}/global").json()
        except Exception:
            glob = None
except Exception as e:
    st.error(f"Cannot reach coordinator: {e}")
    st.stop()

ev = pd.DataFrame(events) if isinstance(events, list) else pd.DataFrame([])
md = pd.DataFrame(metrics) if isinstance(metrics, list) else pd.DataFrame([])

tabs = st.tabs(["Overview", "Details", "📊 FL vs Non‑FL", "🎞️ Animation"])

with tabs[0]:
    st.subheader("Main Task")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", str(status.get("model_type","—")).upper())
    col2.metric("Hidden", str(status.get("hidden_dim","—")) if str(status.get("model_type","")).lower()=="mlp" else "—")
    col3.metric("Task", status.get("task","Predict risk_event"))
    col4.metric("Features", str(len(FEATURES)))

    st.markdown("---")
    st.subheader("🧠 This round learned…")
    if ev.empty or "event" not in ev.columns:
        st.info("No events yet. Submit from both peers to complete a round.")
    else:
        fed = ev[ev["event"] == "fedavg_update_applied"].copy()
        if fed.empty:
            st.info("No FedAvg update yet. Complete round 1 first (submit A then B).")
        else:
            last = fed.iloc[-1].to_dict()
            r = int(last.get("round", 0))
            vec_len = int(last.get("vec_len", 0) or 0)
            dA = last.get("delta_norm_A", None)
            dB = last.get("delta_norm_B", None)
            dAvg = last.get("avg_delta_norm", None)

            a1,a2,a3,a4,a5 = st.columns(5)
            a1.metric("Round", str(r))
            a2.metric("Update dim", str(vec_len))
            a3.metric("Peer A change", f"{float(dA):.3f}" if dA is not None else "—")
            a4.metric("Peer B change", f"{float(dB):.3f}" if dB is not None else "—")
            a5.metric("Combined change", f"{float(dAvg):.3f}" if dAvg is not None else "—")

            if public_mode:
                st.write("Bigger numbers mean the model changed more this round. Each peer learned locally; the coordinator applied the combined update.")
            else:
                st.write(last)

    st.markdown("---")
    st.subheader("📝 A/B clinical note examples (fictional)")
    if ev.empty or "event" not in ev.columns:
        st.info("No note samples yet. Submit from peers.")
    else:
        pk = ev[ev["event"] == "recv_ciphertext"].copy()
        if pk.empty:
            st.info("No recv_ciphertext events yet.")
        else:
            def _last_for(peer: str):
                x = pk[pk.get("peer_id") == peer]
                if x.empty:
                    return None, None, None
                row = x.iloc[-1].to_dict()
                return row.get("note_sample"), row.get("site_stats"), int(row.get("round", 0))

            noteA, statsA, rA = _last_for("A")
            noteB, statsB, rB = _last_for("B")

            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Peer A (e.g., NHS Trust A)**")
                st.caption(f"Latest round: {rA}")
                st.write(noteA or "—")
            with cb:
                st.markdown("**Peer B (e.g., NHS Trust B)**")
                st.caption(f"Latest round: {rB}")
                st.write(noteB or "—")

            st.markdown("### Site snapshot (aggregated, non-sensitive)")
            rows = []
            def _add(peer, stt):
                if isinstance(stt, dict):
                    rows.append({
                        "Peer": peer,
                        "age_mean": stt.get("age_mean"),
                        "sys_bp_mean": stt.get("sys_bp_mean"),
                        "hba1c_mean": stt.get("hba1c_mean"),
                        "smoker_rate": stt.get("smoker_rate"),
                    })
            _add("A", statsA); _add("B", statsB)
            if rows:
                df = pd.DataFrame(rows)
                if "smoker_rate" in df.columns:
                    df["smoker_rate"] = df["smoker_rate"].apply(lambda x: f"{float(x)*100:.1f}%" if x is not None else "—")
                df_show(df, height=160)
            else:
                st.info("No site_stats received yet.")

    st.markdown("---")

    st.markdown("---")
    st.subheader("📡 What is transmitted over the network?")
    st.caption("Public view: we send encrypted, masked *model updates* (Δ) — not patient records. Technical view is optional.")
    if ev.empty or "event" not in ev.columns:
        st.info("No network packets yet. Submit once from Peer A and Peer B.")
    else:
        pk = ev[ev["event"] == "recv_ciphertext"].copy()
        if pk.empty:
            st.info("No recv_ciphertext events yet.")
        else:
            def _last_packet(peer: str):
                x = pk[pk.get("peer_id") == peer]
                if x.empty:
                    return None
                return x.iloc[-1].to_dict()

            pa = _last_packet("A")
            pb = _last_packet("B")

            cA, cB = st.columns(2)
            def _render_packet(col, p, title):
                with col:
                    st.markdown(f"**{title}**")
                    if not p:
                        st.write("—")
                        return
                    # Public summary
                    dim = int(p.get("cipher_len", 0) or 0)
                    size_kb = (float(p.get("bytes_payload", 0)) / 1024.0) if p.get("bytes_payload") is not None else 0.0
                    st.write({
                        "message_type": "Encrypted masked model update (Enc(Δ + sign·mask))",
                        "vector_dim": dim,
                        "size_kb": round(size_kb, 1),
                        "mask_id": p.get("mask_id"),
                        "sign": p.get("sign"),
                        "cipher_hash": p.get("cipher_hash"),
                    })
                    if public_mode:
                        st.caption("This is an encrypted update vector. On its own it reveals nothing about local data.")

                    # Technical expander
                    with st.expander("Technical view: ciphertext sample (first 1–2 elements)", expanded=False):
                        cs = p.get("cipher_sample", None)
                        if cs is None:
                            st.info("Ciphertext sample not provided (Peer 'Show ciphertext sample' is OFF).")
                        else:
                            st.json(cs)

            _render_packet(cA, pa, "Peer A → Coordinator")
            _render_packet(cB, pb, "Peer B → Coordinator")

            st.caption("Coordinator only decrypts the **sum** of the two encrypted masked updates. Masks cancel, then FedAvg is applied.")

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Current round", str(status.get("current_round","?")))
    k2.metric("Received", f"{status.get('received','?')}/{status.get('required',2)}")
    k3.metric("HE bits", str(status.get("he_bits","?")))
    k4.metric("Strict mask", "ON" if status.get("strict_mask_check") else "OFF")
    k5.metric("||w||", f"{status.get('params_norm', 0.0):.4f}" if status.get("params_norm") is not None else "—")

    if md is not None and not md.empty and "round" in md.columns:
        st.markdown("---")
        st.subheader("Training progress (global)")
        fig1, ax1 = plt.subplots()
        ax1.plot(md["round"], md["acc"], marker="o", label="Accuracy")
        if "auc" in md.columns:
            ax1.plot(md["round"], md["auc"], marker="o", label="AUC")
        ax1.set_xlabel("Round"); ax1.set_ylabel("Score")
        ax1.set_title("Global model quality across rounds")
        ax1.grid(True, alpha=0.3); ax1.legend()
        safe_pyplot(fig1)

with tabs[1]:
    st.subheader("Secure collaboration evidence")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Secure messages (encrypted updates)" if public_mode else "### Encrypted packets")
        if ev.empty or "event" not in ev.columns:
            st.info("No events yet. Submit from peers.")
        else:
            pk = ev[ev["event"] == "recv_ciphertext"].copy()
            if pk.empty:
                st.info("No recv_ciphertext events yet.")
            else:
                keep = [c for c in ["ts","round","peer_id","bytes_payload","t_encrypt","mask_id","sign","cipher_hash","delta_norm"] if c in pk.columns]
                pk2 = pk[keep].copy()
                if "ts" in pk2.columns:
                    pk2["ts"] = pd.to_datetime(pk2["ts"], unit="s", errors="coerce")
                df_show(pk2.tail(200), height=360)

    with c2:
        st.markdown("### Protocol timeline" if public_mode else "### Coordinator timeline")
        if ev.empty:
            st.info("No events yet.")
        else:
            ev2 = ev.copy()
            if "ts" in ev2.columns:
                ev2["ts"] = pd.to_datetime(ev2["ts"], unit="s", errors="coerce")
            if public_mode and "event" in ev2.columns:
                ev2["event_pretty"] = ev2["event"].apply(lambda x: pretty_event_name(str(x), True))
                show_cols = [c for c in ["ts","round","event_pretty","peer_id","bytes_payload","mask_id","ok","reason"] if c in ev2.columns]
                df_show(ev2[show_cols].tail(320), height=360)
            else:
                df_show(ev2.tail(320), height=360)

    st.markdown("---")
    st.subheader("Secure aggregated model (global)")
    if not glob or "params" not in glob:
        st.info("Init first to see global model.")
    else:
        params = np.array(glob["params"], dtype=float)
        st.write({"param_vector_length": int(len(params)), "model_type": glob.get("model_type"), "hidden_dim": glob.get("hidden_dim")})
        if str(glob.get("model_type","")).lower() == "logreg":
            w = params[:-1]
            order = np.argsort(np.abs(w))[::-1]
            topk = min(8, len(order))
            top = [{"feature": FEATURES[i], "weight": float(w[i])} for i in order[:topk]]
            df_show(pd.DataFrame(top), height=260)
        else:
            st.info("MLP selected: showing parameter norms (weights are high-dimensional).")
            st.write({"||params||": float(np.linalg.norm(params)), "min": float(np.min(params)), "max": float(np.max(params))})


with tabs[2]:
    st.subheader("📊 Federated Learning vs Centralised (NO-FL) ")
    st.caption("Compare accuracy using the same synthetic data, the same model, and the same training hyper‑parameters. For a fair comparison, set N and Seed here to match the values used in Peer A/B sidebars.")

    # Show current FL status
    mtype = str(status.get("model_type", "logreg")).lower()
    hdim = int(status.get("hidden_dim", 0) or 0)
    rounds_cfg = int(status.get("rounds_total", 5) or 5)

    left, right = st.columns([1.05, 1])
    with right:
        st.markdown("### Current FL run")
        st.write({
            "model_type": mtype,
            "hidden_dim": hdim if mtype == "mlp" else "—",
            "rounds_total": rounds_cfg,
            "fl_rounds_completed": int(md["round"].max()) if (md is not None and (not md.empty) and ("round" in md.columns)) else 0,
        })
        if md is None or md.empty:
            st.info("No FL metrics yet. Run at least one full round (submit A then B) to see the FL accuracy curve.")

    with left:
        st.markdown("### NO-FL Set Up")
        n_patients = st.number_input("Patients per peer (N)", value=1500, step=100, key="cmp_n")
        seed = st.number_input("Seed (match peers)", value=42, step=1, key="cmp_seed")

        st.markdown("**Training hyper‑parameters** (match peers)")
        lr_cmp = st.number_input("LR", value=0.05, step=0.01, format="%.3f", key="cmp_lr")
        batch_cmp = st.number_input("Batch size", value=256, step=64, key="cmp_batch")
        l2_cmp = st.number_input("L2", value=0.001, step=0.001, format="%.3f", key="cmp_l2")
        gc_cmp = st.number_input("Grad clip", value=5.0, step=1.0, format="%.1f", key="cmp_gc")
        per_peer_std = st.checkbox("Standardise per peer (recommended)", value=True, key="cmp_std")

        run_baseline = st.button("Run No-FL training", type="primary", key="cmp_run")

    def _build_val(n: int = 1600, seed_val: int = 2026):
        df_val = make_peer_dataset("VAL", n=int(n), seed=int(seed_val))
        Xv, yv, _, _ = standardize(df_val)
        return Xv, yv

    def _train_baseline():
        # Generate the same synthetic data as peers would (if they use the same N/seed)
        dfA = generate_peer_data("A", n=int(n_patients), seed=int(seed))
        dfB = generate_peer_data("B", n=int(n_patients), seed=int(seed))

        if per_peer_std:
            XA, yA, _, _ = standardize(dfA)
            XB, yB, _, _ = standardize(dfB)
            Xtr = np.vstack([XA, XB]).astype(np.float32)
            ytr = np.concatenate([yA, yB]).astype(np.int64)
        else:
            df_all = pd.concat([dfA, dfB], ignore_index=True)
            Xtr, ytr, _, _ = standardize(df_all)

        Xv, yv = _build_val(n=1600, seed_val=2026)

        # Match coordinator init (seed=123)
        d = len(FEATURES)
        params = init_params(mtype, d, hidden=int(hdim) if mtype == "mlp" else 16, seed=123).astype(np.float32)

        rows = []
        for epoch in range(1, int(rounds_cfg) + 1):
            delta = local_train_one_epoch_model(
                params, Xtr, ytr,
                model_type=mtype,
                hidden=int(hdim) if (mtype == "mlp" and int(hdim) > 0) else 16,
                lr=float(lr_cmp),
                batch_size=int(batch_cmp),
                l2=float(l2_cmp),
                grad_clip=float(gc_cmp),
                seed=int(seed) + int(epoch)
            )
            params = (params + delta).astype(np.float32)

            prob = predict_proba_model(params, Xv, mtype, hidden=int(hdim) if (mtype == "mlp" and int(hdim) > 0) else 16)
            pred = (prob >= 0.5).astype(np.int64)
            acc = float((pred == yv).mean())
            rows.append({"round": int(epoch), "baseline_acc": acc})

        return pd.DataFrame(rows)

    if run_baseline:
        with st.spinner("Running baseline training…"):
            try:
                base_df = _train_baseline()
                st.session_state["baseline_curve"] = base_df
                st.success("Baseline finished.")
            except Exception as e:
                st.error(f"Baseline failed: {e}")
                base_df = None
    else:
        base_df = st.session_state.get("baseline_curve", None)

    st.markdown("---")
    st.markdown("### Accuracy comparison")

    # Prepare curves
    fl_curve = None
    if md is not None and (not md.empty) and ("round" in md.columns) and ("acc" in md.columns):
        fl_curve = md[["round", "acc"]].copy().rename(columns={"acc": "fl_acc"})

    if base_df is None:
        st.info("Click **Run NO-FL training** to compute the Centralised (No‑FL) curve.")
    else:
        cmp = base_df.copy()
        if fl_curve is not None:
            cmp = cmp.merge(fl_curve, on="round", how="left")

        # Plot
        fig, ax = plt.subplots()
        ax.plot(cmp["round"], cmp["baseline_acc"], marker="o", label="Centralised (No‑FL)")
        if "fl_acc" in cmp.columns:
            ax.plot(cmp["round"], cmp["fl_acc"], marker="o", label="Federated Learning")
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy: FL vs Centralised (same model + hyper‑params)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        safe_pyplot(fig)

        # Summary
        last_base = float(cmp["baseline_acc"].iloc[-1]) if ("baseline_acc" in cmp.columns and len(cmp)) else None
        last_fl = None
        if "fl_acc" in cmp.columns and pd.notna(cmp["fl_acc"].iloc[-1]):
            last_fl = float(cmp["fl_acc"].iloc[-1])

        s1, s2, s3 = st.columns(3)
        s1.metric("Centralised final acc", f"{last_base:.4f}" if last_base is not None else "—")
        s2.metric("FL final acc", f"{last_fl:.4f}" if last_fl is not None else "—")
        if (last_fl is not None) and (last_base is not None):
            s3.metric("FL − Centralised", f"{(last_fl - last_base):+.4f}")
        else:
            s3.metric("FL − Centralised", "—")

        st.markdown("#### Per‑round values")
        df_show(cmp, height=260)

        st.caption("Note: this baseline trains on pooled data (Centralised) for the same number of rounds/epochs. FL accuracy is from current demo run achieved.")

with tabs[3]:
    st.subheader("🎞️ Animated Swimlane (message flow)")
    if ev.empty:
        st.info("No events yet. Init and submit from both peers.")
    else:
        if "event" in ev.columns and (ev["event"] == "round_blocked").any():
            st.error("🚨 ROUND BLOCKED detected — likely wrong secret/sign (attack demo).", icon="🚨")

        if "frame" not in st.session_state:
            st.session_state.frame = 0

        s1, s2, s3, s4 = st.columns([1.2, 1, 1, 1])
        round_filter = s1.number_input("Round filter (0=all)", value=0, step=1)
        speed_ms = s2.slider("Frame delay (ms)", 150, 1500, 450, step=50)
        play = s3.button("▶ Play")
        step = s4.button("⏭ Step +1")
        reset = st.button("⟲ Reset")

        rf = None if int(round_filter) == 0 else int(round_filter)

        if rf is not None and "round" in ev.columns:
            ev_f = ev[ev["round"] == rf].reset_index(drop=True)
        else:
            ev_f = ev.reset_index(drop=True)
        total = len(ev_f)

        if reset:
            st.session_state.frame = 0
        if step:
            st.session_state.frame = min(total, st.session_state.frame + 1)

        placeholder = st.empty()

        if play:
            # avoid auto-refresh while playing
            for f in range(1, total + 1):
                st.session_state.frame = f
                hi = f - 1
                fig = draw_swimlane(ev, f, rf, highlight_idx=hi, blink_on=True, public_mode=public_mode)
                placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep((speed_ms/1000.0) * 0.55)
                fig = draw_swimlane(ev, f, rf, highlight_idx=hi, blink_on=False, public_mode=public_mode)
                placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep((speed_ms/1000.0) * 0.45)
        else:
            hi = (st.session_state.frame - 1) if st.session_state.frame > 0 else None
            fig = draw_swimlane(ev, st.session_state.frame, rf, highlight_idx=hi, blink_on=True, public_mode=public_mode)
            safe_pyplot(fig)

if auto:
    time.sleep(refresh)
    rerun()
