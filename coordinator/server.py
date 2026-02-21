#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coordinator/server.py — Federated Coordinator (HE + SMPC cancel + FedAvg)

Features:
- 2 peers (A,B)
- /reset supports model chooser: logreg | mlp (+ hidden_dim)
- HE: Paillier encrypted transport of masked updates
- SMPC-style: A adds +mask, B adds -mask (mask cancels only in the sum)
- Learning Card: stores delta_norm_A/B and avg_delta_norm, update vector length
- Dashboard note compare: stores short fictional note sample + site-level aggregate stats (optional)

This is a PoC for pitch/demo.
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import time, hashlib
import os, json
import requests
from dotenv import load_dotenv
load_dotenv()

# Load environment variables from a local .env file (recommended for demos)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # python-dotenv is optional; environment variables can still be provided by the shell/service manager.
    pass

from common.crypto import generate_paillier, add_cipher_vectors, decrypt_vector
from common.fl_model import init_params, predict_proba_model, model_param_len
from common.data_gen import FEATURES, make_peer_dataset, standardize

from sklearn.metrics import accuracy_score, roc_auc_score

# --- Optional OpenAI (for LLM insights) ---
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


app = FastAPI(title="Federated Coordinator (HE+SMPC+FedAvg)", version="full-clean-bundle")

TASK_DESC  = "Predict a patient risk event (demo label: risk_event)"
LABEL_NAME = "risk_event"

STATE: Dict[str, Any] = {
    "initialised": False,
    "d": len(FEATURES),
    "model_type": "logreg",
    "hidden_dim": 0,
    "rounds_total": 5,
    "current_round": 1,
    "he_bits": 2048,
    "strict_mask_check": True,
    "params": None,
    "pub": None,
    "priv": None,
    "submissions": {},   # peer_id -> submission dict
    "events": [],
    "metrics": [],
    "val": None,
    "round_start_ts": None,
}

def now() -> float:
    return time.time()

def log_event(e: Dict[str, Any]):
    e["ts"] = now()
    e.setdefault("model_type", STATE.get("model_type","logreg"))
    e.setdefault("hidden_dim", STATE.get("hidden_dim",0))
    STATE["events"].append(e)
    if len(STATE["events"]) > 4000:
        STATE["events"] = STATE["events"][-4000:]

def cipher_hash(sample: Any) -> str:
    try:
        return hashlib.sha256(str(sample).encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "na"

def reset_round():
    STATE["submissions"] = {}
    STATE["round_start_ts"] = now()
    log_event({"event":"round_start","round":STATE["current_round"]})

def build_validation(n: int = 1600, seed: int = 2026):
    df_val = make_peer_dataset("VAL", n=n, seed=seed)
    X, y, _, _ = standardize(df_val)
    return X, y

def eval_global(params: np.ndarray):
    X, y = STATE["val"]
    prob = predict_proba_model(params, X, STATE["model_type"], hidden=STATE["hidden_dim"] if STATE["hidden_dim"]>0 else 16)
    pred = (prob >= 0.5).astype(int)
    acc = float(accuracy_score(y, pred))
    try:
        auc = float(roc_auc_score(y, prob))
    except Exception:
        auc = float("nan")
    return acc, auc

class ResetReq(BaseModel):
    rounds_total: int = 5
    he_bits: int = 2048
    strict_mask_check: bool = True
    model_type: str = "logreg"   # "logreg" | "mlp"
    hidden_dim: int = 16         # only if mlp

class PubKeyResp(BaseModel):
    n: str
    he_bits: int

class GlobalResp(BaseModel):
    current_round: int
    rounds_total: int
    params: List[float]
    d: int
    feature_names: List[str]
    model_type: str
    hidden_dim: int
    task: str
    label: str

class SubmitReq(BaseModel):
    peer_id: str
    round: int
    enc_masked_delta: List[dict]
    bytes_payload: int
    t_encrypt: float
    mask_id: str
    sign: int
    preview: Optional[Dict[str, Any]] = None

class SubmitResp(BaseModel):
    ok: bool
    accepted_round: int
    received: int
    required: int
    round_complete: bool
    current_round: int
    error: Optional[str] = None

@app.post("/reset")
def reset(req: ResetReq):
    mt = str(req.model_type).lower().strip()
    if mt not in ("logreg","mlp"):
        raise HTTPException(400, "model_type must be 'logreg' or 'mlp'")

    STATE["initialised"] = True
    STATE["rounds_total"] = int(req.rounds_total)
    STATE["current_round"] = 1
    STATE["he_bits"] = int(req.he_bits)
    STATE["strict_mask_check"] = bool(req.strict_mask_check)
    STATE["model_type"] = mt
    STATE["hidden_dim"] = int(req.hidden_dim) if mt == "mlp" else 0

    d = STATE["d"]
    hidden = STATE["hidden_dim"] if mt == "mlp" else 0
    STATE["params"] = init_params(mt, d, hidden=hidden, seed=123)

    pub, priv = generate_paillier(bits=STATE["he_bits"])
    STATE["pub"], STATE["priv"] = pub, priv

    STATE["events"] = []
    STATE["metrics"] = []
    STATE["val"] = build_validation(n=1600, seed=2026)

    log_event({"event":"init","he_bits":STATE["he_bits"],"rounds_total":STATE["rounds_total"],
               "strict_mask_check":STATE["strict_mask_check"],"task":TASK_DESC,"label":LABEL_NAME})
    reset_round()
    return {"ok": True, "current_round": STATE["current_round"]}

@app.post("/init")
def init_alias(req: ResetReq):
    return reset(req)

@app.get("/status")
def status():
    if not STATE["initialised"]:
        return {"initialised":False,"current_round":1,"rounds_total":STATE.get("rounds_total",5),
                "he_bits":STATE.get("he_bits",2048),"strict_mask_check":STATE.get("strict_mask_check",True),
                "received":0,"required":2,"peers_received":[],"params_norm":None,
                "model_type":"logreg","hidden_dim":0,"task":TASK_DESC,"label":LABEL_NAME}
    params = STATE["params"]
    return {"initialised":True,"current_round":STATE["current_round"],"rounds_total":STATE["rounds_total"],
            "he_bits":STATE["he_bits"],"strict_mask_check":STATE["strict_mask_check"],
            "received":len(STATE["submissions"]),"required":2,"peers_received":list(STATE["submissions"].keys()),
            "params_norm":float(np.linalg.norm(params)) if params is not None else None,
            "model_type":STATE["model_type"],"hidden_dim":STATE["hidden_dim"],"task":TASK_DESC,"label":LABEL_NAME}

@app.get("/pubkey", response_model=PubKeyResp)
def pubkey():
    if not STATE["initialised"] or STATE["pub"] is None:
        raise HTTPException(400, "Not initialised. POST /reset first.")
    return PubKeyResp(n=str(STATE["pub"].n), he_bits=STATE["he_bits"])

@app.get("/global", response_model=GlobalResp)
def global_model():
    if not STATE["initialised"] or STATE["params"] is None:
        raise HTTPException(400, "Not initialised. POST /reset first.")
    return GlobalResp(current_round=STATE["current_round"], rounds_total=STATE["rounds_total"],
                     params=STATE["params"].astype(float).tolist(), d=STATE["d"], feature_names=list(FEATURES),
                     model_type=STATE["model_type"], hidden_dim=STATE["hidden_dim"], task=TASK_DESC, label=LABEL_NAME)

@app.get("/events")
def events(limit: int = 800):
    lim = max(1, min(4000, int(limit)))
    return STATE["events"][-lim:]

@app.get("/metrics")
def metrics():
    return STATE["metrics"]


class LLMInsightReq(BaseModel):
    audience: str = "clinician"   # clinician | partner | investor | technical
    provider: str = "auto"        # auto | openai | ollama
    include_protocol: bool = True
    include_site_stats: bool = True
    max_rounds: int = 12

class LLMInsightResp(BaseModel):
    ok: bool
    audience: str
    model: Optional[str] = None
    text: str
    input_hash: str

def _build_llm_evidence(req: LLMInsightReq) -> Dict[str, Any]:
    # Only send aggregated, non-identifiable information to the LLM.
    metrics = STATE.get("metrics", []) or []
    metrics_tail = metrics[-req.max_rounds:] if req.max_rounds and len(metrics) > req.max_rounds else metrics

    # Site stats in this PoC are synthetic, but keep them aggregate.
    site_stats = STATE.get("site_stats", None)
    if not req.include_site_stats:
        site_stats = None

    # Protocol timings derived from metrics (already aggregated)
    protocol = None
    if req.include_protocol and metrics_tail:
        protocol = {
            "rounds_included": len(metrics_tail),
            "mean_duration_sec": float(np.mean([m.get("duration_sec", 0.0) for m in metrics_tail])),
            "mean_bytes_total": float(np.mean([m.get("bytes_total", 0.0) for m in metrics_tail])),
            "mean_t_encrypt_A": float(np.mean([m.get("t_encrypt_A", 0.0) for m in metrics_tail])),
            "mean_t_encrypt_B": float(np.mean([m.get("t_encrypt_B", 0.0) for m in metrics_tail])),
            "mean_t_he_add": float(np.mean([m.get("t_he_add", 0.0) for m in metrics_tail])),
            "mean_t_decrypt_sum": float(np.mean([m.get("t_decrypt_sum", 0.0) for m in metrics_tail])),
        }

    # Model identity / config
    model_info = {
        "model_type": STATE.get("model_type"),
        "hidden_dim": STATE.get("hidden_dim"),
        "param_dim": int(model_param_len(STATE.get("model_type") or "logreg", STATE.get("hidden_dim") or 32)),
        "task": TASK_DESC,
        "label": getattr(req, "label", None),
        "features": FEATURES,
    }

    # Trend summary (acc only — do NOT invent AUC if not used)
    trend = None
    if metrics_tail:
        accs = [float(m.get("acc", 0.0)) for m in metrics_tail]
        trend = {
            "first_round": int(metrics_tail[0].get("round", 0)),
            "last_round": int(metrics_tail[-1].get("round", 0)),
            "acc_first": accs[0],
            "acc_last": accs[-1],
            "acc_delta": float(accs[-1] - accs[0]),
            "acc_min": float(np.min(accs)),
            "acc_max": float(np.max(accs)),
        }

    evidence = {
        "timestamp_utc": int(time.time()),
        "model_info": model_info,
        "trend": trend,
        "metrics_tail": metrics_tail,
        "site_stats": site_stats,
        "protocol_summary": protocol,
        "notes": [
            "This PoC shares NO patient-level data. Only aggregated metrics and synthetic site summaries are included.",
            "If any field is missing, it should be treated as unavailable."
        ]
    }
    return evidence

def _llm_system_prompt(audience: str) -> str:
    base = (
        "You are MediVault's reporting assistant. "
        "Use ONLY the provided JSON evidence. Do NOT invent numbers, claims, or citations. "
        "If something is not in the evidence, say it is unavailable. "
        "Do NOT provide clinical diagnosis. Provide decision-support wording only. "
        "Always include a short limitations section."
    )
    audience = (audience or "clinician").lower()
    if audience == "investor":
        return base + " Write in a commercial investor tone, focusing on value, traction signals, and next steps for pilot."
    if audience in ("partner", "nhs", "collaborator"):
        return base + " Write for NHS/partner stakeholders: clear, cautious, benefits + governance + pilot steps."
    if audience == "technical":
        return base + " Write for technical reviewers: include protocol summary, assumptions, and what to validate next."
    return base + " Write for clinicians: 60-second plain-English summary, then limitations and pilot recommendations."

def _ollama_generate(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    """Return (text, model_used). Uses Ollama local server."""
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    # Many instruct models work well with an explicit system+user prompt concatenation.
    prompt = f"{system_prompt}\n\n{user_prompt}\n"
    try:
        r = requests.post(
            url + "/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=float(os.environ.get("OLLAMA_TIMEOUT", "90")),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama call failed (connection): {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama call failed: {r.status_code} {r.text}")
    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Ollama call failed (bad JSON): {r.text[:500]}")
    text_out = (data.get("response") or "").strip()
    if not text_out:
        raise HTTPException(status_code=502, detail="Ollama call returned empty response.")
    return text_out, model

def _choose_llm_provider(req_provider: str) -> str:
    """Resolve provider: auto -> prefer OLLAMA if reachable, else OPENAI if key set."""
    p = (req_provider or "auto").strip().lower()
    if p in ("openai", "ollama"):
        return p
    # auto: if OLLAMA_URL reachable quickly, use it; otherwise OpenAI if key set.
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    try:
        rr = requests.get(url + "/api/tags", timeout=1.5)
        if rr.status_code == 200:
            return "ollama"
    except Exception:
        pass
    return "openai"

@app.post("/llm/insight", response_model=LLMInsightResp)
def llm_insight(req: LLMInsightReq):
    # Build evidence payload (shared for both providers)
    evidence = _build_llm_evidence(req)
    raw = json.dumps(evidence, sort_keys=True).encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()[:12]

    system_prompt = _llm_system_prompt(req.audience)
    user_prompt = "Evidence JSON (do not invent anything):\n" + json.dumps(evidence, indent=2)

    provider = _choose_llm_provider(getattr(req, "provider", "auto"))
    if provider == "ollama":
        text_out, model_used = _ollama_generate(system_prompt, user_prompt)
        return LLMInsightResp(ok=True, audience=req.audience, model=f"ollama:{model_used}", text=text_out, input_hash=h)

    # OpenAI path (cloud)
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="OpenAI SDK not installed. Add 'openai' to requirements and reinstall.")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set on the coordinator host.")
    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text_out = getattr(resp, "output_text", None) or ""
        if not text_out:
            try:
                text_out = resp.output[0].content[0].text  # type: ignore
            except Exception:
                text_out = ""
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI call failed: {e}")

    return LLMInsightResp(ok=True, audience=req.audience, model=f"openai:{model}", text=text_out, input_hash=h)


def finish_round():
    r = STATE["current_round"]
    subA = STATE["submissions"].get("A")
    subB = STATE["submissions"].get("B")
    if not subA or not subB:
        return

    mask_ok = (subA["mask_id"] == subB["mask_id"]) and (subA["sign"] == +1) and (subB["sign"] == -1)
    log_event({"event":"mask_check","round":r,"mask_id_A":subA["mask_id"],"mask_id_B":subB["mask_id"],
               "sign_A":subA["sign"],"sign_B":subB["sign"],"ok":bool(mask_ok)})
    if STATE["strict_mask_check"] and not mask_ok:
        log_event({"event":"round_blocked","round":r,"reason":"mask_id/sign mismatch"})
        return

    t0 = time.perf_counter()
    summed = add_cipher_vectors(STATE["pub"], subA["payload"], subB["payload"])
    t_he_add = float(time.perf_counter() - t0)
    log_event({"event":"he_add","round":r,"t_he_add":t_he_add})

    t1 = time.perf_counter()
    dec_sum = decrypt_vector(STATE["priv"], STATE["pub"], summed)
    t_dec = float(time.perf_counter() - t1)
    log_event({"event":"decrypt_sum","round":r,"t_decrypt_sum":t_dec})

    avg_delta = (dec_sum / 2.0).astype(np.float32)
    STATE["params"] = (STATE["params"] + avg_delta).astype(np.float32)
    avg_delta_norm = float(np.linalg.norm(avg_delta))
    w_norm = float(np.linalg.norm(STATE["params"]))

    log_event({"event":"fedavg_update_applied","round":r,
               "avg_delta_norm":avg_delta_norm,"w_norm":w_norm,
               "delta_norm_A": float(subA.get("delta_norm", float("nan"))),
               "delta_norm_B": float(subB.get("delta_norm", float("nan"))),
               "vec_len": int(len(avg_delta)),
               "note":"FedAvg update applied"})

    acc, auc = eval_global(STATE["params"])
    log_event({"event":"model_update","round":r,"acc":acc,"auc":auc})

    bytes_total = int(subA["bytes_payload"]) + int(subB["bytes_payload"])
    duration = float(now() - (STATE["round_start_ts"] or now()))
    STATE["metrics"].append({
        "round": r,
        "mask_id": subA["mask_id"],
        "bytes_total": bytes_total,
        "t_encrypt_A": float(subA["t_encrypt"]),
        "t_encrypt_B": float(subB["t_encrypt"]),
        "t_he_add": t_he_add,
        "t_decrypt_sum": t_dec,
        "duration_sec": duration,
        "acc": acc,
        "auc": auc,
        "avg_delta_norm": avg_delta_norm,
        "w_norm": w_norm,
        "model_type": STATE["model_type"],
        "hidden_dim": STATE["hidden_dim"],
        "vec_len": int(len(avg_delta)),
        "delta_norm_A": float(subA.get("delta_norm", float("nan"))),
        "delta_norm_B": float(subB.get("delta_norm", float("nan"))),
    })
    log_event({"event":"round_complete","round":r,"bytes_total":bytes_total,"duration_sec":duration})

    if STATE["current_round"] < STATE["rounds_total"]:
        STATE["current_round"] += 1
        reset_round()
    else:
        log_event({"event":"demo_complete","rounds_total":STATE["rounds_total"]})

@app.post("/submit", response_model=SubmitResp)
def submit(req: SubmitReq):
    if not STATE["initialised"]:
        raise HTTPException(400, "Not initialised. POST /reset first.")

    peer = str(req.peer_id).strip().upper()
    if peer not in ("A","B"):
        return SubmitResp(ok=False, accepted_round=STATE["current_round"], received=len(STATE["submissions"]),
                         required=2, round_complete=False, current_round=STATE["current_round"],
                         error="peer_id must be A or B")

    if int(req.round) != int(STATE["current_round"]):
        return SubmitResp(ok=False, accepted_round=STATE["current_round"], received=len(STATE["submissions"]),
                         required=2, round_complete=False, current_round=STATE["current_round"],
                         error="Round mismatch")

    if peer in STATE["submissions"]:
        return SubmitResp(ok=False, accepted_round=STATE["current_round"], received=len(STATE["submissions"]),
                         required=2, round_complete=False, current_round=STATE["current_round"],
                         error="Duplicate submission for this peer in current round")

    expected_len = model_param_len(STATE["model_type"], STATE["d"], hidden=max(1, STATE["hidden_dim"]))
    if len(req.enc_masked_delta) != expected_len:
        return SubmitResp(ok=False, accepted_round=STATE["current_round"], received=len(STATE["submissions"]),
                         required=2, round_complete=False, current_round=STATE["current_round"],
                         error=f"Vector length mismatch: got {len(req.enc_masked_delta)} expected {expected_len}")

    preview = req.preview or {}
    chash = cipher_hash(preview.get("ciphertext_sample","")) if isinstance(preview, dict) else "na"
    delta_norm = float(preview.get("delta_norm")) if isinstance(preview, dict) and ("delta_norm" in preview) else float("nan")
    # Optional tiny ciphertext preview (for dashboard tech expander)
    cprev = preview.get("ciphertext_sample") if isinstance(preview, dict) else None
    cipher_sample = None
    if isinstance(cprev, list):
        # keep only first 2 items to avoid big payloads
        cipher_sample = cprev[:2]
    elif isinstance(cprev, dict):
        cipher_sample = cprev
    elif isinstance(cprev, str):
        # e.g., "(hidden)" or short text
        cipher_sample = cprev if (len(cprev) <= 120 and cprev != "(hidden)") else None


    # Optional pitch metadata (non-sensitive)
    note_samples = preview.get("note_samples") if isinstance(preview, dict) else None
    note0 = None
    if isinstance(note_samples, list) and note_samples:
        note0 = str(note_samples[0])
        note0 = (note0[:240].rstrip() + "…") if len(note0) > 240 else note0
    site_stats = preview.get("site_stats") if isinstance(preview, dict) else None
    if not isinstance(site_stats, dict):
        site_stats = None

    log_event({
        "event":"recv_ciphertext","round":int(req.round),"peer_id":peer,"mask_id":req.mask_id,"sign":int(req.sign),
        "bytes_payload":int(req.bytes_payload),"t_encrypt":float(req.t_encrypt),"cipher_len":len(req.enc_masked_delta),
        "cipher_hash":chash,"cipher_sample":cipher_sample,"delta_norm":delta_norm,"note_sample":note0,"site_stats":site_stats
    })

    STATE["submissions"][peer] = {
        "payload": req.enc_masked_delta,
        "bytes_payload": int(req.bytes_payload),
        "t_encrypt": float(req.t_encrypt),
        "mask_id": req.mask_id,
        "sign": int(req.sign),
        "delta_norm": delta_norm,
        "cipher_sample": cipher_sample,
        "note_sample": note0,
        "site_stats": site_stats,
    }

    received = len(STATE["submissions"])
    complete = received >= 2
    if complete:
        before = len(STATE["metrics"])
        finish_round()
        if len(STATE["metrics"]) == before and STATE["strict_mask_check"]:
            return SubmitResp(ok=False, accepted_round=req.round, received=received, required=2,
                              round_complete=False, current_round=STATE["current_round"],
                              error="Round blocked: mask_id/sign mismatch (strict).")

    return SubmitResp(ok=True, accepted_round=req.round, received=received, required=2,
                      round_complete=complete, current_round=STATE["current_round"])
