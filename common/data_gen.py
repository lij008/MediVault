#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common/data_gen.py — Synthetic NHS-like patient data generator (Pitch Demo)

Scheme A included:
- Stable numeric features by distributions (for training stability)
- Optional fictional NHS-style "clinical_note" text:
    * Uses local Ollama (Llama) for a small number of notes if enabled & available
    * Falls back to a strong template generator otherwise (so demo always runs)

No real patient info, no identifiers.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import re
from typing import Optional

FEATURES = [
    "age","sex","bmi","sys_bp","chol","hba1c","hr","steps","sleep","smoker","comorb","deprivation"
]

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))

def _sex_str(sex: int) -> str:
    return "male" if int(sex) == 1 else "female"

def _risk_phrase(p: float) -> str:
    if p >= 0.80: return "high risk"
    if p >= 0.55: return "moderate risk"
    return "lower risk"

def _template_note(row: dict, site: str) -> str:
    age = int(row["age"]); sex = _sex_str(row["sex"])
    bmi = float(row["bmi"]); sbp = float(row["sys_bp"]); hba1c = float(row["hba1c"])
    chol = float(row["chol"]); steps = int(row["steps"]); sleep = float(row["sleep"])
    smoker = int(row["smoker"]); comorb = int(row["comorb"])
    risk = _risk_phrase(float(row.get("_risk_prob", 0.5)))

    lifestyle = []
    if smoker: lifestyle.append("smoker")
    if steps < 5000: lifestyle.append("low activity")
    if sleep < 6.5: lifestyle.append("short sleep")
    lifestyle_txt = ", ".join(lifestyle) if lifestyle else "no major lifestyle red flags"

    comorb_txt = "multiple long-term conditions" if comorb >= 3 else ("some comorbidities" if comorb >= 1 else "no major comorbidities")

    return (f"[{site}] {age}-year-old {sex}. BP {sbp:.0f} mmHg, BMI {bmi:.1f}, HbA1c {hba1c:.1f}%, "
            f"chol {chol:.1f} mmol/L. {comorb_txt}; {lifestyle_txt}. "
            f"Assessment: overall {risk}. Plan: routine monitoring and lifestyle advice.")

def _ollama_generate_note(prompt: str, ollama_url: str, model: str) -> Optional[str]:
    try:
        import httpx
        with httpx.Client(timeout=8.0) as http:
            r = http.post(f"{ollama_url.rstrip('/')}/api/generate",
                          json={"model": model, "prompt": prompt, "stream": False})
            if r.status_code != 200:
                return None
            j = r.json()
            txt = re.sub(r"\s+", " ", (j.get("response","") or "")).strip()
            if not txt:
                return None
            return (txt[:360].rstrip() + "…") if len(txt) > 360 else txt
    except Exception:
        return None

def attach_clinical_notes(df: pd.DataFrame, site: str,
                          use_ollama: bool = False,
                          ollama_url: str = "http://localhost:11434",
                          ollama_model: str = "llama3.1",
                          llm_note_count: int = 8,
                          seed: int = 0) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed + (101 if site == "A" else 202))
    n = len(out)
    notes = [_template_note(out.iloc[i].to_dict(), site=site) for i in range(n)]

    k = int(max(0, min(llm_note_count, n)))
    idx_llm = set(rng.choice(np.arange(n), size=k, replace=False).tolist()) if (use_ollama and k > 0) else set()
    replaced = 0

    if idx_llm:
        rule = ("Write a short fictional NHS-style summary. No names/addresses/DOB. 2-3 sentences.")
        for i in idx_llm:
            row = out.iloc[i].to_dict()
            prompt = (f"{rule}\nSite:{site}\nAge:{int(row['age'])} Sex:{_sex_str(row['sex'])}\n"
                      f"BP:{float(row['sys_bp']):.0f} BMI:{float(row['bmi']):.1f} HbA1c:{float(row['hba1c']):.1f}% "
                      f"Chol:{float(row['chol']):.1f} HR:{float(row['hr']):.0f} Steps:{int(row['steps'])} Sleep:{float(row['sleep']):.1f}h "
                      f"Smoker:{int(row['smoker'])} Comorb:{int(row['comorb'])}\n")
            gen = _ollama_generate_note(prompt, ollama_url, ollama_model)
            if gen:
                notes[i] = f"[{site}] {gen}"
                replaced += 1

    out["clinical_note"] = notes
    out.attrs["ollama_used"] = bool(use_ollama)
    out.attrs["ollama_replaced"] = replaced
    return out

def generate_peer_data(peer_id: str, n: int = 1500, seed: int = 42) -> pd.DataFrame:
    peer_id = str(peer_id).strip().upper()
    if peer_id not in ("A","B"):
        raise ValueError("peer_id must be 'A' or 'B'")

    rng = np.random.default_rng(seed if peer_id == "A" else seed + 777)

    # Non-IID site differences
    if peer_id == "A":
        deprivation = rng.beta(2.5, 2.0, n)
        age = rng.normal(61, 11, n).clip(18, 95)
        steps = rng.normal(5200, 1900, n).clip(500, 20000)
        smoker = rng.binomial(1, 0.22, n)
        comorb = rng.poisson(2.2, n).clip(0, 10)
        hba1c = rng.normal(6.2, 0.9, n).clip(4.0, 12.0)
        sys_bp = rng.normal(142, 16, n).clip(90, 230)
        bmi = rng.normal(29, 5.2, n).clip(16, 55)
        chol = rng.normal(5.6, 1.0, n).clip(2.0, 10.0)
        sleep = rng.normal(6.6, 1.0, n).clip(3.0, 10.5)
    else:
        deprivation = rng.beta(1.8, 2.4, n)
        age = rng.normal(46, 13, n).clip(18, 95)
        steps = rng.normal(7400, 2300, n).clip(700, 25000)
        smoker = rng.binomial(1, 0.16, n)
        comorb = rng.poisson(1.2, n).clip(0, 10)
        hba1c = rng.normal(5.7, 0.7, n).clip(4.0, 11.5)
        sys_bp = rng.normal(128, 14, n).clip(85, 210)
        bmi = rng.normal(26, 4.4, n).clip(16, 50)
        chol = rng.normal(5.0, 0.9, n).clip(2.0, 9.5)
        sleep = rng.normal(7.1, 0.9, n).clip(3.0, 10.5)

    sex = rng.binomial(1, 0.49, n)
    hr = rng.normal(76, 11, n).clip(40, 140)

    # Hidden risk model (demo only)
    z = (0.045*(age-50) + 0.020*(sys_bp-130) + 0.55*(hba1c-5.6) + 0.30*smoker + 0.20*comorb +
         0.018*(bmi-25) + 0.35*deprivation + 0.10*sex - 0.00012*(steps-6500) - 0.12*(sleep-7.0) +
         rng.normal(0, 0.75, n))
    p = _sigmoid(z)
    y = rng.binomial(1, p, n).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "bmi": bmi, "sys_bp": sys_bp, "chol": chol, "hba1c": hba1c, "hr": hr,
        "steps": steps, "sleep": sleep, "smoker": smoker, "comorb": comorb, "deprivation": deprivation,
        "risk_event": y,
    })
    df["_risk_prob"] = p.astype(float)
    df["risk_label"] = df["risk_event"].astype(int)
    return df

def standardize(df: pd.DataFrame):
    X = df[FEATURES].astype("float32").to_numpy()
    y = df["risk_event"].astype("int64").to_numpy()
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype("float32"), y, mu.astype("float32"), sd.astype("float32")

def make_peer_dataset(peer_id: str, n: int = 1500, seed: int = 42) -> pd.DataFrame:
    pid = str(peer_id).strip().upper()
    if pid in ("A","B"):
        return generate_peer_data(pid, n=n, seed=seed)
    nA = n // 2
    nB = n - nA
    dfA = generate_peer_data("A", n=nA, seed=seed + 101)
    dfB = generate_peer_data("B", n=nB, seed=seed + 202)
    df = pd.concat([dfA, dfB], ignore_index=True)
    rng = np.random.default_rng(seed + 999)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)
