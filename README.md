# MediVault PoC — Federated Learning + HE + SMPC (2 peers)

## What this demo shows (in simple terms)
- Two sites (Peer A and Peer B) keep patient data locally.
- Each peer trains locally and produces a **model update** (Δ).
- Before sending, each peer:
  1) adds an SMPC mask (+mask for A, -mask for B) so single-site updates are protected
  2) encrypts the masked update using Paillier homomorphic encryption (HE)
- The coordinator only decrypts the **sum** of the two encrypted updates.
  The masks cancel, and FedAvg is applied to update the global model.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

Then open:
- Dashboard: http://localhost:8600
- Peer A:    http://localhost:8501
- Peer B:    http://localhost:8502

Pitch flow:
1) Dashboard → Init Coordinator (choose LogReg or MLP)
2) Peer A/B → Generate/Reset local data (optional: enable notes)
3) Submit A then B for each round (or enable Auto submit)

## Optional: Llama (Ollama) for a few notes (Scheme A)
- Install and run Ollama locally:
  - `ollama serve`
  - `ollama pull llama3.1`
- In Peer sidebar enable:
  - Use local Ollama (Llama) for a few notes
Notes are **fictional** and only used to make the demo more realistic. Training still uses numeric features.

## Troubleshooting
- If you see Streamlit media cache warnings, keep Dashboard auto-refresh OFF while playing the swimlane.
- If a round is blocked, disable attack toggles and ensure both peers use the same shared secret.
