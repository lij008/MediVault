# MediVault PoC (FL + Dashboard + LLM Insights)

This repository contains a proof-of-concept Federated Learning (FL) demo with:
- A FastAPI “Coordinator” service
- Two Streamlit peer apps (simulating two data holders)
- A Streamlit dashboard (includes FL vs Non-FL + LLM Insights)
- LLM Insights supports **either OpenAI** or **local Ollama**

---

## 1) Prerequisites

### Python environment
Recommended: Conda / Miniconda.

- Python: 3.10+ (3.12 is OK)
- Install dependencies:
```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, at minimum you typically need:
```bash
pip install streamlit fastapi uvicorn requests python-dotenv pandas numpy matplotlib
```

> Note: `requests` is required for Ollama HTTP calls.

---

## 2) Local LLM Option: Ollama (Mac, Linux, Windows)

### 2.1 Install Ollama (macOS)
- Install Ollama from the official installer (Applications)
- Verify:
```bash
ollama --version
```

### 2.2 Start the Ollama server (required)
Open a terminal and run:
```bash
ollama serve
```

Default local API:
- `http://127.0.0.1:11434`

Keep this terminal running while you use the PoC.

### 2.3 Pull a model (example: qwen2.5 7B)
In another terminal:
```bash
ollama pull qwen2.5:7b-instruct
```

Quick test:
```bash
ollama run qwen2.5:7b-instruct "Say OK"
```

HTTP test:
```bash
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"qwen2.5:7b-instruct","prompt":"Say OK","stream":false}'
```

If you see a JSON response with `"response": ...` then Ollama is working.

---

## 3) Choose LLM Provider (OpenAI vs Ollama)

The Dashboard “LLM Insights” supports provider selection (e.g., `auto / ollama / openai`).
You can also control behaviour via environment variables.

### 3.1 Environment variables (recommended)
Create a `.env` file in the project root (same folder as `run_all.py`), for example:

#### Option A: Local Ollama
```env
LLM_PROVIDER=ollama
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_TIMEOUT=90
```

#### Option B: OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=YOUR_KEY_HERE
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TIMEOUT=60
```

#### Option C: Auto (prefer Ollama, fallback to OpenAI)
```env
LLM_PROVIDER=auto
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OPENAI_API_KEY=YOUR_KEY_HERE
OPENAI_MODEL=gpt-4.1-mini
```

> If you select provider in the Dashboard UI, it will override the default behaviour in practice.
> For demos, `auto` is convenient: it uses local Ollama when available.

---

## 4) Run the full demo

From the project root:

```bash
python run_all.py
```

This typically starts (in order):
1) Federated Coordinator (FastAPI / Uvicorn)
2) Peer A (Streamlit)
3) Peer B (Streamlit)
4) Dashboard (Streamlit)

Open the printed local URLs in your browser.

---

## 5) Notes for running Ollama models from an external SSD (macOS)

If your Mac internal disk is small (e.g., 256GB), you can store models on an external SSD.

Example approach (symlink):
```bash
pkill ollama || true
mkdir -p /Volumes/SSD/ollama
[ -d ~/.ollama ] && mv ~/.ollama /Volumes/SSD/ollama/.ollama
ln -s /Volumes/SSD/ollama/.ollama ~/.ollama
```

Verify:
```bash
ls -lah ~/.ollama
du -sh /Volumes/SSD/ollama/.ollama
```

Reboot note:
- The symlink will remain after reboot.
- The SSD must be mounted at `/Volumes/SSD` (same volume name) for Ollama to find models.

---

## 6) Troubleshooting

### 6.1 “Ollama call failed (connection)”
- Make sure Ollama is running:
```bash
ollama serve
```
- Verify the API is reachable:
```bash
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"qwen2.5:7b-instruct","prompt":"Say OK","stream":false}'
```
- Confirm `OLLAMA_URL` in `.env` matches:
  - `http://127.0.0.1:11434`

### 6.2 “name 'requests' is not defined”
- Install requests:
```bash
pip install requests
```
- Ensure `import requests` exists in `coordinator/server.py`
- Restart the coordinator (stop and rerun `python run_all.py`)

### 6.3 OpenAI errors (401/403/429/500)
- Check API key is set:
```bash
python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))"
```
- Confirm you have access to the selected `OPENAI_MODEL`
- If rate limited, try a smaller model or reduce call frequency

### 6.4 Streamlit not updating after code changes
- Stop Streamlit processes and restart:
```bash
pkill -f streamlit || true
python run_all.py
```
- Hard refresh browser: `Cmd+Shift+R`

---

## 7) Recommended demo workflow (quick)
1) Start Ollama:
   ```bash
   ollama serve
   ```
2) Ensure model exists:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```
3) Start PoC:
   ```bash
   python run_all.py
   ```
4) In Dashboard → **LLM Insights**:
   - Choose Provider: `ollama` (or `auto`)
   - Select a prompt template (Clinician / Investor / IG)
   - Generate report

---

## 8) Security note (PoC scope)
This is a proof-of-concept demo. Do not treat it as production-ready.
For real deployments, additional security hardening is required:
- Authentication/authorisation
- TLS everywhere
- Proper key management
- Rate limiting and logging
- Secure storage of secrets and audit trails
