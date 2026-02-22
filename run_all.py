#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — One-command launcher for the Pitch PoC

Starts:
- Coordinator (FastAPI) :8000
- Dashboard (Streamlit) :8600
- Peer A (Streamlit)    :8501
- Peer B (Streamlit)    :8502

Pitch stability:
- Streamlit launched with --server.fileWatcherType none
"""
import os, sys, time, subprocess
from urllib.request import urlopen

COORD_PORT = 8000
DASH_PORT  = 8600
PEER_A_PORT = 8501
PEER_B_PORT = 8502

STREAMLIT_FLAGS = ["--server.fileWatcherType", "none"]

def run(cmd, env=None):
    return subprocess.Popen(cmd, env=env or os.environ.copy())

def wait_http(url: str, timeout_sec: int = 45):
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            with urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.6)
    return False

def main():
    procs = []
    try:
        print("Step 1/4: Starting Federated Coordinator (FastAPI) ...")
        procs.append(run([sys.executable, "-m", "uvicorn", "coordinator.server:app", "--port", str(COORD_PORT)]))
        ok = wait_http(f"http://127.0.0.1:{COORD_PORT}/status", timeout_sec=60)
        print("Coordinator is ready." if ok else "Coordinator not ready in time. Check logs.")

        print("Step 2/4: Starting MediVault Dashboard (Streamlit) ...")
        procs.append(run([sys.executable, "-m", "streamlit", "run", "dashboard/coordinator_dashboard.py",
                          "--server.port", str(DASH_PORT), *STREAMLIT_FLAGS]))
        time.sleep(0.8)

        print("Step 3/4: Starting Peer A GUI ...")
        envA = os.environ.copy(); envA["PEER_ID"] = "A"
        procs.append(run([sys.executable, "-m", "streamlit", "run", "peer/peer_app.py",
                          "--server.port", str(PEER_A_PORT), *STREAMLIT_FLAGS], env=envA))
        time.sleep(0.6)

        print("Step 4/4: Starting Peer B GUI ...")
        envB = os.environ.copy(); envB["PEER_ID"] = "B"
        procs.append(run([sys.executable, "-m", "streamlit", "run", "peer/peer_app.py",
                          "--server.port", str(PEER_B_PORT), *STREAMLIT_FLAGS], env=envB))
        time.sleep(0.6)

        print("\n=== MediVault PoC is running ===")
        print(f"Coordinator API:  http://127.0.0.1:{COORD_PORT}")
        print(f"Dashboard:        http://localhost:{DASH_PORT}")
        print(f"Peer A:           http://localhost:{PEER_A_PORT}")
        print(f"Peer B:           http://localhost:{PEER_B_PORT}\n")
        print("Recommended working flow:")
        print("1) Dashboard → Init Coordinator (choose LogReg/MLP)")
        print("2) Peer A/B → Generate/Reset local data (notes optional)")
        print("3) Submit A then B each round (or enable Auto submit on both)")
        print("\nPress Ctrl+C to stop everything.\n")

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for p in procs:
            try: p.terminate()
            except Exception: pass
        time.sleep(0.6)
        for p in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass
        print("Stopped.")

if __name__ == "__main__":
    main()
