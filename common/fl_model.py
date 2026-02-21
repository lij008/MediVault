#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common/fl_model.py — Federated model utilities (LogReg + MLP)

- "logreg": logistic regression binary classifier
- "mlp"  : 1-hidden-layer MLP (ReLU) + sigmoid output

All models use a single flattened params vector.
"""
from __future__ import annotations
import numpy as np

# ---------- Numeric safety helpers ----------
_CLIP_Z = 50.0          # for sigmoid/logits
_CLIP_XW = 1e6          # pre-matmul clip for X/W
_MAX_EST = 1e150        # matmul magnitude guard
_GRAD_CLIP_DEFAULT = 5.0

def _clean(a: np.ndarray, *, clip: float = _CLIP_XW) -> np.ndarray:
    """Replace NaN/Inf and clip to finite range."""
    a = np.asarray(a, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=clip, neginf=-clip)
    if clip is not None:
        a = np.clip(a, -clip, clip)
    return a

def _safe_matmul(A, B, clip=1e4, target=1e120):
    """
    Numerically safer matmul:
    - replace NaN/Inf with finite values
    - clip magnitudes
    - dynamic scaling to avoid overflow in BLAS matmul
    - suppress numpy runtime warnings inside matmul (we sanitise outputs anyway)
    """
    import numpy as np

    def _clean(x):
        x = np.asarray(x, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
        x = np.clip(x, -clip, clip)
        return x

    Af = _clean(A)
    Bf = _clean(B)

    # Dynamic scaling (rough bound on dot product magnitude)
    # bound ~ max|A| * max|B| * d
    d = Af.shape[1] if Af.ndim == 2 else 1
    maxA = float(np.max(np.abs(Af))) if Af.size else 0.0
    maxB = float(np.max(np.abs(Bf))) if Bf.size else 0.0
    est = maxA * maxB * max(1, d)

    if (not np.isfinite(est)) or est > target:
        # scale both by sqrt to keep product similar scale
        s = np.sqrt(est / target) if np.isfinite(est) and est > 0 else 1e3
        Af = Af / s
        Bf = Bf / s

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Z = Af @ Bf

    Z = np.nan_to_num(Z, nan=0.0, posinf=clip, neginf=-clip)
    Z = np.clip(Z, -clip, clip)
    return Z

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -_CLIP_Z, _CLIP_Z)
    return 1.0 / (1.0 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)
def model_param_len(model_type: str, d: int, hidden: int = 16) -> int:
    mt = str(model_type).lower()
    if mt in ("logreg", "logistic", "lr"):
        return d + 1
    if mt in ("mlp", "nn"):
        h = int(hidden)
        return d*h + h + h + 1  # W1 + b1 + W2 + b2
    raise ValueError(f"unknown model_type: {model_type}")

def init_params(model_type: str, d: int, hidden: int = 16, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mt = str(model_type).lower()
    if mt in ("logreg", "logistic", "lr"):
        w = rng.normal(0, 0.01, d).astype(np.float32)
        b = np.float32(0.0)
        return np.concatenate([w, np.array([b], dtype=np.float32)])
    if mt in ("mlp", "nn"):
        h = int(hidden)
        W1 = (rng.normal(0, 1.0, (d, h)) * np.sqrt(2.0/max(1,d))).astype(np.float32)
        b1 = np.zeros((h,), dtype=np.float32)
        W2 = (rng.normal(0, 1.0, (h,)) * np.sqrt(1.0/max(1,h))).astype(np.float32)
        b2 = np.float32(0.0)
        return pack_mlp(W1, b1, W2, b2)
    raise ValueError(f"unknown model_type: {model_type}")

def unpack_mlp(params: np.ndarray, d: int, h: int):
    params = params.astype(np.float32, copy=False)
    p = 0
    W1 = params[p:p+d*h].reshape(d, h); p += d*h
    b1 = params[p:p+h]; p += h
    W2 = params[p:p+h]; p += h
    b2 = np.float32(params[p]); p += 1
    return W1, b1, W2, b2

def pack_mlp(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.float32) -> np.ndarray:
    return np.concatenate([W1.reshape(-1), b1.reshape(-1), W2.reshape(-1), np.array([b2], dtype=np.float32)])

def predict_proba_model(params: np.ndarray, X: np.ndarray, model_type: str, hidden: int = 16) -> np.ndarray:
    mt = str(model_type).lower()
    if mt in ("logreg", "logistic", "lr"):
        w = params[:-1]
        b = params[-1]
        return sigmoid(_safe_matmul(X, w) + b)
    if mt in ("mlp", "nn"):
        d = X.shape[1]
        h = int(hidden)
        W1, b1, W2, b2 = unpack_mlp(params, d, h)
        z1 = _safe_matmul(X, W1) + b1
        a1 = relu(z1)
        logits = _safe_matmul(a1, W2) + b2
        return sigmoid(logits)
    raise ValueError(f"unknown model_type: {model_type}")

def local_train_one_epoch_model(params: np.ndarray, X: np.ndarray, y: np.ndarray,
                                model_type: str, hidden: int = 16,
                                lr: float = 0.05, batch_size: int = 256, l2: float = 1e-3,
                                grad_clip: float = 5.0, seed: int = 0) -> np.ndarray:
    """
    One epoch mini-batch SGD.
    Returns delta = new_params - params (float32).
    """
    mt = str(model_type).lower()
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)

    if mt in ("logreg", "logistic", "lr"):
        w = params[:-1].astype(np.float32).copy()
        b = np.float32(params[-1])

        for start in range(0, n, batch_size):
            j = idx[start:start+batch_size]
            Xb = X[j].astype(np.float32)
            yb = y[j].astype(np.float32)

            logits = _safe_matmul(Xb, w) + b
            p = sigmoid(logits)
            err = (p - yb) / float(len(yb))
            err = np.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)

            grad_w = _safe_matmul(Xb.T, err).astype(np.float32) + (l2 * w)
            grad_b = np.float32(err.sum())

            gnorm = float(np.linalg.norm(grad_w))
            if gnorm > grad_clip:
                scale = np.float32(grad_clip / (gnorm + 1e-12))
                grad_w *= scale
                grad_b *= scale

            w -= np.float32(lr) * grad_w
            b -= np.float32(lr) * grad_b

        new_params = np.concatenate([w, np.array([b], dtype=np.float32)])
        delta = (new_params - params).astype(np.float32)
        return np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

    if mt in ("mlp", "nn"):
        d = X.shape[1]
        h = int(hidden)
        W1, b1, W2, b2 = unpack_mlp(params, d, h)
        W1 = W1.copy(); b1 = b1.copy(); W2 = W2.copy(); b2 = np.float32(b2)

        for start in range(0, n, batch_size):
            j = idx[start:start+batch_size]
            Xb = X[j].astype(np.float32)
            yb = y[j].astype(np.float32).reshape(-1)

            z1 = _safe_matmul(Xb, W1) + b1
            a1 = np.maximum(0.0, z1)
            logits = _safe_matmul(a1, W2) + b2
            p = sigmoid(logits)

            dlogits = (p - yb) / float(len(yb))
            dlogits = np.nan_to_num(dlogits, nan=0.0, posinf=0.0, neginf=0.0)

            grad_W2 = _safe_matmul(a1.T, dlogits).astype(np.float32) + l2 * W2
            grad_b2 = np.float32(dlogits.sum())
            da1 = (dlogits.reshape(-1,1) * W2.reshape(1,-1)).astype(np.float32)
            dz1 = da1 * (z1 > 0).astype(np.float32)
            grad_W1 = _safe_matmul(Xb.T, dz1).astype(np.float32) + l2 * W1
            grad_b1 = dz1.sum(axis=0).astype(np.float32)

            gnorm = float(np.sqrt(np.sum(grad_W1**2) + np.sum(grad_b1**2) + np.sum(grad_W2**2) + float(grad_b2**2)))
            if gnorm > grad_clip:
                scale = np.float32(grad_clip / (gnorm + 1e-12))
                grad_W1 *= scale; grad_b1 *= scale; grad_W2 *= scale; grad_b2 *= scale

            W1 -= np.float32(lr) * grad_W1
            b1 -= np.float32(lr) * grad_b1
            W2 -= np.float32(lr) * grad_W2
            b2 -= np.float32(lr) * grad_b2

        new_params = pack_mlp(W1, b1, W2, b2)
        delta = (new_params - params).astype(np.float32)
        return np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

    raise ValueError(f"unknown model_type: {model_type}")
