#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common/crypto.py — Demo crypto plumbing for Pitch PoC

Implements:
- Paillier HE transport for model updates (fixed-point int encoding)
- SMPC-style mask cancellation (A adds +mask, B adds -mask)
- Serialization helpers for EncryptedNumber

NOTE: This is a demo PoC. It demonstrates the *message-flow concepts*
of HE transport and SMPC masking, not production-grade crypto engineering.
"""
from __future__ import annotations

import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple
from phe import paillier

SCALE = 1_000_000  # fixed-point scale for floats -> ints

def generate_paillier(bits: int = 2048) -> Tuple[paillier.PaillierPublicKey, paillier.PaillierPrivateKey]:
    pub, priv = paillier.generate_paillier_keypair(n_length=int(bits))
    return pub, priv

def mask_id(secret: str, round_idx: int) -> str:
    h = hashlib.sha256(f"{secret}|{round_idx}".encode("utf-8")).hexdigest()
    return h[:16]

def _prg_mask(secret: str, round_idx: int, length: int) -> np.ndarray:
    # Deterministic PRG: seed from secret+round, generate N(0,1) mask
    seed_bytes = hashlib.sha256(f"MASK|{secret}|{round_idx}".encode("utf-8")).digest()
    seed = int.from_bytes(seed_bytes[:8], "little", signed=False) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=int(length)).astype(np.float32)

def apply_mask(delta: np.ndarray, secret: str, round_idx: int, sign: int) -> np.ndarray:
    sign = int(sign)
    if sign not in (+1, -1):
        raise ValueError("sign must be +1 or -1")
    m = _prg_mask(secret, round_idx, len(delta))
    return (delta + (sign * m)).astype(np.float32)

def _enc_to_dict(enc: paillier.EncryptedNumber) -> Dict[str, Any]:
    return {"c": str(enc.ciphertext()), "e": int(enc.exponent)}

def _dict_to_enc(pub: paillier.PaillierPublicKey, d: Dict[str, Any]) -> paillier.EncryptedNumber:
    return paillier.EncryptedNumber(pub, int(d["c"]), int(d.get("e", 0)))

def encrypt_vector(pub: paillier.PaillierPublicKey, vec: np.ndarray) -> List[Dict[str, Any]]:
    vec = np.asarray(vec, dtype=np.float32)
    out: List[Dict[str, Any]] = []
    for x in vec:
        xi = int(np.round(float(x) * SCALE))
        enc = pub.encrypt(xi)
        out.append(_enc_to_dict(enc))
    return out

def add_cipher_vectors(pub: paillier.PaillierPublicKey, a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(a) != len(b):
        raise ValueError("Cipher vectors must have same length")
    out: List[Dict[str, Any]] = []
    for da, db in zip(a, b):
        ea = _dict_to_enc(pub, da)
        eb = _dict_to_enc(pub, db)
        out.append(_enc_to_dict(ea + eb))
    return out

def decrypt_vector(priv: paillier.PaillierPrivateKey, pub: paillier.PaillierPublicKey, enc_vec: List[Dict[str, Any]]) -> np.ndarray:
    out = np.zeros((len(enc_vec),), dtype=np.float32)
    for i, d in enumerate(enc_vec):
        e = _dict_to_enc(pub, d)
        val_int = priv.decrypt(e)  # int
        out[i] = float(val_int) / float(SCALE)
    return out
