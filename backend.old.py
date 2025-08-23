# --- inside backend.py ---

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import re, hashlib, csv
import os, joblib


HASH_BANK_PATH = "hash_bank.csv"   # this file must be next to backend.py

def normalize(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9$€£\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def md5_ng(ng):
    return hashlib.md5(" ".join(ng).encode("utf-8")).hexdigest()

def load_bank(path: str):
    bank = set()
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            bank.add(row["hash"])          # hash column
    return bank

BANK = load_bank(HASH_BANK_PATH)

def overlap_percent(text: str) -> float:
    w = normalize(text).split()
    hs = [md5_ng(ng) for n in (2,3) for ng in ngrams(w, n)]
    if not hs: 
        return 0.0
    hits = sum(1 for h in hs if h in BANK)
    return 100.0 * hits / len(hs)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

@app.get("/health")
def health():
    return {"ok": True, "bank_size": len(BANK)}

@app.post("/score")
async def score(payload: dict):
    text = payload.get("text", "")
    threshold = float(payload.get("threshold_percent", 20.0))
    p = overlap_percent(text)
    return {
        "overlap_percent": round(p, 1),
        "threshold_percent": threshold,
        "alert": p >= threshold
    }
