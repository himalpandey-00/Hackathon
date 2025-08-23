# backend.py — hash-only FastAPI scorer
# Endpoints:
#   GET  /health -> {"ok": true, "bank_size": N}
#   POST /score  -> {"overlap_percent": x.x, "threshold_percent": y.y, "alert": bool}
#   POST /reload -> reloads hash_bank.csv without restart

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import re, csv, hashlib, os

HERE = Path(__file__).parent
HASH_BANK_PATH = Path(os.getenv("HASH_BANK_FILE", HERE / "hash_bank.csv"))
NG_SIZES = (2, 3)  # bigrams & trigrams

def normalize(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9$€£\s]", " ", t)     # keep currency symbols
    return re.sub(r"\s+", " ", t).strip()

def ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def md5_ng(ng) -> str:
    return hashlib.md5(" ".join(ng).encode("utf-8")).hexdigest()

def load_bank(path: Path):
    bank = set()
    if not path.exists():
        print(f"[WARN] hash bank not found at {path.resolve()}")
        return bank
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "hash" not in r.fieldnames:
            raise RuntimeError("hash_bank.csv must have a 'hash' column")
        for row in r:
            bank.add(row["hash"])
    print(f"[OK] loaded hash bank: {len(bank)} entries from {path.name}")
    return bank

BANK = load_bank(HASH_BANK_PATH)

def overlap_percent(text: str) -> float:
    words = normalize(text).split()
    hashes = [md5_ng(ng) for n in NG_SIZES for ng in ngrams(words, n)]
    if not hashes:
        return 0.0
    hits = sum(1 for h in hashes if h in BANK)
    return 100.0 * hits / len(hashes)

app = FastAPI(title="Romance Scam Detector API (hash-only)")

# allow local demo origins (and "null" if someone opens file:// by mistake)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "null"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

class ScoreIn(BaseModel):
    text: str
    threshold_percent: float = 20.0

@app.get("/health")
def health():
    return {"ok": True, "bank_size": len(BANK)}

@app.post("/score")
def score(payload: ScoreIn):
    p = overlap_percent(payload.text)
    th = float(payload.threshold_percent)
    return {
        "overlap_percent": round(p, 1),
        "threshold_percent": th,
        "alert": p >= th,
    }

@app.post("/reload")
def reload_assets():
    global BANK
    BANK = load_bank(HASH_BANK_PATH)
    return {"ok": True, "bank_size": len(BANK)}
