# score_chat.py
import re, hashlib, pandas as pd

# --- same helpers as before ---
def normalize(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9$€£\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def h(ng):
    return hashlib.md5(" ".join(ng).encode("utf-8")).hexdigest()

# --- load your hash bank built earlier ---
BANK = set(pd.read_csv("/Users/h1mal/hash_bank.csv")["hash"].tolist())

def scam_overlap_percent(text: str) -> float:
    w = normalize(text).split()
    hashed = []
    for n in (2, 3):  # bi+tri-grams
        for ng in ngrams(w, n):
            hashed.append(h(ng))
    if not hashed:
        return 0.0
    hits = sum(1 for x in hashed if x in BANK)
    return 100.0 * hits / len(hashed)

def score_and_alert(text: str, threshold_percent: float = 20.0):
    p = scam_overlap_percent(text)
    return {"overlap_percent": round(p, 1), "alert": p >= threshold_percent}

if __name__ == "__main__":
    # quick manual test
    chat = input("Paste chat: ")
    print(score_and_alert(chat, threshold_percent=20.0))
