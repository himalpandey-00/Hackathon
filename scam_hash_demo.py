
import hashlib
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import gradio as gr

KEYWORDS = [
    "western union", "gift card", "anydesk", "remote access", "insurance payout",
    "reimburse", "wire transfer", "bank account", "bsb", "passport", "id", "bitcoin",
    "crypto", "mule", "voucher"
]

URGENCY = ["urgent", "now", "immediately", "asap", "today", "right away"]

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9$€£\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def ngrams(words: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def hash_ngram(ng: Tuple[str, ...]) -> str:
    joined = " ".join(ng)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()

def build_hash_bank(texts: List[str], top_k:int = 2000) -> Dict[str, int]:
    freq = Counter()
    for t in texts:
        tnorm = normalize(t)
        words = tnorm.split()
        for n in (2, 3):
            for ng in ngrams(words, n):
                freq[hash_ngram(ng)] += 1
    most_common = dict(freq.most_common(top_k))
    return most_common

def extract_features(text: str, hash_bank: Dict[str, int]) -> Dict[str, float]:
    tnorm = normalize(text)
    words = tnorm.split()
    all_hashes = []
    for n in (2, 3):
        for ng in ngrams(words, n):
            all_hashes.append(hash_ngram(ng))
    total = max(len(all_hashes), 1)
    hits = sum(1 for h in all_hashes if h in hash_bank)
    match_ratio = hits / total
    kw_flags = {f"kw_{k.replace(' ', '_')}": (1 if k in tnorm else 0) for k in KEYWORDS}
    urg_count = sum(1 for w in URGENCY if w in tnorm)
    length = len(words)
    feat = {
        "match_ratio": match_ratio,
        "urgency_count": urg_count,
        "length": length,
        "hit_count": hits,
        "ngram_total": total
    }
    feat.update(kw_flags)
    return feat

def df_from_features(texts: List[str], hash_bank: Dict[str,int]) -> pd.DataFrame:
    rows = [extract_features(t, hash_bank) for t in texts]
    return pd.DataFrame(rows)

def load_data(csv_path: str = "sample_chats.csv"):
    df = pd.read_csv(csv_path)
    assert {"text","label"}.issubset(df.columns), "CSV must have 'text' and 'label' columns"
    return df

def train_model(csv_path: str = "sample_chats.csv"):
    df = load_data(csv_path)
    pos_texts = df[df["label"]=="scam"]["text"].tolist()
    hash_bank = build_hash_bank(pos_texts, top_k=2000)
    X = df_from_features(df["text"].tolist(), hash_bank)
    y = (df["label"]=="scam").astype(int)
    num_cols = X.columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols)
    ], remainder="drop")
    clf = LogisticRegression(max_iter=200, class_weight="balanced", C=1.0)
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        X, y, df["text"].tolist(), test_size=0.3, random_state=42, stratify=y
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1]
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=["normal","scam"], zero_division=0)
    state = {
        "hash_bank": hash_bank,
        "model": pipe,
        "columns": X.columns.tolist(),
        "metrics": {"precision": float(p), "recall": float(r), "f1": float(f1)},
        "report": report
    }
    return state

STATE = train_model("sample_chats.csv")

def score_chat(chat_text: str, threshold: float = 0.5):
    feats = extract_features(chat_text, STATE["hash_bank"])
    X = pd.DataFrame([feats])[STATE["columns"]]
    proba = float(STATE["model"].predict_proba(X)[:,1][0])
    label = "Likely SCAM" if proba >= threshold else "Likely NORMAL"
    triggered = [k for k in feats if k.startswith("kw_") and feats[k]==1]
    base = {
        "risk_score": round(proba, 3),
        "label": label,
        "threshold": threshold,
        "triggered_keywords": triggered,
        "match_ratio": round(feats["match_ratio"], 3),
        "hit_count": int(feats["hit_count"]),
        "ngram_total": int(feats["ngram_total"]),
        "urgency_count": int(feats["urgency_count"]),
        "length_tokens": int(feats["length"])
    }
    return json.dumps(base, indent=2)

def ui_predict(chat_text, threshold):
    return score_chat(chat_text, threshold)

with gr.Blocks(title="Hash-Mapping Romance Scam Detector (Hackathon Demo)") as demo:
    gr.Markdown("# Hash-Mapping Romance Scam Detector (Hackathon Demo)")
    gr.Markdown("Paste any chat below. The demo uses hashed n-grams from known scam examples + a simple model to score scam likelihood.")
    with gr.Row():
        chat = gr.Textbox(label="Chat text", lines=8, placeholder="Paste your chat here...")
    threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Alert threshold")
    btn = gr.Button("Score Chat")
    out = gr.Code(label="Result (JSON)", language="json")
    btn.click(fn=ui_predict, inputs=[chat, threshold], outputs=[out])
    with gr.Accordion("Model Metrics", open=False):
        gr.Markdown(f"**Precision**: {STATE['metrics']['precision']:.3f}  \n"
                    f"**Recall**: {STATE['metrics']['recall']:.3f}  \n"
                    f"**F1**: {STATE['metrics']['f1']:.3f}")
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)  # remove server_port=7860

