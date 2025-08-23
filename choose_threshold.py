# choose_threshold.py
import pandas as pd
from score_chat import scam_overlap_percent

df = pd.read_csv("dataset.csv")  # columns: text,label  (label in {'scam','normal'})

for th in [10, 15, 20, 25, 30]:
    tp = fp = tn = fn = 0
    for _, row in df.iterrows():
        p = scam_overlap_percent(row["text"])
        pred = p >= th
        is_scam = (row["label"] == "scam")
        tp += int(pred and is_scam)
        fp += int(pred and not is_scam)
        tn += int((not pred) and (not is_scam))
        fn += int((not pred) and is_scam)
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec  = tp/(tp+fn) if (tp+fn) else 0
    print(f"TH={th}%  Precision={prec:.2f}  Recall={rec:.2f}  TP={tp} FP={fp} TN={tn} FN={fn}")
