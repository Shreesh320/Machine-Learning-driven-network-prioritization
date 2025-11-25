import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="Input CSV file (flow rows)")
parser.add_argument("--out", default="flow_priority_model.pkl", help="Output model file")
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

print("Loading", args.csv)
df = pd.read_csv(args.csv)
needed = ["n_packets","n_bytes","duration_sec","throughput_bps","ip_proto","src_port","dst_port","priority_label"]
for c in needed:
    if c not in df.columns:
        raise SystemExit(f"Missing column: {c}")

df = df.replace("N/A", np.nan)
df = df.dropna(subset=["n_packets","n_bytes","duration_sec","throughput_bps","priority_label"])

# dataframe
X = pd.DataFrame()
X["n_packets"] = df["n_packets"].astype(float)
X["n_bytes"] = df["n_bytes"].astype(float)
X["duration_sec"] = df["duration_sec"].astype(float).replace(0, 1e-3)
X["throughput_bps"] = df["throughput_bps"].astype(float)
X["ip_proto"] = pd.to_numeric(df["ip_proto"].fillna(0), errors="coerce").fillna(0).astype(int)
X["src_port"] = pd.to_numeric(df["src_port"].fillna(0), errors="coerce").fillna(0).astype(int)
X["dst_port"] = pd.to_numeric(df["dst_port"].fillna(0), errors="coerce").fillna(0).astype(int)
X["bytes_per_pkt"] = X["n_bytes"] / X["n_packets"].replace(0,1)
X["is_tcp"] = (X["ip_proto"] == 6).astype(int)
X["is_udp"] = (X["ip_proto"] == 17).astype(int)

y = df["priority_label"].astype(int)

print("Samples:", len(X), "Positive ratio:", y.mean())
# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if len(y.unique())>1 else None)
# train
clf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=args.random_state, n_jobs=-1)
clf.fit(X_train, y_train)
# evaluate
pred = clf.predict(X_test)
print("=== Classification report ===")
print(classification_report(y_test, pred))
print("=== Confusion matrix ===")
print(confusion_matrix(y_test, pred))
# save
joblib.dump(clf, args.out)
print("Saved model to", args.out)
