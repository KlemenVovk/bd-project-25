import os
from typing import Dict, List, Tuple, Iterable, Optional
from collections import Counter
from glob import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,            # kept for console info if you want it
    balanced_accuracy_score,   # <-- macro avg accuracy
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.feature_extraction import FeatureHasher

import xgboost as xgb

from utils.constants import TASK2_OUT_ROOT, LATEX_ROOT, RESULTS_ROOT

FIG_DIR = os.path.join(LATEX_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)

SERVICE_DIR = {
    "Yellow": "yellow_tripdata",
    "Green":  "green_tripdata",
    "FHV":    "fhv_tripdata",
    "FHVHV":  "fhvhv_tripdata",
}

# classes used (FHV excluded: often lacks trip_distance)
CLASSES = ["Yellow", "Green", "FHVHV"]
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}

# features
NUM_COLS = ["trip_distance", "duration_min", "speed_mph", "hour", "dow"]
CAT_COLS = ["pickup_borough", "dropoff_borough"]  # od_pair removed
N_HASH = 2**15
HASH_ALT_SIGN = False
HASHED_LABEL = "Borough categories (hashed total)"   # <-- nicer label

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")

# streaming-friendly transformers
scaler = StandardScaler(with_mean=True)  # fit once, then transform batches
hasher = FeatureHasher(n_features=N_HASH, input_type="string",
                       alternate_sign=HASH_ALT_SIGN)

# SGD config
SGD_KW = dict(
    loss="log_loss",
    penalty="elasticnet", l1_ratio=0.15,
    learning_rate="constant", eta0=0.03,
    average=True,
    random_state=RANDOM_STATE,
)
N_EPOCHS = 2

# XGBoost config (incremental boosting across batches)
XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=len(CLASSES),
    tree_method=os.environ.get("XGB_TREE", "hist"),   # set to "gpu_hist" if you have a GPU
    max_depth=6,
    eta=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    seed=RANDOM_STATE,
)
XGB_ROUNDS_PER_BATCH = 50

# ---------- helpers ----------
def _present_columns(fp: str) -> List[str]:
    try:
        return pq.read_schema(fp).names
    except Exception:
        return []

def _read_part(fp: str, service: str, cap: Optional[int] = None) -> pd.DataFrame:
    want = ["pickup_datetime","dropoff_datetime","pickup_borough","dropoff_borough","trip_distance"]
    avail = set(_present_columns(fp))
    cols = [c for c in want if c in avail]
    if not cols:
        return pd.DataFrame()
    df = pd.read_parquet(fp, engine="pyarrow", columns=cols)
    if df.empty:
        return df
    if cap and len(df) > cap:
        df = df.sample(n=cap, random_state=RANDOM_STATE)
    df["service"] = service
    return df

def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["pickup_datetime","dropoff_datetime","pickup_borough","dropoff_borough","trip_distance","service"]:
        if c not in df.columns: df[c] = np.nan

    dt_pu = pd.to_datetime(df["pickup_datetime"])
    dt_do = pd.to_datetime(df["dropoff_datetime"])
    dur_min = (dt_do - dt_pu).dt.total_seconds() / 60.0
    df["duration_min"] = np.clip(dur_min, 0.1, 6*60)

    with np.errstate(divide="ignore", invalid="ignore"):
        speed = df["trip_distance"].astype("float32").to_numpy() / np.where(df["duration_min"] > 0, df["duration_min"]/60.0, np.nan)
    df["speed_mph"] = np.clip(np.where(np.isfinite(speed), speed, np.nan), 0, 120)

    df["hour"] = dt_pu.dt.hour.astype("Int16")
    df["dow"]  = dt_pu.dt.dayofweek.astype("Int16")

    df["pickup_borough"]  = df["pickup_borough"].astype("string").fillna("NA")
    df["dropoff_borough"] = df["dropoff_borough"].astype("string").fillna("NA")

    return df[["service"] + NUM_COLS + CAT_COLS]

def _design(df: pd.DataFrame) -> Tuple[sparse.csr_matrix, np.ndarray]:
    # numeric
    Xn = df[NUM_COLS].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xn = scaler.transform(Xn).astype("float32")
    Xn = sparse.csr_matrix(Xn)
    # hashed categoricals (PU/DO only)
    tokens = [[f"PU:{a}", f"DO:{b}"] for a,b in df[CAT_COLS].itertuples(index=False)]
    Xc = hasher.transform(tokens).tocsr().astype("float32")
    # stack
    X = sparse.hstack([Xn, Xc], format="csr")
    y = df["service"].map(CLASS2ID).to_numpy(dtype=np.int32, na_value=-1)
    return X, y

# ---------- exact-balanced training batches (caps optional) ----------
def balanced_batches(files_by_service: Dict[str, List[str]],
                     cap_per_file: Optional[Dict[str, int]] = None) -> Iterable[pd.DataFrame]:
    services = list(files_by_service.keys())
    max_len = min([len(files_by_service.get(s, [])) for s in services]) if services else 0
    cap_per_file = cap_per_file or {}
    for i in range(max_len):
        per_service = {}
        for svc in services:
            fp = files_by_service[svc][i]
            df = _read_part(fp, svc, cap=cap_per_file.get(svc))
            df = _engineer(df)
            if df.empty:
                per_service = {}
                break
            per_service[svc] = df
        if len(per_service) != len(services):
            continue
        n_min = min(len(d) for d in per_service.values())
        if n_min == 0:
            continue
        balanced_parts = [
            d.sample(n=n_min, random_state=RANDOM_STATE) if len(d) > n_min else d
            for d in per_service.values()
        ]
        batch = pd.concat(balanced_parts, ignore_index=True)
        batch = batch.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        yield batch

# ---------- unbalanced validation streaming (caps optional) ----------
def stream_validation(files_by_service: Dict[str, List[str]],
                      cap_per_file: Optional[Dict[str, int]] = None) -> Iterable[pd.DataFrame]:
    cap_per_file = cap_per_file or {}
    for svc, files in files_by_service.items():
        for fp in files:
            df = _engineer(_read_part(fp, svc, cap=cap_per_file.get(svc)))
            if not df.empty:
                yield df

# ---------- majority class from parquet metadata ----------
def majority_class_from_files(train_files: Dict[str, List[str]]) -> str:
    counts = {}
    for svc, files in train_files.items():
        total = 0
        for fp in files:
            try:
                total += pq.ParquetFile(fp).metadata.num_rows
            except Exception:
                pass
        counts[svc] = total
    maj = max(counts.items(), key=lambda kv: kv[1])[0] if counts else CLASSES[-1]
    print("Training row estimates (metadata):", counts)
    print("Majority baseline (train):", maj)
    return maj

# ---------- training / evaluation ----------
def train_and_eval(
    train_files: Dict[str, List[str]],
    valid_files: Dict[str, List[str]],
    cap_per_file: Optional[Dict[str, int]] = None,
):
    # Pass 1: fit scaler on balanced batches
    seen = Counter()
    for batch in balanced_batches(train_files, cap_per_file=cap_per_file):
        Xn = batch[NUM_COLS].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scaler.partial_fit(Xn)
        seen.update(batch["service"].tolist())
    print("Seen per-class rows (balanced train batches):", dict(seen))

    # Pass 2: train SGD and XGB together
    clf = SGDClassifier(**SGD_KW)
    inited = False
    booster = None

    for epoch in range(N_EPOCHS):
        for batch in tqdm(balanced_batches(train_files, cap_per_file=cap_per_file)):
            Xb, yb = _design(batch)
            m = (yb >= 0)
            if not np.any(m):
                continue
            Xb, yb = Xb[m], yb[m]

            # SGD
            if not inited:
                clf.partial_fit(Xb, yb, classes=np.arange(len(CLASSES)))
                inited = True
            else:
                clf.partial_fit(Xb, yb)

            # XGBoost — add trees per batch
            dtrain = xgb.DMatrix(Xb, label=yb)
            booster = xgb.train(
                params=XGB_PARAMS,
                dtrain=dtrain,
                num_boost_round=XGB_ROUNDS_PER_BATCH,
                xgb_model=booster,
            )

    # --------- EVAL on unbalanced validation ---------
    y_true_all, y_pred_sgd_all, y_pred_xgb_all = [], [], []
    for dfv in stream_validation(valid_files, cap_per_file=cap_per_file):
        Xv, yv = _design(dfv)
        m = (yv >= 0)
        if not np.any(m):
            continue
        Xv, yv = Xv[m], yv[m]
        y_pred_sgd_all.append(clf.predict(Xv))
        y_pred_xgb_all.append(booster.predict(xgb.DMatrix(Xv)).argmax(axis=1))
        y_true_all.append(yv)

    if not y_true_all:
        print("No validation data — check file lists.")
        return clf, booster

    y_true = np.concatenate(y_true_all)
    y_sgd  = np.concatenate(y_pred_sgd_all)
    y_xgb  = np.concatenate(y_pred_xgb_all)

    # Majority baseline (ignores caps for a strong baseline)
    maj_label = majority_class_from_files(train_files)
    maj_id = CLASS2ID[maj_label]
    y_maj = np.full_like(y_true, maj_id)

    def summarize(name, yhat, fig_suffix):
        # macro-averaged accuracy = balanced accuracy
        macro_acc = balanced_accuracy_score(y_true, yhat)
        macro_f1  = f1_score(y_true, yhat, average="macro")
        acc_raw   = accuracy_score(y_true, yhat)  # optional console info

        print(f"\n{name}: macro-accuracy={macro_acc:.3f}, macro-F1={macro_f1:.3f} (raw acc={acc_raw:.3f})")
        print(classification_report(y_true, yhat, target_names=CLASSES, digits=3))

        # normalized confusion matrix (row-normalized), .2f annotations, PDF only
        cm_norm = confusion_matrix(y_true, yhat,
                                   labels=np.arange(len(CLASSES)),
                                   normalize="true")
        cm_df = pd.DataFrame(cm_norm,
                             index=[f"true_{c}" for c in CLASSES],
                             columns=[f"pred_{c}" for c in CLASSES])
        plt.figure(figsize=(6.4, 5.2))
        sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0)
        plt.title(f"Service classifier — {name} (normalized confusion)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"t7_cm_{fig_suffix}.pdf"))
        plt.close()
        return macro_acc, macro_f1

    acc_maj, f1_maj = summarize("Majority baseline", y_maj, "baseline_majority")
    acc_sgd, f1_sgd = summarize("SGD (partial_fit, balanced train)", y_sgd, "sgd_partialfit_balanced")
    acc_xgb, f1_xgb = summarize("XGBoost (incremental, balanced train)", y_xgb, "xgb_incremental_balanced")

    # ----- Feature importance -----
    # SGD: numeric explicit + hashed cats aggregated (renamed)
    if hasattr(clf, "coef_"):
        coef = clf.coef_
        imp = np.linalg.norm(coef, axis=0)
        n_num = len(NUM_COLS)
        fi_sgd = (pd.DataFrame({
            "feature": NUM_COLS + [HASHED_LABEL],
            "importance": list(imp[:n_num]) + [float(imp[n_num:].sum())],
        }).sort_values("importance", ascending=False))
        fi_sgd.to_csv(os.path.join(RESULTS_ROOT, "t7_feat_importance_sgd_balanced.csv"), index=False)
        plt.figure(figsize=(7.2, 3.8))
        sns.barplot(data=fi_sgd.sort_values("importance", ascending=True),
                    x="importance", y="feature", orient="h")
        plt.title("Feature importance — SGD (numeric explicit; hashed cats aggregated)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "t7_featimp_sgd_balanced.pdf"))
        plt.close()

    # XGB: gain-based; map f0.. to column indices (renamed)
    if booster is not None:
        gains = np.zeros(len(NUM_COLS) + N_HASH, dtype=float)
        for k, v in booster.get_score(importance_type="gain").items():
            idx = int(k[1:])  # 'f123' -> 123
            if 0 <= idx < gains.size:
                gains[idx] += float(v)
        fi_xgb = (pd.DataFrame({
            "feature": NUM_COLS + [HASHED_LABEL],
            "importance": list(gains[:len(NUM_COLS)]) + [float(gains[len(NUM_COLS):].sum())],
        }).sort_values("importance", ascending=False))
        fi_xgb.to_csv(os.path.join(RESULTS_ROOT, "t7_feat_importance_xgb_balanced.csv"), index=False)
        plt.figure(figsize=(7.2, 3.8))
        sns.barplot(data=fi_xgb.sort_values("importance", ascending=True),
                    x="importance", y="feature", orient="h")
        plt.title("Feature importance — XGBoost (numeric explicit; hashed cats aggregated)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "t7_featimp_xgb_balanced.pdf"))
        plt.close()

    # Save summary — macro avg accuracy everywhere
    pd.DataFrame([
        {"model": "Majority baseline", "macro_accuracy": acc_maj, "macro_f1": f1_maj},
        {"model": "SGD(partial_fit, balanced)", "macro_accuracy": acc_sgd, "macro_f1": f1_sgd},
        {"model": "XGB(incremental, balanced)", "macro_accuracy": acc_xgb, "macro_f1": f1_xgb},
    ]).to_csv(os.path.join(RESULTS_ROOT, "t7_service_classifier_results.csv"), index=False)

    return clf, booster

# ---------- convenience ----------
def year_parts(service: str, year: int) -> List[str]:
    ydir = os.path.join(TASK2_OUT_ROOT, SERVICE_DIR[service], f"year={year}")
    if not os.path.isdir(ydir):
        return []
    return sorted(glob(os.path.join(ydir, "part.*.parquet")))

if __name__ == "__main__":
    train_files = {
        "Yellow": year_parts("Yellow", 2023),
        "Green":  year_parts("Green",  2023),
        "FHVHV":  year_parts("FHVHV",  2023),
    }
    valid_files = {
        "Yellow": year_parts("Yellow", 2024),
        "Green":  year_parts("Green",  2024),
        "FHVHV":  year_parts("FHVHV",  2024),
    }

    # Optional caps for quick prototyping:
    caps = {"FHVHV": 150_000, "Yellow": 150_000, "Green": 150_000}
    # caps = None

    sgd, booster = train_and_eval(train_files, valid_files, cap_per_file=caps)
    print("\n[OK] Done.")
