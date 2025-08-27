# t7_service_classifier_experiment.py
import os
from typing import Dict, List, Tuple, Iterable, Optional
from collections import Counter
from glob import glob
import time
import argparse

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
    accuracy_score,
    balanced_accuracy_score,   # macro-averaged accuracy
    f1_score,
    confusion_matrix,
    log_loss,
)
from sklearn.feature_extraction import FeatureHasher
import xgboost as xgb
import psutil

from utils.constants import TASK2_OUT_ROOT, LATEX_ROOT, RESULTS_ROOT

# ---------- IO / paths ----------
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

# features (behavior only; no leakage)
NUM_COLS = ["trip_distance", "duration_min", "speed_mph", "hour", "dow"]
CAT_COLS = ["pickup_borough", "dropoff_borough"]  # od_pair removed
N_HASH = 2**15
HASH_ALT_SIGN = False
HASHED_LABEL = "Borough categories (hashed total)"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")

# streaming-friendly transformers
scaler = StandardScaler(with_mean=True)  # fit once, then transform batches
hasher = FeatureHasher(n_features=N_HASH, input_type="string",
                       alternate_sign=HASH_ALT_SIGN)

# SGD config (kept, but will be skipped if --only_xgb)
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
    tree_method=os.environ.get("XGB_TREE", "hist"),  # set "gpu_hist" if GPU
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

def stream_validation(files_by_service: Dict[str, List[str]],
                      cap_per_file: Optional[Dict[str, int]] = None) -> Iterable[pd.DataFrame]:
    cap_per_file = cap_per_file or {}
    for svc, files in files_by_service.items():
        for fp in files:
            df = _engineer(_read_part(fp, svc, cap=cap_per_file.get(svc)))
            if not df.empty:
                yield df

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
    train_xgb: bool = False,
    only_xgb: bool = False,
    exp_name: str = "",
    xgb_threads: int = 1,
    xgb_rounds: int = 50,
):
    # wire XGB knobs
    XGB_PARAMS["nthread"] = int(xgb_threads)
    global XGB_ROUNDS_PER_BATCH
    XGB_ROUNDS_PER_BATCH = int(xgb_rounds)

    # Pass 1: fit scaler on balanced batches
    seen = Counter()
    for batch in balanced_batches(train_files, cap_per_file=cap_per_file):
        Xn = batch[NUM_COLS].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scaler.partial_fit(Xn)
        seen.update(batch["service"].tolist())
    print("Seen per-class rows (balanced train batches):", dict(seen))

    # resource trackers
    proc = psutil.Process(os.getpid())
    peak_rss = proc.memory_info().rss
    n_batches = 0

    clf = None if only_xgb else SGDClassifier(**SGD_KW)
    inited = False
    booster = None
    start_train_time = time.time()

    for epoch in range(N_EPOCHS):
        for batch in tqdm(balanced_batches(train_files, cap_per_file=cap_per_file)):
            n_batches += 1
            Xb, yb = _design(batch)
            m = (yb >= 0)
            if not np.any(m):
                continue
            Xb, yb = Xb[m], yb[m]

            # SGD (skip entirely if only_xgb)
            if clf is not None:
                if not inited:
                    clf.partial_fit(Xb, yb, classes=np.arange(len(CLASSES)))
                    inited = True
                else:
                    clf.partial_fit(Xb, yb)

            # XGB (optional): add trees per batch
            if train_xgb or only_xgb:
                dtrain = xgb.DMatrix(Xb, label=yb)
                booster = xgb.train(
                    params=XGB_PARAMS,
                    dtrain=dtrain,
                    num_boost_round=XGB_ROUNDS_PER_BATCH,
                    xgb_model=booster if booster is not None else None,
                )

            # peak memory after each batch
            rss_now = proc.memory_info().rss
            if rss_now > peak_rss:
                peak_rss = rss_now

    training_time = time.time() - start_train_time

    # --------- EVAL on unbalanced validation ---------
    y_true_all, y_pred_sgd_all, y_pred_xgb_all = [], [], []
    y_proba_sgd_all, y_proba_xgb_all = [], []

    for dfv in stream_validation(valid_files, cap_per_file=cap_per_file):
        Xv, yv = _design(dfv)
        m = (yv >= 0)
        if not np.any(m):
            continue
        Xv, yv = Xv[m], yv[m]

        if clf is not None:
            y_pred_sgd_all.append(clf.predict(Xv))
            if hasattr(clf, "predict_proba"):
                y_proba_sgd_all.append(clf.predict_proba(Xv))

        if (train_xgb or only_xgb) and booster is not None:
            proba = booster.predict(xgb.DMatrix(Xv))
            y_pred_xgb_all.append(proba.argmax(axis=1))
            y_proba_xgb_all.append(proba)

        y_true_all.append(yv)

    if not y_true_all:
        print("No validation data — check file lists.")
        return clf, booster, training_time

    y_true = np.concatenate(y_true_all)
    # prepare predictions present
    y_sgd = np.concatenate(y_pred_sgd_all) if y_pred_sgd_all else None
    proba_sgd = np.vstack(y_proba_sgd_all) if y_proba_sgd_all else None
    if y_pred_xgb_all:
        y_xgb = np.concatenate(y_pred_xgb_all)
        proba_xgb = np.vstack(y_proba_xgb_all)

    # Majority baseline (metadata-based)
    maj_label = majority_class_from_files(train_files)
    maj_id = CLASS2ID[maj_label]
    y_maj = np.full_like(y_true, maj_id)
    proba_maj = np.zeros((y_true.shape[0], len(CLASSES)), dtype=float)
    proba_maj[:, maj_id] = 1.0

    def summarize(name, yhat, fig_suffix, proba=None):
        macro_acc = balanced_accuracy_score(y_true, yhat)
        macro_f1  = f1_score(y_true, yhat, average="macro")
        err_rate  = 1.0 - macro_acc
        acc_raw   = accuracy_score(y_true, yhat)

        # log loss if probabilities are provided
        ll = np.nan
        if proba is not None:
            try:
                ll = log_loss(y_true, proba, labels=np.arange(len(CLASSES)))
            except Exception:
                ll = np.nan

        print(f"\n{name}: macro-accuracy={macro_acc:.3f}, macro-F1={macro_f1:.3f}, "
              f"raw acc={acc_raw:.3f}, log-loss={ll if np.isnan(ll) else round(ll,3)}")

        # normalized confusion matrix (row-normalized); PDF only
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
        return macro_acc, macro_f1, err_rate, ll

    acc_maj, f1_maj, err_maj, ll_maj = summarize(
        "Majority baseline", y_maj, "baseline_majority", proba=proba_maj
    )

    if y_sgd is not None:
        acc_sgd, f1_sgd, err_sgd, ll_sgd = summarize(
            "SGD (partial_fit, balanced train)", y_sgd, "sgd_partialfit_balanced", proba=proba_sgd
        )

    if (train_xgb or only_xgb) and booster is not None:
        acc_xgb, f1_xgb, err_xgb, ll_xgb = summarize(
            "XGBoost (incremental, balanced train)", y_xgb, "xgb_incremental_balanced", proba=proba_xgb
        )

    # ----- Feature importance -----
    if (not only_xgb) and (clf is not None) and hasattr(clf, "coef_"):
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

    if (train_xgb or only_xgb) and booster is not None:
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

    # ----- Unified summary table (append one row per model) -----
    summary_path = os.path.join(RESULTS_ROOT, "t7_experiment_summary.csv")
    cap_val = None
    if cap_per_file:
        cap_vals = list(set(cap_per_file.values()))
        cap_val = cap_vals[0] if cap_vals else None

    def append_row(model, macro_acc, macro_f1, err_rate, ll):
        row = {
            "exp_name": exp_name or "run",
            "model": model,
            "training_time_sec": training_time,
            "peak_rss_mb": peak_rss / (1024**2),
            "macro_accuracy": macro_acc,
            "error_rate": err_rate,
            "macro_f1": macro_f1,
            "log_loss": ll,
            "cap": cap_val,
            "epochs": N_EPOCHS,
            "n_batches": n_batches,
            "train_rows": int(sum(seen.values())),
            "xgb_threads": xgb_threads if (train_xgb or only_xgb) else None,
            "xgb_rounds_per_batch": xgb_rounds if (train_xgb or only_xgb) else None,
        }
        pd.DataFrame([row]).to_csv(
            summary_path, index=False, mode="a", header=not os.path.exists(summary_path)
        )

    append_row("Majority baseline", acc_maj, f1_maj, err_maj, ll_maj)
    if y_sgd is not None:
        append_row("SGD(partial_fit, balanced)", acc_sgd, f1_sgd, err_sgd, ll_sgd)
    if (train_xgb or only_xgb) and booster is not None:
        append_row("XGB(incremental, balanced)", acc_xgb, f1_xgb, err_xgb, ll_xgb)

    # Per-run CSV (optional)
    rows = [{"model": "Majority baseline", "macro_accuracy": acc_maj, "macro_f1": f1_maj}]
    if y_sgd is not None:
        rows.append({"model": "SGD(partial_fit, balanced)", "macro_accuracy": acc_sgd, "macro_f1": f1_sgd})
    if (train_xgb or only_xgb) and booster is not None:
        rows.append({"model": "XGB(incremental, balanced)", "macro_accuracy": acc_xgb, "macro_f1": f1_xgb})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_ROOT, f"t7_service_classifier_{exp_name or 'experiment'}.csv"),
                              index=False)

    return clf, booster, training_time

# ---------- convenience ----------
def year_parts(service: str, year: int) -> List[str]:
    ydir = os.path.join(TASK2_OUT_ROOT, SERVICE_DIR[service], f"year={year}")
    if not os.path.isdir(ydir):
        return []
    return sorted(glob(os.path.join(ydir, "part.*.parquet")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb", action="store_true", default=False, help="Also train XGBoost incrementally")
    parser.add_argument("--only_xgb", action="store_true", default=False, help="Train ONLY XGBoost (skip SGD)")
    parser.add_argument("--cap", type=int, default=10_000, help="Cap per-file rows for train/valid (quick tests). Set 0 to disable.")
    parser.add_argument("--exp_name", type=str, default="experiment", help="Name of the experiment")
    parser.add_argument("--train_year", type=int, default=2023)
    parser.add_argument("--valid_year", type=int, default=2024)
    parser.add_argument("--xgb_threads", type=int, default=1, help="XGBoost CPU threads (nthread)")
    parser.add_argument("--xgb_rounds", type=int, default=50, help="XGBoost boosting rounds per batch")
    args = parser.parse_args()

    train_files = {
        "Yellow": year_parts("Yellow", args.train_year),
        "Green":  year_parts("Green",  args.train_year),
        "FHVHV":  year_parts("FHVHV",  args.train_year),
    }
    valid_files = {
        "Yellow": year_parts("Yellow", args.valid_year),
        "Green":  year_parts("Green",  args.valid_year),
        "FHVHV":  year_parts("FHVHV",  args.valid_year),
    }

    # Optional caps for quick prototyping; set --cap 0 to disable
    caps = {"FHVHV": args.cap, "Yellow": args.cap, "Green": args.cap} if args.cap and args.cap > 0 else None

    clf, booster, training_time = train_and_eval(
        train_files, valid_files,
        cap_per_file=caps,
        train_xgb=(args.xgb or args.only_xgb),
        only_xgb=args.only_xgb,
        exp_name=args.exp_name,
        xgb_threads=args.xgb_threads,
        xgb_rounds=args.xgb_rounds,
    )
    print(f"\n[OK] Done. Training time: {training_time:.2f} seconds")
