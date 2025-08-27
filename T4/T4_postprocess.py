import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import RESULTS_ROOT, LATEX_ROOT

# ---- Setup ----
sns.set_theme()

FIG_DIR = os.path.join(LATEX_ROOT, "figures")
TAB_DIR = os.path.join(LATEX_ROOT, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

FILES = {
    "Yellow": os.path.join(RESULTS_ROOT, "yellow_tripdata_aggregates.json"),
    "Green":  os.path.join(RESULTS_ROOT, "green_tripdata_aggregates.json"),
    "FHV":    os.path.join(RESULTS_ROOT, "fhv_tripdata_aggregates.json"),
    "FHVHV":  os.path.join(RESULTS_ROOT, "fhvhv_tripdata_aggregates.json"),
}

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

# Load aggregates once
agg = {svc: load_json(p) for svc, p in FILES.items() if load_json(p) is not None}
services = ["Yellow", "Green", "FHV", "FHVHV"]
services_present = [s for s in services if s in agg]

# =============================================================================
# 1) Yearly market share by service (100% stacked area)
# =============================================================================
frames = []
for svc in services_present:
    recs = agg[svc].get("trips_per_year", [])
    if not recs:
        continue
    dfy = pd.DataFrame(recs)
    if dfy.empty or "year" not in dfy or "num_trips" not in dfy:
        continue
    dfy = dfy.groupby("year", as_index=False)["num_trips"].sum()
    dfy["service"] = svc
    frames.append(dfy)

if frames:
    yearly = pd.concat(frames, ignore_index=True)
    wide = (yearly.pivot_table(index="year", columns="service", values="num_trips", aggfunc="sum")
                    .reindex(columns=services_present).fillna(0).sort_index())
    totals = wide.sum(axis=1).replace(0, np.nan)
    shares = (wide.T / totals).T.fillna(0.0) * 100.0

    plt.figure(figsize=(10.5, 4.2))
    x = shares.index.values
    y_stack = [shares[c].values for c in shares.columns]
    plt.stackplot(x, y_stack, labels=list(shares.columns))
    plt.ylim(0, 100)
    plt.ylabel("Share of trips (%)")
    plt.xlabel("Year")
    if len(x) > 0:
        plt.xlim(x[0], x[-1])
    plt.title("Yearly market share by service (100%)")
    plt.legend(title="Service", loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "t4_yearly_market_share.pdf"))
    plt.close()

    shares.round(2).to_csv(os.path.join(RESULTS_ROOT, "t4_yearly_market_share.csv"))

# =============================================================================
# 2) Top-5 OD borough flows per service (2×2 bars)
# =============================================================================
def top5_od_for_service(svc: str) -> pd.DataFrame:
    recs = agg.get(svc, {}).get("top10_od_boroughs", [])
    if not recs:
        return pd.DataFrame(columns=["label", "trips"])
    df = pd.DataFrame(recs).rename(columns={"pickup_borough": "pu", "dropoff_borough": "do"})
    if {"pu", "do", "num_trips"} - set(df.columns):
        return pd.DataFrame(columns=["label", "trips"])
    df = df.nlargest(5, "num_trips").copy()
    df["label"] = df["pu"].fillna("Unknown") + " \u2192 " + df["do"].fillna("Unknown")
    df["trips"] = df["num_trips"].astype(int)
    return df[["label", "trips"]]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()
for ax, svc in zip(axes, services):
    if svc in services_present:
        d = top5_od_for_service(svc)
        if not d.empty:
            d = d.iloc[::-1]  # largest at top
            ax.barh(d["label"], d["trips"])
            ax.set_title(f"{svc}: Top-5 borough OD flows")
            ax.set_xlabel("Trips")
            ax.set_ylabel("")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
    else:
        ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "t4_top5_od_bars.pdf"))
plt.close()

# =============================================================================
# 3) Pickup borough shares (100% stacked bars)
# =============================================================================
bshares = []
for svc in services_present:
    b = pd.DataFrame(agg.get(svc, {}).get("trips_per_borough", []))
    if b.empty:
        continue
    tot = b["num_trips"].sum()
    if tot <= 0:
        continue
    b["share"] = b["num_trips"] / tot
    b["service"] = svc
    b = b.rename(columns={"pickup_borough": "borough"})
    bshares.append(b[["service", "borough", "share"]])

if bshares:
    bs = pd.concat(bshares, ignore_index=True)
    all_boroughs = sorted(bs["borough"].unique())
    bs_pivot = (bs.pivot_table(index="service", columns="borough", values="share", aggfunc="sum")
                  .reindex(services_present).reindex(columns=all_boroughs).fillna(0.0))
    ax = bs_pivot.plot(kind="bar", stacked=True, figsize=(10.5, 4.2), width=0.8)
    ax.set_ylabel("Share of pickups")
    ax.set_xlabel("")
    ax.set_title("Pickup borough shares (100% stacked)")
    plt.legend(title="Borough", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "t4_borough_shares_stacked.pdf"))
    plt.close()

# =============================================================================
# 4) Cosine similarity of pickup borough shares (optional matrix)
# =============================================================================
def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

bvec = {}
if bshares:
    all_b = sorted(
        set(pd.concat(
            [pd.DataFrame(agg.get(s, {}).get("trips_per_borough", []))
             for s in services_present if agg.get(s, {}).get("trips_per_borough")],
            ignore_index=True)["pickup_borough"].unique())
    )
    for svc in services_present:
        b = pd.DataFrame(agg.get(svc, {}).get("trips_per_borough", []))
        if b.empty:
            continue
        v = (b.set_index("pickup_borough")["num_trips"]
               .reindex(all_b).fillna(0.0).values)
        s = v.sum()
        bvec[svc] = (v / s) if s > 0 else v

if bvec:
    svc_present = list(bvec.keys())
    C = np.full((len(svc_present), len(svc_present)), np.nan)
    for i, a in enumerate(svc_present):
        for j, b in enumerate(svc_present):
            C[i, j] = cosine(bvec[a], bvec[b])
    plt.figure(figsize=(5.6, 4.6))
    sns.heatmap(pd.DataFrame(C, index=svc_present, columns=svc_present),
                annot=True, fmt=".2f", vmin=0, vmax=1)
    plt.title("Cosine similarity of pickup borough shares")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "t4_similarity_cosine_borough.pdf"))
    plt.close()

# =============================================================================
# 5) Tips, fares, AND duration — single LaTeX table
# =============================================================================
def extract_first_median(record_list):
    """Return first value from a list[dict] where key contains 'median'."""
    if not isinstance(record_list, list) or not record_list:
        return None
    d0 = record_list[0]
    for k, v in d0.items():
        if "median" in k.lower():
            return v
    return None

rows = []
for svc in services:
    d = agg.get(svc)
    if d is None:
        continue

    # Tipped share (%)
    tipped_share = None
    if isinstance(d.get("tip_percentage"), list):
        for item in d["tip_percentage"]:
            if str(item.get("type", "")).lower().startswith("tipped"):
                try:
                    tipped_share = float(item.get("percentage"))
                except Exception:
                    tipped_share = None
                break

    # Median tip (Yellow/Green: tip_amount; FHVHV: tips)
    med_tip = extract_first_median(d.get("median_tip"))

    # Median fare (Yellow/Green: fare_amount; FHVHV: base_passenger_fare)
    med_fare = extract_first_median(d.get("median_fare"))

    # Median duration (seconds → minutes)
    med_dur_sec = extract_first_median(d.get("median_duration"))
    med_dur_min = (float(med_dur_sec) / 60.0) if med_dur_sec is not None else None

    rows.append({
        "Service": svc,
        "Tipped share [\\%]": f"{tipped_share:.1f}" if tipped_share is not None else "",
        "Median tip [\\$]":   f"{float(med_tip):.2f}"  if med_tip  is not None else "",
        "Median fare [\\$]":  f"{float(med_fare):.2f}" if med_fare is not None else "",
        "Median duration [min]": f"{med_dur_min:.1f}"  if med_dur_min is not None else "",
    })

tf_tbl = pd.DataFrame(rows)
tf_tbl.to_latex(
    os.path.join(TAB_DIR, "t4_tips_fares_duration_summary.tex"),
    index=False,
    escape=True,
    column_format="lrrrr",
    caption="Tips, fares, and duration summary: tipped share (\\%), median tip, median fare, and median duration in minutes (where available).",
    label="tab:t4_tips_fares_duration"
)

print("[OK] Figures →", FIG_DIR)
print("[OK] Tables  →", TAB_DIR)
