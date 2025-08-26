import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.constants import RESULTS_ROOT, LATEX_ROOT

sns.set_style("whitegrid")

def add_hline(latex: str, index: int) -> str:
    """
    Insert \\midrule 'index' lines before the end of the tabular block.
    index=2 -> insert before the last two rows.
    """
    lines = latex.splitlines()
    # place before the '\\end{tabular}' and the last data row
    insert_at = max(0, len(lines) - index - 2)
    lines.insert(insert_at, r'\midrule')
    return '\n'.join(lines).replace('NaN', '')

# ------------------ Inputs ------------------

files = {
    "Yellow taxi": os.path.join(RESULTS_ROOT, "yellow_tripdata_quality_stats.csv"),
    "Green taxi": os.path.join(RESULTS_ROOT, "green_tripdata_quality_stats.csv"),
    "FHV": os.path.join(RESULTS_ROOT, "fhv_tripdata_quality_stats.csv"),
    "FHVHV": os.path.join(RESULTS_ROOT, "fhvhv_tripdata_quality_stats.csv")
}

# used ONLY for charts (visualization); tables aggregate across ALL years
cutoff_years = {
    "Yellow taxi": (2011, 2025),
    "Green taxi": (2013, 2025),
    "FHV": (2014, 2025),
    "FHVHV": (2019, 2025)
}

# candidate issue columns — a CSV may contain only a subset
CANDIDATE_BAD_COLS_ORDERED = [
    "bad_timestamps",
    "bad_distance",
    "bad_coordinates",
    "bad_prices",
    "bad_fares",
    "bad_request_time",
]

PRETTY = {
    "bad_timestamps": "Timestamp issues",
    "bad_distance": "Infeasible distances",
    "bad_coordinates": "Invalid coordinates",
    "bad_prices": "Pricing issues",
    "bad_fares": "Pricing issues",
    "bad_request_time": "Request time issues",
}

# ------------------ Outputs ------------------

tables_dir = os.path.join(LATEX_ROOT, "tables")
figs_dir   = os.path.join(LATEX_ROOT, "figures")
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(figs_dir,   exist_ok=True)

# ------------------ Processing ------------------

def process_dataset(name: str, csv_path: str, cutoff: tuple[int, int]) -> None:
    if not os.path.exists(csv_path):
        print(f"[WARN] Missing file for {name}: {csv_path}")
        return

    # Defensive yearly aggregation
    df = pd.read_csv(csv_path).groupby("year", as_index=False).sum().sort_values("year")

    # Which issue columns exist?
    bad_cols_present = [c for c in CANDIDATE_BAD_COLS_ORDERED if c in df.columns]

    # Ensure clean/invalid columns exist
    if "clean_trips" not in df.columns:
        # NOTE: deriving clean_trips from separate bad_* can undercount due to overlaps;
        # if clean_trips exists upstream, it will be used instead.
        df["clean_trips"] = df["total_trips"] - df[bad_cols_present].sum(axis=1)
    df["invalid_trips"] = df["total_trips"] - df["clean_trips"]

    # ===== TABLE (aggregate ALL years) =====
    sum_cols = ["total_trips", "clean_trips", "invalid_trips"] + bad_cols_present
    totals = df[sum_cols].sum()
    total_trips = totals["total_trips"] if totals["total_trips"] != 0 else 1

    rows = []
    # error rows in fixed order (only those present)
    for c in CANDIDATE_BAD_COLS_ORDERED:
        if c in bad_cols_present:
            pct = (totals[c] / total_trips) * 100.0
            rows.append({
                "Error type": PRETTY[c],
                "Count": int(totals[c]),
                "% of all": f"{round(pct)}%"
            })
    # summary rows
    rows.append({
        "Error type": "Invalid trips",
        "Count": int(totals["invalid_trips"]),
        "% of all": f"{round((totals['invalid_trips'] / total_trips) * 100.0)}%"
    })
    rows.append({
        "Error type": "Clean trips",
        "Count": int(totals["clean_trips"]),
        "% of all": f"{round((totals['clean_trips'] / total_trips) * 100.0)}%"
    })

    table_df = pd.DataFrame(rows, columns=["Error type", "Count", "% of all"])
    table_df["Count"] = table_df["Count"].map(lambda v: f"{int(v):,}")

    # Pandas → LaTeX (escape=True ensures % → \%)
    latex = table_df.to_latex(index=False, escape=True, column_format="lrr")
    latex = add_hline(latex, index=2)  # add \midrule before the last two rows
    table_fname = os.path.join(tables_dir, f"{name.lower().replace(' ', '_')}_dq_table.tex")

    with open(table_fname, "w") as f:
        f.write(latex)

    print(f"[OK] Saved LaTeX table for {name} → {table_fname}")

    # ===== GROUPED BAR CHART (cutoff years only) =====
    yr_min, yr_max = cutoff
    vis = df[(df["year"] >= yr_min) & (df["year"] <= yr_max)].copy()
    if vis.empty or not bad_cols_present:
        print(f"[INFO] No data to visualize for {name} in range {yr_min}-{yr_max}.")
        return

    # Build per-year percentages for present error types
    plot_rows = []
    for _, row in vis.iterrows():
        year = int(row["year"])
        denom = max(int(row["total_trips"]), 1)
        for c in bad_cols_present:
            plot_rows.append({
                "year": year,
                "Issue type": PRETTY[c],
                "% of trips": (row[c] / denom) * 100.0
            })
    plot_df = pd.DataFrame(plot_rows)

    # Seaborn grouped bar (one bar per error type per year)
    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=plot_df,
        y="year", x="% of trips", orient="h", hue="Issue type",
        errorbar=None, dodge=True,
    )
    plt.ylabel("Year", fontsize=14)
    plt.xlabel("% of all trips", fontsize=14)
    plt.legend(title="Issue type", fontsize=14)
    # set x and y ticks font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    fig_fname = os.path.join(figs_dir, f"{name.lower().replace(' ', '_')}_dq_grouped.pdf")
    plt.savefig(fig_fname)
    plt.close()
    print(f"[OK] Saved figure for {name} → {fig_fname}")

# ------------------ Run ------------------

for dataset, csv in files.items():
    process_dataset(dataset, csv, cutoff_years[dataset])