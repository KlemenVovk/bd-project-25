import os
import timeit
import pandas as pd
import numpy as np
import duckdb
from utils.constants import TASK2_OUT_ROOT, TASK3_OUT_ROOT, RESULTS_ROOT, LATEX_ROOT

# ---------------- Setup ----------------
N_REPEATS = 10
os.makedirs(TASK3_OUT_ROOT, exist_ok=True)

# Source partition: Green 2024 (from Task 2 output)
df = pd.read_parquet(os.path.join(TASK2_OUT_ROOT, "green_tripdata", "year=2024"))

def _save_duckdb(path, df_in: pd.DataFrame):
    if os.path.exists(path):
        os.remove(path)
    con = duckdb.connect(path)
    con.register("df_src", df_in)
    con.execute("CREATE TABLE green2024 AS SELECT * FROM df_src")
    con.close()

def _read_duckdb(path) -> pd.DataFrame:
    con = duckdb.connect(path)
    out = con.execute("SELECT * FROM green2024").df()
    con.close()
    return out

# Registry of formats
filename2format = {
    "green_2024.parquet": {
        "save": lambda path: df.to_parquet(path, index=False),
        "read": lambda path: pd.read_parquet(path),
        "desc": "Parquet",
        "index": 0
    },
    "green_2024.csv": {
        "save": lambda path: df.to_csv(path, index=False),
        "read": lambda path: pd.read_csv(path),
        "desc": "CSV",
        "index": 1
    },
    "green_2024.csv.gz": {
        "save": lambda path: df.to_csv(path, index=False, compression="gzip"),
        "read": lambda path: pd.read_csv(path, compression="gzip"),
        "desc": "CSV (gzip)",
        "index": 2
    },
    "green_2024_no_compression.h5": {
        "save": lambda path: df.to_hdf(path, key="df", mode="w", format="table",
                                       complib=None, complevel=0, index=False),
        "read": lambda path: pd.read_hdf(path, key="df"),
        "desc": "HDF5",
        "index": 3
    },
    "green_2024_lz4.h5": {
        "save": lambda path: df.to_hdf(path, key="df", mode="w", format="table",
                                       complib="blosc2:lz4", complevel=9, index=False),
        "read": lambda path: pd.read_hdf(path, key="df"),
        "desc": "HDF5 (LZ4, lvl 9)",
        "index": 4
    },
    "green_2024_zstd.h5": {
        "save": lambda path: df.to_hdf(path, key="df", mode="w", format="table",
                                       complib="blosc2:zstd", complevel=9, index=False),
        "read": lambda path: pd.read_hdf(path, key="df"),
        "desc": "HDF5 (Zstd, lvl 9)",
        "index": 5
    },
    "green_2024.duckdb": {
        "save": lambda path: _save_duckdb(path, df),
        "read": lambda path: _read_duckdb(path),
        "desc": "DuckDB",
        "index": 6
    }
}

# ---------------- Run benchmark ----------------
results = []
for fname, ops in sorted(filename2format.items(), key=lambda kv: kv[1]["index"]):
    path = os.path.join(TASK3_OUT_ROOT, fname)

    # write
    ops["save"](path)

    # read timing
    timer = timeit.Timer(lambda: ops["read"](path))
    times = np.array(timer.repeat(repeat=N_REPEATS, number=1))

    results.append({
        "index": ops["index"],
        "filename": fname,
        "format": ops["desc"],
        "size_MB": os.path.getsize(path) / (1024 ** 2),
        "read_min_s": times.min(),
        "read_avg_s": times.mean(),
        "read_max_s": times.max(),
        "read_std_s": times.std(ddof=1)
    })

df_results = pd.DataFrame(results).sort_values("index")

# Save raw CSV (for reproducibility)
os.makedirs(RESULTS_ROOT, exist_ok=True)
csv_out = os.path.join(RESULTS_ROOT, "file_format_analysis.csv")
df_results_rounded = df_results.copy()
for col in ["size_MB", "read_min_s", "read_avg_s", "read_max_s", "read_std_s"]:
    df_results_rounded[col] = df_results_rounded[col].round(2)
df_results_rounded.to_csv(csv_out, index=False)
print(f"[OK] Saved CSV → {csv_out}")

# ---------------- Build LaTeX table ----------------
# Format: Format | Size (MB) | Read (mean ± std) [s] | Min [s] | Max [s]
df_display = pd.DataFrame({
    "Format": df_results["format"],
    "Size [MB]": df_results["size_MB"].map(lambda x: f"{x:.2f}"),
    "Read [s]": (df_results["read_avg_s"].map(lambda x: f"{x:.2f}")
                               + " ± "
                               + df_results["read_std_s"].map(lambda x: f"{x:.2f}")),
})

# Save LaTeX via pandas (escape=True)
tables_dir = os.path.join(LATEX_ROOT, "tables")
os.makedirs(tables_dir, exist_ok=True)
tex_out = os.path.join(tables_dir, "file_format_comparison.tex")
df_display.to_latex(
    tex_out,
    index=False,
    escape=True,
    column_format="lrr",
)
print(f"[OK] Saved LaTeX table → {tex_out}")
