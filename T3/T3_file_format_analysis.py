import os
import timeit
import pandas as pd
import numpy as np
from utils.constants import TASK2_OUT_ROOT, TASK3_OUT_ROOT, RESULTS_ROOT
from tqdm import tqdm

N_REPEATS = 10
os.makedirs(TASK3_OUT_ROOT, exist_ok=True)
df = pd.read_parquet(os.path.join(TASK2_OUT_ROOT, "green_tripdata", "year=2024"))

filename2format = {
    "green_tripdata_2024.parquet": {
        "save": lambda path: df.to_parquet(path, index=False),
        "read": lambda path: pd.read_parquet(path),
        "desc": "parquet",
        "index": 0
    },
    "green_tripdata_2024.csv": {
        "save": lambda path: df.to_csv(path, index=False),
        "read": lambda path: pd.read_csv(path),
        "desc": "csv (no compression)",
        "index": 1
    },
    "green_tripdata_2024.csv.gz": {
        "save": lambda path: df.to_csv(path, index=False, compression='gzip'),
        "read": lambda path: pd.read_csv(path, compression='gzip'),
        "desc": "csv (gzip compression)",
        "index": 2
    },
    "green_tripdata_2024_no_compression.h5": {
        "save": lambda path: df.to_hdf(path, key='df', mode='w', format='table',
                                       complib=None, complevel=0, index=False),
        "read": lambda path: pd.read_hdf(path, key='df'),
        "desc": "hdf5 (no compression)",
        "index": 3
    },
    "green_tripdata_2024_lz4.h5": {
        "save": lambda path: df.to_hdf(path, key='df', mode='w', format='table',
                                       complib='blosc2:lz4', complevel=9, index=False),
        "read": lambda path: pd.read_hdf(path, key='df'),
        "desc": "hdf5 (lz4 compression level 9)",
        "index": 4
    },
    "green_tripdata_2024_zstd.h5": {
        "save": lambda path: df.to_hdf(path, key='df', mode='w', format='table',
                                       complib='blosc2:zstd', complevel=9, index=False),
        "read": lambda path: pd.read_hdf(path, key='df'),
        "desc": "hdf5 (zstd compression level 9)",
        "index": 5
    }
}

results = []
for fname, ops in tqdm(filename2format.items(), desc="Analyzing file formats"):
    path = os.path.join(TASK3_OUT_ROOT, fname)

    ops["save"](path)
    timer = timeit.Timer(lambda: ops["read"](path))
    times = np.array(timer.repeat(repeat=N_REPEATS, number=1))

    results.append({
        "index": ops["index"],
        "filename": fname,
        "format": ops["desc"],
        "size_MB": round(os.path.getsize(path) / (1024 ** 2), 2),
        "read_min_s": round(times.min(), 2),
        "read_avg_s": round(times.mean(), 2),
        "read_max_s": round(times.max(), 2),
        "read_std_s": round(times.std(ddof=1), 2)
    })

# Create DataFrame and sort by index
df_results = pd.DataFrame(results).sort_values("index")

# Save CSV
results_csv = os.path.join(RESULTS_ROOT, "file_format_analysis.csv")
df_results.to_csv(results_csv, index=False)
print(f"Saved CSV: {results_csv}")

# Save LaTeX table
results_latex = os.path.join(RESULTS_ROOT, "file_format_analysis.tex")
df_results.drop(columns=["index", "filename"]).to_latex(
    results_latex, index=False, float_format="%.2f", caption="File format comparison for NYC Green Taxi 2024 data",
    label="tab:file_format_comparison", column_format="lrrrrr", escape=False
)
print(f"Saved LaTeX table: {results_latex}")
