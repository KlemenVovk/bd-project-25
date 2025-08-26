import dask.dataframe as dd
import os

# from utils.constants import TASK2_OUT_ROOT, TASK6_OUT_ROOT


# Define paths
# yellow_path = os.path.join(TASK2_OUT_ROOT, "yellow_tripdata", "year==2023")
# fhvhv_path = os.path.join(TASK2_OUT_ROOT, "fhvhv_tripdata", "year==2023")

yellow_path = "kafka/jupyter/data/yellow_sample.parquet"
fhvhv_path = "kafka/jupyter/data/fhvhv_sample.parquet"

# Read each set of parquet files separately
yellow_df = dd.read_parquet(yellow_path, engine="pyarrow", gather_statistics=False)
fhvhv_df = dd.read_parquet(fhvhv_path, engine="pyarrow", gather_statistics=False)

yellow_df["source"] = "yellow"
fhvhv_df["source"] = "fhvhv"

# Align schemas
yellow_cols = set(yellow_df.columns)
fhvhv_cols = set(fhvhv_df.columns)
all_cols = sorted(yellow_cols.union(fhvhv_cols))

for col in all_cols:
    if col not in yellow_df.columns:
        yellow_df[col] = None
    if col not in fhvhv_df.columns:
        fhvhv_df[col] = None

# Reorder columns to match
yellow_df = yellow_df[all_cols]
fhvhv_df = fhvhv_df[all_cols]

for col in all_cols:
    yellow_df[col] = yellow_df[col].astype("string")
    fhvhv_df[col] = fhvhv_df[col].astype("string")

# Concatenate the two DataFrames
combined_df = dd.concat([yellow_df, fhvhv_df], axis=0, interleave_partitions=True)

# Save the combined DataFrame to a new parquet file
TASK6_OUT_ROOT = "kafka/jupyter/data/"
output_path = os.path.join(TASK6_OUT_ROOT, "combined")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
combined_df.to_parquet(output_path, engine="pyarrow")
