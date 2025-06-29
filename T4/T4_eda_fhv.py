import os
import dask.dataframe as dd
import duckdb
import json
from dask_sql import Context

from utils.constants import TASK2_OUT_ROOT, RESULTS_ROOT

# Load cleaned data
print("Loading cleaned fhv data...")
dataset_path = os.path.join(TASK2_OUT_ROOT, "fhv_tripdata")
parquet_path = os.path.join(dataset_path, "*", "*.parquet")
df = dd.read_parquet(dataset_path)

# Filter data to 2018 and later

# Prepare result collector
data = {}

# --- Dask-SQL ---
print("Initializing Dask-SQL context...")
c = Context()
df["year"] = df["year"].astype("uint16")  # Required fix for Dask-SQL compatibility
c.create_table("fhv", df)

print("Running Dask-SQL: trips per year...")
data["trips_per_year"] = c.sql("""
    SELECT 
        year,
        COUNT(*) AS num_trips
    FROM fhv
    GROUP BY year
    ORDER BY year
""").compute().to_dict(orient="records")

# --- DuckDB ---
print("Initializing DuckDB and running SQL queries...")
con = duckdb.connect()

print("DuckDB: trips per pickup location ID...")
data["trips_per_PULocationID"] = con.execute(f"""
    SELECT 
        PULocationID,
        COUNT(*) AS num_trips
    FROM read_parquet('{parquet_path}')
    WHERE PULocationID IS NOT NULL
    GROUP BY PULocationID
    ORDER BY num_trips DESC
""").fetchdf().to_dict(orient="records")

print("DuckDB: trips per hour of day...")
data["trips_per_hour"] = con.execute(f"""
    SELECT 
        EXTRACT('hour' FROM pickup_datetime) AS hour_of_day,
        COUNT(*) AS num_trips
    FROM read_parquet('{parquet_path}')
    GROUP BY hour_of_day
    ORDER BY num_trips DESC
""").fetchdf().to_dict(orient="records")

# Volume aggregates
print("DuckDB: trip duration stats using datetime diff...")
data["median_duration"] = con.execute(f"""
    SELECT 
        MEDIAN(EXTRACT(EPOCH FROM (dropoff_datetime - pickup_datetime))) AS median_trip_duration_sec
    FROM read_parquet('{parquet_path}')
    WHERE dropoff_datetime > pickup_datetime AND year >= 2018
""").fetchdf().to_dict(orient="records")

# --- Spatial Aggregates ---
print("Dask: computing trips per pickup borough...")
data["trips_per_borough"] = (
    df.groupby("pickup_borough")["pickup_borough"]
    .count()
    .compute()
    .sort_values(ascending=False)
    .reset_index(name="num_trips")
    .to_dict(orient="records")
)

print("Dask: computing OD matrix (boroughs)...")
data["od_matrix_borough"] = (
    df.groupby(["pickup_borough", "dropoff_borough"])
    .size()
    .compute()
    .reset_index(name="num_trips")
    .to_dict(orient="records")
)

print("Extracting top 10 OD borough pairs...")
data["top10_od_boroughs"] = sorted(data["od_matrix_borough"], key=lambda x: x["num_trips"], reverse=True)[:10]

print("Dask: computing OD matrix (PULocationID, DOLocationID)...")
data["od_matrix_ids"] = (
    df.groupby(["PULocationID", "DOLocationID"])
    .size()
    .compute()
    .reset_index(name="num_trips")
    .to_dict(orient="records")
)

print("Extracting top 10 OD ID pairs...")
data["top10_od_ids"] = sorted(data["od_matrix_ids"], key=lambda x: x["num_trips"], reverse=True)[:10]

# --- Save all results to JSON ---
print("Saving all aggregate results to JSON...")
os.makedirs(RESULTS_ROOT, exist_ok=True)
json_path = os.path.join(RESULTS_ROOT, "fhv_tripdata_aggregates.json")
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"All aggregate results saved to {json_path}.")
