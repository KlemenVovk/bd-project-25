import os
import dask.dataframe as dd
import duckdb
import json
from dask_sql import Context

from utils.constants import TASK2_OUT_ROOT, RESULTS_ROOT

print("Loading cleaned yellow taxi data...")
dataset_path = os.path.join(TASK2_OUT_ROOT, "yellow_tripdata")
parquet_path = os.path.join(dataset_path, "*", "*.parquet")
df = dd.read_parquet(dataset_path)

data = {}

# --- Dask-SQL ---
print("Initializing Dask-SQL context...")
c = Context()
df["year"] = df["year"].astype("uint16")
c.create_table("yellow", df)

print("Running Dask-SQL: trips per year...")
data["trips_per_year"] = c.sql("""
    SELECT 
        year,
        COUNT(*) AS num_trips
    FROM yellow
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

print("DuckDB: median fare per year...")
data["median_fare_per_year"] = con.execute(f"""
    SELECT 
        year,
        MEDIAN(fare_amount) AS median_fare
    FROM read_parquet('{parquet_path}')
    GROUP BY year
    ORDER BY year
""").fetchdf().to_dict(orient="records")

# Volume aggregates
print("DuckDB: median tip amount...")
data["median_tip"] = con.execute(f"""
    SELECT 
        MEDIAN(tip_amount) AS median_tip
    FROM read_parquet('{parquet_path}')
    WHERE tip_amount IS NOT NULL
""").fetchdf().to_dict(orient="records")

print("DuckDB: median trip duration (seconds)...")
data["median_duration"] = con.execute(f"""
    SELECT 
        MEDIAN(DATE_PART('second', dropoff_datetime - pickup_datetime)) AS median_trip_duration_sec
    FROM read_parquet('{parquet_path}')
    WHERE dropoff_datetime > pickup_datetime
""").fetchdf().to_dict(orient="records")

print("DuckDB: median fare amount overall...")
data["median_fare"] = con.execute(f"""
    SELECT 
        MEDIAN(fare_amount) AS median_fare_amount
    FROM read_parquet('{parquet_path}')
    WHERE fare_amount IS NOT NULL
""").fetchdf().to_dict(orient="records")

print("Computing tip vs no-tip percentages (Dask)...")
tipped_pct = (df.tip_amount.gt(0).mean() * 100).compute()
notipped_pct = 100 - tipped_pct
data["tip_percentage"] = [
    {"type": "Tipped", "percentage": tipped_pct},
    {"type": "No Tip", "percentage": notipped_pct}
]

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

# --- Save all results ---
print("Saving all aggregate results to JSON...")
os.makedirs(RESULTS_ROOT, exist_ok=True)
json_path = os.path.join(RESULTS_ROOT, "yellow_tripdata_aggregates.json")
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"All aggregate results saved to {json_path}. Done.")
