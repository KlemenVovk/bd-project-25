from utils.constants import (
    TASK1_OUT_ROOT, TASK2_OUT_ROOT,
    TASK1_FHV_SCHEMA,
    NYC_MOST_EAST_LONGITUDE, NYC_MOST_WEST_LONGITUDE,
    NYC_MOST_NORTH_LATITUDE, NYC_MOST_SOUTH_LATITUDE
)
import dask.dataframe as dd
import numpy as np
import os
from glob import glob
from tqdm import tqdm

REASONABLE_MIN_YEAR = 2014
REASONABLE_MAX_YEAR = 2025

# Load FHV parquet files
parquet_files = sorted(glob(os.path.join(TASK1_OUT_ROOT, "fhv_tripdata", "**", "*.parquet")))
print(f"Found {len(parquet_files)} parquet files in {TASK1_OUT_ROOT}/fhv_tripdata")

dfs = [dd.read_parquet(file) for file in parquet_files]
stats_ddfs = []

for i in tqdm(range(len(dfs))):
    df = dfs[i]
    df['year'] = df['pickup_datetime'].dt.year.astype(np.int16)

    # Timestamp cleaning
    pickup_before_dropoff = df['pickup_datetime'] < df['dropoff_datetime']
    delta_years = df['dropoff_datetime'].dt.year - df['pickup_datetime'].dt.year
    same_year = delta_years == 0
    dropoff_next_year = (
        (delta_years == 1) &
        (df['dropoff_datetime'].dt.month == 1) & (df['dropoff_datetime'].dt.day == 1) &
        (df['pickup_datetime'].dt.month == 12) & (df['pickup_datetime'].dt.day == 31)
    )
    reasonable_year = df['year'].between(REASONABLE_MIN_YEAR, REASONABLE_MAX_YEAR, inclusive='both')
    correct_datetimes = pickup_before_dropoff & (same_year | dropoff_next_year) & reasonable_year

    # Coordinate bounds
    latitudes_not_nan = df['pickup_latitude'].notnull() & df['dropoff_latitude'].notnull()
    longitudes_not_nan = df['pickup_longitude'].notnull() & df['dropoff_longitude'].notnull()
    latitudes_within_bounds = df['pickup_latitude'].between(NYC_MOST_SOUTH_LATITUDE, NYC_MOST_NORTH_LATITUDE) & df['dropoff_latitude'].between(NYC_MOST_SOUTH_LATITUDE, NYC_MOST_NORTH_LATITUDE)
    longitudes_within_bounds = df['pickup_longitude'].between(NYC_MOST_WEST_LONGITUDE, NYC_MOST_EAST_LONGITUDE) & df['dropoff_longitude'].between(NYC_MOST_WEST_LONGITUDE, NYC_MOST_EAST_LONGITUDE)
    coordinates_reasonable = latitudes_not_nan & longitudes_not_nan & latitudes_within_bounds & longitudes_within_bounds

    # Boolean issue flags
    bad_timestamps = ~correct_datetimes
    bad_coordinates = ~coordinates_reasonable
    clean = ~(bad_timestamps | bad_coordinates)

    # Attach issue flags to minimal frame
    df_year = df[['year']].copy()
    df_year['bad_timestamps'] = bad_timestamps
    df_year['bad_coordinates'] = bad_coordinates
    df_year['clean'] = clean

    # Dask-native grouped stats (lazy)
    stats_ddfs.append(
        df_year.groupby("year").agg(
            total_trips=("year", "count"),
            bad_timestamps=("bad_timestamps", "sum"),
            bad_coordinates=("bad_coordinates", "sum"),
            clean_trips=("clean", "sum"),
        )
    )

    # Apply final cleaning mask
    clean_mask = correct_datetimes & coordinates_reasonable
    dfs[i] = df[clean_mask].reset_index(drop=True)

# Final map-reduce over all files
print("Computing stats...")
combined_stats_ddf = dd.concat(stats_ddfs)
final_stats_ddf = combined_stats_ddf.groupby("year").sum()
final_stats = final_stats_ddf.compute().reset_index()

# Save stats to CSV
os.makedirs(os.path.join(TASK2_OUT_ROOT), exist_ok=True)
final_stats.to_csv(os.path.join(TASK2_OUT_ROOT, "fhv_tripdata_quality_stats.csv"), index=False)

# Save cleaned parquet files
print("Saving cleaned data...")
os.makedirs(os.path.join(TASK2_OUT_ROOT, "fhv_tripdata"), exist_ok=True)
for i, ddf in enumerate(tqdm(dfs)):
    assert sorted(set(TASK1_FHV_SCHEMA.keys())) == sorted(set(ddf.columns.tolist())), \
        f"Schema mismatch: found {sorted(ddf.columns.tolist())}, expected {sorted(TASK1_FHV_SCHEMA.keys())}"
    ddf = ddf.astype(TASK1_FHV_SCHEMA)
    ddf.to_parquet(
        os.path.join(TASK2_OUT_ROOT, "fhv_tripdata"),
        engine='pyarrow',
        compression='snappy',
        partition_on=['year'],
        write_index=False,
        append=i > 0,
        overwrite=False,
        row_group_size=2_000_000,
    )
