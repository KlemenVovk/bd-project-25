from utils.constants import (
    TASK1_OUT_ROOT, TASK2_OUT_ROOT,
    TASK1_FHVHV_SCHEMA,
    NYC_MOST_EAST_LONGITUDE, NYC_MOST_WEST_LONGITUDE,
    NYC_MOST_NORTH_LATITUDE, NYC_MOST_SOUTH_LATITUDE,
    RESULTS_ROOT
)
import dask.dataframe as dd
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import pandas as pd

MAX_SPEED = 80  # mph
REASONABLE_PRICE_MIN = 2.5
REASONABLE_PRICE_MAX = 500
REASONABLE_MIN_YEAR = 2019
REASONABLE_MAX_YEAR = 2025
REASONABLE_MIN_TRIP_TIME = 1  # minutes
REASONABLE_MAX_TRIP_TIME = 6 * 60  # minutes

# Load input files
parquet_files = sorted(glob(os.path.join(TASK1_OUT_ROOT, "fhvhv_tripdata", "**", "*.parquet")))
print(f"Found {len(parquet_files)} parquet files in {TASK1_OUT_ROOT}/fhvhv_tripdata")

stats_ddfs = []
dfs = [dd.read_parquet(file) for file in parquet_files]

for i in tqdm(range(len(dfs))):
    df = dfs[i]

    # Extract year
    df['year'] = df['pickup_datetime'].dt.year.astype(np.int16)

    # Datetime validation
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

    # Trip distance/time validation
    REASONABLE_MIN_TRIP_TIME_SEC = 60       # 1 minute (trip_time is given in seconds)
    REASONABLE_MAX_TRIP_TIME_SEC = 6 * 60 * 60  # 6 hours
    duration_valid = df['trip_time'].between(REASONABLE_MIN_TRIP_TIME_SEC, REASONABLE_MAX_TRIP_TIME_SEC, inclusive='both')
    distance_valid = df['trip_distance'].between(0, (df['trip_time'] / 3600) * MAX_SPEED, inclusive='neither')
    valid_trips = duration_valid & distance_valid

    # Coordinate bounds check
    latitudes_not_nan = df['pickup_latitude'].notnull() & df['dropoff_latitude'].notnull()
    longitudes_not_nan = df['pickup_longitude'].notnull() & df['dropoff_longitude'].notnull()
    latitudes_within_bounds = df['pickup_latitude'].between(NYC_MOST_SOUTH_LATITUDE, NYC_MOST_NORTH_LATITUDE) & df['dropoff_latitude'].between(NYC_MOST_SOUTH_LATITUDE, NYC_MOST_NORTH_LATITUDE)
    longitudes_within_bounds = df['pickup_longitude'].between(NYC_MOST_WEST_LONGITUDE, NYC_MOST_EAST_LONGITUDE) & df['dropoff_longitude'].between(NYC_MOST_WEST_LONGITUDE, NYC_MOST_EAST_LONGITUDE)
    coordinates_reasonable = latitudes_not_nan & longitudes_not_nan & latitudes_within_bounds & longitudes_within_bounds

    # Fare validation
    component_sum = (
        df['base_passenger_fare'] +
        df['bcf'] +
        df['sales_tax'] +
        df['tips'] +
        df['tolls'] +
        df['driver_pay'] +
        df['airport_fee'] +
        df['congestion_surcharge'] +
        df['cbd_congestion_fee']
    )
    reasonable_sum = component_sum.between(0, 1000)
    component_nonneg = (
        (df['base_passenger_fare'] >= 0) &
        (df['bcf'] >= 0) &
        (df['sales_tax'] >= 0) &
        (df['tips'] >= 0) &
        (df['tolls'] >= 0) &
        (df['driver_pay'] >= 0) &
        (df['airport_fee'] >= 0) &
        (df['congestion_surcharge'] >= 0) &
        (df['cbd_congestion_fee'] >= 0)
    )
    valid_fares = component_nonneg & reasonable_sum

    # request_datetime check
    valid_request_time = (
        df['request_datetime'].notnull() &
        (df['request_datetime'] <= df['pickup_datetime']) &
        (df['request_datetime'] >= (df['pickup_datetime'] - pd.Timedelta(days=1)))
    )


    # Boolean flags for stats
    bad_timestamps = ~correct_datetimes
    bad_distance = ~valid_trips
    bad_coordinates = ~coordinates_reasonable
    bad_fares = ~valid_fares
    bad_request_time = ~valid_request_time
    clean = ~(bad_timestamps | bad_distance | bad_coordinates | bad_fares | bad_request_time)

    # Year + quality flags
    df_year = df[['year']].copy()
    df_year['bad_timestamps'] = bad_timestamps
    df_year['bad_distance'] = bad_distance
    df_year['bad_coordinates'] = bad_coordinates
    df_year['bad_fares'] = bad_fares
    df_year['bad_request_time'] = bad_request_time
    df_year['clean'] = clean

    # Grouped stats (lazy)
    stats_ddfs.append(
        df_year.groupby("year").agg(
            total_trips=("year", "count"),
            bad_timestamps=("bad_timestamps", "sum"),
            bad_distance=("bad_distance", "sum"),
            bad_coordinates=("bad_coordinates", "sum"),
            bad_fares=("bad_fares", "sum"),
            bad_request_time=("bad_request_time", "sum"),
            clean_trips=("clean", "sum"),
        )
    )

    # Apply cleaning mask
    clean_mask = (
        correct_datetimes &
        valid_trips &
        coordinates_reasonable &
        valid_fares &
        valid_request_time
    )
    dfs[i] = df[clean_mask].reset_index(drop=True)

# Final map-reduce
print("Computing stats...")
combined_stats_ddf = dd.concat(stats_ddfs)
final_stats_ddf = combined_stats_ddf.groupby("year").sum()
final_stats = final_stats_ddf.compute().reset_index().sort_values("year")

# Save stats CSV
os.makedirs(os.path.join(RESULTS_ROOT), exist_ok=True)
final_stats.to_csv(os.path.join(RESULTS_ROOT, "fhvhv_tripdata_quality_stats.csv"), index=False)

# Save cleaned data
print("Saving cleaned data...")
os.makedirs(os.path.join(TASK2_OUT_ROOT, "fhvhv_tripdata"), exist_ok=True)
for i, ddf in enumerate(tqdm(dfs)):
    assert sorted(set(TASK1_FHVHV_SCHEMA.keys())) == sorted(set(ddf.columns.tolist())), \
        f"Schema mismatch: found {sorted(ddf.columns.tolist())}, expected {sorted(TASK1_FHVHV_SCHEMA.keys())}"
    ddf = ddf.astype(TASK1_FHVHV_SCHEMA)
    ddf.to_parquet(
        os.path.join(TASK2_OUT_ROOT, "fhvhv_tripdata"),
        engine='pyarrow',
        compression='snappy',
        partition_on=['year'],
        write_index=False,
        append=i > 0,
        overwrite=False,
        row_group_size=2_000_000,
    )
