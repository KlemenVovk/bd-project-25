from utils.constants import (
    TASK1_OUT_ROOT, TASK2_OUT_ROOT,
    TASK1_GREENTAXI_SCHEMA,
    NYC_MOST_EAST_LONGITUDE, NYC_MOST_WEST_LONGITUDE,
    NYC_MOST_NORTH_LATITUDE, NYC_MOST_SOUTH_LATITUDE
)
import dask.dataframe as dd
import numpy as np
import os
from glob import glob
from tqdm import tqdm

# Thresholds
MAX_SPEED = 80  # mph
REASONABLE_PRICE_MIN = 2.5
REASONABLE_PRICE_MAX = 500
REASONABLE_MIN_TRIP_DURATION = 1  # minutes
REASONABLE_MAX_TRIP_DURATION = 6 * 60  # minutes
REASONABLE_MIN_YEAR = 2013
REASONABLE_MAX_YEAR = 2025

# Load input parquet files
parquet_files = sorted(glob(os.path.join(TASK1_OUT_ROOT, "green_tripdata", "**", "*.parquet")))
print(f"Found {len(parquet_files)} parquet files in {TASK1_OUT_ROOT}/green_tripdata")

dfs = [dd.read_parquet(file) for file in parquet_files]
stats_ddfs = []

for i in tqdm(range(len(dfs))):
    df = dfs[i]
    df['year'] = df['pickup_datetime'].dt.year.astype(np.int16)

    # Timestamp checks
    pickup_before_dropoff = df['pickup_datetime'] < df['dropoff_datetime']
    delta_years = df['dropoff_datetime'].dt.year - df['pickup_datetime'].dt.year
    same_year = delta_years == 0
    dropoff_next_year = (
        (delta_years == 1) &
        (df['dropoff_datetime'].dt.month == 1) & (df['dropoff_datetime'].dt.day == 1) &
        (df['pickup_datetime'].dt.month == 12) & (df['pickup_datetime'].dt.day == 31)
    )
    year_valid = df['year'].between(REASONABLE_MIN_YEAR, REASONABLE_MAX_YEAR)
    time_valid = pickup_before_dropoff & (same_year | dropoff_next_year) & year_valid

    # Trip duration/distance checks
    trip_duration_hr = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 3600
    achievable_trip_distance = trip_duration_hr * MAX_SPEED
    duration_valid = trip_duration_hr.between(REASONABLE_MIN_TRIP_DURATION / 60, REASONABLE_MAX_TRIP_DURATION / 60)
    distance_valid = df['trip_distance'].between(0, achievable_trip_distance, inclusive='neither')
    trip_valid = duration_valid & distance_valid

    # Coordinate checks
    lat_ok = df['pickup_latitude'].notnull() & df['dropoff_latitude'].notnull()
    lon_ok = df['pickup_longitude'].notnull() & df['dropoff_longitude'].notnull()
    lat_bounds = df['pickup_latitude'].between(NYC_MOST_SOUTH_LATITUDE, NYC_MOST_NORTH_LATITUDE) & df['dropoff_latitude'].between(NYC_MOST_SOUTH_LATITUDE, NYC_MOST_NORTH_LATITUDE)
    lon_bounds = df['pickup_longitude'].between(NYC_MOST_WEST_LONGITUDE, NYC_MOST_EAST_LONGITUDE) & df['dropoff_longitude'].between(NYC_MOST_WEST_LONGITUDE, NYC_MOST_EAST_LONGITUDE)
    coord_valid = lat_ok & lon_ok & lat_bounds & lon_bounds

    # Fare and fee checks
    positive_fare = df['fare_amount'] > REASONABLE_PRICE_MIN
    nonneg_components = (
        (df['extra'] >= 0) &
        (df['mta_tax'] >= 0) &
        (df['tip_amount'] >= 0) &
        (df['tolls_amount'] >= 0) &
        (df['improvement_surcharge'] >= 0) &
        (df['congestion_surcharge'] >= 0) &
        (df['cbd_congestion_fee'] >= 0) &
        (df['ehail_fee'] >= 0)
    )
    positive_total = df['total_amount'] > 0
    component_sum = (
        df['fare_amount'] +
        df['mta_tax'] +
        df['improvement_surcharge'] +
        df['tolls_amount'] +
        df['congestion_surcharge'] +
        df['cbd_congestion_fee'] +
        df['ehail_fee']
    )
    sum_valid = component_sum.between(0, df['total_amount'], inclusive='right')
    reasonable_total = df['total_amount'].between(REASONABLE_PRICE_MIN, REASONABLE_PRICE_MAX)
    price_valid = positive_fare & nonneg_components & positive_total & reasonable_total & sum_valid

    # Issue flags
    bad_timestamps = ~time_valid
    bad_distance = ~trip_valid
    bad_coords = ~coord_valid
    bad_prices = ~price_valid
    clean = ~(bad_timestamps | bad_distance | bad_coords | bad_prices)

    # Stats aggregation
    df_year = df[['year']].copy()
    df_year['bad_timestamps'] = bad_timestamps
    df_year['bad_distance'] = bad_distance
    df_year['bad_coordinates'] = bad_coords
    df_year['bad_prices'] = bad_prices
    df_year['clean'] = clean

    stats_ddfs.append(
        df_year.groupby("year").agg(
            total_trips=("year", "count"),
            bad_timestamps=("bad_timestamps", "sum"),
            bad_distance=("bad_distance", "sum"),
            bad_coordinates=("bad_coordinates", "sum"),
            bad_prices=("bad_prices", "sum"),
            clean_trips=("clean", "sum"),
        )
    )

    # Apply clean mask
    clean_mask = time_valid & trip_valid & coord_valid & price_valid
    dfs[i] = df[clean_mask].reset_index(drop=True)

# Reduce all stats with Dask
print("Computing stats...")
final_stats_ddf = dd.concat(stats_ddfs).groupby("year").sum()
final_stats = final_stats_ddf.compute().reset_index()

# Save to disk
os.makedirs(os.path.join(TASK2_OUT_ROOT), exist_ok=True)
final_stats.to_csv(os.path.join(TASK2_OUT_ROOT, "green_tripdata_quality_stats.csv"), index=False)

# Save cleaned Parquet
print("Saving cleaned data...")
os.makedirs(os.path.join(TASK2_OUT_ROOT, "green_tripdata"), exist_ok=True)
for i, ddf in enumerate(tqdm(dfs)):
    assert sorted(set(TASK1_GREENTAXI_SCHEMA.keys())) == sorted(set(ddf.columns.tolist())), \
        f"Schema mismatch: found {sorted(ddf.columns.tolist())}, expected {sorted(TASK1_GREENTAXI_SCHEMA.keys())}"
    ddf = ddf.astype(TASK1_GREENTAXI_SCHEMA)
    ddf.to_parquet(
        os.path.join(TASK2_OUT_ROOT, "green_tripdata"),
        engine='pyarrow',
        compression='snappy',
        partition_on=['year'],
        write_index=False,
        append=i > 0,
        overwrite=False,
        row_group_size=2_000_000,
    )
