import numpy as np
import pandas as pd
import dask.dataframe as dd
from glob import glob
import os
from constants import RAW_DATA_ROOT, TASK1_OUT_ROOT, TASK1_FHVHV_SCHEMA, COLUMN_CONSISTENCY_NAMING_MAP, LOCATIONID_CENTROID_BOROUGH_CSV
from tqdm import tqdm


def map_flag(value):
    # valid is 1 (street hail) or 2 (dispatch), else -1
    valid_flags_map = {
        "Y": 1,
        "N": 0,
    }
    if str(value).strip() in valid_flags_map:
        return valid_flags_map[str(value).strip()]
    else:
        return -1



if __name__ == "__main__":
    pd.set_option('future.no_silent_downcasting', True)

    print("Reading files...")
    files = sorted(glob(os.path.join(RAW_DATA_ROOT, "fhvhv_tripdata/*.parquet")))
    dfs = [dd.read_parquet(f) for f in files]
    
    print("Renaming columns...")
    # Renaming columns for consistency
    for i, f in enumerate(files):
        # Rename columns
        for old_col, new_col in COLUMN_CONSISTENCY_NAMING_MAP.items():
            if old_col in dfs[i].columns.tolist():
                dfs[i] = dfs[i].rename(columns={old_col: new_col})

    # Fix column values (mapping, dtypes...) to be able to concat.
    print("Fixing column values (mapping to consistent values, dtypes, locationids to (lat,lng) centroids etc.)...")
    location_centroid_borough_df = pd.read_csv(LOCATIONID_CENTROID_BOROUGH_CSV)
    for i, f in tqdm(enumerate(files)):
        # map vendor_id column but with dask so i can compute later
        # pickup
        # set dtype to int
        dfs[i]['PULocationID'] = dfs[i]['PULocationID'].fillna(0).astype(int) # 0 - does not exist
        # show any pulocationds that are not in the locationid_to_centers_df
        dfs[i] = dfs[i].merge(location_centroid_borough_df, how='left', left_on='PULocationID', right_on='LocationID')
        # rename to pickup lattitude and longitude
        dfs[i] = dfs[i].rename(columns={'centroid_lat': 'pickup_latitude', 'centroid_lng': 'pickup_longitude', 'Borough': 'pickup_borough'})

        # dropoff
        # set dtype to int
        dfs[i]['DOLocationID'] = dfs[i]['DOLocationID'].fillna(0).astype(int) # 0 - does not exist
        dfs[i] = dfs[i].merge(location_centroid_borough_df, how='left', left_on='DOLocationID', right_on='LocationID')
        # rename to dropoff lattitude and longitude
        dfs[i] = dfs[i].rename(columns={'centroid_lat': 'dropoff_latitude', 'centroid_lng': 'dropoff_longitude', 'Borough': 'dropoff_borough'})
        # drop LocationID columns
        dfs[i] = dfs[i].drop(columns=['LocationID_x', 'LocationID_y', 'Zone_y', 'service_zone_y', 'Zone_x', 'service_zone_x'], errors='ignore')

        # Flags
        dfs[i]['access_a_ride_flag'] = dfs[i]['access_a_ride_flag'].apply(map_flag, meta=('access_a_ride_flag', 'int8'))
        dfs[i]['shared_request_flag'] = dfs[i]['shared_request_flag'].apply(map_flag, meta=('shared_request_flag', 'int8')) 
        dfs[i]['shared_match_flag'] = dfs[i]['shared_match_flag'].apply(map_flag, meta=('shared_match_flag', 'int8'))
        dfs[i]['wav_request_flag'] = dfs[i]['wav_request_flag'].apply(map_flag, meta=('wav_request_flag', 'int8'))
        dfs[i]['wav_match_flag'] = dfs[i]['wav_match_flag'].apply(map_flag, meta=('wav_match_flag', 'int8'))


        dfs[i]['airport_fee'] = dfs[i]['airport_fee'].fillna(0).astype(np.float32)  # Fill NaN with 0 and convert to float32
        dfs[i]['base_passenger_fare'] = dfs[i]['base_passenger_fare'].fillna(0).astype(np.float32)  # Fill NaN with 0 and convert to float32
        dfs[i]['bcf'] = dfs[i]['bcf'].fillna(0).astype(np.float32)  # Fill NaN with 0 and convert to float32
        if 'cbd_congestion_fee' not in dfs[i].columns:
            dfs[i]['cbd_congestion_fee'] = 0.0
        dfs[i]['cbd_congestion_fee'] = dfs[i]['cbd_congestion_fee'].fillna(0).astype(np.float32)  # Fill NaN with 0 and
        dfs[i]['congestion_surcharge'] = dfs[i]['congestion_surcharge'].fillna(0).astype(np.float32)  # Fill NaN with 0 and convert to float32
        dfs[i]['dispatching_base_num'] = dfs[i]['dispatching_base_num'].fillna('').astype('string[pyarrow]').str.strip()  # Fill NaN with empty string and convert to string
        dfs[i]['driver_pay'] = dfs[i]['driver_pay'].fillna(0).astype(np.float32)  # Fill NaN with 0 and convert to float32
        dfs[i]['hvfhs_license_num'] = dfs[i]['hvfhs_license_num'].fillna('').astype('string[pyarrow]').str.strip()  # Fill NaN with
        dfs[i]['originating_base_num'] = dfs[i]['originating_base_num'].fillna('').astype('string[pyarrow]').str.strip()  # Fill NaN with empty string and convert to string
        dfs[i]['sales_tax'] = dfs[i]['sales_tax'].fillna(0).astype(np.float32)  # Fill NaN with 0 and convert to float32
        dfs[i]['tips'] = dfs[i]['tips'].fillna(0).astype(np.float32)  # Fill NaN with 0 and convert to float32
        dfs[i]['tolls'] = dfs[i]['tolls'].fillna(0).astype(np.float32)
        dfs[i]['trip_distance'] = dfs[i]['trip_distance'].fillna(0).astype(np.float32)
        dfs[i]['trip_time'] = dfs[i]['trip_time'].fillna(0).astype(np.float32)

        # parse datetimes
        dfs[i]['on_scene_datetime'] = dd.to_datetime(dfs[i]['on_scene_datetime'], errors='raise')
        dfs[i]['pickup_datetime'] = dd.to_datetime(dfs[i]['pickup_datetime'], errors='raise')
        dfs[i]['request_datetime'] = dd.to_datetime(dfs[i]['request_datetime'], errors='raise')
        dfs[i]['dropoff_datetime'] = dd.to_datetime(dfs[i]['dropoff_datetime'], errors='raise')
        dfs[i]['year'] = dfs[i]['pickup_datetime'].dt.year.astype(np.int16)

    
    # Save everything
    # Up to here, everything is embarrasingly parallelizable (a `dask.compute(*dfs)` would work in parallel - tested).
    # We are saving files sequentially, since multiprocessing append doesn't really work that well in to_parquet or when writing to the same CSV file.
    # Furthermore, even if we do parallelize, the bottleneck is in disk I/O.

    # Compute everything in parallel (persist)
    # dfs = client.persist(dfs)
    # Concat and repartition so that a division is every 100_000 rows.
    
    for ddf in tqdm(dfs):
        assert sorted(set(TASK1_FHVHV_SCHEMA.keys())) == sorted(set(ddf.columns.tolist())), f"Columns in ddf {sorted(ddf.columns.tolist())} do not match the schema {sorted(TASK1_FHVHV_SCHEMA.keys())}"
        ddf = ddf.astype(TASK1_FHVHV_SCHEMA)     
        ddf.to_parquet(
            os.path.join(TASK1_OUT_ROOT, "fhvhv_tripdata"),
            engine='pyarrow',
            compression='snappy',
            partition_on=['year'],
            write_index=False,
            append=i > 0,
            overwrite=False,
            row_group_size=2_000_000, # Row group != partition - dask has 1 partition per file, but row groups are smaller chunks within the file.
        )