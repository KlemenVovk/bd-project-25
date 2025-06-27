import numpy as np
import pandas as pd
import dask.dataframe as dd
from glob import glob
import os
from constants import RAW_DATA_ROOT, TASK1_OUT_ROOT, TASK1_FHV_SCHEMA, COLUMN_CONSISTENCY_NAMING_MAP, LOCATIONID_CENTROID_BOROUGH_CSV
from tqdm import tqdm

if __name__ == "__main__":
    pd.set_option('future.no_silent_downcasting', True)

    print("Reading files...")
    files = sorted(glob(os.path.join(RAW_DATA_ROOT, "fhv_tripdata/*.parquet")))
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

        # parse datetimes
        dfs[i]['pickup_datetime'] = dd.to_datetime(dfs[i]['pickup_datetime'], errors='raise')
        dfs[i]['dropoff_datetime'] = dd.to_datetime(dfs[i]['dropoff_datetime'], errors='raise')
        # Drop rows where timestamp is out of datetimens[64] bounds
        dfs[i] = dfs[i][(dfs[i]['pickup_datetime'] >= pd.Timestamp.min) & (dfs[i]['pickup_datetime'] <= pd.Timestamp.max)]
        dfs[i] = dfs[i][(dfs[i]['dropoff_datetime'] >= pd.Timestamp.min) & (dfs[i]['dropoff_datetime'] <= pd.Timestamp.max)]
        dfs[i]['year'] = dfs[i]['pickup_datetime'].dt.year.astype(np.int16)
        dfs[i]['SR_flag'] = dfs[i]['SR_flag'].fillna(0).astype(np.int8)  # Fill NaN with 0 and convert to int8

    
    # Save everything
    # Up to here, everything is embarrasingly parallelizable (a `dask.compute(*dfs)` would work in parallel - tested).
    # We are saving files sequentially, since multiprocessing append doesn't really work that well in to_parquet or when writing to the same CSV file.
    # Furthermore, even if we do parallelize, the bottleneck is in disk I/O.

    # Compute everything in parallel (persist)
    # dfs = client.persist(dfs)
    # Concat and repartition so that a division is every 100_000 rows.
    
    for ddf in tqdm(dfs):
        # Check that all columns are exactly like the schema and ensure dtypes
        assert sorted(set(TASK1_FHV_SCHEMA.keys())) == sorted(set(ddf.columns.tolist())), f"Columns in ddf {sorted(ddf.columns.tolist())} do not match the schema {sorted(TASK1_FHV_SCHEMA.keys())}"
        ddf = ddf.astype(TASK1_FHV_SCHEMA)     
        ddf.to_parquet(
            os.path.join(TASK1_OUT_ROOT, "fhv_tripdata"),
            engine='pyarrow',
            compression='snappy',
            partition_on=['year'],
            write_index=False,
            append=i > 0,
            overwrite=False,
            row_group_size=2_000_000, # Row group != partition - dask has 1 partition per file, but row groups are smaller chunks within the file.
        )