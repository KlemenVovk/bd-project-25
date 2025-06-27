import numpy as np
import pandas as pd
import dask.dataframe as dd
from glob import glob
import os
from constants import RAW_DATA_ROOT, TASK1_OUT_ROOT, TASK1_YELLOWTAXI_SCHEMA, COLUMN_CONSISTENCY_NAMING_MAP, LOCATIONID_CENTROID_BOROUGH_CSV
from tqdm import tqdm

def map_vendor(value):
    # vendor_id (previously vendor_name but stands for the same thing)
    _old_vendorname2id = {
        "CMT": 1,
        "DDS": 2, # DDS and VTS were merged (Verifone) and later also merged with Curb Mobility
        "VTS": 2,
    }
    valid_vendor_ids= [1, 2, 6, 7]
    if str(value) in _old_vendorname2id:
        return _old_vendorname2id[value]
    if float(value) in valid_vendor_ids:
        return int(value)
    return -1 # Invalid vendor id
    
def map_rate_code(value):
    # rate_code_id
    # 1-6 and 99 (missing or unknown). All other values are set to 99
    valid_rate_code_ids = [1, 2, 3, 4, 5, 6, 99]
    if float(value) in valid_rate_code_ids:
        return int(value)
    return 99 # Invalid rate code id
    
def map_store_and_fwd_flag(value):
    # store_and_fwd_flag
    # npnan to "NA"
    letter_mapping = {
        "Y": 1,
        "N": 0,
    }
    if str(value).strip() in letter_mapping:
        return letter_mapping[str(value)]
    try:
        v = int(float(value))
        if v in [0, 1]:
            return v
    except:
        pass
    return -1

def map_payment_type(value):
    # ignore case and strip
    payment_type_mapping = {
        "credit": 1,
        "crd": 1,
        "cre":1,
        "cash": 2,
        "csh": 2,
        "cas": 2,
        "noc": 3,
        "no charge": 3,
        "no": 3,
        "dis": 4,
        "dispute": 4,
        "na": 5,
    }
    valid_vals = [0, 1, 2, 3, 4, 5, 6]
    if str(value).strip().lower() in payment_type_mapping:
        return payment_type_mapping[str(value).strip().lower()]
    if float(value) in valid_vals:
        return int(float(value))
    return 5 # unknown

if __name__ == "__main__":
    pd.set_option('future.no_silent_downcasting', True)

    print("Reading files...")
    files = sorted(glob(os.path.join(RAW_DATA_ROOT, "yellow_tripdata/*.parquet")))
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
        dfs[i]['vendor_id'] = dfs[i]['vendor_id'].map(map_vendor, meta=('vendor_id', 'int8'))
        dfs[i]['rate_code_id'] = dfs[i]['rate_code_id'].map(map_rate_code, meta=('rate_code_id', 'int8'))
        dfs[i]['store_and_fwd_flag'] = dfs[i]['store_and_fwd_flag'].map(map_store_and_fwd_flag, meta=('store_and_fwd_flag', 'int8'))
        dfs[i]['payment_type'] = dfs[i]['payment_type'].map(map_payment_type, meta=('payment_type', 'int8'))
        # pickup
        if "PULocationID" in dfs[i].columns:
            # set dtype to int
            dfs[i]['PULocationID'] = dfs[i]['PULocationID'].astype(int)
            # show any pulocationds that are not in the locationid_to_centers_df
            dfs[i] = dfs[i].merge(location_centroid_borough_df, how='left', left_on='PULocationID', right_on='LocationID')
            # rename to pickup lattitude and longitude
            dfs[i] = dfs[i].rename(columns={'centroid_lat': 'pickup_latitude', 'centroid_lng': 'pickup_longitude', 'Borough': 'pickup_borough'})

        # dropoff
        if "DOLocationID" in dfs[i].columns:
            # set dtype to int
            dfs[i]['DOLocationID'] = dfs[i]['DOLocationID'].astype(int)
            dfs[i] = dfs[i].merge(location_centroid_borough_df, how='left', left_on='DOLocationID', right_on='LocationID')
            # rename to dropoff lattitude and longitude
            dfs[i] = dfs[i].rename(columns={'centroid_lat': 'dropoff_latitude', 'centroid_lng': 'dropoff_longitude', 'Borough': 'dropoff_borough'})
        # drop LocationID columns
        dfs[i] = dfs[i].drop(columns=['LocationID_x', 'LocationID_y', 'Zone_y', 'service_zone_y', 'Zone_x', 'service_zone_x'], errors='ignore')

        if 'airport_fee' not in dfs[i].columns:
            # set airport_fee to 0
            dfs[i]['airport_fee'] = 0.0
        if 'congestion_surcharge' not in dfs[i].columns:
            dfs[i]['congestion_surcharge'] = 0.0
        if 'improvement_surcharge' not in dfs[i].columns:
            dfs[i]['improvement_surcharge'] = 0.0
        dfs[i]['mta_tax'] = dfs[i]['mta_tax'].fillna(0.0)
        # parse datetimes
        dfs[i]['pickup_datetime'] = dd.to_datetime(dfs[i]['pickup_datetime'], errors='raise')
        dfs[i]['dropoff_datetime'] = dd.to_datetime(dfs[i]['dropoff_datetime'], errors='raise')
        dfs[i]['year'] = dfs[i]['pickup_datetime'].dt.year.astype(np.int16)
        dfs[i]['airport_fee'] = dfs[i]['airport_fee'].fillna(0.0).astype(float)
        dfs[i]['congestion_surcharge'] = dfs[i]['congestion_surcharge'].fillna(0.0).astype(float)
        dfs[i]['improvement_surcharge'] = dfs[i]['improvement_surcharge'].fillna(0.0).astype(float)
        dfs[i]['extra'] = dfs[i]['extra'].fillna(0.0).astype(float)
        dfs[i]['passenger_count'] = dfs[i]['passenger_count'].fillna(0).astype(int)

    
    # Save everything
    # Up to here, everything is embarrasingly parallelizable (a `dask.compute(*dfs)` would work in parallel - tested).
    # We are saving files sequentially, since multiprocessing append doesn't really work that well in to_parquet or when writing to the same CSV file.
    # Furthermore, even if we do parallelize, the bottleneck is in disk I/O.

    # Compute everything in parallel (persist)
    # dfs = client.persist(dfs)
    # Concat and repartition so that a division is every 100_000 rows.
    
    for ddf in tqdm(dfs):
        ddf = ddf.astype(TASK1_YELLOWTAXI_SCHEMA)     
        ddf.to_parquet(
            os.path.join(TASK1_OUT_ROOT, "yellow_tripdata"),
            engine='pyarrow',
            compression='snappy',
            partition_on=['year'],
            write_index=False,
            append=i > 0,
            overwrite=False,
            row_group_size=2_000_000, # Row group != partition - dask has 1 partition per file, but row groups are smaller chunks within the file.
        )