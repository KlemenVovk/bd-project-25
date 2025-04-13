import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import dask.dataframe as dd
from glob import glob
# from dask.distributed import LocalCluster, Client
# from dask_jobqueue import SLURMCluster
import dask
import os
from constants import RAW_DATA_ROOT, YEARS, TASK1_OUT_ROOT, TAXI_ZONES_SHAPEFILE, ZONES_TO_CENTROIDS_MAPPING_CSV, TASK1_SCHEMA, TASK1_NP_SCHEMA, COLUMN_CONSISTENCY_NAMING_MAP
from tqdm import tqdm

def get_raw_files(root_path, year):
    """
    Get all original files in the given path for the specified year.
    """
    return sorted(list(glob(f"{root_path}/yellow_tripdata_{year}*.parquet")))[-LIMIT:]

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

def get_locationid_to_centroid(shapefile):
    # Load zones as GeoDataFrame
    zones = gpd.read_file(shapefile)

    # Reproject to NYC's local projected CRS (US feet) for correct geometry math
    zones_projected = zones.to_crs("EPSG:2263")

    # Calculate centroid in projected space
    zones_projected['centroid'] = zones_projected.geometry.centroid

    # Convert centroid back to WGS84 (lat/lon)
    centroids_wgs84 = zones_projected.set_geometry('centroid').to_crs("EPSG:4326")

    # Extract lat/lng from centroid geometry
    zones['centroid_lat'] = centroids_wgs84.geometry.y
    zones['centroid_lng'] = centroids_wgs84.geometry.x

    zones = zones[['OBJECTID', 'centroid_lat', 'centroid_lng']]
    zones['centroid_lat'] = zones['centroid_lat'].astype(float)
    zones['centroid_lng'] = zones['centroid_lng'].astype(float)
    zones['OBJECTID'] = zones['OBJECTID'].astype(int)

    # create a pandas dataframe with the same columns
    # Apparently, there are duplicate LocationIDs that are mapped to unique Objectids so the matching later doesnt work.
    # Have checked the taxi zone map and visually it is ok (some locationids are shown on the map, that aren't in the LocationID colmun, but are in the objectid column)
    # I.e. ObjectID=56, and 57 are present but both are mapped to LocationID=56 so the join fails...
    zones = pd.DataFrame(zones[['OBJECTID', 'centroid_lat', 'centroid_lng']].copy())
    zones = zones.rename(columns={'OBJECTID': 'LocationID'})
    return zones

def write_parquet_sequentially(dfs, root_dir, partition_on=None, schema=None):
    # Write the first file, as parquet can only append if a dataset already exists.
    assert isinstance(dfs, list), "dfs must be a list of dataframes"
    assert len(dfs) > 0, "No dataframes to write"
    dfs[0].to_parquet(
        path=root_dir,
        partition_on=partition_on,
        engine='pyarrow',
        schema=schema,
        write_index=False,
        compute=True,
    )

    # Write the rest of the dataframes
    for i in tqdm(range(1, len(dfs))):
        dfs[i].to_parquet(
            path=root_dir,
            partition_on=partition_on,
            engine='pyarrow',
            schema=schema,
            write_index=False,
            append=True,
            compute=True,
        )


def write_csv_sequentially(dfs, csv_file):
    for i in tqdm(range(len(dfs))):
        dfs[i].to_csv(
            filename=csv_file,
            single_file=True,
            index=False,
            mode="at", # append/create if not exists
            compute=True
        )

def write_csv_parallel_concat(dfs, csv_file):
    df_concat = dd.concat(dfs, axis=0)
    df_concat.to_csv(
        filename=csv_file,
        single_file=True,
        index=False,
        mode="wt", # append/create if not exists
        compute=True
    )

def write_hdf5_parallel_concat(dfs, hdf_filepath):
    df = dd.concat(dfs, axis=0)
    # convert datetimes to int64 (h5py can't handle datetimes directly)
    for col in df.select_dtypes(include=['datetime64[ns]']):
        df[col] = df[col].astype('int64')

    df = df.compute()
    # to numpy (structured array) as h5py can't handle dataframes directly
    dtype = np.dtype([(col, df[col].dtype) for col in df.columns])
    structured_array = np.empty(len(df), dtype=dtype)
    for col in df.columns:
        structured_array[col] = df[col].to_numpy()

    # write as single dataset
    with h5py.File(hdf_filepath, 'w') as h5f:
        h5f.create_dataset('taxidata', data=structured_array, compression='gzip')


if __name__ == "__main__":
    # Local cluster
    cluster = LocalCluster()
    client = cluster.get_client()
    print(cluster.dashboard_link)

    # # SLURM cluster
    # cluster = SLURMCluster(
    #     cores=4,
    #     processes=1,
    #     memory="100GB",
    #     walltime="12:00:00",
    #     death_timeout=600,
    # )
    # cluster.adapt(minimum=1, maximum=2)
    # print(cluster.job_script())
    # client = Client(cluster)

    pd.set_option('future.no_silent_downcasting', True)

    print("Reading files...")
    files = sum((get_raw_files(RAW_DATA_ROOT, year) for year in YEARS), [])
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
    locationid_to_centers_df = get_locationid_to_centroid(TAXI_ZONES_SHAPEFILE).sort_values(by='LocationID', ascending=True, ignore_index=True)
    locationid_to_centers_df.to_csv(ZONES_TO_CENTROIDS_MAPPING_CSV, index=False)
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
            dfs[i] = dfs[i].merge(locationid_to_centers_df, how='left', left_on='PULocationID', right_on='LocationID')
            # rename to pickup lattitude and longitude
            dfs[i] = dfs[i].rename(columns={'centroid_lat': 'pickup_latitude', 'centroid_lng': 'pickup_longitude'})

        # dropoff
        if "DOLocationID" in dfs[i].columns:
            # set dtype to int
            dfs[i]['DOLocationID'] = dfs[i]['DOLocationID'].astype(int)
            dfs[i] = dfs[i].merge(locationid_to_centers_df, how='left', left_on='DOLocationID', right_on='LocationID')
            # rename to dropoff lattitude and longitude
            dfs[i] = dfs[i].rename(columns={'centroid_lat': 'dropoff_latitude', 'centroid_lng': 'dropoff_longitude'})
        # drop LocationID columns
        dfs[i] = dfs[i].drop(columns=['DOLocationID', 'PULocationID', 'LocationID_x', 'LocationID_y'], errors='ignore')

        if 'airport_fee' not in dfs[i].columns:
            # set airport_fee to 0
            dfs[i]['airport_fee'] = 0.0
        if 'congestion_surcharge' not in dfs[i].columns:
            dfs[i]['congestion_surcharge'] = 0.0
        if 'improvement_surcharge' not in dfs[i].columns:
            dfs[i]['improvement_surcharge'] = 0.0
        year = int(os.path.basename(f).split('_')[2][:4])
        dfs[i]['year'] = year
        dfs[i]['year'] = dfs[i]['year'].astype(np.int16)
        dfs[i]['mta_tax'] = dfs[i]['mta_tax'].fillna(0.0)
        # parse datetimes
        dfs[i]['pickup_datetime'] = dd.to_datetime(dfs[i]['pickup_datetime'], errors='raise')
        dfs[i]['dropoff_datetime'] = dd.to_datetime(dfs[i]['dropoff_datetime'], errors='raise')
        dfs[i]['airport_fee'] = dfs[i]['airport_fee'].fillna(0.0).astype(float)
        dfs[i]['congestion_surcharge'] = dfs[i]['congestion_surcharge'].fillna(0.0).astype(float)
        dfs[i]['improvement_surcharge'] = dfs[i]['improvement_surcharge'].fillna(0.0).astype(float)
        dfs[i]['extra'] = dfs[i]['extra'].fillna(0.0).astype(float)
        dfs[i]['passenger_count'] = dfs[i]['passenger_count'].fillna(0).astype(int)
        # Fix dtypes
        dfs[i] = dfs[i].astype(TASK1_NP_SCHEMA)
        # print shape of each dataframe and the corresponding file

    # Save everything
    # Up to here, everything is embarrasingly parallelizable (a `dask.compute(*dfs)` would work in parallel - tested).
    # We are saving files sequentially, since multiprocessing append doesn't really work that well in to_parquet or when writing to the same CSV file.
    # Furthermore, even if we do parallelize, the bottleneck is in disk I/O.

    # Compute everything in parallel (persist)
    # dfs = client.persist(dfs)

    print("Writing files...")
    # Create needed folders
    os.makedirs(os.path.join(TASK1_OUT_ROOT, "one_year"), exist_ok=True)
    os.makedirs(os.path.join(TASK1_OUT_ROOT, "five_years"), exist_ok=True)
    os.makedirs(os.path.join(TASK1_OUT_ROOT, "all"), exist_ok=True)

    # # Write parquet (all)
    parquet_root = os.path.join(TASK1_OUT_ROOT, "all")
    print(f"Writing Parquet (all) to {parquet_root}...")
    write_parquet_sequentially(dfs, parquet_root, partition_on=['year'], schema=TASK1_SCHEMA)

    # CSV (5 years)
    csv_file = os.path.join(TASK1_OUT_ROOT, "five_years", "2020_2024.csv")
    print(f"Writing CSV (5 years) to {csv_file}...")
    write_csv_sequentially(dfs[-61:-1], csv_file)
    
    # CSV (1 year)
    csv_file = os.path.join(TASK1_OUT_ROOT, "one_year", "2024.csv")
    print(f"Writing CSV (1 year) to {csv_file}...")
    write_csv_sequentially(dfs[-13:-1], csv_file) 

    # HDF (1 year)
    # as a proof of concept - we can concat in parallel, an then write to a single file, however this just uses a lot of memory and is not really needed.
    # since the bottleneck is writing to disk not computation and then workers spend a lot of time just waiting.
    # using h5py as dask's implementation has problems (it can only save in the "tables" format resulting in a larger file than the equivalent CSV...)
    hdf_file = os.path.join(TASK1_OUT_ROOT, "one_year", "2024.h5")
    print(f"Writing HDF5 (1 year) to {hdf_file}...")
    write_hdf5_parallel_concat(dfs[-13:-1], hdf_file)