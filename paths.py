import pyarrow as pa
from glob import glob
import numpy as np

ROOT = "data/raw"
# ROOT = "/d/hpc/projects/FRI/bigdata/data/Taxi/"
YEARS = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
          2020, 2021, 2022, 2023, 2024, 2025]


TASK1_SCHEMA = pa.schema([
    ("vendor_id", pa.int8()),
    ("pickup_datetime", pa.timestamp("ns")),
    ("dropoff_datetime", pa.timestamp("ns")),
    ("passenger_count", pa.uint8()),
    ("trip_distance", pa.float32()),
    ("rate_code_id", pa.uint8()),
    ("store_and_fwd_flag", pa.int8()),
    ("payment_type", pa.uint8()),
    ("fare_amount", pa.float32()),
    ("extra", pa.float32()),
    ("mta_tax", pa.float32()),
    ("tip_amount", pa.float32()),
    ("tolls_amount", pa.float32()),
    ("improvement_surcharge", pa.float32()),
    ("total_amount", pa.float32()),
    ("congestion_surcharge", pa.float32()),
    ("airport_fee", pa.float32()),
    ("pickup_longitude", pa.float32()),
    ("pickup_latitude", pa.float32()),
    ("dropoff_longitude", pa.float32()),
    ("dropoff_latitude", pa.float32()),
    ("year", pa.uint16()),
])

TASK1_NP_SCHEMA = {
    "vendor_id": np.int8,
    "pickup_datetime": 'datetime64[ns]',
    "dropoff_datetime": 'datetime64[ns]',
    "passenger_count": np.uint8,
    "trip_distance": np.float32,
    "rate_code_id": np.uint8,
    "store_and_fwd_flag": np.int8,
    "payment_type": np.uint8,
    "fare_amount": np.float32,
    "extra": np.float32,
    "mta_tax": np.float32,
    "tip_amount": np.float32,
    "tolls_amount": np.float32,
    "improvement_surcharge": np.float32,
    "total_amount": np.float32,
    "congestion_surcharge": np.float32,
    "airport_fee": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "year": np.uint16
}


TASK1_OUT_ROOT = "data/task1"
TAXI_ZONES_SHAPEFILE = "data/zone_data/taxi_zones.shp"
ZONES_TO_CENTROIDS_MAPPING_CSV = "data/zone_data/zones_to_centroids_mapping.csv"



def get_files(root_path, year):
    """
    Get all files in the given path for the specified year.
    """
    return sorted(list(glob(f"{root_path}/yellow_tripdata_{year}*.parquet")))