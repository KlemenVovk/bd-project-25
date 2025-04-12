import pyarrow as pa
from glob import glob

ROOT = "data/raw"
# ROOT = "/d/hpc/projects/FRI/bigdata/data/Taxi/"
YEARS = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
          2020, 2021, 2022, 2023, 2024, 2025]


# # SCHEMA
# TASK1_SCHEMA = pa.schema([
#     ("VendorID", pa.int8()),
#     ("tpep_pickup_datetime", pa.timestamp("us")),
#     ("tpep_dropoff_datetime", pa.timestamp("us")),
#     ("passenger_count", pa.float32()), # can be NaN
#     ("trip_distance", pa.float32()),
#     ("RatecodeID", pa.float32()), # can be NaN
#     ("store_and_fwd_flag", pa.string()),
#     ("PULocationID", pa.int32()),
#     ("DOLocationID", pa.int32()),
#     ("payment_type", pa.int8()),
#     ("fare_amount", pa.float32()),
#     ("extra", pa.float32()),
#     ("mta_tax", pa.float32()),
#     ("tip_amount", pa.float32()),
#     ("tolls_amount", pa.float32()),
#     ("improvement_surcharge", pa.float32()),
#     ("total_amount", pa.float32()),
#     ("congestion_surcharge", pa.float32()),
#     ("Airport_fee", pa.float32()),
#     ("year_from_filename", pa.int32()),
# ])

TASK1_OUT_ROOT = "data/task1"
TAXI_ZONES_SHAPEFILE = "data/zone_data/taxi_zones.shp"
ZONES_TO_CENTROIDS_MAPPING_CSV = "data/zone_data/zones_to_centroids_mapping.csv"



def get_files(root_path, year):
    """
    Get all files in the given path for the specified year.
    """
    return sorted(list(glob(f"{root_path}/yellow_tripdata_{year}*.parquet")))