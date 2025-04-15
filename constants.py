import pyarrow as pa
import numpy as np

RAW_DATA_ROOT = "data/raw"
RESULTS_ROOT = "results"
# RAW_DATA_ROOT = "/d/hpc/projects/FRI/bigdata/data/Taxi/"
YEARS = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
          2020, 2021, 2022, 2023, 2024, 2025]

# TASK 1
TASK1_OUT_ROOT = "data/task1"
# TASK1_OUT_ROOT = "/d/hpc/projects/FRI/kv4582/bd25/task1"
TAXI_ZONES_SHAPEFILE = "data/zone_data/taxi_zones.shp"
ZONES_TO_CENTROIDS_MAPPING_CSV = "data/zone_data/zones_to_centroids_mapping.csv"

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

COLUMN_CONSISTENCY_NAMING_MAP = {
    "End_Lat": "dropoff_latitude",
    "End_Lon": "dropoff_longitude",
    "Start_Lat": "pickup_latitude",
    "Start_Lon": "pickup_longitude",
    "Fare_Amt": "fare_amount",
    "Tip_Amt": "tip_amount",
    "Tolls_Amt": "tolls_amount",
    "Total_Amt": "total_amount",
    "Passenger_Count": "passenger_count",
    "Payment_Type": "payment_type",
    "Rate_Code": "rate_code_id",
    "rate_code": "rate_code_id",
    "RatecodeID": "rate_code_id",
    "Trip_Distance": "trip_distance",
    "Trip_Dropoff_DateTime": "tpep_dropoff_datetime",
    "Trip_Pickup_DateTime": "tpep_pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
    "tpep_pickup_datetime": "pickup_datetime",
    "Airport_fee": "airport_fee",
    "VendorID": "vendor_id",
    "vendor_name": "vendor_id",
    "surcharge": "extra",
    "store_and_forward": "store_and_fwd_flag",
}
