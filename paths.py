import pyarrow as pa
from glob import glob

ROOT = "data/raw"
# ROOT = "/d/hpc/projects/FRI/bigdata/data/Taxi/"
YEARS = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
          2020, 2021, 2022, 2023, 2024, 2025]

SELECTED_COLUMNS = [ # works for all [2020,2021,2022,2023,2024]
  'Airport_fee',
  'DOLocationID',
  'PULocationID',
  'RatecodeID',
  'VendorID',
  'congestion_surcharge',
  'extra',
  'fare_amount',
  'improvement_surcharge',
  'mta_tax',
  'passenger_count',
  'payment_type',
  'store_and_fwd_flag',
  'tip_amount',
  'tolls_amount',
  'total_amount',
  'tpep_dropoff_datetime',
  'tpep_pickup_datetime',
  'trip_distance',
]

# A mapping of seen column names to their expected names (the ones in SELECTED_COLUMNS) so we can concat
CONSISTENCY_COL_MAPPING = {
    "airport_fee": "Airport_fee",
}

TASK1_FINAL_COLUMNS = [
  'Airport_fee',
  'DOLocationID',
  'PULocationID',
  'RatecodeID',
  'VendorID',
  'congestion_surcharge',
  'extra',
  'fare_amount',
  'improvement_surcharge',
  'mta_tax',
  'passenger_count',
  'payment_type',
  'store_and_fwd_flag',
  'tip_amount',
  'tolls_amount',
  'total_amount',
  'tpep_dropoff_datetime',
  'tpep_pickup_datetime',
  'trip_distance',
  'year_from_filename' # added in task1 to partition on
]

# SCHEMA
TASK1_SCHEMA = pa.schema([
    ("VendorID", pa.int8()),
    ("tpep_pickup_datetime", pa.timestamp("us")),
    ("tpep_dropoff_datetime", pa.timestamp("us")),
    ("passenger_count", pa.float32()), # can be NaN
    ("trip_distance", pa.float32()),
    ("RatecodeID", pa.float32()), # can be NaN
    ("store_and_fwd_flag", pa.string()),
    ("PULocationID", pa.int32()),
    ("DOLocationID", pa.int32()),
    ("payment_type", pa.int8()),
    ("fare_amount", pa.float32()),
    ("extra", pa.float32()),
    ("mta_tax", pa.float32()),
    ("tip_amount", pa.float32()),
    ("tolls_amount", pa.float32()),
    ("improvement_surcharge", pa.float32()),
    ("total_amount", pa.float32()),
    ("congestion_surcharge", pa.float32()),
    ("Airport_fee", pa.float32()),
    ("year_from_filename", pa.int32()),
])
TASK1_PARQUET_OUT_ROOT = "data/task1/parquet/"
TAXI_ZONES_SHAPEFILE = "data/zone_data/taxi_zones.shp"


def get_files(root_path, year):
    """
    Get all files in the given path for the specified year.
    """
    return sorted(list(glob(f"{root_path}/yellow_tripdata_{year}*.parquet")))