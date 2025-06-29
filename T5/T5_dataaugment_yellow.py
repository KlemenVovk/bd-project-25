import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_augment import (
    load_weather_data,
    load_schools_data,
    load_business_data,
    load_zip_data,
    merge_business_zip,
    load_precincts_data,
    load_events_data,
    merge_events_precincts,
)
from utils.constants import (
    TASK2_OUT_ROOT,
    TASK5_OUT_ROOT,
    RAW_DATA_ROOT,
)
import dask.dataframe as dd
import dask_geopandas as dgpd
import os
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point
from glob import glob
from tqdm import tqdm

DATA_PATH = os.path.join(RAW_DATA_ROOT, "augment_data")


# Load cleaned data
input_parquet_files = sorted(
    glob(os.path.join(TASK2_OUT_ROOT, "yellow_tripdata", "**", "*.parquet"))
)
print(f"Found {len(input_parquet_files)} input files from: {TASK2_OUT_ROOT}")

dfs = [dd.read_parquet(file, engine="pyarrow") for file in input_parquet_files]
augmented_dfs = []

for i in tqdm(range(len(dfs))):
    df = dfs[i].dropna(
        subset=[
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ]
    )

    df["PU Location"] = df.apply(
        lambda r: Point(r["pickup_longitude"], r["pickup_latitude"]),
        axis=1,
        meta=("PU Location", "geometry"),
    )
    df["DO Location"] = df.apply(
        lambda r: Point(r["dropoff_longitude"], r["dropoff_latitude"]),
        axis=1,
        meta=("DO Location", "geometry"),
    )

    gdf = dgpd.from_dask_dataframe(df, geometry="DO Location")
    gdf = gdf.drop(columns=["PU Location"])
    gdf = gdf.set_crs("EPSG:4326").to_crs(3857)

    # Weather join
    weather_path = os.path.join(DATA_PATH, "weather_data.csv")
    weather_df = load_weather_data(weather_path)
    gdf["dropoff_datetime"] = dd.to_datetime(gdf["dropoff_datetime"])
    gdf["dropoff_time_rounded"] = gdf["dropoff_datetime"].dt.round("h")
    gdf = gdf.merge(
        weather_df, how="left", left_on="dropoff_time_rounded", right_on="time"
    )
    gdf = gdf.drop(columns=["dropoff_time_rounded", "time"])

    # Schools join
    school_path = os.path.join(DATA_PATH, "schools.csv")
    schools_df = load_schools_data(school_path).compute()
    dropoff_df = gdf.compute()

    dropoff_coords = np.array(list(zip(dropoff_df.geometry.x, dropoff_df.geometry.y)))
    school_coords = np.array(list(zip(schools_df.geometry.x, schools_df.geometry.y)))

    # remove schools with NaN or inf coordinates
    valid_schools = ~np.isnan(school_coords).any(axis=1) & ~np.isinf(school_coords).any(
        axis=1
    )
    school_coords = school_coords[valid_schools]

    tree = cKDTree(school_coords)
    distances, indices = tree.query(dropoff_coords, k=1)
    nearest_schools = schools_df.iloc[indices].reset_index(drop=True)
    dropoff_df["dist_from_school"] = distances
    dropoff_df["nearest_school_name"] = nearest_schools["Name"].values

    # Business join
    business_path = os.path.join(DATA_PATH, "businesses.csv")
    ZIP_PATH = os.path.join(DATA_PATH, "US.txt")
    bsns_df = load_business_data(business_path)
    zip_df = load_zip_data(ZIP_PATH)
    bsns_gdf = merge_business_zip(bsns_df, zip_df).compute()
    bsns_coords = np.array(list(zip(bsns_gdf.geometry.x, bsns_gdf.geometry.y)))
    tree = cKDTree(bsns_coords)
    distances, indices = tree.query(dropoff_coords, k=1)
    nearest_bsns = bsns_gdf.iloc[indices].reset_index(drop=True)
    dropoff_df["dist_from_business"] = distances
    dropoff_df["nearest_business_name"] = nearest_bsns["Business Name"].values
    dropoff_df = dropoff_df.drop(columns=["DO Location"])
    # Convert to regular dask dataframe
    dropoff_df = dd.from_pandas(dropoff_df)

    # Events join
    # precincts_path = os.path.join(DATA_PATH, "police_precincts.csv")
    # events_path = os.path.join(DATA_PATH, "events_sample.csv")
    # precincts_df = load_precincts_data(precincts_path)
    # events_df = load_events_data(events_path)
    # merged_events = merge_events_precincts(events_df, precincts_df).compute()
    #
    # dropoff_df = dropoff_df.reset_index(drop=True)
    # dropoff_df["row_id"] = dropoff_df.index
    # dropoff_buffered = dropoff_df.copy()
    # dropoff_buffered["DO Location"] = dropoff_buffered.buffer(100)
    #
    # matches = dgpd.sjoin(dropoff_buffered, merged_events, predicate="intersects")
    # dropoff_geometry = dropoff_df.geometry
    # matches_df = matches.drop(columns="DO Location")
    # matches_df = matches_df[
    #     [
    #         col
    #         for col in matches_df.columns
    #         if col not in dropoff_df.columns or col == "row_id"
    #     ]
    # ]
    #
    # joined = dd.merge(
    #     dd.from_pandas(dropoff_df, npartitions=4),
    #     dd.from_pandas(matches_df, npartitions=2),
    #     how="left",
    #     on="row_id",
    # )
    # joined["DO Location"] = dropoff_geometry
    # joined_gdf = dgpd.from_dask_dataframe(joined, geometry="DO Location")
    #
    # joined_gdf["time_diff"] = np.abs(
    #     (
    #         joined_gdf["Start Date/Time"] - joined_gdf["dropoff_datetime"]
    #     ).dt.total_seconds()
    #     / 3600
    # )
    # joined_gdf = joined_gdf[
    #     (joined_gdf["time_diff"] <= 1) | (joined_gdf["time_diff"].isna())
    # ]
    # joined_gdf = joined_gdf.drop(
    #     columns=["row_id", "index_right", "Start Date/Time", "End Date/Time"]
    # )

    # Save augmented DataFrame
    augmented_dfs.append(dropoff_df)

# Save all augmented data
print("Saving augmented data...")
output_path = os.path.join(TASK5_OUT_ROOT, "yellow_tripdata")
os.makedirs(output_path, exist_ok=True)

for i, ddf in enumerate(tqdm(augmented_dfs)):
    ddf.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        partition_on=["year"],
        write_index=False,
        append=i > 0,
        overwrite=False,
        row_group_size=2_000_000,
    )
