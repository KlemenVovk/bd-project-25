import dask.dataframe as dd
from tqdm import tqdm
import numpy as np
import h5py
import os
import pandas as pd
import geopandas as gpd

def write_parquet_sequentially(dfs, root_dir, partition_on=None, schema=None, row_group_size=None):
    
    if isinstance(dfs, dd.DataFrame):
        dfs.to_parquet(
            path=root_dir,
            partition_on=partition_on,
            engine='pyarrow',
            schema=schema,
            write_index=False,
            compute=True,
            row_group_size = row_group_size if row_group_size else None,
        )
    else:
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
            row_group_size = row_group_size if row_group_size else None,
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
                row_group_size = row_group_size if row_group_size else None,
            )


def write_csv_sequentially(dfs, csv_file):
    for i in tqdm(range(len(dfs))):
        dfs[i].to_csv(
            header=(i == 0), # write header only for the first file
            filename=csv_file,
            single_file=True,
            index=False,
            mode="wt" if i == 0 else "a", # append/create if not exists
            compute=True
        )

def write_csv_parallel_concat(dfs, csv_file):
    df_concat = dd.concat(dfs, axis=0)
    df_concat.to_csv(
        header=True,
        filename=csv_file,
        single_file=True,
        index=False,
        mode="wt", # append/create if not exists
        compute=True,
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

def get_total_size_GB(paths):
    return round(sum(os.path.getsize(p) for p in paths) / (1024 * 1024 * 1024), 3)

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
