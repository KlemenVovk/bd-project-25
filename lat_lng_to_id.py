import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from paths import TAXI_ZONES_SHAPEFILE
import dask_geopandas as dgpd
import dask.dataframe as dd

"""
Taken directly from the dataset:
vendor_name	tpep_pickup_datetime	tpep_dropoff_datetime	passenger_count	trip_distance	pickup_longitude	pickup_latitude	Rate_Code	store_and_forward	dropoff_longitude	dropoff_latitude	payment_type	fare_amount	extra	mta_tax	tip_amount	tolls_amount	total_amount
VTS	         2009-01-04 02:52:00	  2009-01-04 03:02:00	              1	         2.63	      -73.991957	      40.721567	      NaN	              NaN	-73.993803	               40.695922	        CASH	        8.9	  0.5	    NaN	      0.00	         0.0	        9.40

### Pickup location
Pickup LocationID should be 148
https://gps-coordinates.org/my-location.php?lat=40.721567&lng=-73.991957
Visually: https://www.nyc.gov/assets/tlc/images/content/pages/about/taxi_zone_map_manhattan.jpg

### Dropoff location
Dropoff LocationID should be 33
https://gps-coordinates.org/my-location.php?lat=40.695922&lng=-73.993803
https://www.nyc.gov/assets/tlc/images/content/pages/about/taxi_zone_map_brooklyn.jpg

"""



def map_lat_lng_to_shape_id(original_df, shapefile, lng_col, lat_col, shape_id_col, output_col):
    # Load zone shapefile
    zones = gpd.read_file(shapefile)
    # Ensure the same projection
    zones = zones.to_crs("EPSG:4326")

    # Create GeoDataFrame with geometry
    geometry = [Point(xy) for xy in zip(original_df[lng_col], original_df[lat_col])]
    gdf = gpd.GeoDataFrame(original_df.loc[:, [lng_col, lat_col]], geometry=geometry, crs="EPSG:4326")
    
    # Perform spatial join to find the zone for each pickup point (within the zone)
    gdf_with_zone_pickup = gpd.sjoin(gdf, zones, how="left", predicate="within")
    # Rename the column to match the PULocationID
    original_df[output_col] = gdf_with_zone_pickup[shape_id_col]

    return original_df

def map_lat_lng_to_shape_id_dask_gdf(df, shapefile, lng_col, lat_col, shape_id_col, output_col):
    # Load the zone shapefile with GeoPandas
    zones = gpd.read_file(shapefile).to_crs("EPSG:4326")
    zones_gdf = dgpd.from_geopandas(zones)  # small file, no need to partition

    # Create GeoDataFrame with points
    df['geometry'] = df.apply(lambda row: Point(row[lng_col], row[lat_col]), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

    # Convert to Dask GeoDataFrame
    ddf = dgpd.from_geopandas(gdf)

    # Spatial join (this is now distributed!)
    joined = dgpd.sjoin(ddf, zones_gdf, how="inner", predicate="within")

    # Rename and return
    joined[output_col] = joined[shape_id_col]
    return joined.drop(columns=['geometry'])


sample_df = "data/raw/yellow_tripdata_2009-01.parquet"
df = dd.read_parquet(sample_df)
df = map_lat_lng_to_shape_id_dask_gdf(
    df,
    TAXI_ZONES_SHAPEFILE,
    "Start_Lon",
    "Start_Lat",
    "LocationID",
    "PULocationID"
)
df = map_lat_lng_to_shape_id_dask_gdf(
    df,
    TAXI_ZONES_SHAPEFILE,
    "End_Lon",
    "End_Lat",
    "LocationID",
    "DOLocationID"
)
df.to_csv("test2.csv", index=False)

    

