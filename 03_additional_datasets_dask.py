import dask.dataframe as dd
import geopandas as gpd
import dask_geopandas as dgpd
from shapely.geometry import shape, Point
import ast
import numpy as np
import pandas as pd  # still needed for some static data handling
from dask.distributed import Client, LocalCluster



# Constants
ZIP_PATH = 'data/US.txt'
BUSINESS_PATH = 'data/businesses.csv'
WEATHER_PATH = 'data/weather_data.csv'
SCHOOLS_PATH = 'data/schools.csv'
EVENTS_PATH = 'data/events.csv'
PRECINCTS_PATH = 'data/police_precincts.csv'
FILE_LOCATION = 'data/part.191.parquet'
OUTPUT_PATH = 'data/processed_data.csv'
EVENT_DISTANCE = 1000
EVENT_HOUR_LIMIT = 5

def load_zip_data(path):
    zip_df = dd.read_csv(path, sep='\t', header=None, usecols=[1, 4, 9, 10], names=['ZIP Code', 'State', 'lat', 'lng'])
    return zip_df

def load_business_data(path, zip_df):
    bsns = dd.read_csv(path)
    bsns['ZIP Code'] = dd.to_numeric(bsns['ZIP Code'], errors='coerce')
    bsns = bsns.dropna(subset=['ZIP Code'])
    bsns['ZIP Code'] = bsns['ZIP Code'].astype(int)

    bsns = bsns[['Business Name','ZIP Code', 'Latitude', 'Longitude']]
    bsns = bsns.merge(zip_df, how='left', on='ZIP Code')
    bsns['Latitude'] = bsns['Latitude'].fillna(bsns['lat'])
    bsns['Longitude'] = bsns['Longitude'].fillna(bsns['lng'])

    bsns = bsns.drop(columns=['lat', 'lng', 'State', 'ZIP Code'])
    bsns = bsns.dropna(subset=['Latitude', 'Longitude'])

    bsns_pd = bsns.compute()
    bsns_pd['Location'] = bsns_pd.apply(lambda x: Point(x['Longitude'], x['Latitude']), axis=1)
    bsns_gdf = dgpd.GeoDataFrame(bsns_pd, geometry='Location', crs="EPSG:4326").to_crs(epsg=3857)
    bsns_gdf.drop(columns=['Latitude', 'Longitude'], inplace=True)

    return dgpd.from_geopandas(bsns_gdf, npartitions=8)

def load_schools_data(path):
    schools = dd.read_csv(path)
    schools['Name'] = schools['nta_name'].astype(str) + ' ' + schools['location_category_description'].astype(str)
    schools = schools[['Name', 'latitude', 'longitude']]
    schools['Location'] = schools.apply(lambda r: Point(r['longitude'], r['latitude']), axis=1)
    schools_gdf = dgpd.GeoDataFrame(schools, geometry='Location', crs="EPSG:4326").to_crs(epsg=3857)
    schools_gdf.drop(columns=['longitude', 'latitude'], inplace=True)
    return dgpd.from_geopandas(schools_gdf, npartitions=4)

def load_weather_data(path):
    weather = dd.read_csv(path)
    weather['time'] = dd.to_datetime(weather['time'])
    weather['time'] = weather['time'].map(lambda t: t.tz_localize('GMT').tz_convert('America/New_York').tz_localize(None))
    weather['Temperature'] = weather['Temperature'].astype(float)
    weather['Weather Code'] = weather['Weather Code'].astype(int)
    return weather

def preload_events_data(path):
    events = dd.read_csv(path, usecols=['Event Name', 'Start Date/Time', 'End Date/Time', 'Police Precinct'], blocksize="16MB").head(10000)
    events = dd.DataFrame(events)
    events['Police Precinct'] = events['Police Precinct'].apply(lambda x: str(x).split(',')[0])
    events['Police Precinct'] = events['Police Precinct'].astype(int)
    events['Start Date/Time'] = dd.to_datetime(events['Start Date/Time'])
    events['End Date/Time'] = dd.to_datetime(events['End Date/Time'])
    return events

def preload_precincts_data(path):
    precincts = dd.read_csv(path)
    precincts['the_geom'] = precincts['the_geom'].apply(ast.literal_eval)
    precincts['geometry'] = precincts['the_geom'].apply(shape)
    precincts.drop(columns=['the_geom'], inplace=True)

    gdf = dgpd.GeoDataFrame(precincts, geometry='geometry', crs="EPSG:4326")
    gdf['centroid'] = gdf.geometry.centroid
    gdf['lat'] = gdf.centroid.y
    gdf['lng'] = gdf.centroid.x
    gdf['precinct'] = gdf['precinct'].astype(int)
    return gdf.drop(columns=['centroid', 'geometry'])

def load_events_data(events, precincts):
    merged = dd.merge(events, precincts, how='left', left_on='Police Precinct', right_on='precinct')
    merged = merged.dropna(subset=['lat', 'lng'])
    merged = merged.drop(columns=['precinct', 'Police Precinct'])
    merged['Location'] = merged.apply(lambda r: Point(r['lng'], r['lat']), axis=1)
    merged = gpd.GeoDataFrame(merged, geometry='Location', crs="EPSG:4326").to_crs(epsg=3857)
    return dgpd.from_geopandas(merged.drop(columns=['lat', 'lng']), npartitions=4)

def main():
    df = dd.read_parquet(FILE_LOCATION).head(10000)  # Or .persist() if using distributed cluster
    df = df.dropna(subset=['pickup_longitude', 'pickup_latitude'])

    df['PU Location'] = df.apply(lambda r: Point(r['pickup_longitude'], r['pickup_latitude']), axis=1, meta=('PU Location', 'object'))
    pickup_gdf = gpd.GeoDataFrame(df.compute(), geometry='PU Location', crs="EPSG:4326").to_crs(epsg=3857)
    pickup_gdf = dgpd.from_geopandas(pickup_gdf, npartitions=16)

    # Load dependencies
    zip_df = load_zip_data(ZIP_PATH)
    schools = load_schools_data(SCHOOLS_PATH)
    weather = load_weather_data(WEATHER_PATH)
    bsns = load_business_data(BUSINESS_PATH, zip_df)
    events_raw = preload_events_data(EVENTS_PATH)
    precincts = preload_precincts_data(PRECINCTS_PATH)
    events = load_events_data(events_raw, precincts)

    # Spatial joins
    pickup_gdf = dgpd.sjoin_nearest(pickup_gdf, schools, how='left', distance_col='dist_from_school')
    pickup_gdf = pickup_gdf.to_geopandas().groupby(['pickup_datetime', 'dropoff_datetime']).first().reset_index()
    
    pickup_gdf['rounded_time'] = pickup_gdf['pickup_datetime'].dt.round('H')
    weather_pd = weather.compute()
    pickup_gdf = dd.merge(pickup_gdf, weather_pd, how='left', left_on='rounded_time', right_on='time')

    pickup_gdf = dgpd.from_geopandas(pickup_gdf, npartitions=16)
    pickup_gdf = dgpd.sjoin_nearest(pickup_gdf, bsns, how='left', distance_col='dist_from_bsns')
    pickup_gdf = pickup_gdf.to_geopandas().groupby(['pickup_datetime', 'dropoff_datetime']).first().reset_index()

    # Events spatial join and temporal filter
    pickup_gdf = dgpd.from_geopandas(pickup_gdf, npartitions=16)
    pickup_gdf = dgpd.sjoin(pickup_gdf, events, how='left', predicate='dwithin', distance=EVENT_DISTANCE)
    pickup_gdf['time_diff'] = np.abs((pickup_gdf['Start Date/Time'] - pickup_gdf['pickup_datetime']).dt.total_seconds() / 3600)
    pickup_gdf = pickup_gdf[pickup_gdf['time_diff'] <= EVENT_HOUR_LIMIT]

    # Output
    pickup_gdf.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    # Local cluster
    cluster = LocalCluster()
    client = cluster.get_client()
    # print(cluster.dashboard_link)

    # # SLURM cluster
    # cluster = SLURMCluster(
    #     cores=4,
    #     processes=1,
    #     memory="100GB",
    #     walltime="12:00:00",
    #     death_timeout=600,
    # )
    # cluster.adapt(minimum=1, maximum=2)
    main()


