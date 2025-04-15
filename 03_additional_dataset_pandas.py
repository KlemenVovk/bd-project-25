import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Point
import ast
import numpy as np


# ZIP to coordinates
ZIP_PATH = 'data/US.txt'
def load_zip_data(PATH):
    zip = pd.read_csv(PATH, sep='\t', header=None)
    zip = zip.iloc[:, [4, 1, 9, 10]]
    zip.columns = ['State', 'ZIP Code', 'lat', 'lng']
    return zip


# Business data

BUSINESS_PATH = 'data/businesses.csv'
def load_business_data(PATH):
    bsns = pd.read_csv('data/businesses.csv')
    bsns['ZIP Code'] = pd.to_numeric(bsns['ZIP Code'], errors='coerce')
    bsns = bsns.dropna(subset=['ZIP Code'])
    bsns['ZIP Code'] = bsns['ZIP Code'].astype(int)

    bsns = bsns[['Business Name','ZIP Code', 'Latitude', 'Longitude']]
    bsns = pd.merge(bsns, zip, how='left', on='ZIP Code')
    bsns['Latitude'] = bsns['Latitude'].fillna(bsns['lat'])
    bsns['Longitude'] = bsns['Longitude'].fillna(bsns['lng'])
    bsns = bsns.drop(columns=['lat', 'lng', 'State', 'ZIP Code'])
    bsns = bsns.dropna(subset=['Latitude', 'Longitude'])
    bsns['Location'] = bsns.apply(lambda x: Point(x['Longitude'], x['Latitude']), axis=1)
    bsns = gpd.GeoDataFrame(bsns, geometry='Location', crs="EPSG:4326").to_crs(epsg=3857)
    bsns.drop(columns=['Latitude', 'Longitude'], inplace=True)
    bsns['Business Name'] = bsns['Business Name'].astype(str)
    
    return bsns



# Weather data
WEATHER_PATH = 'data/weather_data.csv'
def load_weather_data(PATH):
    weather = pd.read_csv(PATH)
    weather['time'] = pd.to_datetime(weather['time'])
    weather['time'] = weather['time'].dt.tz_localize('GMT').dt.tz_convert('America/New_York')
    weather['time'] = weather['time'].dt.tz_localize(None)
    weather['Temperature'].astype(float)
    weather['Weather Code'].astype(int)
    return weather



# Schools

SCHOOLS_PATH = 'data/schools.csv'
def load_schools_data(PATH):
    schools = pd.read_csv('data/schools.csv')
    schools['Name'] = schools['nta_name'].astype(str) + ' ' + schools['location_category_description'].astype(str)
    schools = schools[['Name', 'latitude', 'longitude']]
    schools['Location'] = schools.apply(lambda r: Point(r['longitude'], r['latitude']), axis=1)
    schools = gpd.GeoDataFrame(schools, geometry='Location', crs="EPSG:4326").to_crs(epsg=3857)
    schools.drop(columns=['longitude', 'latitude'], inplace=True)
    schools['Name'] = schools['Name'].astype(str)
    return schools


# Events

EVENTS_PATH = 'data/events.csv'
def preload_events_data(PATH):
    events = pd.read_csv('data/events.csv', nrows=10000)
    events = events[['Event Name', 'Start Date/Time', 'End Date/Time', 'Police Precinct']]
    events['Police Precinct'] = events['Police Precinct'].apply(lambda x: str(x).split(',')[0])
    events['Police Precinct'] = events['Police Precinct'].astype(int)
    events['Event Name'] = events['Event Name'].astype(str)
    events['Start Date/Time'] = pd.to_datetime(events['Start Date/Time'])
    events['End Date/Time'] = pd.to_datetime(events['End Date/Time'])

    return events

PRECINCTS_PATH = 'data/police_precincts.csv'
def preload_precincts_data(PATH):
    precincts = pd.read_csv('data/police_precincts.csv')
    precincts['the_geom'] = precincts['the_geom'].apply(ast.literal_eval)
    precincts = precincts[['precinct', 'the_geom']]


    precincts['geometry'] = precincts['the_geom'].apply(shape)
    precincts.drop(columns=['the_geom'], inplace=True)
    gdf = gpd.GeoDataFrame(precincts, geometry='geometry')
    gdf['centroid'] = gdf.geometry.centroid


    gdf['lat'] = gdf.centroid.y
    gdf['lng'] = gdf.centroid.x

    gdf.drop(columns=['centroid', 'geometry'], inplace=True)

    gdf['precinct'] = gdf['precinct'].astype(int)
    
    return gdf

def load_events_data(events, precincts):
    merged = pd.merge(events, precincts, how='left', left_on='Police Precinct', right_on='precinct')
    merged = merged.dropna(subset=['lat', 'lng'])
    merged = merged.drop(columns=['precinct', 'Police Precinct'])
    merged['Location'] = merged.apply(lambda r: Point(r['lng'], r['lat']), axis=1)
    merged = gpd.GeoDataFrame(merged, geometry='Location', crs="EPSG:4326").to_crs(epsg=3857)
    merged.drop(columns=['lat', 'lng'], inplace=True)
    
    return merged


if __name__ == "__main__":
    FILE_LOCATION = 'data/part.191.parquet'
    EVENT_DISTANCE = 1000
    EVENT_HOUR_LIMIT = 5
    
    
    df = pd.read_parquet(FILE_LOCATION)
    df = df.head(10000)

    df = df.dropna(subset=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])
    df['PU Location'] = df.apply(lambda r: Point(r['pickup_longitude'], r['pickup_latitude']), axis=1)
    # df['DO Location'] = df.apply(lambda r: Point(r['dropoff_longitude'], r['dropoff_latitude']), axis=1)



    pickup_gdf = gpd.GeoDataFrame(df, 
                                geometry=df['PU Location'], 
                                crs="EPSG:4326").to_crs(epsg=3857).copy()
    # dropoff_gdf = gpd.GeoDataFrame(df, 
    #                                 geometry=df['DO Location'], 
    #                                 crs="EPSG:4326").to_crs(epsg=3857).copy()

    schools = load_schools_data(SCHOOLS_PATH)
    weather = load_weather_data(WEATHER_PATH)
    zip = load_zip_data(ZIP_PATH)
    bsns = load_business_data(BUSINESS_PATH)
    preevents = preload_events_data(EVENTS_PATH)
    precincts = preload_precincts_data(PRECINCTS_PATH)
    events = load_events_data(preevents, precincts)
    
    pickup_gdf = gpd.sjoin_nearest(pickup_gdf, schools, how='left', distance_col='dist_from_school')
    pickup_gdf = pickup_gdf.groupby(['pickup_datetime', 'dropoff_datetime']).first().reset_index()
    pickup_gdf = pickup_gdf.drop(columns=['index_right'])


    pickup_gdf['rounded_time'] = pickup_gdf['pickup_datetime'].dt.round('H')
    pickup_gdf = pd.merge(pickup_gdf, weather, how='left', left_on='rounded_time', right_on='time')
    
    pickup_gdf = pickup_gdf = gpd.sjoin_nearest(pickup_gdf, bsns, how='left', distance_col='dist_from_bsns')
    pickup_gdf = pickup_gdf.groupby(['pickup_datetime', 'dropoff_datetime']).first().reset_index()
    pickup_gdf = pickup_gdf.drop(columns=['index_right'])
    
    
    pickup_gdf = gpd.sjoin(pickup_gdf, events, how='left', distance=f'{EVENT_DISTANCE}', predicate='dwithin')
    pickup_gdf['time_diff'] = np.abs((pickup_gdf['Start Date/Time'] - pickup_gdf['pickup_datetime']).dt.total_seconds() / 3600)
    pickup_gdf = pickup_gdf[pickup_gdf['time_diff'] <= EVENT_HOUR_LIMIT]
    pickup_gdf = pickup_gdf.drop(columns=['index_right'])
    
    pickup_gdf.to_csv('data/processed_data.csv', index=False)
    


