import pandas as pd
from utils.constants import TAXI_ZONES_LOOKUP_CSV, TAXI_ZONES_SHAPEFILE, LOCATIONID_CENTROID_BOROUGH_CSV
import geopandas as gpd

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


if __name__ == "__main__":
    zones_df = pd.read_csv(TAXI_ZONES_LOOKUP_CSV)
    zones_centroids_df = get_locationid_to_centroid(TAXI_ZONES_SHAPEFILE)

    locationid_centroid_borough = zones_df.merge(
        zones_centroids_df,
        on='LocationID',
        how='inner',
    ).to_csv(LOCATIONID_CENTROID_BOROUGH_CSV, index=False)
