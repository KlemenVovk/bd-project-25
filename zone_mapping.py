import pandas as pd
from constants import TAXI_ZONES_LOOKUP_CSV, TAXI_ZONES_SHAPEFILE, LOCATIONID_CENTROID_BOROUGH_CSV
from utils import get_locationid_to_centroid

zones_df = pd.read_csv(TAXI_ZONES_LOOKUP_CSV)
zones_centroids_df = get_locationid_to_centroid(TAXI_ZONES_SHAPEFILE)

locationid_centroid_borough = zones_df.merge(
    zones_centroids_df,
    on='LocationID',
    how='inner',
).to_csv(LOCATIONID_CENTROID_BOROUGH_CSV, index=False)
