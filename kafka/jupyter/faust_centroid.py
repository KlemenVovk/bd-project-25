import faust
import math
from typing import Optional, List
import numpy as np

app = faust.App(
    'taxi-cluster-app',
    broker='kafka://broker1-kr:9092',
    value_serializer='json',
)

class TaxiRecord(faust.Record, serializer='json'):
    pickup_latitude: Optional[float]
    pickup_longitude: Optional[float]
    total_amount: Optional[float]
    passenger_count: Optional[int]

taxi_topic = app.topic('yellow_taxi_stream', value_type=TaxiRecord)

# Number of clusters
K = 3

# Initialize centroids arbitrarily (example coords)
initial_centroids = [
    {'lat': 40.7580, 'lon': -73.9855},  # Times Square approx
    {'lat': 40.7128, 'lon': -74.0060},  # Lower Manhattan approx
    {'lat': 40.730610, 'lon': -73.935242},  # East Village approx
]

# Table: key=cluster_id, value=dict with 'lat', 'lon', 'count'
centroids = app.Table(
    'centroids',
    default=lambda: {'lat': 0.0, 'lon': 0.0, 'count': 0},
    partitions=1,
    changelog_topic=app.topic('custom_stats_changelog_centroid', partitions=1)
)

initialized = False  # Move this outside the agent, at module level

@app.agent(taxi_topic)
async def process(taxis):
    global initialized  # to modify the external variable

    async for taxi in taxis:
        if not initialized:
            for i, c in enumerate(initial_centroids):
                centroids[i] = {'lat': c['lat'], 'lon': c['lon'], 'count': 0}
            initialized = True  # only run once

        if taxi.pickup_latitude is None or taxi.pickup_longitude is None or taxi.pickup_latitude==np.nan or taxi.pickup_longitude==np.nan:
            continue

        def distance(c, lat, lon):
            return math.sqrt((c['lat'] - lat)**2 + (c['lon'] - lon)**2)

        closest_id = min(
            centroids.keys(),
            key=lambda cid: distance(centroids[cid], taxi.pickup_latitude, taxi.pickup_longitude)
        )

        centroid = centroids[closest_id]
        count = centroid['count']

        new_count = count + 1
        new_lat = (centroid['lat'] * count + float(taxi.pickup_latitude)) / new_count
        new_lon = (centroid['lon'] * count + float(taxi.pickup_longitude)) / new_count

        centroids[closest_id] = {'lat': new_lat, 'lon': new_lon, 'count': new_count}
        print(float(taxi.pickup_longitude), float(taxi.pickup_latitude))
        print(f"Cluster {closest_id}: Lat {new_lat:.5f}, Lon {new_lon:.5f}, Count {new_count}")


