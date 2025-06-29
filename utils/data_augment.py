import dask_geopandas as dgpd
import dask.dataframe as dd
from shapely.geometry import Point, shape
import ast

ZIP_PATH = "data/US.txt"
BUSINESS_PATH = "data/businesses.csv"
WEATHER_PATH = "data/weather_data.csv"
SCHOOLS_PATH = "data/schools.csv"
PRECINCTS_PATH = "data/police_precincts.csv"
EVENTS_PATH = "data/events_sample.csv"


def load_zip_data(PATH):
    dtype = {
        1: "int64",  # ZIP Code
        4: "object",  # State
        9: "float64",  # lat
        10: "float64",  # lng
    }
    zip = dd.read_csv(PATH, sep="\t", header=None, dtype=dtype, usecols=[1, 4, 9, 10])
    zip.columns = ["ZIP Code", "State", "lat", "lng"]
    return zip


def load_business_data(PATH):
    bsns = dd.read_csv(
        PATH,
        usecols=["Business Name", "ZIP Code", "Latitude", "Longitude"],
        dtype={
            "ZIP Code": "str",  # Some of the entries are not numeric, manual conversion needed
            "Business Name": "object",
            "Longitude": "float64",
            "Latitude": "float64",
        },  # still useful to enforce type
    )

    bsns["ZIP Code"] = dd.to_numeric(bsns["ZIP Code"], errors="coerce")
    bsns = bsns.dropna(subset=["ZIP Code"])
    bsns["ZIP Code"] = bsns["ZIP Code"].astype(
        "int64"
    )  # This has to be enforced to ensure merge works
    return bsns


def merge_business_zip(bsns, zip_ddf):
    bsns = bsns.merge(zip_ddf, how="left", on="ZIP Code")
    bsns["Latitude"] = bsns["Latitude"].fillna(bsns["lat"])
    bsns["Longitude"] = bsns["Longitude"].fillna(bsns["lng"])
    bsns = bsns.drop(columns=["lat", "lng", "State", "ZIP Code"])
    bsns = bsns.dropna(subset=["Latitude", "Longitude"])

    def make_point(lon, lat):
        return Point(lon, lat)

    # Apply the make_point function to create a 'geometry' column

    bsns = bsns.map_partitions(
        lambda df: df.assign(
            geometry=df.apply(
                lambda x: make_point(x["Longitude"], x["Latitude"]), axis=1
            )
        ),
        meta={
            "Business Name": "str",
            "Latitude": "float64",
            "Longitude": "float64",
            "geometry": "geometry",
        },  # Dask needs to know the type of the new column
    )

    bsns = dgpd.from_dask_dataframe(bsns, geometry="geometry")
    bsns = bsns.set_crs("EPSG:4326").to_crs(3857)
    bsns = bsns.drop(columns=["Latitude", "Longitude"])

    return bsns


def load_weather_data(PATH):
    weather = dd.read_csv(PATH)

    # Handle datetime operations
    weather["time"] = dd.to_datetime(weather["time"])
    weather["time"] = (
        weather["time"].dt.tz_localize("GMT").dt.tz_convert("America/New_York")
    )
    weather["time"] = weather["time"].dt.tz_localize(None)

    # Convert columns to appropriate types
    weather["Temperature"] = weather["Temperature"].astype(float)
    weather["Weather Code"] = weather["Weather Code"].astype(int)

    return weather


def load_schools_data(PATH):
    dtypes = {
        "nta_name": "str",
        "location_category_description": "str",
        "latitude": "float64",
        "longitude": "float64",
    }
    schools = dd.read_csv(
        PATH,
        usecols=["nta_name", "location_category_description", "latitude", "longitude"],
        dtype=dtypes,
    )

    # Combine name columns
    schools["Name"] = (
        schools["nta_name"].astype(str)
        + " "
        + schools["location_category_description"].astype(str)
    )

    # Only keep relevant columns
    schools = schools[["Name", "latitude", "longitude"]]

    # Create geometry column in Dask
    def make_point(row):
        return Point(row["longitude"], row["latitude"])

    # Use map_partitions for custom geometry creation
    schools["Location"] = schools.map_partitions(
        lambda df: df.apply(make_point, axis=1), meta=("Location", "geometry")
    )

    # Convert to GeoDataFrame
    schools_gdf = dgpd.from_dask_dataframe(schools, geometry="Location")
    schools_gdf = schools_gdf.set_crs("EPSG:4326").to_crs(3857)

    # Drop unnecessary columns
    schools_gdf = schools_gdf.drop(columns=["longitude", "latitude"])


    return schools_gdf


def load_precincts_data(PATH):
    precincts = dd.read_csv(
        PATH,
        usecols=["precinct", "the_geom"],
        dtype={"precinct": "int64", "the_geom": "object"},
    )

    # Apply ast.literal_eval on the 'the_geom' column
    precincts["the_geom"] = precincts["the_geom"].apply(
        ast.literal_eval, meta=("x", "object")
    )

    # Create geometry using map_partitions
    precincts["geometry"] = precincts.map_partitions(
        lambda df: df["the_geom"].apply(shape), meta=("geometry", "geometry")
    )
    precincts = precincts.drop(columns=["the_geom"])

    # Create GeoDataFrame
    gdf = dgpd.from_dask_dataframe(precincts, geometry="geometry")

    # Calculate centroids
    gdf["centroid"] = gdf.geometry.centroid

    # Drop unnecessary columns
    gdf = gdf.drop(columns=["geometry"])

    # Ensure 'precinct' column is of integer type
    gdf["precinct"] = gdf["precinct"].astype(int)

    return gdf


def load_events_data(PATH):
    events = dd.read_csv(
        PATH,
        usecols=["Event Name", "Start Date/Time", "End Date/Time", "Police Precinct"],
        dtype={
            "Event Name": "str",
            "Start Date/Time": "str",
            "End Date/Time": "str",
            "Police Precinct": "str",
        },
    )
    events = events[
        ["Event Name", "Start Date/Time", "End Date/Time", "Police Precinct"]
    ]
    events["Police Precinct"] = events["Police Precinct"].apply(
        lambda x: str(x).split(",")[0], meta=("x", "str")
    )
    events["Police Precinct"] = dd.to_numeric(
        events["Police Precinct"], errors="coerce"
    )
    events["Police Precinct"] = events["Police Precinct"].astype(
        "Int64"
    )  # Use Int64 for nullable integers

    # Handle datetime
    events["Start Date/Time"] = dd.to_datetime(events["Start Date/Time"])
    events["End Date/Time"] = dd.to_datetime(events["End Date/Time"])

    return events


def merge_events_precincts(events, precincts):
    events["Police Precinct"] = events["Police Precinct"].astype(
        "Int64"
    )  # Ensure the type matches
    precincts["precinct"] = precincts["precinct"].astype(
        "Int64"
    )  # Ensure the type matches
    precincts = precincts.set_geometry("centroid")
    precincts = precincts.set_crs("EPSG:4326")
    merged = dd.merge(
        events, precincts, how="left", left_on="Police Precinct", right_on="precinct"
    )
    merged = merged.drop(columns=["precinct", "Police Precinct"])
    merged = merged.rename(columns={"centroid": "Location"})
    merged = dgpd.from_dask_dataframe(merged, geometry="Location")
    merged = merged.set_crs("EPSG:4326").to_crs(3857)

    return merged
