import os
import json
import dask.dataframe as dd
from utils.constants import TASK2_OUT_ROOT, TASK8_OUT_ROOT

GREEN_PATH = os.path.join(TASK2_OUT_ROOT, "green_tripdata")
YELLOW_PATH = os.path.join(TASK2_OUT_ROOT, "yellow_tripdata")
FHVHV_PATH = os.path.join(TASK2_OUT_ROOT, "fhvhv_tripdata")

years = range(2009, 2026)

monthly_rides_per_year = {}

for year in years:
    year_path = os.path.join(GREEN_PATH, f"year={year}")

    if os.path.isdir(year_path):
        # Read all parquet files in the year folder
        df = dd.read_parquet(os.path.join(year_path, "*.parquet"))

        # Parse pickup datetime and extract month
        df["month"] = df["pickup_datetime"].dt.month

        # Count number of rides per month
        monthly_counts = df.groupby("month").size().compute()

        # Convert to dictionary {month: ride_count}
        monthly_rides_per_year[year] = {
            int(month): int(count) for month, count in monthly_counts.items()
        }

# Export to JSON
output_path = os.join(TASK8_OUT_ROOT, "green_ride_counts.json")
with open(output_path, "w") as f:
    json.dump(monthly_rides_per_year, f, indent=4)

print(f"Saved monthly ride counts to {output_path}")


monthly_rides_per_year = {}

for year in years:
    year_path = os.path.join(YELLOW_PATH, f"year={year}")

    if os.path.isdir(year_path):
        # Read all parquet files in the year folder
        df = dd.read_parquet(os.path.join(year_path, "*.parquet"))

        # Parse pickup datetime and extract month
        df["month"] = df["pickup_datetime"].dt.month

        # Count number of rides per month
        monthly_counts = df.groupby("month").size().compute()

        # Convert to dictionary {month: ride_count}
        monthly_rides_per_year[year] = {
            int(month): int(count) for month, count in monthly_counts.items()
        }

# Export to JSON
output_path = os.path.join(TASK8_OUT_ROOT, "yellow_ride_count.json")
with open(output_path, "w") as f:
    json.dump(monthly_rides_per_year, f, indent=4)

print(f"Saved monthly ride counts to {output_path}")


fhvhv_monthly_rides = {}

for year in years:
    year_path = os.path.join(FHVHV_PATH, f"year={year}")

    if os.path.isdir(year_path):
        df = dd.read_parquet(os.path.join(year_path, "*.parquet"))

        # Parse pickup datetime
        df["pickup_datetime"] = dd.to_datetime(df["pickup_datetime"], errors="coerce")
        df["month"] = df["pickup_datetime"].dt.month

        # Group by company and month
        grouped = df.groupby(["hvfhs_license_num", "month"]).size().compute()

        # Convert to nested dict {year: {company: {month: count}}}
        if year not in fhvhv_monthly_rides:
            fhvhv_monthly_rides[year] = {}

        for (company, month), count in grouped.items():
            company_dict = fhvhv_monthly_rides[year].setdefault(company, {})
            company_dict[int(month)] = int(count)

# Export to JSON
output_path = os.path.join(TASK8_OUT_ROOT, "fhvhv_monthly_rides_by_company.json")
with open(output_path, "w") as f:
    json.dump(fhvhv_monthly_rides, f, indent=4)

print(f"Saved FHVHV monthly ride counts by company to {output_path}")
