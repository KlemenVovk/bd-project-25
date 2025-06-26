#!/bin/bash

# Dataset and year ranges (format: dataset:start_year:end_year)
datasets_with_years=(
  "yellow_tripdata:2012:2025"
  "green_tripdata:2014:2025"
  "fhv_tripdata:2015:2025"
  "fhvhv_tripdata:2019:2025"
)

# Base URL and destination directory
base_url="https://d37ci6vzurychx.cloudfront.net/trip-data"
dest_dir="./data/raw"

# Create the base destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Log file for missing files
missing_log="missing_files.log"
> "$missing_log"  # Clear previous log

# Loop through each dataset and year range
for entry in "${datasets_with_years[@]}"; do
  IFS=':' read -r dataset start_year end_year <<< "$entry"
  dataset_dir="${dest_dir}/${dataset}"
  mkdir -p "$dataset_dir"

  for year in $(seq $start_year $end_year); do
    for month in {01..12}; do
      filename="${dataset}_${year}-${month}.parquet"
      url="${base_url}/${filename}"

      # Check if the file already exists
      if [ -f "${dataset_dir}/${filename}" ]; then
        echo "File already exists: $filename"
        continue
      fi
      sleep 3  # Sleep to avoid overwhelming the server

      # Download the file
      echo "Trying: $filename"
      wget --directory-prefix "$dataset_dir" --quiet --continue "$url"
      
      # Check if the file was downloaded
      if [ ! -f "${dataset_dir}/${filename}" ]; then
        echo "Missing: $filename" | tee -a "$missing_log"
      fi
    done
  done
done

echo "Download complete. Missing files are logged in $missing_log."
