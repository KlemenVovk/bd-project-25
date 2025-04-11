#!/bin/bash

# Year range
start_year=2009
end_year=2025

# Base URL and destination directory
base_url="https://d37ci6vzurychx.cloudfront.net/trip-data"
dest_dir="./data/raw"

# Create the destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Log file for missing files
missing_log="missing_files.log"
> "$missing_log"  # Clear previous log

# Loop over years and months
for year in $(seq $start_year $end_year); do
  for month in {01..12}; do
    filename="yellow_tripdata_${year}-${month}.parquet"
    url="${base_url}/${filename}"
    echo "Trying: $filename"
    
    wget --directory-prefix "$dest_dir" --quiet --continue "$url"
    
    # Check if the file was downloaded
    if [ ! -f "${dest_dir}/${filename}" ]; then
      echo "Missing: $filename" | tee -a "$missing_log"
    fi
  done
done

echo "Download complete. Missing files are logged in $missing_log."
