#!/bin/bash
echo "Preparing zone data before trip data ingestion..."
python zone_mapping.py
echo "Ingesting yellow taxi trip data..."
python T1_data_ingestion_yellow_taxi.py
echo "Ingesting green taxi trip data..."
python T1_data_ingestion_green_taxi.py
echo "Ingesting fhv trip data..."
python T1_data_ingestion_fhv.py
echo "Ingesting fhvhv trip data..."
python T1_data_ingestion_fhvhv.py