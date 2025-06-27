#!/bin/bash
echo "Running data cleaning for yellow taxi trip data..."
PYTHONPATH="." python T2/T2_data_cleaning_yellow_taxi.py
echo "Running data cleaning for green taxi trip data..."
PYTHONPATH="." python T2/T2_data_cleaning_green_taxi.py
echo "Running data cleaning for fhv trip data..."
PYTHONPATH="." python T2/T2_data_cleaning_fhv.py
echo "Running data cleaning for fhvhv trip data..."
PYTHONPATH="." python T2/T2_data_cleaning_fhvhv.py