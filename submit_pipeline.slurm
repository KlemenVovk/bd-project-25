#!/bin/bash
#SBATCH --job-name=nyc-taxi-pipeline
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kv4582@student.uni-lj.si

# Load modules and activate environment
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

# Ensure slurm_logs directory exists
mkdir -p slurm_logs

#### TASK 1: Ingestion
echo "TASK 1: Data Ingestion"
echo "Preparing zone data before trip data ingestion..."
PYTHONPATH="." python utils/zone_mapping.py

echo "Ingesting yellow taxi trip data..."
PYTHONPATH="." python T1/T1_data_ingestion_yellow_taxi.py
echo "Ingesting green taxi trip data..."
PYTHONPATH="." python T1/T1_data_ingestion_green_taxi.py
echo "Ingesting fhv trip data..."
PYTHONPATH="." python T1/T1_data_ingestion_fhv.py
echo "Ingesting fhvhv trip data..."
PYTHONPATH="." python T1/T1_data_ingestion_fhvhv.py

#### TASK 2: Cleaning
echo "TASK 2: Data Cleaning"
echo "Running data cleaning for yellow taxi trip data..."
PYTHONPATH="." python T2/T2_data_cleaning_yellow_taxi.py
echo "Running data cleaning for green taxi trip data..."
PYTHONPATH="." python T2/T2_data_cleaning_green_taxi.py
echo "Running data cleaning for fhv trip data..."
PYTHONPATH="." python T2/T2_data_cleaning_fhv.py
echo "Running data cleaning for fhvhv trip data..."
PYTHONPATH="." python T2/T2_data_cleaning_fhvhv.py

echo "Pipeline completed successfully."
