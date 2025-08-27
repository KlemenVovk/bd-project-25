#!/bin/bash
#SBATCH --job-name=nyc-taxi-pipeline
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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

echo "TASK 2: Data Cleaning"
echo "Running data cleaning for yellow taxi trip data..."
PYTHONPATH="." python T2/T2_data_cleaning_yellow_taxi.py
echo "Running data cleaning for green taxi trip data..."
PYTHONPATH="." python T2/T2_data_cleaning_green_taxi.py
echo "Running data cleaning for fhv trip data..."
PYTHONPATH="." python T2/T2_data_cleaning_fhv.py
echo "Running data cleaning for fhvhv trip data..."
PYTHONPATH="." python T2/T2_data_cleaning_fhvhv.py
echo "Generating latex output."
PYTHONPATH="." python T2/T2_generate_latex_out.py

echo "TASK 3: File format analysis"
echo "Benchmarking file formats..."
PYTHONPATH="." python T3/T3_file_format_analysis.py

echo "TASK 4: Exploratory Data Analysis"
echo "Running EDA for yellow taxi trip data..."
PYTHONPATH="." python T4/T4_eda_yellow_taxi.py
echo "Running EDA for green taxi trip data..."
PYTHONPATH="." python T4/T4_eda_green_taxi.py
echo "Running EDA for fhv trip data..."
PYTHONPATH="." python T4/T4_eda_fhv.py
echo "Running EDA for fhvhv trip data..."
PYTHONPATH="." python T4/T4_eda_fhvhv.py
echo "Running post-processing for all trip data..."
PYTHONPATH="." python T4/T4_postprocess.py

echo "TASK 7: Modeling Service Types"
echo "Running modeling for service types (scaling experiments)..."
PYTHONPATH="." python T7/T7_modeling_service_types.py

echo "Pipeline completed successfully."
