#!/bin/bash

#SBATCH -J bd-data-in
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 12:00:00

module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate
srun python 01_data_ingestion.py
