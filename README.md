# Big data project 2025
Authors: Urh Prosenc, Klemen Vovk

## Repository structure
- `data`
    - `raw` - contains raw parquet files from `download_yellow_tripdata.sh`.
    - `task1` - created by running 01_data_ingestion.py.
        - `all` - processed dataset, saved to parquet, partitioned on year.
        - `one_year` - only 2024 data in CSV and HDF5 formats for comparing file formats.
        - `five_years` - 2020-2024 data in a single CSV.
    - `zone_data` - contains TAXI zone shapefiles from `download_zone_data.sh`
- `slurm_logs` - proof-of-run of the first task on Arnes HPC.
- `01_data_ingestion.py` - a script that ingests raw parquet files from [NYC TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), and prepares them for concatenation along the time axis (data cleaninig, column mapping, saving to intermediary files etc.).
- `02_format_analysis.ipynb` - a notebook comparison of different file formats (based on the results produced by `01_data_ingestion.py`).
- `03_additional_dataset.ipynb` - a notebook explaining augmentation of original data with external data sources found in `external_sources.txt`
- `03_additional_dataset_pandas.py` and `03_additional_dataset_dask.py` - script versions of the `03_additional_dataset.ipynb` notebook for running augmentation from CLI using Pandas or Dask respectively. 
- `constants.py` - defines paths, schemas and other constants used throught the processing pipeline.
- `download_yellow_tripdata.sh` - downloads the mentioned original parquet files to data/raw for further processing.
- `download_zone_data.sh` - downloads geographical data (shapefile) for taxi zones in NYC. This is used for mapping pickup and dropoff location ids to their geographical coordinates.
- `external_sources.txt` - a textfile containing links to external sources used for augmenting the original taxi trip data.
- `requirements.txt` - exported python 3.12 environment from uv.
- `sanity_checks.py` - quick checks to see if we lost any data during concatenation and cleaning.
- `slurm_job.sh` - a script that submits 01_data_ingestion.py to a SLURM managed cluster (Arnes HPC).

## Getting started

Clone the repository
```bash
git clone https://github.com/KlemenVovk/bd-project-25.git
cd bd-project-25
```

Create a Python virtual environment from requirements.txt with your tool of choice.
```bash
python -m venv .venv
pip install -r requirements.txt
source .venv/bin/activate
```

Download data:
```bash
./download_yellow_tripdata.sh
./download_zone_data.sh
```

Consult external_sources.txt for data used for augmenting the original dataset (weather, nearby businesses, events, schools etc.)

Change paths in constants.py to resemble your data sources and outputs.

Run scripts locally...
```bash
python 01_data_ingestion.py
```

or submit to HPC (SLURM)
```bash
sbatch slurm_job.sh
```