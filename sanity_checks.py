import dask.dataframe as dd
from constants import TASK1_OUT_ROOT, RAW_DATA_ROOT
import os

raw_df_2024 = dd.read_parquet(os.path.join(RAW_DATA_ROOT, "yellow_tripdata_2024*.parquet"), engine='pyarrow')
task1_df_csv_2024 = dd.read_csv(os.path.join(TASK1_OUT_ROOT, 'one_year', "2024.csv"), engine='pyarrow')
task1_df_hdf_2024 = dd.read_hdf(os.path.join(TASK1_OUT_ROOT, 'one_year', "2024.h5"), key='taxidata')
assert raw_df_2024.shape[0].compute() == task1_df_csv_2024.shape[0].compute(), "Row count mismatch between raw data and task1 CSV output"
assert raw_df_2024.shape[0].compute() == task1_df_hdf_2024.shape[0].compute(), "Row count mismatch between raw data and task1 HDF output"