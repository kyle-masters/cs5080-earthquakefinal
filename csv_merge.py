import pandas as pd
import os

# This file is for preprocessing, there are originally 5 csv files with information on events that need to be merged

full_events_file_name = 'data/full.csv'
local_events_file_name = 'data/local.csv'
data_file_name = 'data/data.csv'
error_file_name = 'data/error.csv'

csv_files = [f'raw_data/chunk{i}.csv' for i in range(2, 7)]

os.makedirs('data', exist_ok=True)

if os.path.exists(full_events_file_name):
    os.remove(full_events_file_name)
if os.path.exists(local_events_file_name):
    os.remove(local_events_file_name)
if os.path.exists(data_file_name):
    os.remove(data_file_name)
if os.path.exists(error_file_name):
    os.remove(error_file_name)

for csv_file in csv_files:
    df = pd.read_csv(csv_file, low_memory=False)
    if not os.path.exists(full_events_file_name):
        df.to_csv(full_events_file_name, index=False)
    else:
        df.to_csv(full_events_file_name, index=False, mode='a', header=False)

    df = df[(df.trace_category == 'earthquake_local') & (df.source_depth_km != 'None')]
    if not os.path.exists(local_events_file_name):
        df.to_csv(local_events_file_name, index=False)
    else:
        df.to_csv(local_events_file_name, index=False, mode='a', header=False)

    data_df = df[['network_code', 'source_id', 'source_origin_time', 'source_latitude', 'source_longitude', 'source_depth_km', 'source_magnitude', 'source_distance_deg', 'source_distance_km']]
    if not os.path.exists(data_file_name):
        data_df.to_csv(data_file_name, index=False)
    else:
        data_df.to_csv(data_file_name, index=False, mode='a', header=False)

    error_df = df[['network_code', 'source_id', 'source_origin_uncertainty_sec', 'source_error_sec', 'source_gap_deg', 'source_horizontal_uncertainty_km', 'source_depth_uncertainty_km', 'source_magnitude_type']]
    if not os.path.exists(error_file_name):
        error_df.to_csv(error_file_name, index=False)
    else:
        error_df.to_csv(error_file_name, index=False, mode='a', header=False)
