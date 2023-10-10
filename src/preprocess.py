import glob
import os
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

num_stations = 50
outcome_var = 'avg_speed'


def get_data(input_path: str, output_path: str):
    # input files
    PeMs_daily = os.path.join(input_path, 'PeMS', '*')
    PeMs_metadata = os.path.join(input_path, 'd07_text_meta_2021_03_27.txt')
    Yu_data = os.path.join(input_path, 'Yu_et_al', 'PeMS-M')

    # output files
    output_dir = os.path.join(output_path, 'processed_data', 'graph_inputs')

    # get the columns from the PeMS csv file
    PeMS_columns = [
        'timestamp', 'station', 'district', 'freeway_num',
        'direction_travel', 'lane_type', 'station_length', 'samples',
        'perc_observed', 'total_flow', 'avg_occupancy', 'avg_speed'
    ]

    PeMS_lane_columns = lambda x: [
        'lane_N_samples_{}'.format(x),
        'lane_N_flow_{}'.format(x),
        'lane_N_avg_occ_{}'.format(x),
        'lane_N_avg_speed_{}'.format(x),
        'lane_N_observed_{}'.format(x)
    ]

    PeMS_all_columns = PeMS_columns.copy()
    for i in range(1, 9):
        PeMS_all_columns += PeMS_lane_columns(i)

    # Randomly select stations to build the dataset
    np.random_seed(42)

    files = glob.glob(PeMs_daily)
    station_file = files[0]
    station_file_content = pd.read_csv(station_file, header=0, names=PeMS_all_columns)
    station_file_content = station_file_content[PeMS_columns]

    station_file_content = station_file_content.dropna(subset=[outcome_var])
    unique_stations = station_file_content['station'].unique()
    selected_stations = np.random.choice(unique_stations, size=num_stations, replace=False)

    # Build two months of data for teh selected stations
    station_data = pd.DataFrame({col: [] for col in PeMS_columns})

    for station_file in tqdm(files):
        # file date
        file_date_str = station_file.split(os.path.sep)[-1].split('.')[0]
        file_date = datetime(2021,
                             int(file_date_str.split('_')[-2]),
                             int(file_date_str.split('_')[-1]))

        # Check for weekdays
        if file_date.weekday() < 5:
            # read csv
            station_file_content = pd.read_csv(station_file, header=0, names=PeMS_all_columns)

            # Keep only columns of interest
            station_file_content = station_file_content[PeMS_columns]

            # Keep stations
            station_file_content = station_file_content['station'].isin(selected_stations)

            # Append to dataset
            station_data = pd.concat([station_data, station_file_content])

    # Drop any rows with missing values
    station_data = station_data.dropna(subset=['timestamp', outcome_var])

    # print sample
    print(station_data.head())
    print(f"Data has shape: {station_data.shape}")

    # Download station metadata to compute the distance
    # between stations and build the adjacency matrix
    station_metadata = pd.read_table(PeMs_metadata)
    station_metadata = station_metadata[['ID', 'Latitude', 'Longitude']]
    station_metadata = station_metadata[station_metadata['ID'].isin(selected_stations)]

    print(station_metadata.head())

    # save

