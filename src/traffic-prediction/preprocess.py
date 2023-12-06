import glob
import os
import numpy as np
import pandas as pd

import geopy.distance

from datetime import datetime
from tqdm import tqdm

num_stations = 50
outcome_var = 'avg_speed'

input_path = "data/pems_data"
output_path = "data/processed"

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

# Build two months of data for the selected stations
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

"""
 With the average traffic speeds for each 5-minute interval, and the geographic 
 location of each station on hand, build the inputs to the model
 The inputs are the pair (V, W) where:
 - V includes the node features for each node at each specific point in time.
 - W contains weights for the edges between each pair of nodes
 V -> [T x N] and W => [N x N], where:
      T: number of 5-minute intervals in the selected date range,
      N: number of stations (50) 
"""

# Build the node feature matrix V
station_data = station_data[['timestamp', 'station', outcome_var]]
station_data[outcome_var] = pd.to_numeric(station_data[outcome_var])

# Reshape dataset and aggregate the traffic speeds in each time interval
V = station_data.pivot_table(index=['timestamp'], columns=['station'], values=outcome_var, aggfunc='mean')
print(V.head())

# Build the adjacency matrix W
distances = pd.crosstab(station_metadata.ID, station_metadata.ID, normalize=False)
distances_std= []

for station_i in selected_stations:
    for station_j in selected_stations:
        if station_i == station_j:
            distances.at[station_j, station_i] = 0
        else:
            # Compute the distance between stations
            station_i_meta = station_metadata[station_metadata['ID']==station_i]
            station_j_meta = station_metadata[station_metadata['ID']==station_j]
            # d_ij = geopy.distance.vincenty()
            d_ij = geopy.distance.geodesic(
                (station_i_meta['Latitude'].values[0], station_i_meta['Longitude'].values[0]),
                (station_j_meta['Latitude'].values[0], station_j_meta['Longitude'].values[0])
            ).m
            distances.at[station_j, station_i] = d_ij
            distances_std.append(d_ij)

# Standardize the distances
distances_std = np.std(distances_std)

#  compute 𝑤𝑖𝑗 for each pair of nodes (𝑖,𝑗) in the graph
W = pd.crosstab(station_metadata.ID, station_metadata.ID, normalize=True)

epsilon = 0.1
sigma = distances_std

for station_i in selected_stations:
    for station_j in selected_stations:
        if station_i == station_j:
            W.at[station_j, station_i] = 0
        else:
            # Compute the distance between stations
            d_ij = distances.loc[station_j, station_i]

            # Compute weight w_ij
            w_ij = np.exp(-d_ij**2 / sigma**2)
            if w_ij >= epsilon:
                W.at[station_j, station_i] = w_ij

# Save to file
V.to_csv(os.path.join(output_dir, 'V_{}.csv'.format(num_stations)), index=False)
W.to_csv(os.path.join(output_dir, 'W_{}.csv'.format(num_stations)), index=False)

station_metadata.to_csv(os.path.join(output_dir, 'station_metadata_{}.csv'.format(num_stations)), index=False)
