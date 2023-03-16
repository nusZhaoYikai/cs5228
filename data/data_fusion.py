import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os

# load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Define the file name and corresponding facility type
auxiliary_files = {
    "sg-commerical-centres.csv": "commercial_centre",
    "sg-gov-markets-hawker-centres.csv": "hawker_centre",
    "sg-population-demographics.csv": "population",
    "sg-primary-schools.csv": "primary_school",
    "sg-secondary-schools.csv": "secondary_school",
    "sg-shopping-malls.csv": "shopping_mall",
    "sg-train-stations.csv": "train_station"
}

# Read additional data from the folder
auxiliary_data = pd.DataFrame()
for file, amenity_type in auxiliary_files.items():
    file_data = pd.read_csv(os.path.join("auxiliary-data", file))
    file_data["amenity_type"] = amenity_type
    auxiliary_data = pd.concat([auxiliary_data, file_data], ignore_index=True)


# Calculate the closest proximity of each house to various facilities
def integrate_auxiliary_data(house_data, auxiliary_data):
    for amenity_type in auxiliary_data['amenity_type'].unique():
        amenity_data = auxiliary_data[auxiliary_data['amenity_type'] == amenity_type][['lat', 'lng']]

        # Remove NaN and infinity values
        amenity_data = amenity_data.dropna()
        amenity_data = amenity_data.replace([np.inf, -np.inf], np.nan).dropna()

        house_coords = house_data[['latitude', 'longitude']].values
        # Check if the facility data is empty and skip if it is
        if amenity_data.shape[0] == 0:
            continue

        # Use NearestNeighbors to calculate the closest facility distance
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(amenity_data)
        distances, _ = nbrs.kneighbors(house_coords)

        house_data[f'nearest_{amenity_type}_distance'] = distances[:, 0]

    return house_data


# Blend additional data
train_data_with_aux = integrate_auxiliary_data(train_data, auxiliary_data)
test_data_with_aux = integrate_auxiliary_data(test_data, auxiliary_data)


# save data
train_data_with_aux.to_csv("train_with_aux.csv", index=False)
test_data_with_aux.to_csv("test_with_aux.csv", index=False)
