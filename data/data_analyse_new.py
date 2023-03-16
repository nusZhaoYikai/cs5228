import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_data(train_data):
    # Encode non-numeric features
    encoded_data = pd.get_dummies(train_data,
                                  columns=['town', 'flat_type', 'storey_range', 'flat_model', 'eco_category', 'subzone',
                                           'planning_area', 'region'])

    # Calculate the correlation coefficient matrix
    corr_matrix = encoded_data.corr()

    # Extract features that are highly correlated with the resale price
    # Remove its own relevance
    corr_matrix = corr_matrix.drop('resale_price', axis=0)
    resale_price_corr = corr_matrix['resale_price'].sort_values(ascending=False)

    # Set a threshold, such as 0.1, to filter out features with a correlation coefficient greater than that threshold
    threshold = 0.3
    important_features = resale_price_corr[abs(resale_price_corr) > threshold]

    # Displays important features
    print("important features:")
    print(important_features)

    # Visualize part of the data of the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix.loc[important_features.index, important_features.index], annot=True, cmap='coolwarm')
    plt.show()

# load data
print("Before adding auxiliary features:")
plot_data(pd.read_csv('train.csv'))
print("After adding auxiliary features:")
plot_data(pd.read_csv('train_with_aux.csv'))