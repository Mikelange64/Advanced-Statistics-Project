#1.)LOAD DATASET(10)  2.)EXPLORE THE DATASET(19)  3.)FILTER THE DATA(28)  4.)HANDLE MISSING VALUES(53)  5.)STANDARDIZE THE DATA(65)  6.)PERFORM PCA(76)  7.)VISUALIZE PCA RESULTS(95)  8.)INTERPRET AND SAVE RESULTS(123)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#-----------------------------------------------------------------------------------------------------------------------
#1.) LOAD DATASET

data = pd.read_csv('WDICSV.csv')
print(data.head())
# Display column names
print(data.columns)

#-----------------------------------------------------------------------------------------------------------------------
#2.) EXPLORE THE DATASET

# Check for missing values
print(data.isnull().sum())

# Get basic statistics of numerical columns
print(data.describe())

#-----------------------------------------------------------------------------------------------------------------------
#3.)FILTER THE DATA

# Specify the indicators of interest
indicators = [
    'GDP (current US$)',
    'Inflation, consumer prices (annual %)',
    'Trade (% of GDP)'
]

# Filter data for the selected indicators
filtered_data = data[data['Indicator Name'].isin(indicators)]
print(filtered_data.head())

# Select specific years (e.g., 2000 to 2020)
years = [str(year) for year in range(2000, 2021)]

# Filter columns for selected years
filtered_data = filtered_data[['Country Name', 'Indicator Name'] + years]
print(filtered_data.head())

# Pivot the data so that indicators become columns
pivot_data = filtered_data.pivot(index='Country Name', columns='Indicator Name', values=years)
print(pivot_data.head())

#-----------------------------------------------------------------------------------------------------------------------
#4.)HANDLE MISSING VALUES

# Check for missing values
print(pivot_data.isnull().sum())

# Drop rows with missing values
pivot_data_cleaned = pivot_data.dropna()

# Alternatively, fill missing values with the mean
# pivot_data_cleaned = pivot_data.fillna(pivot_data.mean())

#-----------------------------------------------------------------------------------------------------------------------
#5.)STANDARDIZE THE DATA

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pivot_data_cleaned)

# Convert back to a DataFrame for readability
scaled_df = pd.DataFrame(scaled_data, index=pivot_data_cleaned.index, columns=pivot_data_cleaned.columns)
print(scaled_df.head())

#-----------------------------------------------------------------------------------------------------------------------
#6.)PERFORM PCA

# Initialize PCA and fit to the data
pca = PCA(n_components=None)  # Keep all components initially
pca_data = pca.fit_transform(scaled_data)

# Create a DataFrame with principal components
pca_df = pd.DataFrame(pca_data, index=scaled_df.index, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
print(pca_df.head())

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

# Cumulative variance
cumulative_variance = np.cumsum(explained_variance)
print("Cumulative Variance:", cumulative_variance)

#-----------------------------------------------------------------------------------------------------------------------
#7.)VISUALIZE PCA RESULTS
# Plot cumulative variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Variance Explained by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.grid()

# Save the plot to a file
plt.savefig('cumulative_variance_plot.png')  # Provide a valid file name
print("Plot saved as 'cumulative_variance_plot.png'")

# Plot the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
plt.title('PCA: PC1 vs PC2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()

# Save the PCA scatter plot to a file
plt.savefig('pca_scatter_plot.png')  # Provide a valid file name
print("Plot saved as 'pca_scatter_plot.png'")

#-----------------------------------------------------------------------------------------------------------------------
#8.)INTERPRET AND SAVE RESULTS

# Get feature loadings
loadings = pd.DataFrame(pca.components_.T, index=scaled_df.columns, columns=[f'PC{i+1}' for i in range(len(pca.components_))])
print("Feature Loadings:\n", loadings)

# Save PCA-transformed data
pca_df.to_csv('pca_results.csv', index=True)











