import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the dataset
data = pd.read_csv('Data/retail_sales_dataset.csv')


data.head() #Display the first 5 rows of the dataset

# Clean the data: Handle missing values and outliers
data.dropna(subset=['Customer ID', 'Date', 'Total Amount'], inplace=True)  # Drop critical missing values
data['Date'] = pd.to_datetime(data['Date'])  # Ensure date is in datetime format

# Calculate RFM metrics
snapshot_date = data['Date'].max() + pd.Timedelta(days=1)  # snapshot date set to one day after most recent transaction
rfm = data.groupby('Customer ID').agg({
    'Date': lambda x: (snapshot_date - x.max()).days,  # Recency
    'Transaction ID': 'nunique',                       # Frequency
    'Total Amount': 'sum'                              # Monetary value
}).rename(columns={
    'Date': 'Recency',
    'Transaction ID': 'Frequency',
    'Total Amount': 'Monetary'
})


# Debug: Print RFM metrics before normalization
print("RFM metrics before normalization:")
print(rfm.head())

# Normalize RFM metrics
rfm_normalized = (rfm - rfm.mean()) / rfm.std() #normalized to have mean=0 and std= 1

# Handle NaN values in the normalized data
rfm_normalized.fillna(0, inplace=True)


# Debug: Print RFM metrics after dropping NaN values
print("RFM metrics after dropping NaN values:")
print(rfm_normalized.head())


##Clustering 
#Perform K-means clustering on the normalized RFM data

# Determine optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_normalized)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Apply K-means with optimal number of clusters (e.g., k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_normalized) #adding cluster labels to the rfm dataframe

# Analyze cluster characteristics
cluster_analysis = rfm.groupby('Cluster').mean().reset_index()
print(cluster_analysis)

#Visualizing Cluster Characteristics
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='Monetary', data=rfm.reset_index())
plt.title('Monetary Value Across Clusters')
plt.show()

# Recommendations:
# - High-value customers (high RFM scores): Offer loyalty programs.
# - Low-value customers (low RFM scores): Target with discounts or promotions.
# - Mid-tier customers: Engage with personalized offers.