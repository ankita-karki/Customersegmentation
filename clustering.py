import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    'Date': lambda x: (snapshot_date - x.max()).days,  # Recency- how recently the customer made a purchase
    'Transaction ID': 'nunique',                       # Frequency - Number of unique transcations
    'Total Amount': 'sum'                              # Monetary value # Total amount spent by the customer
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

#Handling skewness
rfm[['Recency', 'Frequency', 'Monetary']].hist()

# Log-transform skewed features (e.g., Frequency/Monetary)
rfm['Monetary'] = np.log1p(rfm['Monetary']) # reduce impact of outliers in skewed distributions

# Standardize features 
#ensure equal importance of each feature to the clustering algorithm
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Elbow method 
# Elbow point is where the rate of decrease in SSE starts to slow down, indicating the optimal number of clusters.
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()

# Choose K=4 based on the elbow plot
kmeans = KMeans(n_clusters=4, random_state=42) #assign each customer to one of four clusters 
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)


# Cluster summary
cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).reset_index()

print(cluster_summary)


# PCA for 2D visualization
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=rfm_pca[:, 0], y=rfm_pca[:, 1], hue=rfm['Cluster'], palette='viridis')
plt.title('Customer Clusters (PCA)')
plt.show()

# Label clusters based on RFM metrics
segment_map = {
    0: 'At-Risk Customers', #low recency, low frequency, low monetary
    1: 'High-Value Loyalists', #high recency, high frequency, high monetary
    2: 'Potential Loyalists', #moderate recency, moderate frequency, moderate monetary
    3: 'Low-Engagement' #high recency, low frequency, low monetary
}
rfm['Segment'] = rfm['Cluster'].map(segment_map)

