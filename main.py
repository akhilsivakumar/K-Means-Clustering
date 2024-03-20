# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv("/content/customer_shopping_data.csv")

# Visualize the data
plt.scatter(data['quantity'], data['price'])
plt.xlabel("Quantity")
plt.ylabel("Price")
plt.title("Customer Shopping Data")
plt.show()

# Extract features for clustering
x = data.iloc[:, 5:6].values

# Perform K-Means Clustering with 3 clusters
kmeans = KMeans(n_clusters=3)
identified_clusters = kmeans.fit_predict(x)

# Add cluster information to the dataset
data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters

# Visualize clustered data
plt.scatter(data_with_clusters['quantity'], data_with_clusters['price'], c=data_with_clusters['Clusters'], cmap='rainbow')
plt.xlabel("Quantity")
plt.ylabel("Price")
plt.title("Customer Shopping Data with Clusters")
plt.show()
