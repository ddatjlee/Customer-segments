import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class CustomerSegmentationModel:
    def __init__(self, n_clusters=4, random_state=42):
        """
        Initialize the model with the number of clusters and random state.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.data_scaled = None

    def preprocess_data(self, data):
        """
        Preprocess data for clustering.
        - Remove missing CustomerID
        - Compute TotalSpending, Frequency, and Recency metrics.
        """
        # Drop missing customer IDs
        data_clean = data.dropna(subset=['CustomerID'])

        # Calculate TotalSpending
        data_clean['TotalSpending'] = data_clean['Quantity'] * data_clean['UnitPrice']

        # Aggregate metrics by CustomerID
        metrics = data_clean.groupby('CustomerID').agg({
            'InvoiceNo': 'nunique',  # Frequency
            'TotalSpending': 'sum',  # Monetary
            'InvoiceDate': 'max'     # Recency
        }).rename(columns={'InvoiceNo': 'Frequency', 'TotalSpending': 'Monetary', 'InvoiceDate': 'Recency'})

        # Calculate Recency in days
        metrics['Recency'] = (data_clean['InvoiceDate'].max() - metrics['Recency']).dt.days

        return metrics

    def fit(self, data):
        """
        Fit the K-means model on the customer data.
        """
        # Preprocess data
        customer_metrics = self.preprocess_data(data)

        # Lưu CustomerID để sử dụng lại sau này
        customer_ids = customer_metrics.index

        # Extract the features
        features = customer_metrics[['Frequency', 'Monetary', 'Recency']]

        # Scale the features
        self.data_scaled = self.scaler.fit_transform(features)

        # Fit the K-means model
        customer_metrics['Cluster'] = self.kmeans.fit_predict(self.data_scaled)

        # Thêm CustomerID trở lại DataFrame
        customer_metrics['CustomerID'] = customer_ids

        # Sắp xếp lại thứ tự cột để giữ CustomerID ở đầu
        customer_metrics = customer_metrics[['CustomerID', 'Frequency', 'Monetary', 'Recency', 'Cluster']]

        return customer_metrics

    def predict(self, new_data):
        """
        Predict cluster labels for new data.
        """
        if self.data_scaled is None:
            raise Exception("Model has not been fitted yet. Please call fit() before predict().")

        # Scale the new data
        new_data_scaled = self.scaler.transform(new_data)

        # Predict clusters
        return self.kmeans.predict(new_data_scaled)

    def get_cluster_centers(self):
        """
        Retrieve the cluster centers.
        """
        return self.kmeans.cluster_centers_
