import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

# Create a figure and axis for plotting
fig, axes = plt.subplots(figsize=(6, 3))
axes.set_xlabel('Feature Dimensions (m)')
axes.set_ylabel('log(d_max/d_min)')
axes.set_title('d_max/d_min vs. Number of Features')

# Define plot line styles for different sample sizes
plot_styles = {100: 'ro-', 200: 'b^-', 500: 'gs-', 1000: 'cv-'}

# Define sample sizes and feature range for analysis
sample_sizes = [100, 200, 500, 1000]
feature_set = range(1, 101)

# Main plotting loop
for samples in sample_sizes:
    log_ratios = []  # Store log(d_max/d_min) ratios for this sample size

    for features in feature_set:
        # Determine the number of informative features and clusters per class
        n_informative = min(features, 2)  # Ensure at least 1 informative feature, max 2
        clusters = 1 if n_informative == 1 else 2  # Number of clusters based on informative features

        # Generate synthetic classification data
        X_data, _ = make_classification(n_samples=samples, n_features=features, 
                                        n_informative=n_informative, n_redundant=0, n_repeated=0,
                                        n_clusters_per_class=clusters, random_state=42)

        # Randomly select a query point and exclude it from distance calculations
        query_idx = np.random.choice(X_data.shape[0])
        query_point = X_data[query_idx]
        X_data = np.delete(X_data, query_idx, axis=0)

        # Calculate the Euclidean distances from the query point to all other points
        dist_to_query = np.linalg.norm(X_data - query_point, axis=1)

        # Calculate the ratio of the max to min distance
        ratio = np.max(dist_to_query) / np.min(dist_to_query)
        log_ratios.append(np.log(ratio))  # Store the log of the ratio

    # Plot the results for this sample size
    axes.plot(feature_set, log_ratios, plot_styles[samples], label=f'N={samples:,}')

# Add the legend, grid, and adjust layout for readability
axes.legend()
axes.grid(True)
plt.tight_layout()

# Display the plot
# plt.show()
