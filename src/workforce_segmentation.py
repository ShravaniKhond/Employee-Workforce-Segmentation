# Employee Workforce Segmentation
# K-Means + Hierarchical Clustering + PCA

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Create output folder
os.makedirs("outputs", exist_ok=True)

# 2. Load dataset
data_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Dataset not found: {data_path}\n"
        "Please place the CSV file inside the data folder."
    )

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 3. Select numerical features
# Exclude ID, target, and constant columns
exclude_cols = [
    "EmployeeNumber",
    "Attrition",
    "EmployeeCount",
    "StandardHours",
    "Over18"
]

numeric_df = df.select_dtypes(include="number")
numeric_df = numeric_df.drop(
    columns=[c for c in exclude_cols if c in numeric_df.columns],
    errors="ignore"
)

# Remove columns with no variation
numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

# Handle missing values, if any
numeric_df = numeric_df.fillna(numeric_df.median())

# 4. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

print("Features used:", numeric_df.shape[1])

# 5. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

kmeans_score = silhouette_score(X_scaled, df["KMeans_Cluster"])
print("\nK-Means Silhouette Score:", round(kmeans_score, 3))

# 6. Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X_scaled)

hierarchical_score = silhouette_score(
    X_scaled, df["Hierarchical_Cluster"]
)
print("Hierarchical Silhouette Score:", round(hierarchical_score, 3))

# 7. PCA: Reduce features to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

explained_variance = pca.explained_variance_ratio_.sum()
print("PCA explained variance (2 components):",
      round(explained_variance * 100, 2), "%")

# 8. Plot K-Means clusters using PCA
plt.figure(figsize=(8, 6))
plt.scatter(
    df["PCA1"], df["PCA2"],
    c=df["KMeans_Cluster"], cmap="viridis",
    s=35, alpha=0.7
)
plt.title("Employee Segmentation - K-Means with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("outputs/kmeans_pca.png", dpi=300)
plt.show()

# 9. Plot Hierarchical clusters using PCA
plt.figure(figsize=(8, 6))
plt.scatter(
    df["PCA1"], df["PCA2"],
    c=df["Hierarchical_Cluster"], cmap="viridis",
    s=35, alpha=0.7
)
plt.title("Employee Segmentation - Hierarchical Clustering with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("outputs/hierarchical_pca.png", dpi=300)
plt.show()

# 10. Create dendrogram using a sample for readability
sample_size = min(200, len(X_scaled))
sample_data = X_scaled[:sample_size]
linked = linkage(sample_data, method="ward")

plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode="lastp", p=30)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster / Sample")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("outputs/dendrogram.png", dpi=300)
plt.show()

# 11. Save results
df.to_csv("outputs/employee_segments.csv", index=False)

comparison = pd.DataFrame({
    "Model": ["K-Means", "Hierarchical Clustering"],
    "Number_of_Clusters": [3, 3],
    "Silhouette_Score": [kmeans_score, hierarchical_score]
})

comparison.to_csv("outputs/model_comparison.csv", index=False)

print("\nSUCCESS! Clustering and PCA completed.")
print("Results saved in the outputs folder.")
print(comparison)