import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('D:/VIT Bhopal/GitHub/ML Projects/Customer Segmentation using Unsupervised Learning/Data.csv')

# Data exploration
print("Dataset Shape:", df.shape)
print(df.info())
print(df.describe().T)

# Data Cleaning
# Check if 'Accepted' column exists before modifying
if 'Accepted' in df.columns:
    df['Accepted'] = df['Accepted'].str.replace('Accepted', '')

# Identify and drop null values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        print(f"Column {col} contains {df[col].isnull().sum()} null values.")

df.dropna(inplace=True)
print("Total missing values after dropping nulls:", len(df))

# Split 'Dt_Customer' into 'day', 'month', 'year'
if 'Dt_Customer' in df.columns:
    parts = df["Dt_Customer"].str.split("-", n=3, expand=True)
    df["day"] = parts[0].astype('int')
    df["month"] = parts[1].astype('int')
    df["year"] = parts[2].astype('int')

# Drop irrelevant columns
cols_to_drop = ['Z_CostContact', 'Z_Revenue', 'Dt_Customer']
for col in cols_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Data Visualization
floats, objects = [], []
for col in df.columns:
    if df[col].dtype == object:
        objects.append(col)
    elif df[col].dtype == float:
        floats.append(col)

print("Categorical Columns:", objects)
print("Numerical Columns:", floats)

# Count plot for categorical variables
plt.figure(figsize=(15, 10))
for i, col in enumerate(objects):
    plt.subplot(2, 2, i + 1)
    sb.countplot(df[col])
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Comparison of features with respect to 'Response'
if 'Response' in df.columns:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(objects):
        plt.subplot(2, 2, i + 1)
        df_melted = df.melt(id_vars=[col], value_vars=['Response'], var_name='hue')
        sb.countplot(x=col, hue='value', data=df_melted)
        plt.title(f"{col} vs Response")
    plt.tight_layout()
    plt.show()

# Label Encoding for categorical variables
for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Correlation Heatmap
plt.figure(figsize=(15, 15))
sb.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Standardization
scaler = StandardScaler()
data = scaler.fit_transform(df)

# Dimensionality Reduction using t-SNE
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(df)
plt.figure(figsize=(7, 7))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c='blue', alpha=0.5)
plt.title("t-SNE Visualization of Data")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

# K-Means Clustering - Finding Optimal Clusters
error = []
silhouette_scores = []
for n_clusters in range(2, 21):
    model = KMeans(init='k-means++', n_clusters=n_clusters, max_iter=500, random_state=22)
    labels = model.fit_predict(df)
    error.append(model.inertia_)
    silhouette_scores.append(silhouette_score(df, labels))

plt.figure(figsize=(10, 5))
sb.lineplot(x=range(2, 21), y=error, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

plt.figure(figsize=(10, 5))
sb.lineplot(x=range(2, 21), y=silhouette_scores, marker='o')
plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# K-Means with optimal clusters (k=5)
model = KMeans(init='k-means++', n_clusters=5, max_iter=500, random_state=22)
segments = model.fit_predict(df)
print("Silhouette Score for K=5:", silhouette_score(df, segments))

# Visualize Clusters
plt.figure(figsize=(7, 7))
df_tsne = pd.DataFrame({'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'segment': segments})
sb.scatterplot(x='x', y='y', hue='segment', data=df_tsne, palette='Set1')
plt.title("K-Means Clusters Visualized with t-SNE")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Cluster")
plt.show()
