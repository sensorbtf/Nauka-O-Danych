import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Load data
file_path = r"C:\Users\Sensorbtf\OneDrive\Studia\Magisterka\Dane\IHME_GBD_2019_SMOKING_AGE_1990_2019_INIT_AGE_Y2021M05D27.CSV"
df = pd.read_csv(file_path)

# Inspect the dataset
print(df.info())
print(df.head())

# Select relevant columns
selected_columns = ['val', 'upper', 'lower']  # Numeric columns for analysis
categorical_columns = ['measure_name', 'location_name', 'sex_name', 'age_group_name', 'year_id']

# Drop rows with missing values in relevant columns
numeric_df = df[selected_columns].dropna()

# Isolation Forest for Outlier Detection
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
numeric_df['outliers'] = isolation_forest.fit_predict(numeric_df)

# Visualize outliers
sns.scatterplot(data=numeric_df, x='val', y='upper', hue='outliers', palette='coolwarm')
plt.title("Outliers Detected by Isolation Forest")
plt.xlabel("Value (val)")
plt.ylabel("Upper Bound (upper)")
plt.grid()
plt.show()

# Standardize data for PCA, t-SNE, and UMAP
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df[selected_columns])

# PCA for Dimensionality Reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
numeric_df['PCA1'] = pca_result[:, 0]
numeric_df['PCA2'] = pca_result[:, 1]

# Visualize PCA
sns.scatterplot(data=numeric_df, x='PCA1', y='PCA2', hue='outliers', palette='viridis')
plt.title("PCA Results")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
plt.show()

# t-SNE for Advanced Visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(scaled_data)
numeric_df['tSNE1'] = tsne_result[:, 0]
numeric_df['tSNE2'] = tsne_result[:, 1]

# Visualize t-SNE
sns.scatterplot(data=numeric_df, x='tSNE1', y='tSNE2', hue='outliers', palette='coolwarm')
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid()
plt.show()

# UMAP for Advanced Visualization
reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
umap_result = reducer.fit_transform(scaled_data)
numeric_df['UMAP1'] = umap_result[:, 0]
numeric_df['UMAP2'] = umap_result[:, 1]

# Visualize UMAP
sns.scatterplot(data=numeric_df, x='UMAP1', y='UMAP2', hue='outliers', palette='Spectral')
plt.title("UMAP Visualization")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.grid()
plt.show()

# Interactive 3D Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    numeric_df['PCA1'], 
    numeric_df['PCA2'], 
    numeric_df['val'], 
    c=numeric_df['outliers'], 
    cmap='coolwarm'
)
ax.set_title("3D Interactive PCA Visualization")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("Original Value (val)")
plt.show()

# Correlation Matrix
corr_matrix = numeric_df[selected_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# ANOVA: Testing differences in groups
if 'measure_name' in df.columns:
    df['measure_name'] = df['measure_name'].astype('category')
    anova_model = ols('val ~ C(measure_name)', data=df).fit()
    anova_results = anova_lm(anova_model)
    print("ANOVA Results:\n", anova_results)
else:
    print("No grouping column found for ANOVA.")
