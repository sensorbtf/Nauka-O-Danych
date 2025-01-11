# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:17:00 2025
W Pythonie przeanalizuj wplyw normalizacji cech na wyniki k-means 
dla danych Iris. Wykorzystaj funkcje StandardScaler z sklearn.preprocessing
@author: Sensorbtf
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()
X = iris.data
y = iris.target

kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)

cm = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)
print(f'Accuracy without scaling: {accuracy:.2f}')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
y_pred_scaled = kmeans.fit_predict(X_scaled)

cm_scaled = confusion_matrix(y, y_pred_scaled)
accuracy_scaled = accuracy_score(y, y_pred_scaled)
print(f'Accuracy with scaling: {accuracy_scaled:.2f}')
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=50)
plt.title('K-Means without Scaling')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_scaled, cmap='viridis', edgecolor='k', s=50)
plt.title('K-Means with Scaling')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.tight_layout()
plt.show()
