# Importowanie bibliotek
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import plotly.express as px
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Wczytanie danych
file_path = r"C:\Users\Sensorbtf\OneDrive\Studia\Magisterka\Dane\IHME_GBD_2019_SMOKING_AGE_1990_2019_INIT_AGE_Y2021M05D27.CSV"
df = pd.read_csv(file_path)

# Wstępna eksploracja danych
print(df.info())
print(df.describe())

# Wybór kolumn do analizy
numerical_columns = ['val', 'upper', 'lower']
categorical_columns = ['location_name', 'sex_name', 'year_id']

# Identyfikacja wartości odstających za pomocą Isolation Forest
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
df['outliers'] = isolation_forest.fit_predict(df[numerical_columns])
outliers = df[df['outliers'] == -1]
print("Wartości odstające:")
print(outliers)

# Wizualizacja wartości odstających
fig = px.scatter_matrix(df, dimensions=numerical_columns, color='outliers',
                        labels={'outliers': 'Odstające'},
                        title="Wartości odstające w danych")
fig.show()

# Redukcja wymiarowości za pomocą PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numerical_columns])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]
print("Wariancja wyjaśniona przez PCA:", pca.explained_variance_ratio_)

# Wizualizacja wyników PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='outliers', data=df, palette='coolwarm')
plt.title("Wizualizacja PCA")
plt.xlabel("Główna składowa 1")
plt.ylabel("Główna składowa 2")
plt.show()

# Wizualizacja t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(scaled_data)
df['tSNE1'], df['tSNE2'] = tsne_result[:, 0], tsne_result[:, 1]
sns.scatterplot(x='tSNE1', y='tSNE2', hue='location_name', data=df, palette='coolwarm', legend=None)
plt.title("Wizualizacja t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

# Wizualizacja UMAP
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_result = umap_reducer.fit_transform(scaled_data)
df['UMAP1'], df['UMAP2'] = umap_result[:, 0], umap_result[:, 1]
sns.scatterplot(x='UMAP1', y='UMAP2', hue='sex_name', data=df, palette='Spectral', legend=None)
plt.title("Wizualizacja UMAP")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()

# Macierz korelacji
correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Macierz korelacji")
plt.show()

# Testy statystyczne - ANOVA
# Porównanie średnich wartości `val` w zależności od `sex_name`
anova_model = ols('val ~ C(sex_name)', data=df).fit()
anova_results = anova_lm(anova_model)
print("Wyniki testu ANOVA:")
print(anova_results)
