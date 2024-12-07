

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from plotnine import *

file_path = "C:\\Users\\Sensorbtf\\OneDrive\\studia\\Magisterka\\Dane\\IHME_GBD_2019_SMOKING_AGE_1990_2019_INIT_AGE_Y2021M05D27.CSV"
data = pd.read_csv(file_path)

# 1. Wykres Liniowy (Line Plot)
years = data['year_id']
values = data['val']

plt.plot(years, values, marker='o', linestyle='-', color='b', label="Wartość")
plt.xlabel("Rok")
plt.ylabel("Średnia wartość")
plt.title("Wykres Liniowy - Średnia Wartość w Czasie")
plt.legend()
plt.show()

# 2. Wykres Słupkowy (Bar Plot)
locations = data['location_name'].head(10)
avg_values = data.groupby('location_name')['val'].mean().head(10)

plt.bar(locations, avg_values, color='orange')
plt.xlabel("Lokalizacja")
plt.ylabel("Średnia wartość")
plt.title("Wykres Słupkowy - Średnia Wartość na Lokalizację")
plt.xticks(rotation=45)
plt.show()

# 3. Histogram
plt.hist(data['val'].dropna(), bins=30, color='purple', alpha=0.7)
plt.xlabel("Wartości")
plt.ylabel("Częstotliwość")
plt.title("Histogram - Rozkład Wartości")
plt.show()

# 4. Wykres Kołowy (Pie Chart)
top_locations = data['location_name'].value_counts().head(4)
plt.pie(top_locations, labels=top_locations.index, autopct='%1.1f%%', startangle=90)
plt.title("Wykres Kołowy - Top 4 Lokalizacje")
plt.show()

# 5. Dostosowany Wykres Liniowy (Customized Line Plot)
plt.plot(years, values, color='green', marker='x', linestyle='--', linewidth=2)
plt.xlabel("Rok")
plt.ylabel("Średnia wartość")
plt.title("Dostosowany Wykres Liniowy")
plt.show()

# 6. Wykres z Siatką (Line Plot with Grid)
plt.plot(years, values, color='blue', marker='o')
plt.grid(True)
plt.xlabel("Rok")
plt.ylabel("Średnia wartość")
plt.title("Wykres z Siatką")
plt.show()

# 7. Wykres Punktowy (Scatter Plot)
ages = data['age_group_name'].head(50)  # assuming ages are numeric categories
plt.scatter(years.head(50), values.head(50), s=100, c=np.random.rand(50), alpha=0.5, cmap='viridis')
plt.colorbar()
plt.xlabel("Rok")
plt.ylabel("Średnia wartość")
plt.title("Wykres Punktowy - Wartości w Czasie")
plt.show()

# 8. Wykres 3D (3D Plot)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.linspace(0, len(years), 100), np.linspace(0, max(values), 100))
Z = np.sin(np.sqrt(X**2 + Y**2))
ax.plot_surface(X, Y, Z, cmap='plasma')
plt.title("Wykres 3D")
plt.show()

# Using Plotnine for ggplot-style plots
# Wykres Liniowy w Plotnine
(ggplot(data) +
 aes(x='year_id', y='val') +
 geom_line(color='blue') +
 ggtitle("Wykres Liniowy w Plotnine") +
 xlab("Rok") +
 ylab("Średnia wartość"))

# Wykres Słupkowy w Plotnine
data_bar = data.groupby('location_name', as_index=False)['val'].mean().head(10)
(ggplot(data_bar) +
 aes(x='location_name', y='val') +
 geom_bar(stat='identity', fill='skyblue') +
 ggtitle("Wykres Słupkowy w Plotnine") +
 xlab("Lokalizacja") +
 ylab("Średnia wartość"))

# Histogram w Plotnine
data_hist = data[['val']].dropna()
(ggplot(data_hist) +
 aes(x='val') +
 geom_histogram(bins=30, fill='purple', alpha=0.7) +
 ggtitle("Histogram w Plotnine") +
 xlab("Wartość") +
 ylab("Częstotliwość"))

# Wykres Pudłowy (Box Plot) w Plotnine
(ggplot(data) +
 aes(x='sex_name', y='val') +
 geom_boxplot(fill='lightblue') +
 ggtitle("Wykres Pudłowy w Plotnine") +
 xlab("Płeć") +
 ylab("Wartość"))
# Tworzenie wykresu pudłowego
(ggplot(data_box) +
 aes(x='kategorie', y='wartość') +
 geom_boxplot(fill='lightblue') +
 ggtitle("Wykres Pudłowy") +
 xlab("Kategorie") +
 ylab("Wartość"))

# 18 Dodawanie Motywów
# Dodanie motywu do wykresu
(ggplot(data) +
 aes(x='x', y='y') +
 geom_point() +
 ggtitle("Wykres z Motywem") +
 theme_minimal())
