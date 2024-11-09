import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from plotnine import *


# Dane do wykresu liniowego
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Tworzenie wykresu liniowego
plt.plot(x, y, marker='o', linestyle='-', color='b', label="Liniowy")
plt.xlabel("Oś X")
plt.ylabel("Oś Y")
plt.title("Wykres Liniowy")
plt.legend()
plt.show()

# Dane do wykresu słupkowego
kategorie = ['A', 'B', 'C', 'D']
wartosci = [5, 7, 3, 8]

# Tworzenie wykresu słupkowego
plt.bar(kategorie, wartosci, color='orange')
plt.xlabel("Kategorie")
plt.ylabel("Wartości")
plt.title("Wykres Słupkowy")
plt.show()

# Dane do histogramu
dane = np.random.normal(0, 1, 1000)

# Tworzenie histogramu
plt.hist(dane, bins=30, color='purple', alpha=0.7)
plt.xlabel("Wartości")
plt.ylabel("Częstotliwość")
plt.title("Histogram")
plt.show()

# Dane do wykresu kołowego
kategorie = ['A', 'B', 'C', 'D']
wartosci = [15, 30, 45, 10]

# Tworzenie wykresu kołowego
plt.pie(wartosci, labels=kategorie, autopct='%1.1f%%', startangle=90)
plt.title("Wykres Kołowy")
plt.show()

#Matplotlib
#Matplotlib#Matplotlib
#Matplotlib

x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 35]
plt.plot(x, y, color='green', marker='x', linestyle='--', linewidth=2)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Dostosowany Wykres Liniowy")
plt.show()

# Grid w Matplodlib
plt.plot(x, y, color='blue', marker='o')
plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Wykres z Siatką")
plt.show()

# Informacje w Matplodlib
plt.plot(x, y, marker='o')
plt.annotate('Najwyższy punkt', xy=(5, 35), xytext=(3, 30), 
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.title("Wykres z Adnotacją")
plt.show()

# ROzproszone linie w Matplodlib
x = np.random.rand(50)
y = np.random.rand(50)
sizes = 1000 * np.random.rand(50)
colors = np.random.rand(50)
plt.scatter(x, y, s=sizes, c=colors, alpha=0.5, cmap='viridis')
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Wykres Punktowy")
plt.show()

# 3D w Matplodlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
ax.plot_surface(X, Y, Z, cmap='plasma')
plt.title("Wykres 3D")
plt.show()


# Przykładowe dane
data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 6, 8, 10]
})

# Tworzenie podstawowego wykresu punktowego
(ggplot(data) + aes(x='x', y='y') + geom_point())

# 15.4 Wykres Punktowy (Scatter Plot)
# Wykres punktowy do wizualizacji zależności między zmiennymi
(ggplot(data) +
 aes(x='x', y='y') +
 geom_point() +
 ggtitle("Wykres Punktowy") +
 xlab("Oś X") +
 ylab("Oś Y"))

# 15.5 Wykres Liniowy
# Wykres liniowy do przedstawienia trendów na danych ciągłych
(ggplot(data) +
 aes(x='x', y='y') +
 geom_line(color='blue') +
 ggtitle("Wykres Liniowy") +
 xlab("Oś X") +
 ylab("Oś Y"))

# 15.6 Wykres Słupkowy
# Dane do wykresu słupkowego
data_bar = pd.DataFrame({
    'kategorie': ['A', 'B', 'C', 'D'],
    'wartość': [4, 7, 1, 8]
})

# Tworzenie wykresu słupkowego
(ggplot(data_bar) +
 aes(x='kategorie', y='wartość') +
 geom_bar(stat='identity', fill='skyblue') +
 ggtitle("Wykres Słupkowy") +
 xlab("Kategorie") +
 ylab("Wartość"))


# Dane do histogramu
data_hist = pd.DataFrame({
    'wartość': np.random.normal(0, 1, 1000)
})
# Tworzenie histogramu
(ggplot(data_hist) +
 aes(x='wartość') +
 geom_histogram(bins=30, fill='purple', alpha=0.7) +
 ggtitle("Histogram") +
 xlab("Wartość") +
 ylab("Częstotliwość"))

# 16.1 Kolory i Style w Plotnine
# Wykres punktowy z dostosowanymi kolorami
data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [10, 20, 25, 30, 35]
})
(ggplot(data) +
 aes(x='x', y='y') +
 geom_point(color='red', size=3) +
 ggtitle("Wykres Punktowy z Dostosowanymi Kolorami") +
 xlab("Oś X") +
 ylab("Oś Y"))

# 16.2 Dodawanie Linii Trendu
# Wykres punktowy z linią trendu
(ggplot(data) +
 aes(x='x', y='y') +
 geom_point() +
 geom_smooth(method='lm') +
 ggtitle("Wykres z Linią Trendu") +
 xlab("Oś X") +
 ylab("Oś Y"))

# 17.1 Facetowanie w Plotnine
# Przykładowe dane do facetowania
data_facet = pd.DataFrame({
    'x': np.tile([1, 2, 3, 4], 2),
    'y': [1, 2, 3, 4, 2, 3, 4, 5],
    'grupa': ['A']*4 + ['B']*4
})
# Tworzenie wykresu z facetowaniem
(ggplot(data_facet) +
 aes(x='x', y='y') +
 geom_point() +
 facet_wrap('~grupa') +
 ggtitle("Wykres z Facetowaniem") +
 xlab("Oś X") +
 ylab("Oś Y"))

# 17.2 Wykres Pudłowy (Box Plot)
# Przykładowe dane do wykresu pudłowego
data_box = pd.DataFrame({
    'kategorie': np.random.choice(['A', 'B', 'C'], 100),
    'wartość': np.random.randn(100)
})
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
