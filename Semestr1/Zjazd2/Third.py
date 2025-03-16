import pandas as pd

# Ścieżka do pliku CSV
csv_file_path = r"C:\Users\Sensorbtf\OneDrive\studia\Magisterka\Dane\IHME_GBD_2019_SMOKING_AGE_1990_2019_INIT_AGE_Y2021M05D27.csv"
df = pd.read_csv(csv_file_path)

# Zadanie 1: Wczytywanie danych i wyświetlanie podstawowych informacji
print("\n=====ZADANIE 1=====\n")
print("\n===== Pierwsze 5 wierszy danych =====\n")
print(df.head())s

print("\n===== Podstawowe informacje o danych =====\n")
print(df.info())

print("\n===== Statystyki opisowe =====\n")
print(df.describe())

# Zadanie 2: Obliczanie podstawowych statystyk
print("\n=====ZADANIE 2=====\n")
mean_val = df['val'].mean()
print(f"\nŚrednia wartość rozpowszechnienia palenia: {mean_val:.2f}")

median_val = df['val'].median()
print(f"Mediana wartości rozpowszechnienia palenia: {median_val:.2f}")

std_val = df['val'].std()
print(f"Odchylenie standardowe wartości rozpowszechnienia palenia: {std_val:.2f}")

# Zadanie 3: Identyfikacja i obsługa brakujących danych
print("\n=====ZADANIE 3=====\n")
missing_values = df.isnull().sum()
print("\n===== Brakujące wartości w każdej kolumnie =====\n")
print(missing_values)

# Uzupełnianie brakujących danych w kolumnie 'val' (bez inplace)
df['val'] = df['val'].fillna(df['val'].mean())

# Usuwanie wierszy z brakującymi danymi w 'year_id'
df.dropna(subset=['year_id'], inplace=True)

# Zadanie 4: Wykrywanie wartości odstających (IQR)
print("\n=====ZADANIE 4=====\n")
Q1 = df['val'].quantile(0.25)
Q3 = df['val'].quantile(0.75)
IQR = Q3 - Q1

# Wartości odstające
outliers = df[(df['val'] < (Q1 - 1.5 * IQR)) | (df['val'] > (Q3 + 1.5 * IQR))]
print("\n===== Wartości odstające =====\n")
print(outliers)

# Zadanie 5: Analiza zależności między kolumnami
print("\n=====ZADANIE 5=====\n")
# Filtrujemy kolumny liczbowe do obliczenia korelacji
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr()

print("\n===== Macierz korelacji (tylko dane liczbowe) =====\n")
print(correlation_matrix)

# Zadanie 6: Przekształcanie danych
print("\n=====ZADANIE 6=====\n")
df['val_normalized'] = (df['val'] - df['val'].min()) / (df['val'].max() - df['val'].min())
print("\n===== Nowa kolumna 'val_normalized' =====\n")
print(df[['val', 'val_normalized']].head())

grouped = df.groupby('sex_name')['val'].mean()
print("\n===== Średnie wartości rozpowszechnienia palenia w zależności od płci =====\n")
print(grouped)

df_sorted = df.sort_values(by='val', ascending=False)
print("\n===== Dane posortowane według wartości 'val' =====\n")
print(df_sorted.head())


