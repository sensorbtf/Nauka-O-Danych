import pandas as pd

# Ścieżka do pliku CSV
csv_file_path = r"C:\Users\Sensorbtf\OneDrive\studia\Magisterka\Dane\IHME_GBD_2019_SMOKING_AGE_1990_2019_INIT_AGE_Y2021M05D27.csv"
df = pd.read_csv(csv_file_path)
df_cleaned = df.dropna(subset=['year_id', 'val'])

# Obliczamy średnie rozpowszechnienie palenia dla każdej płci
mean_prevalence_by_sex = df_cleaned.groupby('sex_name')['val'].mean().round(2)
print("\n===== Średnie rozpowszechnienie palenia w zależności od płci =====\n")
print(mean_prevalence_by_sex)

# 2. Obliczamy medianę rozpowszechnienia palenia (z użyciem funkcji round() z Pythona)
median_prevalence = round(df_cleaned['val'].median(), 2)
print("\n===== Mediana rozpowszechnienia palenia =====\n")
print(f"Mediana rozpowszechnienia palenia: {median_prevalence}")

# 3. Obliczamy odchylenie standardowe dla każdej płci
std_prevalence_by_sex = df_cleaned.groupby('sex_name')['val'].std().round(2)
print("\n===== Odchylenie standardowe rozpowszechnienia palenia w zależności od płci =====\n")
print(std_prevalence_by_sex)

# 4. Obliczamy kowariancję między 'sex_id' a 'val' (rozpowszechnieniem palenia)
covariance_sex = df_cleaned[['sex_id', 'val']].cov().round(4)
print("\n===== Kowariancja między płcią a rozpowszechnieniem palenia =====\n")
print(covariance_sex)

# 5. Obliczamy korelację między 'sex_id' a 'val' (rozpowszechnieniem palenia)
correlation_sex = df_cleaned['sex_id'].corr(df_cleaned['val']).round(4)
print("\n===== Korelacja między płcią a rozpowszechnieniem palenia =====\n")
print(f"Korelacja między płcią a rozpowszechnieniem palenia: {correlation_sex}")
