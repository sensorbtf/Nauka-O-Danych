import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import durbin_watson
import scipy.stats as stats
from scipy.stats import shapiro

# Load dataset
data_path = r"D:\OneDrive\Studia\Magisterka\Dane\IHME_GBD_2019_SMOKING_AGE_1990_2019_INIT_AGE_Y2021M05D27.CSV"
data = pd.read_csv(data_path)

# Fix duplicate column names and clean up
data.columns = ["measure_name", "location_id", "location_name", "sex_id", "sex_name", "age_group_id", "age_group_name", "year_id", "val", "upper", "lower"]

# Filter or clean data if necessary (example: drop NaN values)
data.dropna(inplace=True)

# Define features and target
X = data[["location_id", "sex_id", "age_group_id", "year_id"]]  # Adjust feature columns as needed
y = data["val"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results.append({'Model': name, 'MSE': mse, 'R2': r2})
    print(f"{name}: MSE={mse}, R2={r2}")

# Feature importance for Ridge
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': ridge_model.coef_
}).sort_values(by='Importance', ascending=False)
print("Feature Importance for Ridge:")
print(feature_importance)

# Visualize feature importance
feature_importance.plot(kind='bar', x='Feature', y='Importance', legend=False)
plt.title('Feature Importance for Ridge')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.show()

# Correlation matrix for multicollinearity
correlation_matrix = pd.DataFrame(X_train_scaled, columns=X.columns).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Residual analysis for Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
residuals = y_test - linear_model.predict(X_test_scaled)

# Plot residual histogram
plt.hist(residuals, bins=30, edgecolor='k')
plt.title('Residuals Histogram')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Q-Q plot for residuals
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot for Residuals')
plt.show()

# Shapiro-Wilk test for normality
shapiro_test = shapiro(residuals)
print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")

# Durbin-Watson test for autocorrelation
dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson Statistic: {dw_stat}")

# Visualize results
results_df = pd.DataFrame(results)
sns.barplot(x='Model', y='R2', data=results_df)
plt.title('R2 Scores by Model')
plt.show()

sns.barplot(x='Model', y='MSE', data=results_df)
plt.title('MSE by Model')
plt.show()

# Compare normalized vs. original data
original_data_results = []
scaled_data_results = []

for name, model in models.items():
    # Original data
    model.fit(X_train, y_train)
    predictions_orig = model.predict(X_test)
    mse_orig = mean_squared_error(y_test, predictions_orig)
    r2_orig = r2_score(y_test, predictions_orig)
    original_data_results.append({'Model': name, 'MSE': mse_orig, 'R2': r2_orig})

    # Normalized data
    model.fit(X_train_scaled, y_train)
    predictions_scaled = model.predict(X_test_scaled)
    mse_scaled = mean_squared_error(y_test, predictions_scaled)
    r2_scaled = r2_score(y_test, predictions_scaled)
    scaled_data_results.append({'Model': name, 'MSE': mse_scaled, 'R2': r2_scaled})

original_df = pd.DataFrame(original_data_results)
scaled_df = pd.DataFrame(scaled_data_results)

# Combine and visualize results
comparison_df = pd.concat([original_df, scaled_df], keys=['Original', 'Scaled'])
sns.barplot(x='Model', y='R2', hue=comparison_df.index.get_level_values(0), data=comparison_df.reset_index())
plt.title('Model Performance: Original vs. Scaled Data')
plt.show()

print("Original Data Results:")
print(original_df)

print("Scaled Data Results:")
print(scaled_df)
