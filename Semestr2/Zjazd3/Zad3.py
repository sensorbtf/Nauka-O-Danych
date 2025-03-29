# -*- coding: utf-8 -*-


# Importy
from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# === Wczytanie danych ===
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 1. Porównaj dokładności modeli: Random Forest, XGBoost i Stacking ===
acc_scores = {}

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
acc_scores['Random Forest'] = accuracy_score(y_test, rf.predict(X_test))
print("Random Forest accuracy:", acc_scores['Random Forest'])

xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
acc_scores['XGBoost'] = accuracy_score(y_test, xgb.predict(X_test))
print("XGBoost accuracy:", acc_scores['XGBoost'])

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True))
]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack.fit(X_train, y_train)
acc_scores['Stacking'] = accuracy_score(y_test, stack.predict(X_test))
print("Stacking (RF+SVC) accuracy:", acc_scores['Stacking'])

# === 2. Tuning hiperparametrów dla modelu XGBoost ===
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), param_grid, cv=3)
grid.fit(X_train, y_train)
best_xgb = grid.best_estimator_
acc_scores['XGBoost Tuned'] = accuracy_score(y_test, best_xgb.predict(X_test))
print("XGBoost Tuned accuracy:", acc_scores['XGBoost Tuned'])
print("Best params for XGBoost:", grid.best_params_)

# === 3. Nowy model w stackingu (KNN + Decision Tree) ===
new_estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=3)),
    ('tree', DecisionTreeClassifier(max_depth=5))
]
new_stack = StackingClassifier(estimators=new_estimators, final_estimator=LogisticRegression())
new_stack.fit(X_train, y_train)
acc_scores['Stacking (KNN + Tree)'] = accuracy_score(y_test, new_stack.predict(X_test))
print("Stacking (KNN + Tree) accuracy:", acc_scores['Stacking (KNN + Tree)'])

# === 4. Test na innym zbiorze danych (Wine) ===
X2, y2 = load_wine(return_X_y=True)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

rf2 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X2_train, y2_train)
xgb2 = XGBClassifier(eval_metric='mlogloss', random_state=42).fit(X2_train, y2_train)
stack2 = StackingClassifier(
    estimators=[('rf', rf2), ('svc', SVC(probability=True))],
    final_estimator=LogisticRegression(max_iter=500)
).fit(X2_train, y2_train)

print("\n=== Wyniki na zbiorze Wine ===")
print("Random Forest:", accuracy_score(y2_test, rf2.predict(X2_test)))
print("XGBoost:", accuracy_score(y2_test, xgb2.predict(X2_test)))
print("Stacking:", accuracy_score(y2_test, stack2.predict(X2_test)))

# === 5. Wykres porównujący dokładność ===
plt.figure(figsize=(10, 6))
plt.bar(acc_scores.keys(), acc_scores.values(), color='skyblue')
plt.ylabel('Dokładność')
plt.title('Porównanie dokładności modeli (Iris dataset)')
plt.ylim(0.8, 1.05)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
