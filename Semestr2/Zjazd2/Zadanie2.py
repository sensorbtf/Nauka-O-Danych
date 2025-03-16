# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 12:19:18 2025

@author: mateu
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Generowanie danych
X = np.random.randint(0, 1000, (500, 1))
y = (X % 2 == 0).astype(int)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Klasyfikator drzewa decyzyjnego
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predykcja i ocena modelu
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Dokładność modelu: {accuracy:.2f}")
