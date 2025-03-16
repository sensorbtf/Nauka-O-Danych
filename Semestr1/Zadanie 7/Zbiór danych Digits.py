# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 07:48:41 2024

@author: mateu
"""

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the Digits Dataset
digits = load_digits()
X = digits.data  # Features (8x8 pixel values flattened into 64 features)
y = digits.target  # Labels (digits 0–9)

# Step 2: Data Exploration
print("Dataset Overview:")
print(f"Feature Shape: {X.shape}")
print(f"Target Shape: {y.shape}")
print(f"Classes: {np.unique(y)}")

# Visualize some digit images
fig, axes = plt.subplots(1, 10, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')
plt.show()

# Step 3: Split and Standardize Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train Classifiers
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)

# SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# Step 5: Evaluate Models
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

evaluate_model(y_test, log_reg_pred, "Logistic Regression")
evaluate_model(y_test, svm_pred, "SVM")
evaluate_model(y_test, knn_pred, "k-NN")

# Step 6: Visualize Some Predictions
def visualize_predictions(images, true_labels, predicted_labels):
    fig, axes = plt.subplots(1, 10, figsize=(15, 4))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}")
        ax.axis('off')
    plt.show()

# Visualize predictions for the first 10 test samples
visualize_predictions(digits.images[-len(y_test):][:10], y_test[:10], log_reg_pred[:10])
