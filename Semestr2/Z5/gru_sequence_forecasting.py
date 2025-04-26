# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 09:16:34 2025

@author: mateu
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Sztuczne dane
X = np.random.rand(1000, 10, 32)
y = np.random.rand(1000, 1)

# Budowa modelu
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, dropout=0.3, input_shape=(10, 32)),
    tf.keras.layers.Dense(1)
])

# Kompilacja
model.compile(optimizer='adam', loss='mse')

# Trening
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Wykres
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.show()
