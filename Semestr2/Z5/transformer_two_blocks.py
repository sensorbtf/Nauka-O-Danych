# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 09:16:46 2025

@author: mateu
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# Funkcja pomocnicza: blok Transformer Encoder
def transformer_encoder(inputs, num_heads, key_dim):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)

    dense_output = Dense(512, activation='relu')(attention_output)
    dense_output = LayerNormalization(epsilon=1e-6)(dense_output + attention_output)
    return dense_output

# Dane sztuczne
X = np.random.rand(1000, 10, 512)
Y = np.random.rand(1000, 512)

# Wejście
inputs = Input(shape=(10, 512))
x = transformer_encoder(inputs, num_heads=4, key_dim=64)
x = transformer_encoder(x, num_heads=4, key_dim=64)  # Drugi blok
x = GlobalAveragePooling1D()(x)
outputs = Dense(512, activation='linear')(x)

model = Model(inputs=inputs, outputs=outputs)

# Kompilacja
model.compile(optimizer='adam', loss='mse')

# Trening
history = model.fit(X, Y, epochs=10, batch_size=32)

# Wykres
plt.plot(history.history['loss'], label='Strata treningowa')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.show()
