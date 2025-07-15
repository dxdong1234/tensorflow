# TensorFlow 2.x example: Linear Regression

import tensorflow as tf
import numpy as np

# 1. Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# 2. Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# 3. Define the loss function and optimizer
model.compile(optimizer='sgd', loss='mean_squared_error')

# 4. Train the model
model.fit(X, y, epochs=100)

# 5. Print the learned parameters
weights, bias = model.layers[0].get_weights()
print(f"Learned weights: {weights[0][0]:.4f}")
print(f"Learned bias: {bias[0]:.4f}")
