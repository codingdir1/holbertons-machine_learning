#!/usr/bin/env python3

import os
import numpy as np
import GPyOpt
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Setup Data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define the Objective Function (Black-box for GPyOpt)
def keras_objective(x):
    # Parameters provided by GPyOpt as a 2D array
    lr       = float(x[:, 0])
    units    = int(x[:, 1])
    dropout  = float(x[:, 2])
    l2_reg   = float(x[:, 3])
    batch_sz = int(x[:, 4])

    # Construct the Model
    model = Sequential([
        Dense(units, activation='relu', kernel_regularizer=l2(l2_reg), input_shape=(20,)),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

    # Early Stopping to prevent overfitting
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Satisficing Metric Checkpointing
    # Filename encodes hyperparameter values
    ckpt_name = f"model_lr{lr:.4f}_u{units}_dr{dropout:.2f}_l2{l2_reg:.4f}_bs{batch_sz}.keras"
    mc = ModelCheckpoint(ckpt_name, monitor='val_accuracy', save_best_only=True, mode='max')

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=batch_sz,
        callbacks=[es, mc],
        verbose=0
    )

    # Return the best validation loss (GPyOpt minimizes by default)
    return np.min(history.history['val_loss'])

# 3. Define Hyperparameter Search Space
domain = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.01)},
    {'name': 'units',         'type': 'discrete',   'domain': (16, 32, 64, 128)},
    {'name': 'dropout_rate',  'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'l2_weight',     'type': 'continuous', 'domain': (0.0001, 0.01)},
    {'name': 'batch_size',    'type': 'discrete',   'domain': (16, 32, 64)}
]

# 4. Initialize and Run Bayesian Optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=keras_objective, 
    domain=domain,
    model_type='GP',
    acquisition_type='EI'
)

# Limit to 30 iterations as requested
optimizer.run_optimization(max_iter=30)

# 5. Output Report and Convergence Plot
print("\n--- Optimization Complete ---")
print(f"Best Validation Loss: {optimizer.fx_opt:.4f}")
print(f"Optimal Hyperparameters: {optimizer.x_opt}")

# Save report to text file
with open('bayes_opt.txt', 'w') as f:
    f.write(f"Best Loss: {optimizer.fx_opt}\n")
    f.write(f"Best Parameters: {optimizer.x_opt}\n")

# Plot and show convergence
optimizer.plot_convergence()
plt.show()


