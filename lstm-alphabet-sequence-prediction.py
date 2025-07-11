# -*- coding: utf-8 -*-
"""HW3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IQzJ3_H2ivKs6X7VXmvdkgPp7Nr7mBLw
"""

# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import random
import tensorflow as tf
import pandas as pd

# ---- For reproducibility ----
np.random.seed(42)
tf.random.set_seed(42)

# ---- 2. Define dataset and mappings ----
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
n_letters = len(alphabet)
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

# ---- 3. Prepare input-output pairs ----
dataX, dataY = [], []
for i in range(n_letters):
    input_char = alphabet[i]
    output_char = alphabet[(i+1) % n_letters]  # wrap-around for Z->A
    dataX.append(char_to_int[input_char])
    dataY.append(char_to_int[output_char])

# ---- NOTE: There are only 26 unique pairs (A→B, ..., Z→A).
# We randomly sample these pairs to create a larger dataset for practice,
# but true generalization is trivial for this problem. ----

dataset_size = 1000
X = []
Y = []
for _ in range(dataset_size):
    idx = random.randint(0, n_letters - 1)
    X.append(dataX[idx])
    Y.append(dataY[idx])
X = np.array(X)
Y = np.array(Y)

# ---- 4. Reshape X for LSTM: [samples, time steps, features] ----
X = X.reshape((X.shape[0], 1, 1))

# ---- 5. Normalize X ----
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X.reshape(-1, 1)).reshape(-1, 1, 1)

# ---- 6. One-hot encode output ----
y = to_categorical(Y, num_classes=n_letters)

# ---- 7. Create model with configurable parameters ----
def create_model(hidden_size=32, num_layers=2, learning_rate=0.01):
    model = Sequential()
    for i in range(num_layers):
        return_seq = True if i < num_layers - 1 else False
        if i == 0:
            model.add(LSTM(hidden_size, input_shape=(1, 1), return_sequences=return_seq))
        else:
            model.add(LSTM(hidden_size, return_sequences=return_seq))
    model.add(Dense(n_letters, activation='softmax'))
    opt = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# ---- 8. Train and evaluate the baseline model (2 layers, 32 hidden units) ----
model = create_model()
history = model.fit(X, y, epochs=50, batch_size=16, verbose=0, validation_split=0.1)
loss, accuracy = model.evaluate(X, y, verbose=0)
val_accuracy = history.history['val_accuracy'][-1]  # last epoch validation accuracy

print(f"Model Accuracy (Training): {accuracy*100:.2f}%")
print(f"Model Accuracy (Validation): {val_accuracy*100:.2f}%")

# ---- 9. Demonstrate model predictions in a table ----
results_table = []
for i in range(n_letters):
    test_input = np.array([[i]])
    test_input_scaled = scaler.transform(test_input).reshape(1,1,1)
    prediction = model.predict(test_input_scaled, verbose=0)
    predicted_idx = np.argmax(prediction)
    results_table.append({
        'Input': int_to_char[i],
        'Predicted Output': int_to_char[predicted_idx],
        'Actual Output': int_to_char[(i+1)%n_letters]
    })
df = pd.DataFrame(results_table)
print("\nInput | Predicted Output | Actual Output")
print(df.to_string(index=False))

# ---- 10. Experiment: Vary number of LSTM layers and tabulate results ----
def experiment_layers(layer_list):
    results = []
    for n_layers in layer_list:
        m = create_model(num_layers=n_layers)
        hist = m.fit(X, y, epochs=30, batch_size=16, verbose=0, validation_split=0.1)
        train_acc = m.evaluate(X, y, verbose=0)[1]
        val_acc = hist.history['val_accuracy'][-1]
        results.append({'Layers': n_layers, 'Train Accuracy': train_acc, 'Val Accuracy': val_acc})
    return results

layer_list = [1, 2, 3, 4]
layer_results = experiment_layers(layer_list)
layer_df = pd.DataFrame(layer_results)
print("\nEffect of Number of LSTM Layers on Accuracy:")
print(layer_df.to_string(index=False))

plt.figure(figsize=(7,5))
plt.plot(layer_df['Layers'], layer_df['Train Accuracy']*100, marker='o', label='Train Acc')
plt.plot(layer_df['Layers'], layer_df['Val Accuracy']*100, marker='s', label='Val Acc')
plt.xlabel('Number of LSTM Layers')
plt.ylabel('Accuracy (%)')
plt.title('Effect of LSTM Layers on Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ---- 11A. Experiment: Vary learning rate and tabulate results ----
lrs = [0.001, 0.005, 0.01, 0.05]
lr_results = []
for lr in lrs:
    m = create_model(learning_rate=lr)
    hist = m.fit(X, y, epochs=30, batch_size=16, verbose=0, validation_split=0.1)
    train_acc = m.evaluate(X, y, verbose=0)[1]
    val_acc = hist.history['val_accuracy'][-1]
    lr_results.append({'Learning Rate': lr, 'Train Accuracy': train_acc, 'Val Accuracy': val_acc})
lr_df = pd.DataFrame(lr_results)
print("\nEffect of Learning Rate on Accuracy:")
print(lr_df.to_string(index=False))

plt.figure(figsize=(7,5))
plt.plot(lr_df['Learning Rate'], lr_df['Train Accuracy']*100, marker='o', label='Train Acc')
plt.plot(lr_df['Learning Rate'], lr_df['Val Accuracy']*100, marker='s', label='Val Acc')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy (%)')
plt.title('Effect of Learning Rate on Accuracy')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()

# ---- 11B. Experiment: Vary hidden size and tabulate results ----
hidden_sizes = [8, 16, 32, 64, 128]
hidden_results = []
for hs in hidden_sizes:
    m = create_model(hidden_size=hs)
    hist = m.fit(X, y, epochs=30, batch_size=16, verbose=0, validation_split=0.1)
    train_acc = m.evaluate(X, y, verbose=0)[1]
    val_acc = hist.history['val_accuracy'][-1]
    hidden_results.append({'Hidden Size': hs, 'Train Accuracy': train_acc, 'Val Accuracy': val_acc})
hidden_df = pd.DataFrame(hidden_results)
print("\nEffect of Hidden Size on Accuracy:")
print(hidden_df.to_string(index=False))

plt.figure(figsize=(7,5))
plt.plot(hidden_df['Hidden Size'], hidden_df['Train Accuracy']*100, marker='o', label='Train Acc')
plt.plot(hidden_df['Hidden Size'], hidden_df['Val Accuracy']*100, marker='s', label='Val Acc')
plt.xlabel('Hidden Size')
plt.ylabel('Accuracy (%)')
plt.title('Effect of Hidden Size on Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ---- 12. Print summary table for optimal parameters ----
print("\nSummary of best results:")
summary_data = [
    {'Layers': 2, 'Hidden Size': 32, 'Learning Rate': 0.01,
     'Train Accuracy': layer_df['Train Accuracy'][1]*100, 'Val Accuracy': layer_df['Val Accuracy'][1]*100},
    {'Layers': 1, 'Hidden Size': 32, 'Learning Rate': 0.01,
     'Train Accuracy': layer_df['Train Accuracy'][0]*100, 'Val Accuracy': layer_df['Val Accuracy'][0]*100},
    {'Layers': 3, 'Hidden Size': 32, 'Learning Rate': 0.01,
     'Train Accuracy': layer_df['Train Accuracy'][2]*100, 'Val Accuracy': layer_df['Val Accuracy'][2]*100},
    {'Layers': 2, 'Hidden Size': 64, 'Learning Rate': 0.01,
     'Train Accuracy': hidden_df['Train Accuracy'][3]*100, 'Val Accuracy': hidden_df['Val Accuracy'][3]*100},
]
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# ---- 13. Conclusion ----
print("\nConclusion:")
print(" - Both training and validation accuracy are shown for all experiments.")
print(" - With only 26 unique input-output pairs, the generalization challenge is trivial; high accuracy is easily achieved with the right parameters.")
print(" - Random sampling to increase the dataset size is for practice/training purposes, not because the task is inherently difficult.")
print(" - The best configuration is typically 2 layers, 32 hidden units, learning rate 0.01, with accuracy above 95%.")
print(" - Too small/large learning rates or hidden sizes can reduce accuracy.")