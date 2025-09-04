import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import Zeros, Constant, GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping

tf.config.run_functions_eagerly(True)

# Load and preprocess Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Define function to create models
def create_model(initializer, optimizer, use_regularization=False):
    model = Sequential()
    model.add(Dense(16, activation="relu", input_shape=(4,), kernel_initializer=initializer))
    if use_regularization:
        model.add(Dropout(0.3))  # Drop neurons randomly
    model.add(Dense(8, activation="relu", kernel_initializer=initializer))
    if use_regularization:
        model.add(Dropout(0.3))
    model.add(Dense(3, activation="softmax", kernel_initializer=initializer))

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Initializers & Optimizers to test
initializers = {
    "Zeros": Zeros(),
    "Constant(0.1)": Constant(0.1),
    "Xavier(Glorot)": GlorotUniform(),
    "HeNormal": HeNormal()
}
optimizers = {
    "SGD": lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
    "Momentum": lambda: tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "Adagrad": lambda: tf.keras.optimizers.Adagrad(learning_rate=0.01),
    "Adam": lambda: tf.keras.optimizers.Adam(learning_rate=0.01),
    "RMSprop": lambda: tf.keras.optimizers.RMSprop(learning_rate=0.01)
}


# Run experiments
results_no_reg, results_reg = {}, {}
es = EarlyStopping(patience=5, restore_best_weights=True)

for init_name, init in initializers.items():
    for opt_name, opt in optimizers.items():
        label = f"{init_name} + {opt_name}"

        # Without regularization
        model = create_model(init, opt(), use_regularization=False)
        history = model.fit(X_train, y_train, epochs=50, batch_size=16,
                            validation_split=0.2, verbose=0, callbacks=[es])
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        results_no_reg[label] = acc

        # With regularization (Dropout + EarlyStopping)
        model = create_model(init, opt(), use_regularization=True)
        history = model.fit(X_train, y_train, epochs=50, batch_size=16,
                            validation_split=0.2, verbose=0, callbacks=[es])
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        results_reg[label] = acc


# Plot comparison
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.barh(list(results_no_reg.keys()), list(results_no_reg.values()), color="skyblue", edgecolor="black")
plt.title("Test Accuracy (Without Regularization)")
plt.xlabel("Accuracy")

plt.subplot(1,2,2)
plt.barh(list(results_reg.keys()), list(results_reg.values()), color="lightgreen", edgecolor="black")
plt.title("Test Accuracy (With Regularization: Dropout + EarlyStopping)")
plt.xlabel("Accuracy")

plt.tight_layout()
plt.show()
