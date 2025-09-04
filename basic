import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ===============================
# Sigmoid and derivatives
# ===============================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ===============================
# Training data (XOR)
# ===============================
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# Seed for reproducibility
np.random.seed(42)

# ===============================
# Initialize weights and biases
# ===============================
input_layer_size = 2
hidden_layer_size = 4
output_layer_size = 1

W1 = np.random.randn(input_layer_size, hidden_layer_size)
b1 = np.zeros((1, hidden_layer_size))
W2 = np.random.randn(hidden_layer_size, output_layer_size)
b2 = np.zeros((1, output_layer_size))

# Training parameters
learning_rate = 0.1
epochs = 10000
losses = []

# ===============================
# Training loop
# ===============================
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Loss (MSE)
    loss = np.mean((y - a2) ** 2)
    losses.append(loss)

    # Backpropagation
    error_output = (a2 - y) * sigmoid_derivative(a2)
    error_hidden = np.dot(error_output, W2.T) * sigmoid_derivative(a1)

    # Update weights
    W2 -= learning_rate * np.dot(a1.T, error_output)
    b2 -= learning_rate * np.sum(error_output, axis=0, keepdims=True)
    W1 -= learning_rate * np.dot(X.T, error_hidden)
    b1 -= learning_rate * np.sum(error_hidden, axis=0, keepdims=True)

# ===============================
# Evaluation
# ===============================
pred_probs = a2
preds = (pred_probs > 0.5).astype(int)

# Accuracy
acc = accuracy_score(y, preds)
print("Predictions:", preds.flatten())
print("True labels:", y.flatten())
print(f"Accuracy: {acc*100:.2f}%")

# ===============================
# Visualization
# ===============================
# 1. Loss curve
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(losses, label="Training Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Loss Convergence")
plt.legend()

# 2. Confusion Matrix
cm = confusion_matrix(y, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
plt.subplot(1,2,2)
disp.plot(cmap=plt.cm.Blues, values_format="d", ax=plt.gca(), colorbar=False)
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()

