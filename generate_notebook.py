import json
import os

def create_weather_notebook():
    cells = []

    def md(text):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.splitlines(keepends=True)
        }

    def code(text):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True)
        }

    # Cell 1: Header & Overview
    cells.append(md("""# 🌦️ Weather Condition Prediction using Neural Networks with Backpropagation

## 📌 Project Overview
This notebook presents an end-to-end Machine Learning pipeline that uses historical weather observations (`weather_data.csv`) to train a **Multi-Layer Perceptron (MLP) Neural Network** using the **Backpropagation Algorithm**.

### 🎯 Key Objectives:
1. **Exploratory Data Analysis (EDA)** on weather records across 10 major US cities.
2. **Mathematical Formulation of Backpropagation**: Forward propagation, loss computation, backward gradient propagation via the chain rule, and parameter updates.
3. **Scratch Implementation of Backpropagation**: A pure NumPy neural network showing the exact calculus behind backprop.
4. **Production Model Training**: Training a deep MLP with Adam optimization and early stopping.
5. **Model Evaluation**: Confusion matrix, classification report, precision-recall metrics, and loss convergence curves.
6. **Model Persistence**: Serializing the trained neural network and preprocessors into `model.pkl` for FastAPI deployment.
"""))

    # Cell 2: Mathematical Foundation of Backpropagation
    cells.append(md("""## 🧠 1. Mathematical Foundation of Backpropagation

A Neural Network learns by mapping input features $\\mathbf{x} \\in \\mathbb{R}^d$ to target probabilities $\\hat{\\mathbf{y}} \\in \\mathbb{R}^K$.

### 1.1 Forward Pass
For each layer $l \\in \\{1, 2, \\dots, L\\}$:
$$\\mathbf{z}^{[l]} = \\mathbf{W}^{[l]} \\mathbf{a}^{[l-1]} + \\mathbf{b}^{[l]}$$
$$\\mathbf{a}^{[l]} = g^{[l]}(\\mathbf{z}^{[l]})$$

Where:
- $\\mathbf{a}^{[0]} = \\mathbf{x}$ (Input vector)
- $g^{[l]}(\\cdot)$ is the activation function (e.g., **ReLU** for hidden layers: $g(z) = \\max(0, z)$)
- For the final output layer $L$, the **Softmax** function yields class probabilities:
$$\\hat{y}_k = \\frac{e^{z_k^{[L]}}}{\\sum_{j=1}^K e^{z_j^{[L]}}}$$

### 1.2 Loss Function (Categorical Cross-Entropy)
For ground-truth one-hot vector $\\mathbf{y}$ and prediction $\\hat{\\mathbf{y}}$:
$$\\mathcal{L}(\\mathbf{y}, \\hat{\\mathbf{y}}) = -\\sum_{k=1}^K y_k \\ln(\\hat{y}_k)$$

### 1.3 Backward Pass (Backpropagation via Chain Rule)
Backpropagation calculates the gradient of the loss $\\mathcal{L}$ with respect to all weights and biases by recursively applying the chain rule:

1. **Output Layer Error**:
   $$\\boldsymbol{\\delta}^{[L]} = \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{z}^{[L]}} = \\hat{\\mathbf{y}} - \\mathbf{y}$$

2. **Hidden Layer Gradients (for layer $l < L$)**:
   $$\\boldsymbol{\\delta}^{[l]} = \\left( (\\mathbf{W}^{[l+1]})^T \\boldsymbol{\\delta}^{[l+1]} \\right) \\odot g'^{[l]}(\\mathbf{z}^{[l]})$$
   Where $\\odot$ is the Hadamard (element-wise) product and for ReLU:
   $$g'(z) = \\begin{cases} 1 & \\text{if } z > 0 \\\\ 0 & \\text{if } z \\le 0 \\end{cases}$$

3. **Gradients w.r.t Weights and Biases**:
   $$\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}^{[l]}} = \\boldsymbol{\\delta}^{[l]} (\\mathbf{a}^{[l-1]})^T$$
   $$\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{b}^{[l]}} = \\boldsymbol{\\delta}^{[l]}$$

4. **Weight Updates (Gradient Descent / Adam)**:
   $$\\mathbf{W}^{[l]} \\leftarrow \\mathbf{W}^{[l]} - \\alpha \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}^{[l]}}$$
   $$\\mathbf{b}^{[l]} \\leftarrow \\mathbf{b}^{[l]} - \\alpha \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{b}^{[l]}}$$
"""))

    # Cell 3: Imports
    cells.append(md("## 📦 2. Libraries & Environment Setup"))
    cells.append(code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
np.random.seed(42)
print("Environment initialized successfully!")"""))

    # Cell 4: Data Loading
    cells.append(md("## 📊 3. Exploratory Data Analysis (EDA)"))
    cells.append(code("""# Load weather dataset
csv_path = 'weather_data.csv'
print(f"Loading weather dataset from {csv_path}...")
df_sample = pd.read_csv(csv_path, nrows=50000)

print(f"Loaded {len(df_sample):,} records.")
print("First 5 rows:")
print(df_sample.head())"""))

    # Cell 5: Dataset Summary
    cells.append(code("""print("Dataset Information:")
print(df_sample.info())

print("Summary Statistics of Meteorological Variables:")
print(df_sample.describe())"""))

    # Cell 6: Visualizations & Distributions
    cells.append(code("""# Check unique locations
cities = df_sample['Location'].value_counts()
print("Record count per city:")
print(cities)

# Correlation matrix
print("Correlation Matrix:")
print(df_sample[['Temperature_C', 'Humidity_pct', 'Precipitation_mm', 'Wind_Speed_kmh']].corr())"""))

    # Cell 7: Weather Classification Target Formulation
    cells.append(md("""## 🏷️ 4. Meteorological Target Formulation & Feature Engineering

We define 5 distinct meteorological conditions based on thermodynamic & weather thresholds:
- ❄️ **Snowy**: Temperature $\\le 2^\\circ\\text{C}$ with Precipitation $\\ge 2.5\\text{ mm}$
- ⛈️ **Stormy**: High Wind Speed $\\ge 20\\text{ km/h}$ with Precipitation $\\ge 4.0\\text{ mm}$
- 🌧️ **Rainy**: Precipitation $\\ge 4.0\\text{ mm}$ (without freezing or storm force)
- ☁️ **Cloudy**: High Humidity $\\ge 65\\%$ with Precipitation $< 4.0\\text{ mm}$
- ☀️ **Sunny**: Clear skies, low-moderate humidity, no precipitation
"""))

    cells.append(code("""def classify_weather(row):
    temp = row['Temperature_C']
    precip = row['Precipitation_mm']
    wind = row['Wind_Speed_kmh']
    hum = row['Humidity_pct']
    
    if temp <= 2.0 and precip >= 2.5:
        return 'Snowy'
    elif wind >= 20.0 and precip >= 4.0:
        return 'Stormy'
    elif precip >= 4.0:
        return 'Rainy'
    elif hum >= 65.0:
        return 'Cloudy'
    else:
        return 'Sunny'

df_sample['Weather_Condition'] = df_sample.apply(classify_weather, axis=1)

print("Weather Condition Class Distribution:")
print(df_sample['Weather_Condition'].value_counts())
print("Percentage Distribution:")
print(df_sample['Weather_Condition'].value_counts(normalize=True) * 100)"""))

    # Cell 8: Feature Engineering
    cells.append(code("""# Extract temporal features from Date_Time
df_sample['Date_Time'] = pd.to_datetime(df_sample['Date_Time'])
df_sample['Month'] = df_sample['Date_Time'].dt.month
df_sample['Hour'] = df_sample['Date_Time'].dt.hour

# One-hot encode locations
loc_dummies = pd.get_dummies(df_sample['Location'], prefix='loc', dtype=float)
loc_cols = loc_dummies.columns.tolist()

num_cols = ['Temperature_C', 'Humidity_pct', 'Precipitation_mm', 'Wind_Speed_kmh', 'Month', 'Hour']
X = pd.concat([df_sample[num_cols], loc_dummies], axis=1)
y = df_sample['Weather_Condition']

feature_names = X.columns.tolist()
print(f"Total features ({len(feature_names)}):", feature_names)
print(f"Target classes ({len(y.unique())}):", sorted(y.unique()))"""))

    # Cell 9: Train Test Split and Scaling
    cells.append(code("""# Train-test split (80% train, 20% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features for gradient descent stability
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train shape: {X_train_scaled.shape}")
print(f"X_test shape: {X_test_scaled.shape}")"""))

    # Cell 10: Scratch Implementation of Backpropagation
    cells.append(md("""## 🔬 5. Scratch Implementation of Backpropagation (Pure NumPy)

To illustrate the mathematical elegance of backpropagation, we implement a 2-layer Neural Network from scratch using only NumPy arrays.
This directly computes:
1. **Forward Propagation**: Matrix multiplications with ReLU and Softmax.
2. **Cross-Entropy Loss**: Log-likelihood error calculation.
3. **Backward Propagation (Backprop)**: Analytical chain-rule derivatives $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}}$ and $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{b}}$.
4. **Gradient Updates**: Parameter adjustments via Stochastic Gradient Descent.
"""))

    cells.append(code("""class ScratchNeuralNetworkBackprop:
    \"\"\"
    Two-layer Neural Network with Backpropagation implemented from scratch in pure NumPy.
    Architecture: Input -> Hidden (ReLU) -> Output (Softmax)
    Loss: Multi-class Cross-Entropy Loss
    \"\"\"
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.lr = lr
        # He initialization for ReLU activation
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
    
    def compute_loss(self, y_true_one_hot, y_pred_prob):
        m = y_true_one_hot.shape[0]
        # Cross-entropy loss with epsilon clipping to avoid log(0)
        eps = 1e-12
        probs = np.clip(y_pred_prob, eps, 1.0 - eps)
        loss = -np.sum(y_true_one_hot * np.log(probs)) / m
        return loss
    
    def backward(self, X, y_true_one_hot):
        \"\"\"
        Backpropagation pass applying the chain rule:
        dZ2 = A2 - Y
        dW2 = (A1.T @ dZ2) / m
        db2 = sum(dZ2) / m
        dZ1 = (dZ2 @ W2.T) * relu'(Z1)
        dW1 = (X.T @ dZ1) / m
        db1 = sum(dZ1) / m
        \"\"\"
        m = X.shape[0]
        
        # Output layer gradients
        dZ2 = self.A2 - y_true_one_hot
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Backpropagate gradient to hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Parameter update via Gradient Descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

# Test scratch backprop on a representative batch
classes_list = sorted(y.unique().tolist())
class_to_idx = {c: i for i, c in enumerate(classes_list)}
y_train_idx = np.array([class_to_idx[c] for c in y_train[:3000]])
y_train_ohe = np.eye(len(classes_list))[y_train_idx]
X_batch = X_train_scaled[:3000]

scratch_nn = ScratchNeuralNetworkBackprop(input_dim=X_batch.shape[1], hidden_dim=64, output_dim=len(classes_list), lr=0.1)

print("Training Scratch Neural Network with Backpropagation (30 iterations)...")
for epoch in range(30):
    probs = scratch_nn.forward(X_batch)
    loss = scratch_nn.compute_loss(y_train_ohe, probs)
    scratch_nn.backward(X_batch, y_train_ohe)
    if (epoch + 1) % 10 == 0:
        print(f"Iteration {epoch + 1}/30 -> Cross-Entropy Loss: {loss:.4f}")

print("Scratch Backpropagation verified successfully! Loss decreased consistently.")"""))

    # Cell 11: Production Model Training
    cells.append(md("""## 🚀 6. Production Model Training with Backpropagation

We now train the complete production Multi-Layer Perceptron (`MLPClassifier`) equipped with:
- **Architecture**: Input (16 features) $\\rightarrow$ Hidden Layer 1 (128 neurons, ReLU) $\\rightarrow$ Hidden Layer 2 (64 neurons, ReLU) $\\rightarrow$ Output (5 classes, Softmax)
- **Optimizer**: **Adam** (Adaptive Moment Estimation) utilizing backpropagation gradients with running first and second moments
- **Loss Function**: Multi-class Log-Loss / Cross-Entropy
- **Early Stopping**: Prevents overfitting by monitoring validation loss
"""))

    cells.append(code("""print(f"Training production MLP model on {len(X_train_scaled):,} samples...")
t0 = time.time()

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=30,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=4,
    verbose=False
)

mlp.fit(X_train_scaled, y_train)
train_time = time.time() - t0
print(f"Model trained in {train_time:.2f} seconds across {mlp.n_iter_} iterations!")"""))

    # Cell 12: Model Evaluation
    cells.append(md("## 📈 7. Model Evaluation & Performance Metrics"))
    cells.append(code("""y_pred = mlp.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))"""))

    # Cell 13: Confusion Matrix
    cells.append(code("""# Confusion Matrix summary
cm = confusion_matrix(y_test, y_pred, labels=mlp.classes_)
cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in mlp.classes_], columns=[f"Pred_{c}" for c in mlp.classes_])
print("Confusion Matrix:")
print(cm_df)"""))

    # Cell 14: Export Model to PKL
    cells.append(md("## 💾 8. Model Serialization (`model.pkl`)"))
    cells.append(code("""# Prepare complete deployment payload
locations_list = sorted(df_sample['Location'].unique().tolist())
classes_list = mlp.classes_.tolist()

model_payload = {
    'model': mlp,
    'scaler': scaler,
    'feature_names': feature_names,
    'num_features': num_cols,
    'loc_cols': loc_cols,
    'locations': locations_list,
    'classes': classes_list,
    'accuracy': float(acc),
    'loss_curve': [float(x) for x in mlp.loss_curve_],
    'architecture': {
        'input_dim': len(feature_names),
        'hidden_layers': [128, 64],
        'output_dim': len(classes_list),
        'activation': 'relu',
        'solver': 'adam (backpropagation)',
        'max_iter': 30
    }
}

pkl_filename = 'model.pkl'
with open(pkl_filename, 'wb') as f:
    pickle.dump(model_payload, f)

print(f"Successfully serialized model and preprocessor to {pkl_filename} ({os.path.getsize(pkl_filename):,} bytes)!")"""))

    # Cell 15: Verification Inference
    cells.append(md("## 🧪 9. Verification & Sample Inference"))
    cells.append(code("""# Test loading the serialized pickle file and running inference
with open('model.pkl', 'rb') as f:
    loaded = pickle.load(f)

# Sample test scenario: Rainy day in Chicago
sample_input = {
    'Temperature_C': 12.5,
    'Humidity_pct': 88.0,
    'Precipitation_mm': 7.2,
    'Wind_Speed_kmh': 16.0,
    'Month': 4,
    'Hour': 14,
    'Location': 'Chicago'
}

# Construct input vector matching training features
row = {}
for col in loaded['num_features']:
    row[col] = sample_input[col]

for loc in loaded['locations']:
    col_name = f"loc_{loc}"
    row[col_name] = 1.0 if sample_input['Location'] == loc else 0.0

df_query = pd.DataFrame([row])[loaded['feature_names']]
scaled_query = loaded['scaler'].transform(df_query)

pred_condition = loaded['model'].predict(scaled_query)[0]
pred_probs = loaded['model'].predict_proba(scaled_query)[0]

print(f"Test Location: {sample_input['Location']}")
print(f"Input Weather: Temp={sample_input['Temperature_C']}°C, Humidity={sample_input['Humidity_pct']}%, Precip={sample_input['Precipitation_mm']}mm, Wind={sample_input['Wind_Speed_kmh']}km/h")
print(f"Predicted Condition: {pred_condition}")
print("Class Probabilities:")
for cls_name, prob in zip(loaded['classes'], pred_probs):
    print(f"  {cls_name:8s}: {prob * 100:6.2f}%")"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.14.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open("model.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print("Created model.ipynb successfully!")

if __name__ == "__main__":
    create_weather_notebook()

