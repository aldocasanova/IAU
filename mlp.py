import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- Configurar Dispositivo CUDA ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# --- Cargar el Dataset ---
train_df = pd.read_csv("train_dataset.csv")
val_df = pd.read_csv("val_dataset.csv")
test_df = pd.read_csv("test_dataset.csv")

X_train, y_train = train_df.drop("label", axis=1).values, train_df["label"].values
X_val, y_val = val_df.drop("label", axis=1).values, val_df["label"].values
X_test, y_test = test_df.drop("label", axis=1).values, test_df["label"].values

# --- Dataset y DataLoader ---
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_dataset = TextDataset(X_train, y_train)
val_dataset = TextDataset(X_val, y_val)
test_dataset = TextDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Definir MLP ---
class MLP(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_classes=8):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

model = MLP(input_size=X_train.shape[1]).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# --- Entrenamiento ---
num_epochs = 100  # Mantengo las 150 épocas que usaste
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = (correct / total) * 100  # Convertir a porcentaje
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validación
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = (correct / total) * 100  # Convertir a porcentaje
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# --- Evaluación en Test ---
model.eval()
test_loss, correct, total = 0.0, 0, 0
test_predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_predictions.extend(predicted.cpu().numpy())

test_acc = (correct / total) * 100  # Convertir a porcentaje
print(f"Test Accuracy: {test_acc:.2f}%")

# --- Gráficas ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('MLP Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.title('MLP Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')  # Actualizar etiqueta del eje Y
plt.legend()
plt.tight_layout()
plt.savefig("mlp_curves.png")
plt.show()

# --- Guardar Predicciones ---
results_df = pd.DataFrame({"true_label": y_test, "predicted_label": test_predictions})
results_df = pd.concat([pd.DataFrame(X_test, columns=train_df.columns[:-1]), results_df], axis=1)
results_df.to_csv("mlp_predictions.csv", index=False)
print("Predicciones guardadas en 'mlp_predictions.csv'")