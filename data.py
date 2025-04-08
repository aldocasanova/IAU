import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random

# Descargar recursos de NLTK
nltk.download('stopwords')

# --- Configuración Inicial ---
np.random.seed(42)

# Lista de 20 palabras relevantes (seleccionadas como representativas de textos religiosos)
relevant_words = [
    "pray", "love", "god", "peace", "soul", "wisdom", "faith", "holy", "spirit", "truth",
    "sin", "grace", "heaven", "earth", "light", "dark", "hope", "joy", "mercy", "bless"
]

# Clases objetivo (8 clases)
classes = ["Hinduism", "Buddhism", "Taoism", "Confucianism", "Judaism", "Christianity", "Islam", "Others"]

# --- Generación de Frecuencias por Clase ---
# Simulamos frecuencias base para cada palabra en cada clase (20 palabras x 8 clases)
base_frequencies = np.random.randint(1, 20, size=(len(relevant_words), len(classes)))  # Frecuencias entre 1 y 19

# --- Generación del Dataset ---
data = []
labels = []

for class_idx in range(len(classes)):
    # Generar 74 o 73 instancias para train/val por clase (total 590)
    num_train_val = 74 if class_idx < 2 else 73  # Total 590
    num_test = 13 if class_idx < 4 else 12       # Total 100
    
    # Train/val instances
    for _ in range(num_train_val):
        # Generar frecuencias con variación aleatoria basada en las frecuencias base de la clase
        freqs = base_frequencies[:, class_idx] + np.random.randint(-5, 6, size=len(relevant_words))
        freqs = np.clip(freqs, 0, None)  # Asegurar que no haya frecuencias negativas
        data.append(freqs)
        labels.append(class_idx)
    
    # Test instances
    for _ in range(num_test):
        freqs = base_frequencies[:, class_idx] + np.random.randint(-5, 6, size=len(relevant_words))
        freqs = np.clip(freqs, 0, None)
        data.append(freqs)
        labels.append(class_idx)

# Convertir a array numpy y normalizar
data = np.array(data)
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Crear DataFrame
df = pd.DataFrame(data_normalized, columns=relevant_words)
df["label"] = labels

# --- Verificación ---
print(f"Total instancias: {len(df)}")
print(f"Distribución de etiquetas: {np.bincount(df['label'])}")
print("Primeras filas del dataset (normalizado):")
print(df.head())

# --- División del Dataset ---
X_train_val, X_test, y_train_val, y_test = train_test_split(
    df.drop("label", axis=1), df["label"], test_size=100/690, stratify=df["label"], random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42
)

# Reconstruir DataFrames para guardar
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# --- Verificación de la división ---
print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Train labels: {np.bincount(train_df['label'])}")
print(f"Val labels: {np.bincount(val_df['label'])}")
print(f"Test labels: {np.bincount(test_df['label'])}")

# --- Guardar el Dataset ---
train_df.to_csv("train_dataset.csv", index=False)
val_df.to_csv("val_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)

print("\nDataset generado y guardado en 'train_dataset.csv', 'val_dataset.csv', 'test_dataset.csv'")