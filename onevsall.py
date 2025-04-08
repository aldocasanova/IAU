import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Cargar el Dataset ---
train_df = pd.read_csv("train_dataset.csv")
val_df = pd.read_csv("val_dataset.csv")
test_df = pd.read_csv("test_dataset.csv")

X_train, y_train = train_df.drop("label", axis=1), train_df["label"]
X_val, y_val = val_df.drop("label", axis=1), val_df["label"]
X_test, y_test = test_df.drop("label", axis=1), test_df["label"]

# --- Definir y Entrenar Modelo One-vs-All ---
ovr_model = LogisticRegression(multi_class='ovr', max_iter=1000, C=1.0)
ovr_model.fit(X_train, y_train)

# --- Evaluación del Modelo ---
y_train_pred = ovr_model.predict(X_train)
y_val_pred = ovr_model.predict(X_val)
y_test_pred = ovr_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred) * 100  # Convertir a porcentaje
val_acc = accuracy_score(y_val, y_val_pred) * 100        # Convertir a porcentaje
test_acc = accuracy_score(y_test, y_test_pred) * 100     # Convertir a porcentaje

train_metrics = precision_recall_fscore_support(y_train, y_train_pred, average='weighted')
val_metrics = precision_recall_fscore_support(y_val, y_val_pred, average='weighted')
test_metrics = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')

# --- Resultados ---
print("=== Resultados One-vs-All ===")
print(f"Porcentaje de aciertos en Train: {train_acc:.2f}%")
print(f"Porcentaje de aciertos en Validation: {val_acc:.2f}%")
print(f"Porcentaje de aciertos en Test: {test_acc:.2f}%")
print(f"\nMétricas detalladas (Precision, Recall, F1):")
print(f"Train: Precision: {train_metrics[0]:.4f}, Recall: {train_metrics[1]:.4f}, F1: {train_metrics[2]:.4f}")
print(f"Validation: Precision: {val_metrics[0]:.4f}, Recall: {val_metrics[1]:.4f}, F1: {val_metrics[2]:.4f}")
print(f"Test: Precision: {test_metrics[0]:.4f}, Recall: {test_metrics[1]:.4f}, F1: {test_metrics[2]:.4f}")

# --- Verificación de Predicciones Perfectas ---
if train_acc == 100.0 and val_acc == 100.0 and test_acc == 100.0:
    print("\n¡El modelo One-vs-All ha alcanzado un 100% de aciertos en todos los conjuntos!")
else:
    print("\nNota: Hay algunos errores de clasificación. Revisa las predicciones para más detalles.")

# --- Guardar Predicciones ---
results_df = pd.DataFrame({"true_label": y_test, "predicted_label": y_test_pred})
results_df = pd.concat([X_test.reset_index(drop=True), results_df], axis=1)
results_df.to_csv("one_vs_all_predictions.csv", index=False)
print("\nPredicciones guardadas en 'one_vs_all_predictions.csv'")