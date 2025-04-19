import pandas as pd

# Carga tu CSV
df = pd.read_csv("outputs/predictions_convnext_base_unfreezed5_epochs7_acc95.csv")

# Filtra solo las predicciones incorrectas
errores = df[df["real_class"] != df["predicted_class"]]

# Agrupa por clase real y clase predicha para contar errores específicos
confusion = errores.groupby(["real_class", "predicted_class"]).size().reset_index(name="count")

# Ordena para ver los errores más comunes
confusion_ordenada = confusion.sort_values(by="count", ascending=False)

# Muestra los top errores
print("🔍 Principales confusiones del modelo:")
print(confusion_ordenada.head(10))