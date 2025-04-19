import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pandas as pd
from cnn_model import CNN  # Tu clase personalizada
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
MODEL_NAME = "convnext_base_unfreezed5_epochs7_acc95"
MODEL_PATH = "/Users/martaalvarez/MLII-Grupo6/Project_Grupo6_MLII/models/convnext_base_unfreezed5_epochs7_acc95.pt"
IMG_SIZE = 224
BATCH_SIZE = 1
VALID_DIR = "/Users/martaalvarez/MLII-Grupo6/Project_Grupo6_MLII/dataset/validation"

# -------------------------------
# TRANSFORMACIONES
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# CARGA DE DATOS
# -------------------------------
dataset = datasets.ImageFolder(VALID_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

# -------------------------------
# CARGA DEL MODELO
# -------------------------------
weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
base_model = convnext_base(weights=weights)

model = CNN(base_model, num_classes=len(idx_to_class), unfreezed_layers=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# -------------------------------
# PREDICCIONES
# -------------------------------
results = []
softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloader):
        outputs = model(inputs)
        probs = softmax(outputs)
        conf, preds = torch.max(probs, 1)

        real_class = idx_to_class[labels.item()]
        pred_class = idx_to_class[preds.item()]
        confidence = round(conf.item() * 100, 2)
        image_path = dataset.samples[i][0]

        results.append({
            "model": MODEL_NAME,
            "real_class": real_class,
            "predicted_class": pred_class,
            "confidence (%)": confidence,
            "image_path": image_path
        })

# -------------------------------
# GUARDAR CSV
# -------------------------------
os.makedirs("outputs", exist_ok=True)
csv_name = f"outputs/predictions_{MODEL_NAME}.csv"
df = pd.DataFrame(results)
df.to_csv(csv_name, index=False)
print(f"✅ CSV guardado como: {csv_name}")
