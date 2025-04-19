import os
import streamlit as st
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import numpy as np
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

# --- Configuración general ---
num_classes = 15
classnames = [
    "Bedroom", "Coast", "Forest", "Highway", "Industrial",
    "Inside city", "Kitchen", "Living room", "Mountain", "Office",
    "Open country", "Store", "Street", "Suburb", "Tall building"
]
Images_size = 224
Images_types = ['jpg', 'jpeg', 'png']
models_dir = "/Users/martaalvarez/MLII-Grupo6/Project_Grupo6_MLII/models"
selected_model_name = "convnext_base_unfreezed5_epochs7_acc95"
model_path = os.path.join(models_dir, selected_model_name + ".pt")

# --- Arquitectura CNN ---
class CNN(nn.Module):
    def __init__(self, base_model, num_classes, unfreezed_layers=0):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        if unfreezed_layers > 0:
            for layer in list(self.base_model.children())[-unfreezed_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.fc = nn.Sequential(
            nn.Linear(self.base_model.classifier[2].in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )
        self.base_model.classifier = nn.Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- Dataset personalizado ---
class CustomImageDataset(Dataset):
    def __init__(self, image, transform=None):
        self.image = image
        self.transform = transform
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        image = self.transform(self.image) if self.transform else self.image
        return image, 0

# --- Gradiente rojo-naranja según confianza ---
def get_gradient_color(conf, min_conf, max_conf):
    f = (conf - min_conf) / (max_conf - min_conf) if max_conf != min_conf else 1
    G = int(165 * f)
    return f"#ff{G:02x}00"

# --- App principal ---
def main():
    st.set_page_config(page_title="Clasificación de Imágenes con ConvNeXt", layout="centered")
    st.title("Clasificación de Imágenes con ConvNeXt")
    st.markdown("""
    Esta app clasifica imágenes usando un modelo entrenado con imágenes en color (RGB)  
    basado en **ConvNeXt Base**.
    """)

    image_file = st.file_uploader("📷 Cargar imagen", type=Images_types)

    if image_file is not None:
        with st.spinner("Procesando imagen..."):
            image = Image.open(image_file)

            # Asegurar que la imagen esté en RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            transform_pipeline = transforms.Compose([
                transforms.Resize((Images_size, Images_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            dataset = CustomImageDataset(image, transform=transform_pipeline)
            loader = DataLoader(dataset, batch_size=1)

        with st.spinner("Cargando el modelo y clasificando..."):
            if not os.path.exists(model_path):
                st.error(f"No se encontró el modelo en: {model_path}")
                return

            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
            base_model = convnext_base(weights=weights)

            model = CNN(base_model, num_classes, unfreezed_layers=5)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

            with torch.no_grad():
                for img, _ in loader:
                    outputs = model(img)
                    probs = outputs[0].detach().cpu().numpy()
                    sorted_indices = np.argsort(probs)[::-1]
                    top_n = 5

                    st.subheader("📊 Predicciones:")
                    other_confidences = [probs[i] * 100 for i in sorted_indices[1:top_n]]
                    min_conf = min(other_confidences) if other_confidences else 0
                    max_conf = max(other_confidences) if other_confidences else 0

                    top_idx = sorted_indices[0]
                    top_conf = probs[top_idx] * 100
                    nombre_top = classnames[top_idx]
                    st.markdown(f"""
                    <div style="background-color: lightgreen; color: black; text-align: center; padding: 10px; border-radius: 5px; font-size: 20px; margin-bottom: 5px;">
                        <b>{nombre_top}</b>: {top_conf:.2f}%
                    </div>
                    """, unsafe_allow_html=True)

                    for idx in sorted_indices[1:top_n]:
                        conf = probs[idx] * 100
                        color = get_gradient_color(conf, min_conf, max_conf)
                        nombre_clase = classnames[idx]
                        st.markdown(f"""
                        <div style="background-color: {color}; color: black; text-align: center; padding: 10px; border-radius: 5px; font-size: 20px; margin-bottom: 5px;">
                            <b>{nombre_clase}</b>: {conf:.2f}%
                        </div>
                        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Imagen cargada (RGB)", use_column_width=True)

        st.write("¿La predicción es correcta?")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Sí"):
                st.success("Gracias por confirmar que la predicción es correcta.")
        with col_b:
            if st.button("❌ No"):
                st.error("Gracias por tu feedback. Se marcará como error la predicción.")
    else:
        st.info("Por favor, carga una imagen para realizar la clasificación.")

if __name__ == "__main__":
    main()