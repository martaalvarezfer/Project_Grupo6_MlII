# **Machine Learning II – Deep Learning Project**

## **Canonist.ia**

El objetivo de este proyecto es aplicar *transfer learning* a un caso de negocio simulado relacionado con portales inmobiliarios y desplegar el modelo en una aplicación **Streamlit** accesible públicamente.

La aplicación recibe una imagen y la clasifica según su entorno (**habitación**, **cocina**, **bosque**, **industria**, etc.), utilizando diversas **CNN preentrenadas** que se ajustan a nuestras necesidades. Para ello, se entrenan varios modelos empleando la plataforma **Weights & Biases (W&B)**, donde se reportan métricas detalladas y se comparan entre ellos. 

Una vez entrenado el modelo, se integra en una app **Streamlit** que permite a los usuarios cargar imágenes y obtener una predicción de forma sencilla y en tiempo real.

Este proyecto se ha desarrollado como parte de la asignatura *Machine Learning II* del **Máster en Big Data** de la Universidad **Comillas ICAI**.

### 👥 **Equipo del proyecto**

| Nombre                      | Email                              |
|----------------------------|------------------------------------|
| Marta Álvarez Fernández    | 202402675@alu.comillas.edu         |
| Leticia Cólgan Valero      | leticiacologan@alu.comillas.edu    |
| Ana Vera Peña              | 202416868@alu.comillas.edu         |
| Antonio Bajo Gómez-Madurga | 202410510@alu.comillas.edu         |

---

## ⚙️ **Requisitos previos**

1. Clona el repositorio

```bash
git clone https://github.com/martaalvarezfer/Project_Grupo6_MLII.git
```

2. Crea el entorno virtual e instala las dependencias

```bash
cd Project_Grupo6_MLII
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Asegúrate de incluir en `.gitignore` tus archivos sensibles y los `.pt`.

---

## 📁 **Estructura de carpetas**

- `src/`  
  - `project/`: scripts de entrenamiento, arquitectura CNN, comparaciones y generación de CSVs de predicción.  
  - `streamlit/`: script para lanzar la app.
- `models/`: contiene los archivos `.pt` con los pesos entrenados (NO incluidos en GitHub).
- `dataset/`: conjunto de imágenes utilizadas para entrenamiento y validación.
- `img/`: imágenes ilustrativas como capturas de la app.

⚠️ **Importante**  
Los archivos `.pt` superan los **100 MB**, por lo tanto no se incluyen en GitHub. Además, tampoco pudimos alojarlos correctamente en W&B.  
Para usar la app, debes descargarlos desde esta carpeta de Google Drive y colocarlos en el directorio `models/`:

🔗 https://drive.google.com/drive/folders/16Kv4ADU2d1ow-46PWrHoT4E0jnmw3cOn?usp=sharing

---

## 🧠 **Modelo seleccionado**

**ConvNeXt Base** entrenado con **Cross-Entropy Loss**

Este modelo ha sido el mejor evaluado. Se ha utilizado la arquitectura `convnext_base`, preentrenada en ImageNet, ajustada mediante *fine-tuning* descongelando las últimas **5 capas**. Entrenado durante **7 épocas**, con imágenes RGB de tamaño **224x224**, usando:

- Optimizador: `Adam`
- Learning rate: `0.0001`
- Función de pérdida: `Cross-Entropy`

📊 **Resultados:**

- Entrenamiento: **91.06%**
- Validación: **93.60%**
- Train loss: `0.275`
- Val loss: `0.178`

🔍 **Errores más comunes del modelo:**

El modelo tiene dificultades para diferenciar entornos visualmente similares, por ejemplo:

- `Bedroom` → `Living room` 
- `Office` → `Kitchen` 

Esto sugiere que las confusiones se dan principalmente en interiores con elementos comunes como iluminación o distribución.

---

## 🚀 **Lanzar la app**

Puedes probar el modelo directamente con la app ejecutando:

```bash
streamlit run src/streamlit/app.py
```

### Ejemplo de interfaz:

![App](img/app.png)

---

## 🔧 **Mejoras propuestas**

1. **Aumentar la variedad del dataset**  
   Aplicar técnicas como *data augmentation* (rotaciones, escalado, etc.).

2. **Modelos de segundo nivel**  
   Hemos detectado que nuestros modelos son mejores en cierto tipo de imagen que en otras, es decir en algunas no presenta un porcentaje       alto de confianza en ninguna de ellas. Por tanto estas dudodas se pasarían a submodelos que están especializados en diversos tipos de       imágenes y se seleccionaria el que presente el mayor porcentaje de confianza. Por tanto podríamos aplicar un primer modelo que detecte.     Esta técnica puede aumentar la precisión general, porque permite hacer predicciones más cuidadosas y específicas en los casos más           difíciles, en lugar de forzar una decisión poco segura con el modelo principal.

3. **Entrenamiento con GPU**  
   Limitación clave del proyecto. Modelos más pesados como `resnext101_64x4d` no se han podido entrenar completamente.

4. **Descongelar capas específicas**  
   En lugar de hacerlo por número, elegir capas concretas relevantes.

5. **Función de pérdida personalizada**  
   Adaptarla a desequilibrios o dificultades del dataset.

---

Este proyecto nos ha permitido simular un flujo real de desarrollo de un modelo en producción, incluyendo la monitorización, evaluación, y despliegue en una aplicación funcional para el usuario final.
