# 🖼️ CIFAR-100 Image Classification — Deep Vision

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christianirshool-glitch/cifar100-image-classification/blob/main/ET_4.ipynb)
![TensorFlow](https://img.shields.io/badge/Engine-TensorFlow%2FKeras-orange?style=flat-square&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
[![License](https://img.shields.io/badge/Licencia-MIT-green)](LICENSE)

Proyecto de Deep Learning enfocado en la clasificación de imágenes sobre el dataset **CIFAR-100** (100 clases, 60.000 imágenes). Se diseñan, entrenan y comparan dos estrategias: **Transfer Learning con fine-tuning** sobre arquitecturas preentrenadas y una **CNN entrenada desde cero**.

---

## 📌 Índice

- [Contexto](#-contexto)
- [Objetivo](#-objetivo)
- [Dataset](#-dataset)
- [Metodología](#-metodología)
- [Modelos desarrollados](#-modelos-desarrollados)
- [Resultados](#-resultados)
- [Tech Stack](#️-tech-stack)
- [Cómo ejecutarlo](#-cómo-ejecutarlo)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Autor](#-autor)
- [Licencia](#-licencia)

---

## 📌 Contexto

CIFAR-100 es uno de los benchmarks más utilizados en visión por computador. Con 100 clases distribuidas en 20 superclases y sólo 600 imágenes por clase (32×32 px), representa un reto real de clasificación multiclase en el que la arquitectura del modelo y las técnicas de optimización son determinantes para el rendimiento final.

La baja resolución espacial del dataset supone una restricción arquitectónica importante: modelos como Xception o InceptionV3 fueron descartados por requerir tamaños mínimos de entrada (71×71 o 75×75 px), evidenciando que no todas las arquitecturas preentrenadas son compatibles con este tipo de datos.

---

## 🎯 Objetivo

Diseñar y comparar dos estrategias de clasificación de imágenes:

- Aprovechar el conocimiento de redes preentrenadas en ImageNet mediante Transfer Learning, evaluando múltiples arquitecturas y aplicando Fine-Tuning.
- Diseñar una red convolucional propia entrenada desde cero, con justificación empírica de cada decisión arquitectónica.
- Evaluar y comparar ambos enfoques en el conjunto de test de CIFAR-100.

---

## 📊 Dataset

| Característica | Detalle |
|---|---|
| Dataset | [CIFAR-100](https://keras.io/api/datasets/cifar100/) |
| Número de clases | 100 (agrupadas en 20 superclases) |
| Imágenes de entrenamiento | 50.000 |
| Imágenes de test | 10.000 |
| Resolución | 32×32 píxeles · RGB |

---

## 🔧 Metodología

### 1. Preparación de datos
- Carga y normalización del dataset desde `tf.keras.datasets`.
- Codificación One-Hot de las etiquetas.
- Preprocesamiento específico por arquitectura (`preprocess_input`).

### 2. Análisis exploratorio
- Visualización de muestras por clase.
- Análisis de la distribución de clases (histograma de frecuencias).

### 3. Modelado

| Notebook | Enfoque |
|---|---|
| `ET_4.ipynb` — Estrategia 1 | Transfer Learning + Fine-Tuning (EfficientNetB0 y ConvNeXtTiny) |
| `ET_4.ipynb` — Estrategia 2 | CNN diseñada y entrenada desde cero |

### 4. Optimización
Ambas estrategias aplican de forma justificada:
- **Weight Regularization** (L2)
- **Dropout** progresivo (0.25 en capas conv., 0.5 en capas densas)
- **Batch Normalization**
- **Data Augmentation**
- **Label Smoothing** (Estrategia 2)
- **Callbacks**: `ReduceLROnPlateau` y `EarlyStopping`

### 5. Evaluación
- Accuracy y Loss en entrenamiento y validación.
- `classification_report` sobre el conjunto de test.
- Curvas de aprendizaje (Loss/Accuracy por época).

---

## 🧠 Modelos desarrollados

### 🔹 Estrategia 1: Transfer Learning + Fine-Tuning

Se evaluaron 6 arquitecturas preentrenadas en ImageNet. Xception e InceptionV3 fueron descartadas por incompatibilidad con la resolución 32×32. Las dos arquitecturas finalistas fueron:

**EfficientNetB0** (selección por robustez):
- Seleccionada por su excelente equilibrio y menor nivel de overfitting entre todas las arquitecturas evaluadas.
- Diferencia mínima entre entrenamiento (~42%) y validación (~34%), indicando alta capacidad de generalización.
- Top model: `GlobalAveragePooling2D → Dense(512, relu) → Dense(100, softmax)`.
- Fine-Tuning aplicado: sin mejoras significativas por la baja resolución del dataset (32×32), que limita la extracción de nuevas características en capas profundas.
- Modificaciones del Top Model (hasta 1024-512-256 neuronas) no incrementaron el accuracy; se mantuvo la arquitectura base.
- Data Augmentation afectó negativamente al mantener la base congelada.
- Única técnica con impacto positivo: **Batch Normalization** (ligera estabilización).

**ConvNeXtTiny** (mejor accuracy final):
- Arquitectura de convoluciones modernas con diseño inspirado en Transformers.
- Alcanzó un **~52.70% de val_accuracy** y ~61.90% en entrenamiento tras 20 épocas.
- Destacó en categorías con rasgos visuales distintivos: `keyboard` (precisión 0.91), `apple` (0.71).
- Dropout y regularización mitigaron el sobreajuste inherente a su densidad arquitectónica.
- Posicionada como la arquitectura más robusta de la Estrategia 1 para imágenes de baja resolución.

### 🔹 Estrategia 2: CNN desde cero
- Arquitectura secuencial diseñada empíricamente con bloques `Conv2D (3×3) → BatchNorm → Activation → MaxPooling → Dropout`.
- Filtros incrementales: 32 → 64 → 128 para capturar características de complejidad creciente.
- Sustitución de `Flatten` por `GlobalAveragePooling2D` para reducir drásticamente los parámetros y el overfitting.
- Técnicas combinadas: **Data Augmentation** (traslaciones, rotaciones, volteos), **Dropout**, **L2** y **Label Smoothing**.
- Callbacks `ReduceLROnPlateau` y `EarlyStopping` para optimizar el entrenamiento.
- Techo de rendimiento alcanzado: ~**61-62% de accuracy**, limitado por la pérdida de información espacial de la API Sequential.

---

## 📈 Resultados

| Modelo | Train Accuracy | Val Accuracy |
|---|---|---|
| EfficientNetB0 (Transfer Learning) | ~42% | ~34% |
| ConvNeXtTiny (Transfer Learning) | ~61.90% | ~52.70% |
| CNN desde cero | ~61-62% | — |

> **Conclusión:** El Transfer Learning con ConvNeXtTiny superó al resto en accuracy absoluto. EfficientNetB0 destacó por su equilibrio y menor overfitting. La CNN desde cero demostró el valor de las técnicas de regularización pero encontró un techo arquitectónico propio de los modelos secuenciales.

---

## 🛠️ Tech Stack

| Herramienta | Uso |
|---|---|
| `TensorFlow / Keras` | Definición, entrenamiento y evaluación de modelos |
| `scikit-learn` | Métricas de evaluación (`classification_report`) |
| `NumPy` | Operaciones numéricas y manipulación de arrays |
| `Matplotlib` | Visualización de curvas de aprendizaje y muestras |
| `Google Colab` | Entorno de ejecución con GPU |

---

## 🚀 Cómo ejecutarlo

El notebook está diseñado para ejecutarse directamente en **Google Colaboratory** sin ninguna configuración local.

```bash
# Opción 1: Abre directamente en Colab con el badge de arriba

# Opción 2: Clona el repositorio
git clone https://github.com/christianirshool-glitch/cifar100-image-classification.git
cd cifar100-image-classification

# Instala las dependencias
pip install tensorflow scikit-learn matplotlib

# Ejecuta el notebook
jupyter notebook ET_4.ipynb
```

> ⚠️ Se recomienda usar GPU (disponible gratuitamente en Google Colab) para reducir los tiempos de entrenamiento.

---

## 📁 Estructura del proyecto
cifar100-image-classification/
├── ET_4.ipynb        # Notebook principal (ambas estrategias)
├── requirements.txt  # Dependencias
├── LICENSE           # Licencia MIT
└── README.md


---

## 👤 Autor

**Christian Méndez Giraldo**  
Data Scientist · MSc in Data Science & AI  
[GitHub](https://github.com/christianirshool-glitch)

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT** — consulta el archivo [LICENSE](LICENSE) para más detalles.
