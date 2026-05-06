# 🖼️ CIFAR-100 Image Classification — Deep Vision

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christianirshool-glitch/cifar100-image-classification/blob/main/ET_4.ipynb)
![TensorFlow](https://img.shields.io/badge/Engine-TensorFlow%2FKeras-orange?style=flat-square&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
[![License](https://img.shields.io/badge/Licencia-MIT-green)](LICENSE)

Proyecto de Deep Learning enfocado en la clasificación de imágenes sobre el dataset **CIFAR-100** (100 clases, 60.000 imágenes). Se diseñan, entrenan y comparan dos estrategias: **Transfer Learning con fine-tuning** sobre una red preentrenada y una **CNN entrenada desde cero**.

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

---

## 🎯 Objetivo

Diseñar y comparar dos estrategias de clasificación de imágenes:

- Aprovechar el conocimiento de redes preentrenadas en ImageNet mediante Transfer Learning y Fine-Tuning.
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
| `ET_4.ipynb` — Estrategia 1 | Transfer Learning + Fine-Tuning (MobileNetV2) |
| `ET_4.ipynb` — Estrategia 2 | CNN diseñada y entrenada desde cero |

### 4. Optimización
Ambas estrategias aplican de forma justificada:
- **Weight Regularization** (L2)
- **Dropout**
- **Batch Normalization**
- **Data Augmentation**

### 5. Evaluación
- Accuracy y Loss en entrenamiento y validación.
- `classification_report` sobre el conjunto de test.
- Curvas de aprendizaje (Loss/Accuracy por época).

---

## 🧠 Modelos desarrollados

### 🔹 Estrategia 1: Transfer Learning + Fine-Tuning
- **Base model:** MobileNetV2 preentrenada en ImageNet (`include_top=False`).
- Congelación inicial de todas las capas convolucionales.
- Top model personalizado: `Flatten → Dense(512, relu) → Dense(100, softmax)`.
- Fine-Tuning progresivo: descongelación de capas superiores en una segunda fase de entrenamiento.
- Optimizador: Adam · Loss: Categorical Crossentropy.

### 🔹 Estrategia 2: CNN desde cero
- Arquitectura convolucional diseñada empíricamente.
- Bloques `Conv2D → BatchNorm → Activation → MaxPooling → Dropout`.
- Justificación de la selección de hiperparámetros: learning rate, batch size, número de capas y neuronas.
- Mismas técnicas de regularización y data augmentation aplicadas de forma controlada.

---

## 📈 Resultados

| Modelo | Test Accuracy |
|---|---|
| MobileNetV2 (Transfer Learning + Fine-Tuning) | XX% |
| CNN desde cero | XX% |

> Rellena los valores con los resultados obtenidos en tu ejecución.

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

```
cifar100-image-classification/
├── ET_4.ipynb        # Notebook principal (ambas estrategias)
├── requirements.txt  # Dependencias
├── LICENSE           # Licencia MIT
└── README.md
```

---

## 👤 Autor

**Christian Méndez Giraldo**  
Data Scientist · MSc in Data Science & AI  
[GitHub](https://github.com/christianirshool-glitch)

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT** — consulta el archivo [LICENSE](LICENSE) para más detalles.
