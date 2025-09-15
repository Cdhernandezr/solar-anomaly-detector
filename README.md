# 🛰️ Pipeline de MLOps para Detección de Anomalías en Paneles Solares

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![MLflow](https://img.shields.io/badge/mlflow-%230194E2.svg?style=for-the-badge&logo=mlflow&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

> La inspección manual de granjas solares es lenta, costosa y propensa a errores. Este proyecto resuelve ese problema implementando un **pipeline de MLOps de extremo a extremo** que automatiza la detección de paneles defectuosos usando Deep Learning, listo para un entorno de producción.

---
## 🏗️ Arquitectura del Pipeline de MLOps

El sistema sigue un flujo de trabajo modular y automatizado que abarca desde la ingesta de datos hasta el registro de un modelo validado, desacoplando el entrenamiento del servicio de predicción.

![Arquitectura del Pipeline](https://i.imgur.com/tuUaE5G.png)

1.  **Ingesta y Preparación:** Un pipeline en **Mage** descarga, procesa y divide automáticamente las imágenes termográficas.
2.  **Entrenamiento y Seguimiento:** El modelo de PyTorch se entrena y cada experimento (parámetros, métricas y artefactos) se registra en **MLflow**.
3.  **Evaluación y Registro:** El modelo se evalúa contra un umbral de calidad. Si lo supera, se versiona y registra en el **Model Registry** de MLflow.
4.  **Servicio de Inferencia:** Una API construida con **FastAPI** y containerizada con **Docker** carga el modelo registrado para servir predicciones en tiempo real.

---
## 💻 Stack Tecnológico

| Categoría | Tecnologías |
| :--- | :--- |
| **Modelado y Análisis** 🧠| `🐍 Python` `🔥 PyTorch` `🐼 Pandas` `📊 Scikit-learn` `🖼️ OpenCV` |
| **API y Servicio** 🚀 | ` FastAPI` ` Uvicorn` `🐳 Docker` |
| **MLOps y Orquestación** ⚙️ | `🧪 MLflow` `🪄 Mage AI` |
| **Herramientas** 🛠️ | ` Git & GitHub` ` Visual Studio Code` |

---
## 🚀 Demo en Acción

Una demostración visual del pipeline y la API en funcionamiento.

![Demo del Proyecto](ruta/a/tu/demo.gif)

---
## 📊 Resultados

El modelo final, `ThermalNet`, alcanzó una **precisión del 88.55%** en el conjunto de prueba. La matriz de confusión muestra un buen equilibrio en la clasificación de ambas clases.

**Matriz de Confusión del Modelo:**
![Matriz de Confusión](ruta/a/tu/matriz_de_confusion.png)

---
## ⚙️ Instalación y Uso

Instrucciones detalladas para replicar el entorno y ejecutar el proyecto.

### **Instalación**
1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu_usuario/tu_repositorio.git](https://github.com/tu_usuario/tu_repositorio.git)
    cd tu_repositorio
    ```
2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate | macOS/Linux: source venv/bin/activate
    ```
3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

### **Ejecución**

#### **1. Pipeline de Entrenamiento (Mage + MLflow)**
```bash
# Iniciar el servidor de Mage
mage start solar_mage_project
 ```
#### ➡️ Abre http://localhost:6789, selecciona el pipeline y ejecútalo. Los experimentos se registrarán automáticamente en MLflow.

#### **2. Servidor de la API (Docker)**
```bash
# Construir la imagen de Docker
docker build -t solar_anomaly_api:v1 -f api/Dockerfile .

# Ejecutar el contenedor
docker run -p 8000:8000 solar_anomaly_api:v1
 ```
➡️ Abre http://127.0.0.1:8000/docs para interactuar con la API.

---
## 👤 Contacto
Cristopher Hernández Romanos

* LinkedIn: [Cristopher Hernandez Romanos](https://www.linkedin.com/in/cristopherhr/)

* GitHub: [Cdhernandezr](https://github.com/Cdhernandezr)
