# 🔍 Sistema de Detección de Anomalías en Paneles Solares con Deep Learning

Este proyecto utiliza técnicas de visión por computadora y aprendizaje profundo para detectar defectos en paneles solares a partir de imágenes termográficas infrarrojas.

## 📁 Estructura del Proyecto

- `solar-anomaly-detector.ipynb`: Notebook principal con todo el flujo del proyecto.
- Dataset utilizado: [InfraredSolarModules - RaptorMaps](https://github.com/RaptorMaps/InfraredSolarModules)

## 🧠 Modelos Implementados

Se implementan y comparan 3 arquitecturas:

- `CompactCNN`: Red neuronal convolucional compacta.
- `ModifiedResNet`: Versión modificada de ResNet18 para imágenes en escala de grises.
- `ThermalNet`: Arquitectura personalizada con mecanismo de atención espacial.

## 📊 Resultados

- Se evalúan los modelos usando precisión, matriz de confusión y reporte de clasificación.
- Entrenamiento con early stopping y reducción dinámica de tasa de aprendizaje.

## ⚙️ Requisitos

Este proyecto se desarrolló en Google Colab. Asegúrate de tener activada la GPU en `Entorno de ejecución > Cambiar tipo de entorno`.

## 📌 Ejecución

```bash
# Solo necesitas abrir el notebook en Google Colab

## 👨‍💻 Autor

**Cristopher Hernández Romanos**  
Ingeniero Electrónico  
Universidad del Magdalena  
GitHub: [@tuusuario](https://github.com/Cdhernadnezr)  
LinkedIn: [Tu perfil](https://linkedin.com/in/cristopherhr) 
