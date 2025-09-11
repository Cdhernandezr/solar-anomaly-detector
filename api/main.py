# api/main.py

import sys
import io
import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image

# --- Configuración de Módulos ---
# Esto es crucial para que Python encuentre tus módulos en la carpeta src/
# Añadimos el directorio raíz del proyecto al path del sistema
sys.path.append('..')
from src.models import ThermalNet
from src.data_utils import DataTransforms

# --- Inicialización de la Aplicación y el Modelo ---

app = FastAPI(
    title="API de Detección de Anomalías en Paneles Solares",
    description="Una API para clasificar imágenes termográficas de paneles solares como 'Normal' o 'Defectuoso'.",
    version="1.0.0"
)

# Cargamos el modelo UNA SOLA VEZ cuando la aplicación se inicia.
# Esto es una optimización clave para que la API sea rápida.
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalNet(num_classes=2)
    
    # Ruta relativa al modelo desde la carpeta raíz del proyecto
    model_path = "models/ThermalNet_best_model.pth"
    
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval() # Ponemos el modelo en modo de evaluación

    # Obtenemos las transformaciones de validación (deben ser las mismas que en el entrenamiento)
    val_transforms = DataTransforms.get_val_transforms()
    
    print("✅ Modelo y transformaciones cargados exitosamente.")

except FileNotFoundError:
    print(f"❌ ERROR: No se encontró el archivo del modelo en '{model_path}'.")
    print("Asegúrate de haber ejecutado el script de entrenamiento primero (python -m src.train).")
    model = None


# --- Definición de los Endpoints de la API ---

@app.get("/", tags=["General"])
def read_root():
    """Endpoint raíz para verificar que la API está funcionando."""
    return {"status": "OK", "message": "API activa. Visita /docs para la documentación interactiva."}


@app.post("/predict", tags=["Predicciones"])
async def predict(image_file: UploadFile = File(...)):
    """
    Recibe una imagen (🖼️), la procesa y devuelve la predicción del modelo.
    """
    if not model:
        return {"error": "El modelo no está cargado. Revisa los logs del servidor."}

    # 1. Leer el contenido del archivo de imagen en memoria
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")  # Convertir a escala de grises

    # 2. Aplicar las mismas transformaciones que en la validación
    image_tensor = val_transforms(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # 3. Realizar la predicción
    with torch.no_grad(): # Desactivamos el cálculo de gradientes para la inferencia
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities.max().item()

    label_map = {0: 'Normal', 1: 'Defectuoso'}
    
    # 4. Devolver un resultado claro en formato JSON
    return {
        "filename": image_file.filename,
        "predicted_class": label_map[prediction_idx],
        "confidence": f"{confidence:.4f}"
    }