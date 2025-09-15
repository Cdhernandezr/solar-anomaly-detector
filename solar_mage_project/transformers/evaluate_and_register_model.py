# Importaciones necesarias para este bloque
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# El decorador @transformer indica que este bloque transforma datos
@transformer
def evaluate_and_register_model(trainer, *args, **kwargs):
    """
    Evalúa el modelo entrenado en el conjunto de prueba y, si es suficientemente bueno,
    lo registra en el Model Registry de MLflow.
    
    Args:
        trainer: El objeto ModelTrainer que contiene el modelo entrenado del bloque anterior.
    """
    print("🧪 Iniciando evaluación del modelo final...")
    
    # --- 1. Evaluar el modelo ---
    # La función evaluate_on_test devuelve las métricas y la figura de la matriz de confusión
    final_metrics, confusion_matrix_fig = trainer.evaluate_on_test()
    
    # --- 2. Registrar métricas y artefactos finales en el "Run" activo de MLflow ---
    # MLflow sigue registrando en el mismo "Run" iniciado en el bloque anterior
    
    print("📊 Registrando métricas y artefactos finales en MLflow...")
    mlflow.log_metrics({
        "final_accuracy": final_metrics['accuracy'],
        "final_precision": final_metrics['precision'],
        "final_recall": final_metrics['recall'],
        "final_f1_score": final_metrics['f1_score']
    })
    
    # Guarda la figura de la matriz de confusión
    mlflow.log_figure(confusion_matrix_fig, "confusion_matrix.png")
    
    # Guarda el modelo usando la integración de MLflow con PyTorch
    # Esto es crucial para poder registrarlo después
    mlflow.pytorch.log_model(trainer.model, "model")

    print("✅ Modelo, métricas y artefactos registrados en el Run de MLflow.")

    # --- 3. Lógica de MLOps: El Control de Calidad y Registro del Modelo ---
    
    ACCURACY_THRESHOLD = 0.88  # Definimos nuestro umbral de calidad
    
    print(f"⚖️ Verificando si la precisión ({final_metrics['accuracy']:.4f}) supera el umbral ({ACCURACY_THRESHOLD})...")
    
    if final_metrics['accuracy'] >= ACCURACY_THRESHOLD:
        print(f"👍 ¡El modelo superó el umbral! Registrando en el Model Registry...")
        
        # Obtenemos el ID del Run actual
        run_id = mlflow.active_run().info.run_id
        
        # La URI (dirección) del modelo que acabamos de guardar como artefacto
        model_uri = f"runs:/{run_id}/model"
        
        # Registramos el modelo dándole un nombre único en el registro
        mlflow.register_model(
            model_uri=model_uri,
            name="solar_anomaly_detector"
        )
        
        print("🎉 ¡Modelo registrado exitosamente!")
    else:
        print("👎 El modelo no superó el umbral de calidad. No se registrará.")
        
    return {"accuracy": final_metrics['accuracy'], "threshold": ACCURACY_THRESHOLD}