# src/train.py
import os
import zipfile
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Importamos nuestros módulos locales
from .config import Config
from .data_utils import SolarDataManager, ThermalImageDataset, DataTransforms
from .models import ThermalNet # Nos enfocaremos en el mejor modelo


class Logger:
    """Sistema de logging personalizado para el proyecto"""
    STYLES = {
        'header': '\033[95m\033[1m',
        'info': '\033[94m',
        'success': '\033[92m',
        'warning': '\033[93m',
        'error': '\033[91m',
        'bold': '\033[1m',
        'end': '\033[0m'
    }

    @classmethod
    def log(cls, message, style='info'):
        print(f"{cls.STYLES[style]}{message}{cls.STYLES['end']}")

    @classmethod
    def section(cls, title):
        print(f"\n{cls.STYLES['header']}{'='*80}")
        print(f"{title}")
        print(f"{'='*80}{cls.STYLES['end']}")


def plot_model_history(model_name, train_history, val_history):

    train_color = '#1f77b4'
    val_color = '#ff7f0e'
    epochs = range(1, len(train_history['accuracy']) + 1)

    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Historial de entrenamiento - {model_name}", fontsize=16, fontweight='bold')

    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_history['accuracy'], linestyle='--', color=train_color, label='Entrenamiento')
    plt.plot(epochs, val_history['accuracy'], linestyle='-', color=val_color, label='Validación')
    plt.xlabel('Épocas', fontsize=12, fontweight='bold')
    plt.ylabel('Precisión (%)', fontsize=12, fontweight='bold')
    plt.title('Evolución de precisión', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_history['loss'], linestyle='--', color=train_color, label='Entrenamiento')
    plt.plot(epochs, val_history['loss'], linestyle='-', color=val_color, label='Validación')
    plt.xlabel('Épocas', fontsize=12, fontweight='bold')
    plt.ylabel('Pérdida', fontsize=12, fontweight='bold')
    plt.title('Evolución de pérdida', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

class ModelTrainer:
    class ModelTrainer:
    """Clase para entrenar y evaluar modelos de clasificación"""

    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mover modelo al dispositivo
        self.model.to(self.device)

        # Configurar optimizador y criterio
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )

        # Métricas de seguimiento
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        Logger.log(f"Entrenador configurado - Dispositivo: {self.device}", 'info')

    def train_epoch(self):
        """Entrena el modelo por una época"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self):
        """Evalúa el modelo en el conjunto de validación"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train_model(self, epochs=Config.MAX_EPOCHS):
        """Entrena el modelo con early stopping"""
        Logger.section(f"ENTRENANDO {self.model.__class__.__name__}")

        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Entrenar
            train_loss, train_acc = self.train_epoch()
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)

            # Validar
            val_loss, val_acc = self.validate()
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)

            # Actualizar scheduler
            self.scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start

            # Mostrar progreso
            Logger.log(
                f"Época {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                f"Tiempo: {epoch_time:.1f}s",
                'info'
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss, val_acc)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    Logger.log("Early stopping activado", 'warning')
                    break

        total_time = time.time() - start_time
        Logger.log(f"Entrenamiento completado en {total_time:.1f}s", 'success')

        return self.train_history, self.val_history

    def save_checkpoint(self, epoch, val_loss, val_acc):
        """Guarda el mejor checkpoint del modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }

        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        filename = os.path.join(models_dir, f"{self.model.__class__.__name__}_best_model.pth")
        torch.save(checkpoint, filename)
        Logger.log(f"Checkpoint guardado: {filename}", 'success')

    def evaluate_on_test(self):
        """Evalúa el modelo en el conjunto de prueba"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        # Calcular métricas
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100

        # Mostrar matriz de confusión
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Defective'],
                   yticklabels=['Normal', 'Defective'])
        plt.title(f'Matriz de Confusión - {self.model.__class__.__name__}')
        plt.ylabel('Valor Real')
        plt.xlabel('Predicción')
        plt.show()

        plot_model_history(
        self.model.__class__.__name__,
        self.train_history,
        self.val_history
        )
        # Reporte de clasificación
        report = classification_report(all_labels, all_preds,
                                     target_names=['Normal', 'Defective'])
        print(f"\nReporte de Clasificación:\n{report}")

        return accuracy


def download_and_extract_dataset():
    """Descarga y extrae el dataset de paneles solares infrarrojos"""
    url = "https://github.com/RaptorMaps/InfraredSolarModules/raw/master/2020-02-14_InfraredSolarModules.zip"
    
    # Crear la carpeta ../data/ si no existe
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    zip_path = os.path.join(data_dir, "solar_dataset.zip")
    extract_path = data_dir

    Logger.log("📥 Descargando dataset...", 'info')
    os.system(f"wget {url} -O {zip_path}")

    Logger.log("📂 Extrayendo archivos...", 'info')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    return os.path.join(data_dir, "InfraredSolarModules")

def run_training():
    """Función principal que encapsula el proceso de entrenamiento."""
    Logger.section("INICIANDO PROCESO DE ENTRENAMIENTO")

    # Configurar semilla para reproducibilidad
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    # Preparar datos
    dataset_path = download_and_extract_dataset()
    data_manager = SolarDataManager(dataset_path)
    train_df, val_df, test_df = data_manager.create_stratified_split()

    # Crear datasets y dataloaders
    train_transforms = DataTransforms.get_train_transforms()
    val_transforms = DataTransforms.get_val_transforms()
    
    train_dataset = ThermalImageDataset(train_df, train_transforms)
    val_dataset = ThermalImageDataset(val_df, val_transforms)
    test_dataset = ThermalImageDataset(test_df, val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Entrenar el mejor modelo: ThermalNet
    thermal_net_model = ThermalNet(num_classes=2)
    trainer = ModelTrainer(thermal_net_model, train_loader, val_loader, test_loader)
    
    Logger.section(f"ENTRENANDO MODELO: {thermal_net_model.__class__.__name__}")
    trainer.train_model()
    
    Logger.section(f"EVALUANDO MODELO: {thermal_net_model.__class__.__name__}")
    trainer.evaluate_on_test()
    
    Logger.section("PROCESO DE ENTRENAMIENTO FINALIZADO")

# Este bloque permite que el script sea ejecutable desde la terminal
if __name__ == '__main__':
    run_training()