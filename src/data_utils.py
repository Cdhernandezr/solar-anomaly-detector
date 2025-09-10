import os
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from .utils import Logger
from .config import Config

class SolarDataManager:
    """Gestor de datos para el dataset de paneles solares"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.load_metadata()

    def load_metadata(self):
        """Carga y procesa los metadatos del dataset"""
        metadata_path = os.path.join(self.dataset_path, 'module_metadata.json')
        self.df = pd.read_json(metadata_path, orient='index').sort_index()

        # Construir rutas completas
        self.df['full_path'] = self.df['image_filepath'].apply(
            lambda x: os.path.join(self.dataset_path, x)
        )

        # Crear etiquetas binarias
        self.df['defect_status'] = self.df['anomaly_class'].apply(
            lambda x: 'Defective' if x != 'No-Anomaly' else 'Normal'
        )

        Logger.log(f"Dataset cargado: {len(self.df)} imágenes", 'success')
        self.show_distribution()

    def show_distribution(self):
        """Muestra la distribución de clases en el dataset"""
        Logger.log("📊 Distribución de clases:", 'info')
        distribution = self.df['defect_status'].value_counts()
        for status, count in distribution.items():
            percentage = (count / len(self.df)) * 100
            print(f"  • {status}: {count} ({percentage:.1f}%)")

    def create_stratified_split(self):
        """Crea división estratificada del dataset"""
        # Primera división: train+val vs test
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=Config.TEST_RATIO,
            stratify=self.df['defect_status'],
            random_state=Config.RANDOM_SEED
        )

        # Segunda división: train vs val
        val_size_adjusted = Config.VAL_RATIO / (Config.TRAIN_RATIO + Config.VAL_RATIO)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df['defect_status'],
            random_state=Config.RANDOM_SEED
        )

        Logger.log(f"División completada - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}", 'success')
        return train_df, val_df, test_df


class ThermalImageDataset(Dataset):
    """Dataset personalizado para imágenes termográficas de paneles solares"""

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_encoder = {'Normal': 0, 'Defective': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Cargar imagen
        img_path = self.df.iloc[idx]['full_path']
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")

        # Convertir a PIL para transformaciones
        image = Image.fromarray(image)

        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)

        # Obtener etiqueta
        label = self.label_encoder[self.df.iloc[idx]['defect_status']]

        return image, label

class DataTransforms:
    """Transformaciones de datos para entrenamiento y validación"""

    @staticmethod
    def get_train_transforms():
        return transforms.Compose([
            transforms.Resize((48, 32)),  # Redimensionar para mejor procesamiento
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    @staticmethod
    def get_val_transforms():
        return transforms.Compose([
            transforms.Resize((48, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
